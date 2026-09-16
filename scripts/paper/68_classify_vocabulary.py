#!/usr/bin/env python3
"""
68_classify_vocabulary.py — Vocabulaire observe hors ontologie : classement

Les 118 termes extraits sans contrainte lexicale (run D) ne sont pas
homogenes. Trois categories ne relevent PAS du vocabulaire geologique et
doivent etre separees avant toute conclusion :

    pdf_artifact     ligature non decodee ("well stratiﬁed"), cesure non
                     recollee ("hyper-bolic") — defaut d'extraction PDF
    extraction_noise adjectifs nus, quantifieurs, syntagmes verbaux
                     ("numerous", "move downslope beyond it") — rejet correct
    morphometric     valeurs chiffrees ("run-out distance of 80 km") — ce
                     sont des MESURES, que l'ontologie ne peut pas
                     representer faute d'attribut numerique. Constat reel,
                     mais distinct d'une lacune de vocabulaire.

Le reste constitue le SIGNAL GEOLOGIQUE : des termes descriptifs que la
litterature emploie et que le vocabulaire de reference n'encode pas.

CLASSEMENT ORDONNE : les regles sont evaluees dans l'ordre, premiere
correspondance retenue. Artefacts et bruit d'abord, pour qu'un terme
bruite contenant un mot-cle geologique ne soit pas compte comme signal.

Usage :
    cd ~/ontogeorag && export PYTHONPATH=$PWD
    python 68_classify_vocabulary.py \
        --run ~/ontogeorag_D/output/run16D \
        --out output/paper_results/vocabulary
"""
import argparse, collections, csv, json, re, unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# --- termes traites nommement (priorite absolue) -----------------------
EXPLICIT = {
    # bruit : adjectifs nus, quantifieurs, syntagmes verbaux
    "numerous": "extraction_noise", "large": "extraction_noise",
    "rare": "extraction_noise", "positive": "extraction_noise",
    "coarse": "extraction_noise", "bounded": "extraction_noise",
    "amalgamate": "extraction_noise", "rotation": "extraction_noise",
    "width": "extraction_noise", "rafting": "extraction_noise",
    "downslope": "extraction_noise", "landwards": "extraction_noise",
    "smaller sizes": "extraction_noise", "limited distribution": "extraction_noise",
    "extends regionally": "extraction_noise",
    "wholesale development": "extraction_noise",
    "move downslope beyond it": "extraction_noise",
    "distinct seismic facies": "extraction_noise",
    "rather distinct seismic facies": "extraction_noise",
    "distinct sharp": "extraction_noise",
    "large reflection events": "extraction_noise",
    "heterogeneous": "extraction_noise",
    "dominantly turbidite channel–levee": "extraction_noise",
    # signal : attribut sismique
    "high-frequency": "seismic_attribute",
    "highly reflective": "seismic_attribute",
    "high reflection strength": "seismic_attribute",
    "translucent": "seismic_attribute",
    "coherent": "seismic_attribute",
    # signal : terminaison stratigraphique
    "onlaps": "stratigraphic_termination",
    "downlap": "stratigraphic_termination",
    "unconformity": "stratigraphic_termination",
    "incisional": "stratigraphic_termination",
    "incised systems": "stratigraphic_termination",
    "lap out against high-relief areas": "stratigraphic_termination",
    # signal : geometrie de section
    "bulges up": "section_geometry",
    "flat-laying": "section_geometry",
    "abrupt vertical surfaces": "section_geometry",
    "megaspread": "section_geometry",
    "sheared": "deformation_structure",
    "transparency": "amplitude_variant",
    "transverse": "deformation_structure",
    "radial": "deformation_structure",
    "low-angle": "deformation_structure",
    "longitudinal": "basal_surface_marking",
    "laminar": "lithology_texture",
    "internal": "extraction_noise",
    "landwards-dipping": "section_geometry",
    "sharp pointy features": "section_geometry",
    "seaward structural tilting": "deformation_structure",
    "sea-bed structural uplifts": "deformation_structure",
    "unfailed layers": "lithology_texture",
    "uprising overpressured shale": "lithology_texture",
    "ramps up and down stratigraphy": "extraction_noise",
}

# --- regles ordonnees : (famille, motif regex) -------------------------
RULES = [
    # 1. artefacts d'extraction PDF
    ("pdf_artifact", r"hyper-\s*bolic|\uFB01|\uFB02|\u00AD"),
    # 2. mesures chiffrees
    ("morphometric", r"\d|\brun-?out\b|\bup to\b|\bthickness of\b|\bkm\b|\bm across\b"),
    # 3. blocs et structures de deformation
    ("deformation_structure",
     r"\bblock|\bglide\b|\brafted\b|\boutrunner\b|listric|thrust|\bramp\b|duplex|"
     r"compressi|extensional|imbricat|\bfold|pressure ridge|\bshear\b|\bfault|"
     r"tilted|inclined|slump|rotated|fractured|broken|disrupt"),
    # 4. marques de la surface basale
    ("basal_surface_marking",
     r"groov|striation|longitudinal linear|flow line|lineation|scour|gouge"),
    # 5. terminaisons stratigraphiques
    ("stratigraphic_termination",
     r"onlap|downlap|toplap|truncat|unconform|incis|lap out"),
    # 6. attributs sismiques hors amplitude/continuite
    ("seismic_attribute",
     r"frequen|reflectiv|reflection strength|translucen|coheren|acoustic"),
    # 7. variantes d'amplitude
    ("amplitude_variant", r"amplitude|reflector|bright|\bdim\b|weak|strong"),
    # 8. geometrie de section
    ("section_geometry",
     r"tapering|dipping flank|cross-section|topograph|vertical surface|"
     r"lens|wedge|sheet|lobate|arcuate|planar|bulge|flat-lay"),
    # 9. lithologie et texture
    ("lithology_texture",
     r"debrit|\bsand\b|clast|lamination|interbed|matrix|graded|homogeneous|"
     r"structureless|stratified|featureless|undisturbed"),
]

GEOLOGICAL = {"deformation_structure", "basal_surface_marking",
              "stratigraphic_termination", "seismic_attribute",
              "amplitude_variant", "section_geometry", "lithology_texture"}
NON_VOCABULARY = {"pdf_artifact", "extraction_noise", "morphometric"}

COLORS = {"deformation_structure": "#1565c0", "basal_surface_marking": "#00838f",
          "stratigraphic_termination": "#2e7d32", "seismic_attribute": "#6a1b9a",
          "amplitude_variant": "#8e24aa", "section_geometry": "#0277bd",
          "lithology_texture": "#558b2f", "morphometric": "#ef6c00",
          "extraction_noise": "#9e9e9e", "pdf_artifact": "#c62828",
          "unclassified": "#424242"}


def classify(term):
    t = unicodedata.normalize("NFC", term.lower().strip())
    if t in EXPLICIT:
        return EXPLICIT[t]
    for fam, pat in RULES:
        if re.search(pat, t, re.I):
            return fam
    return "unclassified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root, out = Path(a.run).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    obs = collections.Counter()
    for ps in ("pass_a", "pass_b"):
        f = root / ps / "rejected_triples_v5.jsonl"
        if not f.exists():
            raise SystemExit(f"introuvable : {f}")
        for line in f.open(encoding="utf-8"):
            t = json.loads(line)
            if t.get("_reject_reason") == "type_constraint" and \
               (t.get("relation_norm") or t.get("relation")) == "hasDescriptor":
                obs[(t.get("target_norm") or t.get("target", "")).lower().strip()] += 1

    fam = collections.defaultdict(list)
    for term, n in obs.items():
        fam[classify(term)].append((term, n))
    for f in fam:
        fam[f].sort(key=lambda x: -x[1])

    def tot(keys):
        return (sum(len(fam[k]) for k in keys if k in fam),
                sum(n for k in keys if k in fam for _, n in fam[k]))

    g_terms, g_occ = tot(GEOLOGICAL)
    n_terms, n_occ = tot(NON_VOCABULARY)
    u_terms, u_occ = tot({"unclassified"})

    print(f"termes observes hors lexique : {len(obs)}  "
          f"({sum(obs.values())} occurrences)\n")
    print("--- SIGNAL GEOLOGIQUE ---")
    for f in sorted(GEOLOGICAL, key=lambda k: -sum(n for _, n in fam.get(k, []))):
        if f in fam:
            print(f"  {f:28s} {len(fam[f]):3d} termes / {sum(n for _, n in fam[f]):3d} occ."
                  f"   ex: {', '.join(t for t, _ in fam[f][:3])[:60]}")
    print(f"  {'TOTAL':28s} {g_terms:3d} termes / {g_occ:3d} occ.\n")
    print("--- HORS VOCABULAIRE (a exclure du resultat geologique) ---")
    for f in sorted(NON_VOCABULARY, key=lambda k: -sum(n for _, n in fam.get(k, []))):
        if f in fam:
            print(f"  {f:28s} {len(fam[f]):3d} termes / {sum(n for _, n in fam[f]):3d} occ."
                  f"   ex: {', '.join(t for t, _ in fam[f][:3])[:60]}")
    print(f"  {'TOTAL':28s} {n_terms:3d} termes / {n_occ:3d} occ.\n")
    print(f"--- NON CLASSES : {u_terms} termes / {u_occ} occ.")
    for t, n in fam.get("unclassified", [])[:20]:
        print(f"      {n:3d}  {t}")

    # --- CSV de relecture -------------------------------------------
    with (out / "vocabulary_classification.csv").open("w", newline="",
                                                      encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "occurrences", "family_auto", "is_geological_signal",
                    "family_expert", "comment"])
        for famname, items in sorted(fam.items()):
            for t, n in items:
                w.writerow([t, n, famname, famname in GEOLOGICAL, "", ""])

    # --- figure -----------------------------------------------------
    order = ([f for f in sorted(GEOLOGICAL, key=lambda k: -sum(n for _, n in fam.get(k, [])))
              if f in fam]
             + [f for f in ("morphometric", "extraction_noise", "pdf_artifact")
                if f in fam]
             + (["unclassified"] if "unclassified" in fam else []))
    labels = [f.replace("_", " ") for f in order]
    counts = [sum(n for _, n in fam[f]) for f in order]
    nterms = [len(fam[f]) for f in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = range(len(order))
    ax.barh(list(y), counts, color=[COLORS.get(f, "#777") for f in order], height=0.68)
    for i, (c, t) in enumerate(zip(counts, nterms)):
        ax.text(c + max(counts) * 0.012, i, f"{c} occ. / {t} terms",
                va="center", fontsize=9.5, color="#333")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.28)
    ax.set_xlabel("occurrences in extracted triples", fontsize=12)
    ax.set_title("Descriptive terms outside reference vocabulary",
                 fontsize=17, fontweight="bold", pad=12)
    if len(order) > len(GEOLOGICAL & set(fam)):
        ax.axhline(len(GEOLOGICAL & set(fam)) - 0.5, color="#b0b0b0",
                   linestyle="--", linewidth=1.2)
        ax.text(max(counts) * 1.26, len(GEOLOGICAL & set(fam)) - 0.62,
                "geological signal", ha="right", fontsize=10, style="italic",
                color="#555")
        ax.text(max(counts) * 1.26, len(GEOLOGICAL & set(fam)) - 0.30,
                "not vocabulary", ha="right", fontsize=10, style="italic",
                color="#555")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_observed_vocabulary.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (out / "vocabulary_classification.json").write_text(json.dumps({
        "run": str(root), "n_terms": len(obs), "n_occurrences": sum(obs.values()),
        "geological_signal": {"n_terms": g_terms, "n_occurrences": g_occ},
        "not_vocabulary": {"n_terms": n_terms, "n_occurrences": n_occ},
        "unclassified": {"n_terms": u_terms, "n_occurrences": u_occ},
        "by_family": {f: {"n_terms": len(v), "n_occurrences": sum(n for _, n in v),
                          "terms": dict(v)} for f, v in fam.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nEcrit : {out}/vocabulary_classification.json")
    print(f"Ecrit : {out}/vocabulary_classification.csv  (colonne family_expert a relire)")
    print(f"Ecrit : {out}/fig_observed_vocabulary.png|.pdf")


if __name__ == "__main__":
    main()