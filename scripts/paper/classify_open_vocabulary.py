#!/usr/bin/env python3
"""Classification of the descriptor values extracted outside the extraction vocabulary (Section 3.6).

Reads the descriptor values that schema enforcement rejected as outside the vocabulary
(hasDescriptor triples with reject reason type_constraint, in
pass_a/rejected_triples_v5.jsonl and pass_b/rejected_triples_v5.jsonl of a run) and
assigns each term to one family.

Families
    not vocabulary     pdf_artifact (undecoded ligature, unjoined hyphenation),
                       extraction_noise (bare adjectives, quantifiers, verb phrases),
                       morphometric (numeric measurements, which the ontology has no
                       numeric attribute to record)
    geological signal  deformation_structure, basal_surface_marking,
                       stratigraphic_termination, seismic_attribute, amplitude_variant,
                       section_geometry, lithology_texture

Assignment. A table of terms classified by name (EXPLICIT) takes precedence; otherwise
ordered regular-expression rules are applied and the first match wins. Artefacts and
noise are tested first, so that a noisy term containing a geological keyword is not
counted as signal. The table and the rules were written by the authors: the grouping is
a judgement, released here so that it can be inspected and changed.

Published results
    configuration D (no vocabulary in the prompt): 118 terms, 287 occurrences;
    84 geological terms, 225 occurrences, in seven families; 34 not vocabulary
    (7 morphometric, 24 extraction noise, 3 PDF artefacts)
    configuration E (vocabulary suggested): 42 terms, 39 geological (126 occurrences)
    and 3 extraction noise

Outputs (--out): vocabulary_classification.json, vocabulary_classification.csv (with an
empty family_expert column for review), fig_observed_vocabulary.png and .pdf. That
figure is a working view, not a figure of the article.

Historical name: 68_classify_vocabulary.py

Usage
    python scripts/paper/classify_open_vocabulary.py --run /path/to/run16D --out output/vocabulary_D
    python scripts/paper/classify_open_vocabulary.py --run /path/to/run16E --out output/vocabulary_E
"""
import argparse, collections, csv, json, re, unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# --- terms classified by name (take precedence over the rules) ---------
EXPLICIT = {
    # noise: bare adjectives, quantifiers, verb phrases
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
    # signal: seismic attribute
    "high-frequency": "seismic_attribute",
    "highly reflective": "seismic_attribute",
    "high reflection strength": "seismic_attribute",
    "translucent": "seismic_attribute",
    "coherent": "seismic_attribute",
    # signal: stratigraphic termination
    "onlaps": "stratigraphic_termination",
    "downlap": "stratigraphic_termination",
    "unconformity": "stratigraphic_termination",
    "incisional": "stratigraphic_termination",
    "incised systems": "stratigraphic_termination",
    "lap out against high-relief areas": "stratigraphic_termination",
    # signal: section geometry
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

# --- ordered rules: (family, regular expression); first match wins -----
RULES = [
    # 1. PDF extraction artefacts
    ("pdf_artifact", r"hyper-\s*bolic|\uFB01|\uFB02|\u00AD"),
    # 2. numeric measurements
    ("morphometric", r"\d|\brun-?out\b|\bup to\b|\bthickness of\b|\bkm\b|\bm across\b"),
    # 3. blocks and deformation structures
    ("deformation_structure",
     r"\bblock|\bglide\b|\brafted\b|\boutrunner\b|listric|thrust|\bramp\b|duplex|"
     r"compressi|extensional|imbricat|\bfold|pressure ridge|\bshear\b|\bfault|"
     r"tilted|inclined|slump|rotated|fractured|broken|disrupt"),
    # 4. basal-surface markings
    ("basal_surface_marking",
     r"groov|striation|longitudinal linear|flow line|lineation|scour|gouge"),
    # 5. stratigraphic terminations
    ("stratigraphic_termination",
     r"onlap|downlap|toplap|truncat|unconform|incis|lap out"),
    # 6. seismic attributes other than amplitude and continuity
    ("seismic_attribute",
     r"frequen|reflectiv|reflection strength|translucen|coheren|acoustic"),
    # 7. amplitude variants
    ("amplitude_variant", r"amplitude|reflector|bright|\bdim\b|weak|strong"),
    # 8. section geometry
    ("section_geometry",
     r"tapering|dipping flank|cross-section|topograph|vertical surface|"
     r"lens|wedge|sheet|lobate|arcuate|planar|bulge|flat-lay"),
    # 9. lithology and texture
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
            raise SystemExit(f"not found: {f}")
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

    print(f"terms observed outside the vocabulary: {len(obs)}  "
          f"({sum(obs.values())} occurrences)\n")
    print("--- GEOLOGICAL SIGNAL ---")
    for f in sorted(GEOLOGICAL, key=lambda k: -sum(n for _, n in fam.get(k, []))):
        if f in fam:
            print(f"  {f:28s} {len(fam[f]):3d} terms / {sum(n for _, n in fam[f]):3d} occ."
                  f"   ex: {', '.join(t for t, _ in fam[f][:3])[:60]}")
    print(f"  {'TOTAL':28s} {g_terms:3d} terms / {g_occ:3d} occ.\n")
    print("--- NOT VOCABULARY (excluded from the geological result) ---")
    for f in sorted(NON_VOCABULARY, key=lambda k: -sum(n for _, n in fam.get(k, []))):
        if f in fam:
            print(f"  {f:28s} {len(fam[f]):3d} terms / {sum(n for _, n in fam[f]):3d} occ."
                  f"   ex: {', '.join(t for t, _ in fam[f][:3])[:60]}")
    print(f"  {'TOTAL':28s} {n_terms:3d} terms / {n_occ:3d} occ.\n")
    print(f"--- UNCLASSIFIED: {u_terms} terms / {u_occ} occ.")
    for t, n in fam.get("unclassified", [])[:20]:
        print(f"      {n:3d}  {t}")

    # --- review CSV --------------------------------------------------
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

    print(f"\nwritten: {out}/vocabulary_classification.json")
    print(f"written: {out}/vocabulary_classification.csv  (family_expert column left empty for review)")
    print(f"written: {out}/fig_observed_vocabulary.png|.pdf")


if __name__ == "__main__":
    main()