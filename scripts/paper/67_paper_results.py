#!/usr/bin/env python3
"""
67_paper_results.py — Tous les resultats du manuscrit, en une passe

Lit les runs sur disque et produit :
    paper_results.json    chaque chiffre, avec le fichier qui l'atteste
    tables/*.tex          tableaux prets a \\input
    figures/*.png|.pdf    figure d'ablation
    MISSING.txt           ce qui n'a pas pu etre calcule, et pourquoi

REGLE : rien n'est invente. Un resultat introuvable est marque MISSING avec
le chemin attendu. Chaque valeur du JSON porte sa source.

Sections couvertes (numerotation du plan de manuscrit) :
    3.1  graphe livre
    3.2  rappel contre benchmark expert
    3.3  ablation quatre configurations
    3.4  vocabulaire descriptif du corpus
    3.5  ce que l'ontologie ne couvre pas
    3.6  metrologie du verificateur
    3.7  erreurs de direction et de type de relation
    3.8  validation structurelle du graphe

Usage :
    cd ~/ontogeorag && export PYTHONPATH=$PWD
    python 67_paper_results.py --out output/paper_results
"""
import argparse, collections, json, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# --- configurations comparees ------------------------------------------
RUNS = {
    "run15":  {"root": "~/ontogeorag/output/run15",       "flat": "flat_run15_enforced_dedup.json",
               "label": "run15 (repaired run14)", "queries": 249, "prompt": "13 imposed"},
    "A":      {"root": "~/ontogeorag/output/run16",       "flat": "flat_run16_enforced_dedup.json",
               "label": "A: full from PDF",       "queries": 249, "prompt": "13 imposed"},
    "B":      {"root": "~/ontogeorag/output/run16B",      "flat": "flat_run16B_enforced_dedup.json",
               "label": "B: + descriptor queries", "queries": 287, "prompt": "13 imposed"},
    "C":      {"root": "~/ontogeorag/output/run16C",      "flat": "flat_run16C_enforced_dedup.json",
               "label": "C: + extended lexicon",  "queries": 287, "prompt": "38 imposed"},
    "D":      {"root": "~/ontogeorag_D/output/run16D",    "flat": "flat_run16D_enforced_dedup.json",
               "label": "D: open vocabulary",     "queries": 249, "prompt": "open"},
    "E":      {"root": "~/ontogeorag_D/output/run16E",    "flat": "flat_run16E_enforced_dedup.json",
               "label": "E: suggested lexicon",   "queries": 287, "prompt": "suggested"},
}

# --- familles geologiques des termes hors lexique (section 3.5) --------
# Classement a valider par un geologue ; le script rapporte aussi les
# termes non classes plutot que de les forcer dans une famille.
FAMILIES = {
    "deformation_structure": [
        "rotated block", "glide block", "listric", "thrust", "ramp", "duplex",
        "compressive", "compressional", "extensional", "imbricat", "fold",
        "pressure ridge", "shear", "fault", "tilted", "inclined", "slump"],
    "basal_surface_marking": [
        "groove", "striation", "longitudinal linear", "flow line", "lineation",
        "scour", "gouge"],
    "amplitude_variant": [
        "amplitude", "reflector", "reflectivity", "bright", "dim", "weak", "strong"],
    "geometry_scale": [
        "km-scale", "lens", "wedge", "sheet", "lobate", "arcuate", "planar",
        "thickness", "thick", "thin"],
    "lithology_texture": [
        "debrite", "sand", "clast", "lamination", "interbedded", "matrix",
        "graded", "homogeneous", "structureless"],
}


def p(x):
    return Path(x).expanduser()


def jload(path, missing):
    f = p(path)
    if not f.exists():
        missing.append(str(f))
        return None
    return json.load(open(f, encoding="utf-8"))


def jl(path, missing):
    f = p(path)
    if not f.exists():
        missing.append(str(f))
        return []
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]


def family_of(term):
    t = term.lower()
    for fam, keys in FAMILIES.items():
        if any(k in t for k in keys):
            return fam
    return "unclassified"


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", default="~/ontogeorag/output/run14/step1/chunks.jsonl")
    ap.add_argument("--primary", default="D", help="run presente comme graphe livre")
    a = ap.parse_args()

    out = p(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    MISSING, R = [], {}

    # ---------------- 3.1 / 3.2 : graphe livre et rappel ---------------
    print("=== 3.1-3.2  graphe livre et rappel ===")
    per_run = {}
    for k, cfg in RUNS.items():
        root = p(cfg["root"])
        m = jload(root / "kg_final/article_metrics.json", MISSING)
        if m is None:
            m = jload(root / "kg_final/metrics_lb_extended_benchmark.json", MISSING)
        flat = jload(root / f"kg_final/{cfg['flat']}", MISSING)
        if m is None or flat is None:
            print(f"  {k:6s} MANQUANT"); continue
        t = flat["triples"]
        rels = collections.Counter(x["relation"] for x in t)
        ents = {(x.get(s) or "").lower() for x in t for s in ("subject", "object")}
        ents.discard("")
        r34 = m.get("recall_vs_lb2019") or m.get("recall_vs_reference") or {}
        r26 = m.get("recall_vs_lb2019_orig26", {})
        cov = m.get("descriptor_coverage", {})
        per_run[k] = {
            "label": cfg["label"], "queries": cfg["queries"], "prompt": cfg["prompt"],
            "n_triples": len(t), "n_entities": len(ents),
            "n_relations_instantiated": len(rels),
            "relation_counts": dict(rels.most_common()),
            "n_descriptors_distinct": len({(x.get("object") or "").lower()
                                           for x in t if x["relation"] == "hasDescriptor"}),
            "recall34_t1": (r34.get("tier1") or {}).get("hits"),
            "recall34_t12": (r34.get("tier12") or {}).get("hits"),
            "recall34_t12_strict": ((r34.get("tier12") or {}).get("lower_bound") or {}).get("hits"),
            "recall26_t1": (r26.get("tier1") or {}).get("hits"),
            "recall26_t12": (r26.get("tier12") or {}).get("hits"),
            "desc_cov_t12": (cov.get("tier12") or {}).get("n_found"),
            "desc_missing_t12": (cov.get("tier12") or {}).get("missing"),
            "source": str(root / "kg_final/article_metrics.json"),
        }
        v = per_run[k]
        print(f"  {k:6s} {v['n_triples']:4d} triples | /34 {v['recall34_t12']}/34 "
              f"| /26 {v['recall26_t12']}/26 | desc {v['desc_cov_t12']}/13 "
              f"| {v['n_descriptors_distinct']} desc. distincts")
    R["runs"] = per_run
    R["primary_run"] = a.primary

    # ---------------- 3.4 : vocabulaire canonique du corpus -----------
    print("\n=== 3.4  vocabulaire descriptif du corpus ===")
    chunks = jl(a.chunks, MISSING)
    voc = {}
    if chunks:
        from pipeline.rag.constants import KNOWN_DESCRIPTORS, LB2019_DESCRIPTORS
        prim = per_run.get(a.primary)
        used = set()
        if prim:
            flat = jload(p(RUNS[a.primary]["root"]) / f"kg_final/{RUNS[a.primary]['flat']}", MISSING)
            used = {(x.get("object") or "").lower() for x in flat["triples"]
                    if x["relation"] == "hasDescriptor"}
        for d in sorted(KNOWN_DESCRIPTORS):
            pat = re.compile(r"\b" + re.escape(d) + r"\b", re.I)
            hits = [c for c in chunks if pat.search(c.get("text", ""))]
            docs = {c.get("doc_id") or c.get("source") or c.get("paper") for c in hits}
            voc[d] = {"chunks": len(hits), "docs": len(docs),
                      "in_kg": d in used, "is_lb13": d in LB2019_DESCRIPTORS}
        dead = [d for d, v in voc.items() if v["chunks"] == 0]
        absent = [d for d, v in voc.items() if v["chunks"] > 0 and not v["in_kg"]]
        print(f"  descripteurs canoniques         : {len(voc)}")
        print(f"  jamais attestes dans le corpus  : {len(dead)} {dead}")
        print(f"  attestes mais absents du graphe : {len(absent)}")
        R["canonical_vocabulary"] = {"per_descriptor": voc, "dead": dead,
                                     "attested_but_absent": absent,
                                     "source": str(p(a.chunks))}

    # ---------------- 3.5 : vocabulaire observe (produit par 68) ------
    print("\n=== 3.5  vocabulaire observe hors ontologie ===")
    vc = jload("output/paper_results/vocabulary_E/vocabulary_classification.json",
               MISSING)
    if vc:
        print(f"  termes : {vc['n_terms']}  ({vc['n_occurrences']} occurrences)")
        print(f"  signal geologique : {vc['geological_signal']['n_terms']} termes / "
              f"{vc['geological_signal']['n_occurrences']} occ.")
        print(f"  hors vocabulaire  : {vc['not_vocabulary']['n_terms']} termes / "
              f"{vc['not_vocabulary']['n_occurrences']} occ.")
        R["observed_vocabulary"] = vc
    else:
        print("  MANQUANT — lancer 68_classify_vocabulary.py d'abord")

    # ---------------- 3.6 : metrologie du verificateur ----------------
    print("\n=== 3.6  metrologie du verificateur ===")
    neg = jload("~/ontogeorag_D/output/run16E/m4/negatives/negatives_report.json", MISSING)
    if neg:
        o = neg["overall"]
        print(f"  n={neg['n_scored']}  spec {o['specificity']:.3f}  "
              f"prec {o['precision']:.3f}  ROC-AUC {neg['roc_auc_graded_score']:.3f}")
        for cl, v in neg["per_corruption_class"].items():
            print(f"    {cl:16s} sens {v['sensitivity']:.3f}  AUC {v['roc_auc']:.3f}")
        R["verifier_metrology"] = neg

    # ---------------- 3.7 : direction --------------------------------
    print("\n=== 3.7  controle directionnel ===")
    dirsum = jload("~/ontogeorag_D/output/run16E/m4/direction_check/m4_direction_summary.json",
                   MISSING)
    if dirsum:
        print(f"  n={dirsum['n_checked']}  {dirsum['verdicts']}")
        R["direction_check"] = dirsum

    # ---------------- 3.8 : validation structurelle -------------------
    print("\n=== 3.8  validation structurelle ===")
    R["graph_validation"] = {}
    for k in ("A", "C", "D", "E"):
        gv = jload(p(RUNS[k]["root"]) / "validation/graph_validation_report.json", MISSING)
        if gv:
            n = gv["nodes"]
            red = round(100 * (1 - n["estimated_concepts"] / n["distinct_surface_forms"]), 1)
            print(f"  {k}: {n['distinct_surface_forms']} formes / "
                  f"~{n['estimated_concepts']} concepts = {red}% redondance | "
                  f"{len(gv['edges']['contradictions'])} contradictions | "
                  f"{len(gv['connectivity']['unreachable'])} entites isolees")
            R["graph_validation"][k] = {
                "surface_forms": n["distinct_surface_forms"],
                "estimated_concepts": n["estimated_concepts"],
                "redundancy_pct": red,
                "contradictions": len(gv["edges"]["contradictions"]),
                "duplicates": len(gv["edges"]["duplicates"]),
                "evidence_anchored": gv["edges"]["evidence_anchored"],
                "unreachable_entities": len(gv["connectivity"]["unreachable"]),
                "instance_leakage": len(n["instance_leakage"])}

    # ---------------- precision experte (si rendue) -------------------
    prec = jload(p(RUNS[a.primary]["root"]) / "validation/precision_report.json", MISSING)
    if prec:
        R["expert_precision"] = prec
        print(f"\n=== precision experte : {prec['precision_strict']['p']:.1%} "
              f"IC95 {prec['precision_strict']['ci95']}")
    else:
        print("\n=== precision experte : EN ATTENTE (paquet genere, verdicts non rendus)")

    # ---------------- tableaux LaTeX ----------------------------------
    tex = ["% Table: configuration ablation — genere par 67_paper_results.py",
           "\\begin{tabular}{llccccc}", "\\toprule",
           "Config & Prompt vocabulary & Queries & Triples & Distinct desc. "
           "& Recall/34 & Recall/26 \\\\", "\\midrule"]
    for k in ("A", "B", "C", "D", "E"):
        v = per_run.get(k)
        if not v:
            continue
        tex.append(f"{k} & {v['prompt']} & {v['queries']} & {v['n_triples']} & "
                   f"{v['n_descriptors_distinct']} & {v['recall34_t12']}/34 & "
                   f"{v['recall26_t12']}/26 \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables/ablation.tex").write_text("\n".join(tex), encoding="utf-8")

    if "verifier_metrology" in R:
        t2 = ["\\begin{tabular}{lccc}", "\\toprule",
              "Corruption class & Sensitivity & ROC-AUC & $n$ \\\\", "\\midrule"]
        for cl, v in R["verifier_metrology"]["per_corruption_class"].items():
            name = cl.replace("_", "\\_")
            n_cl = v["tp"] + v["fn"]
            t2.append("\\texttt{%s} & %.3f & %.3f & %d \\\\"
                      % (name, v["sensitivity"], v["roc_auc"], n_cl))
        t2 += ["\\bottomrule", "\\end{tabular}"]
        (out / "tables/verifier_metrology.tex").write_text("\n".join(t2), encoding="utf-8")

    # ---------------- figure d'ablation --------------------------------
    ks = [k for k in ("A", "B", "C", "D", "E") if k in per_run]
    if ks:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        v0 = [per_run[k]["recall34_t12"] or 0 for k in ks]
        ax[0].plot(ks, v0, "o-", color="#1565c0", linewidth=2.4, markersize=9)
        ax[0].set_ylim(min(v0) - 2, max(v0) + 2)
        for i, y in enumerate(v0):
            ax[0].annotate(f"{y}/34", (i, y), textcoords="offset points",
                           xytext=(0, 9), ha="center", fontsize=10)
        ax[0].set_ylabel("benchmark edges recovered (/34)", fontsize=11)
        ax[0].set_title("(a) Recall", fontsize=15, fontweight="bold")
        ax[1].bar(ks, [per_run[k]["n_descriptors_distinct"] for k in ks], color="#2e7d32")
        ax[1].set_ylabel("distinct descriptors in graph", fontsize=11)
        ax[1].set_title("(b) Descriptive breadth", fontsize=15, fontweight="bold")
        ax[2].bar(ks, [per_run[k]["n_triples"] for k in ks], color="#6a1b9a")
        ax[2].set_ylabel("triples in delivered graph", fontsize=11)
        ax[2].set_title("(c) Graph size", fontsize=15, fontweight="bold")
        for x in ax:
            x.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"figures/fig_ablation.{ext}", dpi=300,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"\nEcrit : {out}/figures/fig_ablation.png|.pdf")

    # ---------------- sorties ------------------------------------------
    (out / "paper_results.json").write_text(
        json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
    if MISSING:
        (out / "MISSING.txt").write_text("\n".join(sorted(set(MISSING))), encoding="utf-8")
    print(f"\nEcrit : {out}/paper_results.json")
    print(f"Ecrit : {out}/tables/*.tex")
    print(f"Fichiers manquants : {len(set(MISSING))}"
          + (f"  -> {out}/MISSING.txt" if MISSING else ""))


if __name__ == "__main__":
    main()