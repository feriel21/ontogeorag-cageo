#!/usr/bin/env python3
"""154_table5_ablation.py — Table 5 du manuscrit, par un script minimal et sans chemin personnel.

POURQUOI
    67_paper_results.py produit la Table 5, mais il contient aussi des sections
    que le manuscrit n'utilise plus (vocabulaire compté sur l'index run14,
    familles à 5 classes, précision experte « en attente ») et des chemins
    ~/ontogeorag en dur. Ce script ne garde que le calcul de la Table 5, avec
    le même code de lecture que 67, et vérifie qu'il retombe sur les mêmes valeurs.

RÈGLES FIXÉES AVANT EXÉCUTION
    T1  Pour chaque configuration : relations = nombre de triples du graphe aplati ;
        descripteurs = objets distincts (minuscules) des relations hasDescriptor ;
        rappel /34 = article_metrics["recall_vs_lb2019" ou "recall_vs_reference"]["tier12"]["hits"] ;
        rappel /26 = article_metrics["recall_vs_lb2019_orig26"]["tier12"]["hits"]. (Identique à 67.)
    T2  Nombre de requêtes lu dans le manifeste de la run (argv de l'étape extract_pass_a,
        fichier --queries compté ligne par ligne), et non recopié à la main.
    T3  Le vocabulaire du prompt n'est pas lisible dans le manifeste : il est donné
        par la configuration (commit de la run) et recopié tel quel.
    T4  L'appariement strict (lower_bound) est relevé et comparé au rappel, sans filtrer.

GARDE-FOUS (STOP)
    - dossier de sortie inexistant ;
    - les cinq configurations A–E fournies, avec graphe, métriques et manifeste ;
    - valeurs identiques à la Table 5 du manuscrit (relations, descripteurs, rappels, requêtes) ;
    - si paper_results.json de 67 est fourni : valeurs identiques à celles de 67.

USAGE (CPU)
    python -u 154_table5_ablation.py \\
        --run A=~/ontogeorag/output/run16 --run B=~/ontogeorag/output/run16B \\
        --run C=~/ontogeorag/output/run16C --run D=~/ontogeorag_D/output/run16D \\
        --run E=~/ontogeorag_D/output/run16E \\
        --check-67 ~/ontogeorag/output/paper_results_final/paper_results.json \\
        --out output/audit/154_table5
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

CONFIG = {  # T3 : vocabulaire du prompt, par configuration
    "A": "13 imposed", "B": "13 imposed", "C": "38 imposed",
    "D": "none", "E": "38 suggested",
}
MANUSCRIPT = {  # Table 5 du manuscrit : queries, relations, descriptors, /34, /26
    "A": (249, 207, 21, 25, 19), "B": (287, 209, 23, 25, 19),
    "C": (287, 226, 30, 22, 17), "D": (249, 216, 23, 21, 16),
    "E": (287, 232, 31, 18, 13),
}
FIELDS = ("queries", "relations", "descriptors", "recall34", "recall26")


def stop(msg):
    print("STOP: " + msg, flush=True)
    sys.exit(1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flat_graph(run_dir):
    cands = sorted((run_dir / "kg_final").glob("flat_*_enforced_dedup.json"))
    if len(cands) != 1:
        stop("graphe aplati introuvable ou multiple dans " + str(run_dir / "kg_final") + " : " + str(cands))
    return cands[0]


def queries_from_manifest(manifest, run_dir):
    steps = manifest.get("steps")
    steps = list(steps.values()) if isinstance(steps, dict) else steps
    for st in steps:
        if st.get("step") == "extract_pass_a":
            argv = st.get("argv")
            argv = argv.split() if isinstance(argv, str) else list(argv)
            if "--queries" not in argv:
                stop("argv sans --queries dans " + str(run_dir))
            qpath = Path(argv[argv.index("--queries") + 1]).expanduser()
            return qpath
    stop("étape extract_pass_a absente du manifeste de " + str(run_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="LABEL=dossier de la run (A à E)")
    ap.add_argument("--queries-dir", default=None,
                    help="dossier où chercher le fichier de requêtes par son nom si le chemin du manifeste n'existe pas")
    ap.add_argument("--check-67", default=None, help="paper_results.json produit par 67 (optionnel)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out).expanduser()
    if out.exists():
        stop("le dossier de sortie existe déjà : " + str(out))
    runs = {}
    for item in a.run:
        if "=" not in item:
            stop("--run attend LABEL=dossier : " + item)
        k, v = item.split("=", 1)
        runs[k] = Path(v).expanduser()
    if sorted(runs) != sorted(CONFIG):
        stop("configurations fournies " + str(sorted(runs)) + " au lieu de A–E")

    rows, sources, problems = {}, {}, []
    for k in sorted(CONFIG):
        d = runs[k]
        metrics_p = d / "kg_final" / "article_metrics.json"
        manifest_p = d / "manifest.json"
        for need in (metrics_p, manifest_p):
            if not need.is_file():
                stop("fichier absent : " + str(need))
        flat_p = flat_graph(d)
        m, flat, man = load_json(metrics_p), load_json(flat_p), load_json(manifest_p)
        t = flat["triples"]
        r34 = m.get("recall_vs_lb2019") or m.get("recall_vs_reference") or {}
        r26 = m.get("recall_vs_lb2019_orig26", {})
        qpath = queries_from_manifest(man, d)
        if not qpath.is_file() and a.queries_dir:
            qpath = Path(a.queries_dir).expanduser() / qpath.name
        if not qpath.is_file():
            stop("fichier de requêtes introuvable : " + str(qpath))
        n_queries = sum(1 for line in open(qpath, encoding="utf-8") if line.strip())
        rows[k] = {
            "config": k, "vocabulary": CONFIG[k], "queries": n_queries,
            "relations": len(t),
            "descriptors": len({(x.get("object") or "").lower() for x in t if x["relation"] == "hasDescriptor"}),
            "recall34": (r34.get("tier12") or {}).get("hits"),
            "recall26": (r26.get("tier12") or {}).get("hits"),
            "recall34_strict": ((r34.get("tier12") or {}).get("lower_bound") or {}).get("hits"),
            "recall26_strict": ((r26.get("tier12") or {}).get("lower_bound") or {}).get("hits"),
            "git_commit": man.get("git_commit"),
        }
        sources[k] = {"metrics": str(metrics_p), "metrics_sha256": sha256(metrics_p),
                      "flat": str(flat_p), "flat_sha256": sha256(flat_p),
                      "manifest": str(manifest_p), "queries_file": str(qpath),
                      "queries_sha256": sha256(qpath)}
        got = tuple(rows[k][f] for f in FIELDS)
        if got != MANUSCRIPT[k]:
            problems.append(k + " : obtenu " + str(dict(zip(FIELDS, got))) + " ; manuscrit " + str(dict(zip(FIELDS, MANUSCRIPT[k]))))

    if a.check_67:
        pr = load_json(Path(a.check_67).expanduser())["runs"]
        for k in sorted(CONFIG):
            v = pr.get(k)
            if v is None:
                problems.append("67 : configuration absente " + k)
                continue
            pairs = [("relations", "n_triples"), ("descriptors", "n_descriptors_distinct"),
                     ("recall34", "recall34_t12"), ("recall26", "recall26_t12"), ("queries", "queries")]
            for mine, theirs in pairs:
                if rows[k][mine] != v.get(theirs):
                    problems.append("67 : " + k + " " + mine + " " + str(rows[k][mine]) + " != " + str(v.get(theirs)))
    if problems:
        stop("\n  - " + "\n  - ".join(problems))

    out.mkdir(parents=True)
    with open(out / "table5_ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows["A"].keys()))
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))
    tex = ["% Table 5 — produit par 154_table5_ablation.py", "\\begin{tabular}{clrrrrr}", "\\toprule",
           " & Descriptor vocabulary & Queries & Relations & Descriptors & Recovered (of 34) & Recovered (of 26) \\\\",
           "\\midrule"]
    for k in sorted(rows):
        r = rows[k]
        tex.append("%s & %s & %d & %d & %d & %d & %d \\\\" % (k, r["vocabulary"], r["queries"], r["relations"],
                                                             r["descriptors"], r["recall34"], r["recall26"]))
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "table5_ablation.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    strict_equal = {k: (rows[k]["recall34"] == rows[k]["recall34_strict"],
                        rows[k]["recall26"] == rows[k]["recall26_strict"]) for k in rows}
    report = {"script": "154_table5_ablation.py", "rows": rows, "sources": sources,
              "strict_equals_reported": strict_equal,
              "checked_against_manuscript": True, "checked_against_67": bool(a.check_67)}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for k in sorted(rows):
        r = rows[k]
        print("%s  %-13s q=%d rel=%d desc=%d /34=%s (strict %s) /26=%s (strict %s)"
              % (k, r["vocabulary"], r["queries"], r["relations"], r["descriptors"],
                 r["recall34"], r["recall34_strict"], r["recall26"], r["recall26_strict"]), flush=True)
    print("sortie: " + str(out), flush=True)


if __name__ == "__main__":
    main()