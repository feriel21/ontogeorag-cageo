#!/usr/bin/env python3
"""Reproduce Table 5 (ablation across configurations A-E).

For each configuration the script reads three files produced by the pipeline
run, and nothing else:

    <run>/kg_final/flat_*_enforced_dedup.json   delivered graph
    <run>/kg_final/article_metrics.json         benchmark recall (stage 07)
    <run>/manifest.json                         command lines of the run

Columns
    queries      lines in the query file passed to extraction (read from the manifest)
    relations    triples in the delivered graph
    descriptors  distinct lower-cased objects of hasDescriptor relations
    recall34     recalled edges of the 34-edge benchmark, Tier 1+2
    recall26     recalled edges of the 26-edge benchmark, Tier 1+2

Recall uses the headline matcher of stage 07 (subject and object matched by
containment after normalisation, relation normalised). The strict matcher
(exact tuple equality) is reported alongside it in the CSV.

The script stops if any value differs from the published table, or from a
previous results file when --check-previous is given.

Usage
    python scripts/paper/table5_ablation.py \\
        --run A=output/run16  --run B=output/run16B --run C=output/run16C \\
        --run D=output/run16D --run E=output/run16E --out output/table5
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

# Descriptor vocabulary supplied in the extraction prompt. Not recorded in the
# manifest: it is fixed by the commit of each run.
VOCABULARY = {"A": "13 imposed", "B": "13 imposed", "C": "38 imposed",
              "D": "none", "E": "38 suggested"}

# Published Table 5: queries, relations, descriptors, recall /34, recall /26.
PUBLISHED = {"A": (249, 207, 21, 25, 19), "B": (287, 209, 23, 25, 19),
             "C": (287, 226, 30, 22, 17), "D": (249, 216, 23, 21, 16),
             "E": (287, 232, 31, 18, 13)}
FIELDS = ("queries", "relations", "descriptors", "recall34", "recall26")


def fail(msg):
    print("ERROR: " + msg, flush=True)
    sys.exit(1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_flat_graph(run_dir):
    found = sorted((run_dir / "kg_final").glob("flat_*_enforced_dedup.json"))
    if len(found) != 1:
        fail("expected one flat graph in %s, found %s" % (run_dir / "kg_final", found))
    return found[0]


def query_file_from_manifest(manifest, run_dir, queries_dir):
    """Return the --queries file of the extraction step (pass a)."""
    steps = manifest["steps"]
    steps = list(steps.values()) if isinstance(steps, dict) else steps
    for step in steps:
        if step.get("step") == "extract_pass_a":
            argv = step["argv"].split() if isinstance(step["argv"], str) else list(step["argv"])
            if "--queries" not in argv:
                fail("no --queries argument in the manifest of " + str(run_dir))
            path = Path(argv[argv.index("--queries") + 1]).expanduser()
            # Absolute paths in the manifest point to the original machine;
            # fall back to the file of the same name in --queries-dir.
            if not path.is_file() and queries_dir:
                path = Path(queries_dir) / path.name
            if not path.is_file():
                fail("query file not found: " + str(path))
            return path
    fail("step extract_pass_a missing from the manifest of " + str(run_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                    help="run directory of one configuration (A to E)")
    ap.add_argument("--queries-dir", default="configs",
                    help="directory holding the query files (default: configs)")
    ap.add_argument("--check-previous", default=None,
                    help="optional paper_results.json from an earlier run, to compare against")
    ap.add_argument("--out", required=True, help="output directory (must not exist)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        fail("output directory already exists: " + str(out))
    runs = {}
    for item in args.run:
        label, _, directory = item.partition("=")
        runs[label] = Path(directory).expanduser()
    if sorted(runs) != sorted(VOCABULARY):
        fail("expected configurations A-E, got " + str(sorted(runs)))

    rows, sources, problems = {}, {}, []
    for label in sorted(VOCABULARY):
        run_dir = runs[label]
        metrics_path = run_dir / "kg_final" / "article_metrics.json"
        manifest_path = run_dir / "manifest.json"
        for path in (metrics_path, manifest_path):
            if not path.is_file():
                fail("missing file: " + str(path))
        flat_path = find_flat_graph(run_dir)
        metrics, graph, manifest = read_json(metrics_path), read_json(flat_path), read_json(manifest_path)
        triples = graph["triples"]
        r34 = metrics.get("recall_vs_lb2019") or metrics.get("recall_vs_reference") or {}
        r26 = metrics.get("recall_vs_lb2019_orig26", {})
        query_path = query_file_from_manifest(manifest, run_dir, args.queries_dir)

        rows[label] = {
            "config": label,
            "vocabulary": VOCABULARY[label],
            "queries": sum(1 for line in open(query_path, encoding="utf-8") if line.strip()),
            "relations": len(triples),
            "descriptors": len({(t.get("object") or "").lower()
                                for t in triples if t["relation"] == "hasDescriptor"}),
            "recall34": (r34.get("tier12") or {}).get("hits"),
            "recall26": (r26.get("tier12") or {}).get("hits"),
            "recall34_strict": ((r34.get("tier12") or {}).get("lower_bound") or {}).get("hits"),
            "recall26_strict": ((r26.get("tier12") or {}).get("lower_bound") or {}).get("hits"),
            "git_commit": manifest.get("git_commit"),
        }
        sources[label] = {"metrics_sha256": sha256(metrics_path), "graph_sha256": sha256(flat_path),
                          "queries_file": query_path.name, "queries_sha256": sha256(query_path)}

        obtained = tuple(rows[label][f] for f in FIELDS)
        if obtained != PUBLISHED[label]:
            problems.append("%s: obtained %s, published %s"
                            % (label, dict(zip(FIELDS, obtained)), dict(zip(FIELDS, PUBLISHED[label]))))

    if args.check_previous:
        previous = read_json(args.check_previous)["runs"]
        pairs = [("relations", "n_triples"), ("descriptors", "n_descriptors_distinct"),
                 ("recall34", "recall34_t12"), ("recall26", "recall26_t12"), ("queries", "queries")]
        for label in sorted(VOCABULARY):
            for ours, theirs in pairs:
                if rows[label][ours] != (previous.get(label) or {}).get(theirs):
                    problems.append("previous results differ: %s %s" % (label, ours))
    if problems:
        fail("values do not match:\n  - " + "\n  - ".join(problems))

    out.mkdir(parents=True)
    with open(out / "table5_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows["A"]))
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))

    tex = ["% Table 5, generated by scripts/paper/table5_ablation.py",
           "\\begin{tabular}{clrrrrr}", "\\toprule",
           " & Descriptor vocabulary & Queries & Relations & Descriptors"
           " & Recovered (of 34) & Recovered (of 26) \\\\", "\\midrule"]
    for k in sorted(rows):
        r = rows[k]
        tex.append("%s & %s & %d & %d & %d & %d & %d \\\\"
                   % (k, r["vocabulary"], r["queries"], r["relations"], r["descriptors"],
                      r["recall34"], r["recall26"]))
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "table5_ablation.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (out / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")

    for k in sorted(rows):
        r = rows[k]
        print("%s  %-13s queries=%d relations=%d descriptors=%d /34=%s (strict %s) /26=%s (strict %s)"
              % (k, r["vocabulary"], r["queries"], r["relations"], r["descriptors"],
                 r["recall34"], r["recall34_strict"], r["recall26"], r["recall26_strict"]))
    print("written: " + str(out))


if __name__ == "__main__":
    main()