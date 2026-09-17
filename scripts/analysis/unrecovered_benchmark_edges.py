#!/usr/bin/env python3
"""Benchmark edges not recovered by the delivered graph, and whether the corpus supports them.

Supports Section 3.3 (which reference relations were not recovered):
    - edges of the 34-edge benchmark absent from the delivered graph (16);
    - those whose two arguments never co-occur in an indexed passage (2);
    - for the others, the number of passages in which both arguments occur.

Matching between benchmark and graph is exact on (subject, relation, object)
after lower-casing and whitespace normalisation.

Co-occurrence is counted over the full run16E index (1,758 passages), with
whole-word matching. Inflected forms are not matched ("turbidity currents"
does not match "turbidity current"), so an edge with no co-occurrence must be
checked by hand before it is reported as unsupported.

This script does not separate retrieval failures from extraction failures.
That separation is made by the retrieval replay in scripts/evaluation/.

Read-only. Python 3.10.
Historical name: 94_failure_modes.py

Usage
    python scripts/analysis/unrecovered_benchmark_edges.py --root /path/to/data --root /path/to/repo
"""
import argparse
import json
import re
import sys
from pathlib import Path


def locate(roots: list[Path], patterns: list[str]) -> Path | None:
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            found = sorted(root.glob(pattern))
            if found:
                return found[0]
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except Exception:
                continue
    return rows


def field(d: dict, *candidates):
    """First non-empty value among candidate keys, then the same keys case-insensitively."""
    for c in candidates:
        if c in d and d[c] not in (None, ""):
            return d[c]
    lower = {k.lower(): v for k, v in d.items()}
    for c in candidates:
        if c.lower() in lower and lower[c.lower()] not in (None, ""):
            return lower[c.lower()]
    return None


def extract_edges(obj) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("edges", "relations", "benchmark", "items"):
            if key in obj and isinstance(obj[key], list):
                return [x for x in obj[key] if isinstance(x, dict)]
    return []


def extract_delivered_triples(obj) -> list[dict]:
    if isinstance(obj, dict) and "triples" in obj:
        return [t for t in obj["triples"] if isinstance(t, dict)]
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)]
    return []


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def edge_key(s: str, r: str, o: str) -> tuple:
    return (normalise(s), normalise(r), normalise(o))


def term_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Unrecovered edges of the 34-edge benchmark.")
    ap.add_argument("--root", action="append", default=None,
                    help="directory to search (repeatable). Default: ~/ontogeorag and ~/ontogeorag_D")
    args = ap.parse_args()
    roots = [Path(r).expanduser() for r in (args.root or ["~/ontogeorag", "~/ontogeorag_D"])]

    bench_path = locate(roots, ["**/lb_reference_edges.json"])
    graph_path = locate(roots, ["**/run16E/kg_final/flat_run16E_enforced_dedup.json"])
    chunks_path = locate(roots, ["**/run16E/index/chunks.jsonl"])

    print("benchmark (34)  : " + (str(bench_path) if bench_path else "NOT FOUND"))
    print("delivered graph : " + (str(graph_path) if graph_path else "NOT FOUND"))
    print("run16E chunks   : " + (str(chunks_path) if chunks_path else "NOT FOUND"))
    print("=" * 78)

    if not (bench_path and graph_path and chunks_path):
        print("")
        print("One or more required files not found. Stopping.")
        return 1

    edges = extract_edges(json.loads(bench_path.read_text(encoding="utf-8", errors="replace")))
    triples = extract_delivered_triples(json.loads(graph_path.read_text(encoding="utf-8", errors="replace")))
    chunks = load_jsonl(chunks_path)

    texts = []
    for c in chunks:
        t = field(c, "text", "content", "chunk", "body")
        if isinstance(t, str):
            texts.append(t.lower())

    delivered = set()
    for t in triples:
        s = field(t, "subject", "head", "s")
        r = field(t, "relation", "predicate", "rel", "p")
        o = field(t, "object", "tail", "o")
        if isinstance(s, str) and isinstance(r, str) and isinstance(o, str):
            delivered.add(edge_key(s, r, o))

    print("")
    print(str(len(edges)) + " benchmark edges, "
          + str(len(delivered)) + " distinct relations in the delivered graph, "
          + str(len(texts)) + " indexed chunks.")

    unrecovered = []
    for e in edges:
        s = field(e, "subject", "head", "s")
        r = field(e, "relation", "predicate", "rel", "p")
        o = field(e, "object", "tail", "o")
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(o, str)):
            continue
        if edge_key(s, r, o) not in delivered:
            unrecovered.append((s, r, o))

    print(str(len(unrecovered)) + " benchmark edge(s) not found as such"
          + " in the delivered graph.")
    print("-" * 78)

    no_evidence, with_evidence = [], []
    for s, r, o in unrecovered:
        m_s, m_o = term_pattern(s), term_pattern(o)
        n_chunks_both = sum(1 for t in texts if m_s.search(t) and m_o.search(t))
        entry = (s, r, o, n_chunks_both)
        if n_chunks_both == 0:
            no_evidence.append(entry)
        else:
            with_evidence.append(entry)

    print("")
    print("NO CO-OCCURRENCE in any chunk (" + str(len(no_evidence))
          + "), check inflected forms by hand:")
    for s, r, o, n in no_evidence:
        print("  " + s + " --[" + r + "]--> " + o)

    print("")
    print("BOTH ARGUMENTS CO-OCCUR in at least one chunk ("
          + str(len(with_evidence)) + "):")
    for s, r, o, n in sorted(with_evidence, key=lambda x: -x[3]):
        print("  " + s + " --[" + r + "]--> " + o + "   (" + str(n) + " co-occurring chunks)")

    print("")
    print("=" * 78)
    print("Retrieval versus extraction for these "
          + str(len(with_evidence)) + " edges: see the retrieval replay in scripts/evaluation/.")
    print("Edges to replay: "
          + str(len(with_evidence)) + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())