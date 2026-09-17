#!/usr/bin/env python3
"""Literal frequency of each extraction-vocabulary descriptor in the run16E index.

Partitions the 40 descriptors of KNOWN_DESCRIPTORS into three lists:
    - absent from the corpus (no occurrence);
    - attested in the corpus but absent from the delivered graph;
    - present in the delivered graph.
It reports whether the three lists add up to the vocabulary size.

Matching is literal and whole-word: a hyphenated descriptor is searched with
its hyphen only. The partition reported in Section 3.5 and Figure 4 uses
hyphen/whitespace equivalence instead (descriptor_partition.py). This script
is kept because surface_forms_and_entity_counts.py compares both conventions.

KNOWN_DESCRIPTORS is imported from pipeline/rag/constants.py, not copied.

Read-only. Python 3.10.
Historical name: 96_descriptor_frequencies_run16E.py

Usage
    python scripts/analysis/descriptor_frequencies_literal.py --root /path/to/data --root /path/to/repo
"""
import argparse
import importlib.util
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


def load_constants(path: Path):
    spec = importlib.util.spec_from_file_location("cible_verif", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_texts(path: Path) -> list[str]:
    texts = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("text") or d.get("content") or d.get("chunk") or ""
            if isinstance(t, str):
                texts.append(t.lower())
    return texts


def field(d: dict, *candidates):
    for c in candidates:
        if c in d and d[c] not in (None, ""):
            return d[c]
    return None


def load_graph(path: Path) -> set:
    """Lower-cased objects of hasDescriptor relations in the delivered graph."""
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    triples = obj["triples"] if isinstance(obj, dict) and "triples" in obj else obj
    in_graph = set()
    for t in triples:
        if not isinstance(t, dict):
            continue
        r = field(t, "relation", "predicate")
        o = field(t, "object", "tail")
        if r == "hasDescriptor" and isinstance(o, str):
            in_graph.add(o.strip().lower())
    return in_graph


def term_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Literal descriptor frequencies over the run16E index.")
    ap.add_argument("--root", action="append", default=None,
                    help="directory to search (repeatable). Default: ~/ontogeorag and ~/ontogeorag_D")
    ap.add_argument("--constants", default=None, help="path to pipeline/rag/constants.py")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in (args.root or ["~/ontogeorag", "~/ontogeorag_D"])]

    constants_path = Path(args.constants).expanduser() if args.constants else \
        locate(roots, ["**/pipeline/rag/constants.py"])
    chunks_path = locate(roots, ["**/run16E/index/chunks.jsonl"])
    graph_path = locate(roots, ["**/run16E/kg_final/flat_run16E_enforced_dedup.json"])

    print("constants.py    : " + (str(constants_path) if constants_path else "NOT FOUND"))
    print("run16E chunks   : " + (str(chunks_path) if chunks_path else "NOT FOUND"))
    print("delivered graph : " + (str(graph_path) if graph_path else "NOT FOUND"))
    print("=" * 78)

    if not (constants_path and chunks_path and graph_path):
        print("")
        print("One or more required files not found. Stopping.")
        return 1

    try:
        c = load_constants(constants_path)
    except Exception as err:
        print("IMPORT FAILURE for constants.py: " + type(err).__name__ + ": " + str(err),
              file=sys.stderr)
        return 1

    if not hasattr(c, "KNOWN_DESCRIPTORS"):
        print("KNOWN_DESCRIPTORS not found in " + str(constants_path), file=sys.stderr)
        return 1

    descriptors = sorted(c.KNOWN_DESCRIPTORS)
    texts = load_texts(chunks_path)
    in_graph = load_graph(graph_path)

    print("")
    print(str(len(descriptors)) + " descriptors in KNOWN_DESCRIPTORS, "
          + str(len(texts)) + " run16E chunks, "
          + str(len(in_graph)) + " distinct descriptors in the delivered graph.")
    print("-" * 78)

    absent, attested_not_extracted, extracted = [], [], []

    for d in descriptors:
        m = term_pattern(d)
        n_occ = sum(len(m.findall(t)) for t in texts)
        n_doc = sum(1 for t in texts if m.search(t))
        if n_occ == 0:
            absent.append(d)
        elif d not in in_graph:
            attested_not_extracted.append((d, n_occ, n_doc))
        else:
            extracted.append((d, n_occ, n_doc))

    attested_not_extracted.sort(key=lambda x: -x[1])
    extracted.sort(key=lambda x: -x[1])

    print("")
    print("ABSENT FROM THE CORPUS (no literal occurrence, inflections not tested): "
          + str(len(absent)))
    for d in absent:
        print("  - " + d)

    print("")
    print("ATTESTED BUT NEVER EXTRACTED: " + str(len(attested_not_extracted)))
    for d, n, k in attested_not_extracted:
        print("  - " + d + "  (" + str(n) + " occurrences / " + str(k) + " chunks)")

    print("")
    print("EXTRACTED IN THE DELIVERED GRAPH: " + str(len(extracted)))
    for d, n, k in extracted:
        print("  - " + d + "  (" + str(n) + " occurrences / " + str(k) + " chunks)")

    total = len(absent) + len(attested_not_extracted) + len(extracted)
    print("")
    print("=" * 78)
    print(str(len(absent)) + " + " + str(len(attested_not_extracted)) + " + "
          + str(len(extracted)) + " = " + str(total)
          + ("  -- adds up to " + str(len(descriptors))
             if total == len(descriptors)
             else "  !! does NOT add up to " + str(len(descriptors)) + ", check by hand"))
    return 0


if __name__ == "__main__":
    sys.exit(main())