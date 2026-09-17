#!/usr/bin/env python3
"""Structural properties of the delivered graph, recomputed from its triples.

Supports Section 3.10 (structural properties of the graph):
    - number of relations and of distinct entities (232 and 216);
    - entities with no path to "mass transport deposit" (61);
    - surface forms containing "pressure", inspected for the redundancy example.

Everything is recomputed from the flat graph file, not read from an earlier
validation report, so the figures cannot come from another run.

Surface redundancy is estimated here by string normalisation only (case,
punctuation, simple plural). This underestimates redundancy, because
synonyms that share no characters ("overpressure", "excess pore pressure")
are not merged. The figure is labelled as a heuristic in the output.

Read-only. Python 3.10.
Historical name: 93_graph_validation_run16E.py

Usage
    python scripts/analysis/graph_structure.py --graph /path/to/run16E/kg_final/flat_run16E_enforced_dedup.json
    python scripts/analysis/graph_structure.py --root /path/to/runs
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def locate_graph(roots: list[Path]) -> Path | None:
    patterns = ["**/run16E/kg_final/flat_run16E_enforced_dedup.json"]
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            found = sorted(root.glob(pattern))
            if found:
                return found[0]
    return None


def load_triples(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(obj, dict) and "triples" in obj:
        return [t for t in obj["triples"] if isinstance(t, dict)]
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)]
    raise ValueError("unrecognised file structure (neither {'triples': [...]} nor [...])")


def entities(triples: list[dict]) -> set:
    seen = set()
    for t in triples:
        for field in ("subject", "object"):
            v = t.get(field)
            if isinstance(v, str) and v.strip():
                seen.add(v.strip().lower())
    return seen


def normalised_key(name: str) -> str:
    """Case, punctuation and simple -s/-es plural. Non-lexical synonyms are not merged."""
    n = name.lower().strip()
    n = re.sub(r"[-_]", " ", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    words = n.split()
    if words and words[-1].endswith("es") and len(words[-1]) > 4:
        words[-1] = words[-1][:-2]
    elif words and words[-1].endswith("s") and not words[-1].endswith("ss") and len(words[-1]) > 3:
        words[-1] = words[-1][:-1]
    return " ".join(words)


def group_by_key(ents: set) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for e in ents:
        groups[normalised_key(e)].append(e)
    return groups


def connectivity(triples: list[dict], ents: set, anchor_patterns=("mass transport deposit", "mass-transport deposit")):
    """Undirected reachability from every entity whose name contains the anchor."""
    neighbours: dict[str, set] = defaultdict(set)
    for t in triples:
        s, o = t.get("subject"), t.get("object")
        if isinstance(s, str) and isinstance(o, str):
            s, o = s.strip().lower(), o.strip().lower()
            neighbours[s].add(o)
            neighbours[o].add(s)
    starts = [e for e in ents if any(m in e for m in anchor_patterns)]
    if not starts:
        return None, [], ents
    seen = set(starts)
    stack = list(starts)
    while stack:
        cur = stack.pop()
        for v in neighbours.get(cur, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    isolated = ents - (seen & ents)
    return len(isolated), sorted(starts), isolated


def pressure_forms(ents: set) -> list[str]:
    return sorted(e for e in ents if "pressure" in e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Structural properties of the delivered run16E graph.")
    ap.add_argument("--root", action="append", default=None,
                    help="directory to search (repeatable). Default: ~/ontogeorag and ~/ontogeorag_D")
    ap.add_argument("--graph", default=None,
                    help="direct path to flat_run16E_enforced_dedup.json")
    args = ap.parse_args()

    if args.graph:
        path = Path(args.graph).expanduser()
    else:
        roots = [Path(r).expanduser() for r in (args.root or ["~/ontogeorag", "~/ontogeorag_D"])]
        path = locate_graph(roots)

    if path is None or not path.is_file():
        print("NOT FOUND: flat_run16E_enforced_dedup.json", file=sys.stderr)
        return 1

    print("Graph: " + str(path))
    print("=" * 78)

    try:
        triples = load_triples(path)
    except Exception as err:
        print("READ FAILURE: " + type(err).__name__ + ": " + str(err), file=sys.stderr)
        return 1

    ents = entities(triples)
    print("")
    print("relations: " + str(len(triples)))
    print("distinct entities (raw, lower-cased): " + str(len(ents)))

    # surface redundancy, string normalisation only
    groups = group_by_key(ents)
    n_concepts_heuristic = len(groups)
    redundancy_heuristic = 100 * (1 - n_concepts_heuristic / len(ents)) if ents else 0
    print("")
    print("--- Surface redundancy: HEURISTIC (string normalisation) ---")
    print("[underestimates redundancy: synonyms sharing no characters are not merged]")
    print("concepts after heuristic grouping: " + str(n_concepts_heuristic))
    print("heuristic redundancy: " + str(round(redundancy_heuristic, 1)) + " %")
    multi = sorted((g for g in groups.values() if len(g) > 1), key=lambda g: -len(g))
    if multi:
        print("non-trivial groups (" + str(len(multi)) + " in total), the 10 largest:")
        for g in multi[:10]:
            print("  " + str(len(g)) + "x: " + ", ".join(sorted(g)))

    # connectivity
    n_isolated, starts, isolated = connectivity(triples, ents)
    print("")
    print("--- Connectivity ---")
    if n_isolated is None:
        print("no 'mass transport deposit' entity found: not computable")
    else:
        print("start node(s): " + ", ".join(starts))
        print("entities with no path to 'mass transport deposit': " + str(n_isolated)
              + " / " + str(len(ents)))
        if isolated:
            sample = sorted(isolated)[:15]
            print("sample (15 max): " + ", ".join(sample))

    # surface forms around "pressure"
    forms = pressure_forms(ents)
    print("")
    print("--- Surface forms containing 'pressure' ---")
    print("distinct forms: " + str(len(forms)))
    for f in forms:
        print("  - " + f)
    return 0


if __name__ == "__main__":
    sys.exit(main())