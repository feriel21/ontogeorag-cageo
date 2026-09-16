#!/usr/bin/env python3
"""
95i_provenance_benchmark26.py

Two questions left open once the real reference graph was located.

  A  How was the 26-edge benchmark built? The frozen file carries
     `description`, `source` and `descriptors_13` fields that record it.
     Print them verbatim rather than inferring.

  B  How much of the reference graph is expressible in the extraction
     schema at all? The reference uses two generic relation types
     (`impacts`, `related_to`) over 90 descriptive-variable nodes, while
     the extraction schema declares 13 typed relations over a validation
     vocabulary. Count how many of the 174 reference relations have both
     arguments in that vocabulary and a relation that maps to an allowed
     type.

DECISION RULE
-------------
  R9  A benchmark edge is SELECTED from the reference iff it appears in
      the reference edge list under content-key matching. Otherwise it
      is DERIVED, and recall against it is not recall against the
      reference as published.

Read-only. Writes one JSON report.

Usage:
    python3 95i_provenance_benchmark26.py
    python3 95i_provenance_benchmark26.py --reference /path/to/reference_kg.json
"""

import argparse
import importlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
SCRIPTS_ROOT = HOME / "ontogeorag"
DATA_ROOT = HOME / "ontogeorag_D"

BENCH_26 = SCRIPTS_ROOT / "configs" / "lb_reference_edges_original26.json"
BENCH_34 = SCRIPTS_ROOT / "configs" / "lb_reference_edges.json"

REFERENCE_GLOBS = ["**/reference*/**/*.json", "**/reference_kg*.json", "**/reference*.json"]

OUT_DIR = SCRIPTS_ROOT / "output" / "audit"
OUT_JSON = OUT_DIR / "95i_benchmark_provenance_report.json"

SEP = "=" * 78
SUB = "-" * 78


def norm(s):
    return " ".join(str(s).split()).strip().lower()


def load_constants():
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        return importlib.import_module("pipeline.rag.constants")
    except Exception:
        path = SCRIPTS_ROOT / "pipeline" / "rag" / "constants.py"
        spec = importlib.util.spec_from_file_location("ontogeorag_constants", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def find_reference(explicit):
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    seen = []
    for root in (SCRIPTS_ROOT, DATA_ROOT):
        if not root.exists():
            continue
        for pattern in REFERENCE_GLOBS:
            for p in root.glob(pattern):
                if not p.is_file() or p.suffix != ".json":
                    continue
                try:
                    obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("edges"), list) and isinstance(obj.get("nodes"), dict):
                    seen.append((len(obj["edges"]), p))
    if not seen:
        return None
    seen.sort(reverse=True)
    print("reference candidates found:")
    for n, p in seen:
        print("  " + str(n).rjust(5) + " edges   " + str(p))
    return seen[0][1]


def ref_edges(obj):
    out = []
    for e in obj.get("edges", []):
        if not isinstance(e, dict):
            continue
        out.append({"s": norm(e.get("source")), "r": norm(e.get("relation")),
                    "o": norm(e.get("target")),
                    "citations": e.get("citations") or [],
                    "directed": e.get("directed"),
                    "raw_label": e.get("raw_label", "")})
    return out


def bench_edges(path):
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    edges = obj.get("edges", obj if isinstance(obj, list) else [])
    out = []
    for e in edges:
        if isinstance(e, dict):
            out.append({"s": norm(e.get("subject")), "r": norm(e.get("relation")),
                        "o": norm(e.get("object")), "raw": e})
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            out.append({"s": norm(e[0]), "r": norm(e[1]), "o": norm(e[2]), "raw": list(e)})
    return obj, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()

    report = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "rules": ["R9 a benchmark edge is selected iff it appears in the reference edge list; otherwise derived"]}

    # ---------------------------------------------------------------- A
    print(SEP)
    print("PART A - how the 26-edge benchmark records its own provenance")
    print(SEP)
    if not BENCH_26.exists():
        print("FATAL: " + str(BENCH_26) + " not found")
        sys.exit(2)
    obj26, e26 = bench_edges(BENCH_26)
    for field in ("description", "source", "provenance", "note"):
        if field in obj26:
            print(field + ": " + json.dumps(obj26[field], ensure_ascii=False)[:600])
    if "descriptors_13" in obj26:
        d13 = obj26["descriptors_13"]
        print("descriptors_13 (" + str(len(d13)) + "): " + ", ".join(sorted(norm(x) for x in d13)))
    print("edges: " + str(len(e26)))
    print("relation types used by the benchmark: "
          + ", ".join(r + "=" + str(n) for r, n in Counter(e["r"] for e in e26).most_common()))
    report["bench26"] = {k: obj26[k] for k in obj26 if k != "edges"}
    report["bench26"]["n_edges"] = len(e26)
    print()

    # ---------------------------------------------------------------- B
    print(SEP)
    print("PART B - the reference graph")
    print(SEP)
    ref_path = find_reference(args.reference)
    if ref_path is None:
        print("FATAL: no reference graph found. Pass --reference /path/to/file.json")
        sys.exit(2)
    print("reference: " + str(ref_path))
    robj = json.loads(ref_path.read_text(encoding="utf-8", errors="replace"))
    R = ref_edges(robj)
    nodes = robj.get("nodes", {})
    print("relations: " + str(len(R)) + "   nodes: " + str(len(nodes)))
    print("relation types: " + ", ".join(r + "=" + str(n)
                                         for r, n in Counter(e["r"] for e in R).most_common()))
    cit = Counter(len(e["citations"]) for e in R)
    none_c = cit.get(0, 0)
    one_c = cit.get(1, 0)
    multi_c = sum(n for k, n in cit.items() if k > 1)
    print("citations: none=" + str(none_c) + "  exactly one=" + str(one_c)
          + "  two or more=" + str(multi_c))
    refs = set()
    for e in R:
        for c in e["citations"]:
            refs.add(str(c)[:80])
    print("distinct cited works: " + str(len(refs)))
    report["reference"] = {"path": str(ref_path), "n_relations": len(R), "n_nodes": len(nodes),
                           "relation_types": dict(Counter(e["r"] for e in R)),
                           "citations_none": none_c, "citations_one": one_c,
                           "citations_multi": multi_c, "n_cited_works": len(refs)}
    print(SUB)

    # R9
    rkeys = {e["s"] + " | " + e["o"] for e in R} | {e["o"] + " | " + e["s"] for e in R}
    selected = [e for e in e26 if (e["s"] + " | " + e["o"]) in rkeys]
    print("benchmark edges whose argument pair appears in the reference: "
          + str(len(selected)) + " of " + str(len(e26)))
    if len(selected) == 0:
        print("VERDICT (rule R9): the 26-edge benchmark is DERIVED, not selected.")
        print("  Do not write that N of the reference relations 'meet the criteria'.")
        print("  Write that a typed benchmark was derived from the reference.")
    else:
        print("VERDICT (rule R9): " + str(len(selected)) + " benchmark edges are selected from the reference.")
        for e in selected[:10]:
            print("  " + e["s"] + " | " + e["r"] + " | " + e["o"])
    report["selected_from_reference"] = len(selected)
    print(SUB)

    # expressibility
    mod = load_constants()

    def items(name):
        val = getattr(mod, name, None)
        if isinstance(val, dict):
            out = {norm(k) for k in val}
            out |= {norm(v) for v in val.values() if isinstance(v, str)}
            return out
        if isinstance(val, (list, tuple, set, frozenset)):
            return {norm(v) for v in val if isinstance(v, str)}
        return set()

    allowed = items("ALLOWED_RELATIONS")
    rel_map = items("RELATION_MAP")
    vocabs = {
        "descriptors+settings+mtd": items("KNOWN_DESCRIPTORS") | items("KNOWN_SETTINGS") | items("MTD_VARIANTS"),
    }
    vocabs["  + entity norms"] = vocabs["descriptors+settings+mtd"] | items("ENTITY_NORMS")

    print("relation types of the reference against the extraction schema:")
    for r, n in Counter(e["r"] for e in R).most_common():
        status = "in ALLOWED_RELATIONS" if r in allowed else (
            "in RELATION_MAP" if r in rel_map else "NOT mappable")
        print("  " + r.ljust(16) + str(n).rjust(5) + "   " + status)
    print(SUB)
    print("reference relations whose BOTH arguments are in the validation vocabulary:")
    expr = {}
    for vname, vocab in vocabs.items():
        ok = [e for e in R if e["s"] in vocab and e["o"] in vocab]
        expr[vname.strip()] = len(ok)
        print("  " + vname.ljust(26) + str(len(ok)).rjust(5) + " of " + str(len(R)))
    print(SUB)
    print("-> the complement is the part of the reference the extraction schema")
    print("   cannot express at all. Report it as a property of the two")
    print("   representations, not as a recall failure.")
    report["expressibility"] = {"allowed_relations": sorted(allowed),
                                "reference_relations_in_vocabulary": expr}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print()
    print(SEP)
    print("report written: " + str(OUT_JSON))
    print(SEP)


if __name__ == "__main__":
    main()