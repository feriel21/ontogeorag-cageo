#!/usr/bin/env python3
"""
95h_mesurer_graphe_reference.py

Replaces the untraced "173 LB2019 relations" with a measured figure, and
turns the sentence that follows it into counts instead of a list of
reasons.

  A  Inventory every artefact on disk that could be the reference graph,
     with its edge count. Nothing is assumed to be the reference.

  B  Compare the two frozen benchmark files (26 and 34) against each
     other and against the full reference set.

  C  Funnel. For every edge of the full reference set that is NOT in the
     26-edge benchmark, determine which admission criterion it fails:
       C1  both arguments present in the validation vocabulary
       C2  relation mappable to an allowed type
       C3  supported by a single cited passage
     Criterion C3 is only testable if the reference carries a provenance
     field; if it does not, the script says so instead of inventing it.

DECISION RULES
--------------
  R6 (unchanged)  a definition EXPLAINS a published value iff it
                  reproduces it exactly.
  R8  The reference set is whichever artefact the frozen 26-edge
      benchmark is a subset of. An artefact the benchmark is not a
      subset of is not the reference, whatever its name.

Read-only. Writes one JSON report.

Usage:
    python3 95h_mesurer_graphe_reference.py
"""

import importlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
SCRIPTS_ROOT = HOME / "ontogeorag"
DATA_ROOT = HOME / "ontogeorag_D"

BENCH_34 = SCRIPTS_ROOT / "configs" / "lb_reference_edges.json"
BENCH_26 = SCRIPTS_ROOT / "configs" / "lb_reference_edges_original26.json"

SEARCH_DIRS = [SCRIPTS_ROOT / "configs", SCRIPTS_ROOT / "output",
               SCRIPTS_ROOT / "reviewer-experiments", SCRIPTS_ROOT / "pipeline",
               DATA_ROOT / "output"]
NAME_HINT = re.compile(r"lb2019|lb_reference|reference_edges|ontology|schema", re.IGNORECASE)
SKIP_DIRS = {".git", "venv", "__pycache__", "node_modules"}

OUT_DIR = SCRIPTS_ROOT / "output" / "audit"
OUT_JSON = OUT_DIR / "95h_reference_graph_report.json"

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


def edges_from(obj):
    """Return list of dicts with subject/relation/object plus the raw record."""
    out = []
    if isinstance(obj, dict):
        for key in ("edges", "triples", "relations", "reference_edges", "data"):
            if key in obj:
                return edges_from(obj[key])
        # dict of lists?
        vals = list(obj.values())
        if vals and all(isinstance(v, (list, tuple)) for v in vals):
            return []
        return []
    if not isinstance(obj, list):
        return []
    for item in obj:
        if isinstance(item, dict):
            s = item.get("subject", item.get("s", item.get("head")))
            r = item.get("relation", item.get("r", item.get("predicate")))
            o = item.get("object", item.get("o", item.get("tail")))
            if s is not None and r is not None and o is not None:
                out.append({"subject": norm(s), "relation": norm(r), "object": norm(o),
                            "raw": item})
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            out.append({"subject": norm(item[0]), "relation": norm(item[1]),
                        "object": norm(item[2]), "raw": list(item)})
    return out


def key_of(e):
    return e["subject"] + " | " + e["relation"] + " | " + e["object"]


def part_a(report):
    print(SEP)
    print("PART A - inventory of candidate reference artefacts")
    print(SEP)
    seen = {}
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if any(p in SKIP_DIRS for p in path.parts):
                continue
            if not NAME_HINT.search(path.name):
                continue
            try:
                obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                print("  " + str(path) + "   [unparsable: " + repr(exc)[:60] + "]")
                continue
            edges = edges_from(obj)
            top = sorted(obj)[:6] if isinstance(obj, dict) else ["<list>"]
            print("  " + str(len(edges)).rjust(5) + " edges   " + str(path))
            print("            top-level: " + ", ".join(str(t) for t in top))
            seen[str(path)] = {"n_edges": len(edges),
                               "top_level": [str(t) for t in top]}
    report["part_a"] = seen
    if not seen:
        print("  no candidate artefact found under the searched directories.")
    print()
    return seen


def part_b(mod, report):
    print(SEP)
    print("PART B - the two frozen benchmarks, and the reference set")
    print(SEP)
    sets = {}
    for label, path in (("bench26", BENCH_26), ("bench34", BENCH_34)):
        if not path.exists():
            print(label + ": ABSENT at " + str(path))
            continue
        edges = edges_from(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        sets[label] = edges
        print(label + ": " + str(len(edges)) + " edges   " + str(path))

    const_edges = edges_from(getattr(mod, "LB2019_REFERENCE_EDGES", []))
    if const_edges:
        sets["constants"] = const_edges
        print("constants.LB2019_REFERENCE_EDGES: " + str(len(const_edges)) + " edges")

    if "bench26" in sets and "bench34" in sets:
        k26 = {key_of(e) for e in sets["bench26"]}
        k34 = {key_of(e) for e in sets["bench34"]}
        print(SUB)
        print("bench26 subset of bench34: " + str(k26 <= k34))
        extra = sorted(k34 - k26)
        print("edges in 34 not in 26: " + str(len(extra)))
        for k in extra:
            print("  " + k)
        report["bench_diff"] = {"n26": len(k26), "n34": len(k34),
                                "subset": k26 <= k34, "extra": extra}
    print()
    return sets


def build_vocabularies(mod):
    def items(name):
        val = getattr(mod, name, None)
        if isinstance(val, dict):
            out = {norm(k) for k in val}
            out |= {norm(v) for v in val.values() if isinstance(v, str)}
            return out
        if isinstance(val, (list, tuple, set, frozenset)):
            return {norm(v) for v in val if isinstance(v, str)}
        return set()

    desc = items("KNOWN_DESCRIPTORS")
    setting = items("KNOWN_SETTINGS")
    mtd = items("MTD_VARIANTS")
    ent = items("ENTITY_NORMS")
    lb = items("LB2019_DESCRIPTORS")
    return {
        "descriptors+settings+mtd": desc | setting | mtd,
        "  + entity norms": desc | setting | mtd | ent,
        "  + LB2019 descriptors": desc | setting | mtd | ent | lb,
    }


def part_c(mod, sets, inventory, report):
    print(SEP)
    print("PART C - funnel from the reference set to the 26-edge benchmark")
    print(SEP)

    if "bench26" not in sets:
        print("cannot run: bench26 missing.")
        return
    k26 = {key_of(e) for e in sets["bench26"]}

    # R8: the reference is the artefact bench26 is a subset of, and which
    # is strictly larger than bench34.
    candidates = []
    for label, edges in sets.items():
        if label in ("bench26",):
            continue
        keys = {key_of(e) for e in edges}
        candidates.append((label, edges, keys, k26 <= keys))
    for path, meta in inventory.items():
        pass  # inventory is reported in part A; files are re-read below only if needed

    print("candidate reference sets (rule R8):")
    reference = None
    for label, edges, keys, ok in candidates:
        print("  " + label.ljust(22) + str(len(keys)).rjust(5) + " edges   contains bench26: " + str(ok))
        if ok and (reference is None or len(keys) > len(reference[2])):
            reference = (label, edges, keys, ok)

    if reference is None:
        print(SUB)
        print("NO candidate contains the 26-edge benchmark.")
        print("  -> the reference graph is not among the artefacts found. Do not")
        print("     write a total until it is located; the 173 stays unsupported.")
        report["part_c"] = {"reference": None}
        return

    label, edges, keys, _ = reference
    print(SUB)
    print("reference set: " + label + "   " + str(len(keys)) + " edges")
    print("MEASURED TOTAL for the manuscript sentence: " + str(len(keys)))
    print()

    rel_counts = Counter(e["relation"] for e in edges)
    print("relation types in the reference set (" + str(len(rel_counts)) + "):")
    for r, n in rel_counts.most_common():
        print("  " + r.ljust(24) + str(n))
    print()

    allowed = {norm(x) for x in getattr(mod, "ALLOWED_RELATIONS", [])}
    rel_map = getattr(mod, "RELATION_MAP", {})
    mappable = allowed | {norm(k) for k in rel_map} if isinstance(rel_map, dict) else allowed

    vocabs = build_vocabularies(mod)
    excluded = [e for e in edges if key_of(e) not in k26]
    print("edges outside the 26-edge benchmark: " + str(len(excluded)))
    print(SUB)

    print("criterion C2, relation mappable to an allowed type:")
    fail_c2 = [e for e in excluded if e["relation"] not in mappable]
    print("  fail: " + str(len(fail_c2)) + " of " + str(len(excluded)))
    for r, n in Counter(e["relation"] for e in fail_c2).most_common(10):
        print("    " + r.ljust(24) + str(n))
    print(SUB)

    print("criterion C1, both arguments in the validation vocabulary:")
    c1_results = {}
    for vname, vocab in vocabs.items():
        fail = [e for e in excluded
                if e["subject"] not in vocab or e["object"] not in vocab]
        c1_results[vname.strip()] = len(fail)
        print("  " + vname.ljust(26) + "fail: " + str(len(fail)) + " of " + str(len(excluded)))
        kept26 = [e for e in sets["bench26"]
                  if e["subject"] in vocab and e["object"] in vocab]
        mark = "  *" if len(kept26) == len(sets["bench26"]) else ""
        print("    " + "all 26 benchmark edges satisfy it: "
              + str(len(kept26) == len(sets["bench26"])) + mark)
    print("  * a definition under which every admitted edge passes is the one the")
    print("    benchmark was built with (rule R6).")
    print(SUB)

    prov_keys = set()
    for e in edges[:50]:
        raw = e.get("raw")
        if isinstance(raw, dict):
            prov_keys |= set(raw.keys())
    prov_candidates = sorted(k for k in prov_keys
                             if re.search(r"passage|source|cite|paper|doi|ref|evidence", str(k), re.I))
    print("criterion C3, single cited passage:")
    print("  record keys available: " + (", ".join(sorted(str(k) for k in prov_keys)) or "(tuples, no keys)"))
    if prov_candidates:
        print("  provenance-like fields: " + ", ".join(prov_candidates))
        for field in prov_candidates:
            multi = 0
            single = 0
            missing = 0
            for e in excluded:
                raw = e.get("raw")
                v = raw.get(field) if isinstance(raw, dict) else None
                if v is None or v == "":
                    missing += 1
                elif isinstance(v, (list, tuple, set)):
                    if len(v) > 1:
                        multi += 1
                    else:
                        single += 1
                else:
                    single += 1
            print("    " + str(field).ljust(20) + "single: " + str(single).rjust(4)
                  + "   multiple: " + str(multi).rjust(4) + "   absent: " + str(missing).rjust(4))
    else:
        print("  NO provenance field in the reference records.")
        print("  -> criterion C3 is not testable from this artefact. Say in the paper")
        print("     that the single-passage criterion was applied at benchmark")
        print("     construction, and do not report a count for it.")

    print(SUB)
    only_c2 = {key_of(e) for e in fail_c2}
    remaining = [e for e in excluded if key_of(e) not in only_c2]
    print("SUGGESTED SENTENCE SHAPE")
    print("  Of the " + str(len(keys)) + " reference relations, " + str(len(k26))
          + " meet the admission criteria.")
    print("  Of the " + str(len(excluded)) + " excluded, " + str(len(fail_c2))
          + " carry a relation type outside the extraction schema;")
    print("  the remaining " + str(len(remaining))
          + " fail on argument coverage or on the single-passage requirement.")
    print("  (check the C1 table above before splitting that remainder further)")

    report["part_c"] = {"reference_label": label,
                        "n_reference": len(keys),
                        "n_bench26": len(k26),
                        "n_excluded": len(excluded),
                        "relation_types": dict(rel_counts),
                        "fail_relation_type": len(fail_c2),
                        "fail_relation_type_by_relation": dict(Counter(e["relation"] for e in fail_c2)),
                        "fail_vocabulary_by_definition": c1_results,
                        "provenance_fields": prov_candidates}
    print()


def main():
    report = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "rules": ["R6 a definition explains a published value iff it reproduces it exactly",
                        "R8 the reference set is the artefact the frozen 26-edge benchmark is a subset of"]}
    inventory = part_a(report)
    mod = load_constants()
    sets = part_b(mod, report)
    part_c(mod, sets, inventory, report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(SEP)
    print("report written: " + str(OUT_JSON))
    print(SEP)


if __name__ == "__main__":
    main()