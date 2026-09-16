#!/usr/bin/env python3
"""
95m_argv_manifeste_et_ciblage.py

Corrects two defects in 95l.

  Defect 1. 95l reported the extraction parameters as NOT RECORDED. The
  manifest does record them, in an `argv` list where a flag and its
  value are separate elements. A regular expression looking for
  "--flag value" inside one string can never match that. This script
  parses argv as a list, which is what it is.

  Defect 2. 95l matched benchmark arguments by substring, so a query
  anchored on `discontinuous` was counted as targeting an edge whose
  object is `continuous`. Matching here is on whole terms only, and the
  count reported is the number of distinct benchmark EDGES targeted,
  not the number of queries, since several queries can name one edge.

DECISION RULES
--------------
  R11 (unchanged)  a parameter is reported as used only if a recorded
                   invocation shows it; otherwise the code default ran
                   and is reported as a default.
  R12  Argument matching is on whole terms. A term contained inside a
       longer term is not a match.

Read-only. Writes one JSON report.

Usage:
    python3 95m_argv_manifeste_et_ciblage.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
SCRIPTS_ROOT = HOME / "ontogeorag"
DATA_ROOT = HOME / "ontogeorag_D"

MANIFESTS = [SCRIPTS_ROOT / "output" / "run16E" / "manifest.json",
             DATA_ROOT / "output" / "run16E" / "manifest.json"]
Q287 = SCRIPTS_ROOT / "configs" / "descriptor_queries_v2.jsonl"
Q249 = SCRIPTS_ROOT / "configs" / "descriptor_queries.jsonl"
BENCH = {"26": SCRIPTS_ROOT / "configs" / "lb_reference_edges_original26.json",
         "34": SCRIPTS_ROOT / "configs" / "lb_reference_edges.json"}

WATCH = ["--reranker", "--bm25-topn", "--bm25_topn", "--top-k", "--top_k",
         "--temperature", "--model", "--queries", "--index-dir", "--schema",
         "--out", "--seed", "--max-new-tokens", "--pass"]

DEFAULTS = {"--reranker": "none, so no re-ranking", "--bm25-topn": "25",
            "--top-k": "3", "--temperature": "0.1",
            "--model": "Qwen/Qwen2.5-7B-Instruct"}

OUT_DIR = SCRIPTS_ROOT / "output" / "audit"
OUT_JSON = OUT_DIR / "95m_argv_and_targeting_report.json"

SEP = "=" * 78
SUB = "-" * 78


def norm(s):
    return " ".join(str(s).split()).strip().lower()


def parse_argv(argv):
    """argv is a list of tokens. Return {flag: value}."""
    out = {}
    if isinstance(argv, str):
        argv = argv.split()
    if not isinstance(argv, list):
        return out
    i = 0
    while i < len(argv):
        tok = str(argv[i])
        if tok.startswith("--"):
            if "=" in tok:
                k, v = tok.split("=", 1)
                out.setdefault(k, []).append(v)
                i += 1
                continue
            nxt = str(argv[i + 1]) if i + 1 < len(argv) else ""
            if nxt and not nxt.startswith("--"):
                out.setdefault(tok, []).append(nxt)
                i += 2
                continue
            out.setdefault(tok, []).append(True)
        i += 1
    return out


def read_jsonl(path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def part_a(report):
    print(SEP)
    print("PART A - argv as recorded, parsed as a list  (rule R11)")
    print(SEP)
    man_path = None
    for p in MANIFESTS:
        if p.exists():
            man_path = p
            break
    if man_path is None:
        print("manifest not found.")
        report["part_a"] = None
        return
    man = json.loads(man_path.read_text(encoding="utf-8", errors="replace"))
    steps = man.get("steps") or []
    print("manifest: " + str(man_path) + "   steps: " + str(len(steps)))

    per_step = {}
    for s in steps:
        if not isinstance(s, dict):
            continue
        name = str(s.get("step", "?"))
        argv = s.get("argv") or s.get("args") or s.get("cmd")
        flags = parse_argv(argv)
        per_step[name] = {"script": s.get("script"), "flags": {k: v for k, v in flags.items()},
                          "started": s.get("started"), "ended": s.get("ended")}
        print(SUB)
        print("step: " + name + "   script: " + str(s.get("script")))
        if isinstance(argv, list):
            print("  argv (" + str(len(argv)) + " tokens), full:")
            line = " ".join(str(x) for x in argv)
            for chunk in [line[i:i + 110] for i in range(0, len(line), 110)]:
                print("    " + chunk)
        else:
            print("  no argv recorded for this step")
            continue
        for flag in WATCH:
            if flag in flags:
                print("  " + flag.ljust(18) + str(flags[flag]))

    print(SEP)
    print("PARAMETERS THAT DECIDE THE MANUSCRIPT SENTENCES")
    print(SEP)
    extract_steps = {k: v for k, v in per_step.items() if k.startswith("extract")}
    if not extract_steps:
        print("no extraction step carries argv.")
    aliases = {"--bm25-topn": ["--bm25-topn", "--bm25_topn"],
               "--top-k": ["--top-k", "--top_k"]}
    for flag in ("--reranker", "--bm25-topn", "--top-k", "--temperature", "--model", "--queries"):
        keys = aliases.get(flag, [flag])
        vals = []
        for name, info in extract_steps.items():
            for k in keys:
                if k in info["flags"]:
                    vals.append(name + "=" + str(info["flags"][k]))
        print(SUB)
        print(flag)
        if vals:
            for v in vals:
                print("  RECORDED  " + v)
        else:
            print("  ABSENT from argv -> the code default applied: "
                  + str(DEFAULTS.get(flag, "(see the script)")))
    print(SUB)
    rr = any("--reranker" in info["flags"] for info in extract_steps.values())
    if rr:
        print("A re-ranker WAS passed. Two-stage retrieval is accurate; quote the")
        print("model identifier from the argv above.")
    else:
        print("No re-ranker in argv. Under rule R11 the delivered run used lexical")
        print("retrieval alone, and the paper must not describe a re-ranking stage.")
        print("This is now read from the recorded argv, not inferred from its absence")
        print("elsewhere.")
    report["part_a"] = {"manifest": str(man_path), "steps": per_step}
    print()


def load_bench(path):
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    edges = obj.get("edges", obj if isinstance(obj, list) else [])
    out = []
    for e in edges:
        if isinstance(e, dict):
            out.append((norm(e.get("subject")), norm(e.get("relation")), norm(e.get("object"))))
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            out.append((norm(e[0]), norm(e[1]), norm(e[2])))
    return out


def anchor_terms(focus):
    f = str(focus)
    for prefix in ("combo:", "gapfill:", "lb:", "LB:"):
        if f.lower().startswith(prefix.lower()):
            f = f[len(prefix):]
    f = f.replace("_", " ")
    parts = [norm(p) for p in re.split(r"->|\+|:", f) if norm(p)]
    return parts, norm(f)


def mentions(term, parts, whole):
    """R12: whole-term match only."""
    if not term:
        return False
    if term in parts:
        return True
    return re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", whole) is not None


def part_b(report):
    print(SEP)
    print("PART B - benchmark targeting, whole-term matching  (rule R12)")
    print(SEP)
    queries = read_jsonl(Q287) or read_jsonl(Q249)
    print("queries inspected: " + str(len(queries)))
    out = {}
    for label, path in BENCH.items():
        if not path.exists():
            continue
        edges = load_bench(path)
        targeted_edges = {}
        per_query = []
        for r in queries:
            parts, whole = anchor_terms(r.get("focus"))
            if not parts and not whole:
                continue
            for s, rel, o in edges:
                if mentions(s, parts, whole) and mentions(o, parts, whole):
                    key = s + " | " + rel + " | " + o
                    targeted_edges.setdefault(key, []).append(str(r.get("focus")))
                    per_query.append((str(r.get("focus")), str(r.get("strategy")), key))
                    break
        print(SUB)
        print("benchmark " + label + ": " + str(len(edges)) + " edges")
        print("  distinct edges with at least one query written for them: "
              + str(len(targeted_edges)) + " of " + str(len(edges)))
        print("  queries involved: " + str(len(per_query)))
        for key in sorted(targeted_edges):
            anchors = sorted(set(targeted_edges[key]))
            print("    " + key)
            for a in anchors[:3]:
                print("        " + a[:76])
        out[label] = {"n_edges": len(edges),
                      "n_targeted_edges": len(targeted_edges),
                      "n_queries": len(per_query),
                      "targeted": {k: sorted(set(v)) for k, v in targeted_edges.items()}}
    print(SUB)
    print("-> the figure to report is the number of DISTINCT benchmark edges for")
    print("   which a query was written, since recall is computed over edges.")
    print("   Report it beside recall, in the same sentence.")
    report["part_b"] = out
    print()


def part_c(report):
    print(SEP)
    print("PART C - which query file the delivered run actually used")
    print(SEP)
    info = (report.get("part_a") or {}).get("steps") or {}
    used = set()
    for name, s in info.items():
        if not name.startswith("extract"):
            continue
        for v in s.get("flags", {}).get("--queries", []):
            used.add(str(v))
    if used:
        for u in sorted(used):
            p = Path(u)
            n = len(read_jsonl(p)) if p.exists() else None
            print("  " + u + ("   (" + str(n) + " queries)" if n is not None else "   (file not found)"))
        print(SUB)
        print("-> this decides whether the delivered run used the 249 base set or the")
        print("   287 extended set, which the ablation table asserts for configuration E.")
    else:
        print("  no --queries flag in the extraction argv.")
    report["part_c"] = sorted(used)
    print()


def main():
    report = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "rules": ["R11 a parameter is reported as used only if a recorded invocation shows it",
                        "R12 argument matching is on whole terms"]}
    part_a(report)
    part_b(report)
    part_c(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(SEP)
    print("report written: " + str(OUT_JSON))
    print(SEP)


if __name__ == "__main__":
    main()