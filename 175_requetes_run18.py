#!/usr/bin/env python3
"""175 — Query set of run18: queries that name an evaluation edge are removed.

RULE (fixed before run18)
    A query of the input file is removed if
      R1  its anchor (focus) belongs to the families LB: or gapfill:
          (anchors written from the expert graph or for a targeted edge), or
      R2  its anchor OR its text names both arguments of an edge of
          configs/lb_reference_edges_original26.json, as whole terms
          (a term contained in a longer term does not count).
    All other queries are kept unchanged and in their original order.
    The whole-term match does not cover plurals or abbreviations such as MTD:
    16 of the 210 kept queries still name both terms of an expert edge.

OUTPUTS
    --out            JSONL file of the kept queries
    --out.meta.json  rule, sha256 of input and output, counts, and the index
                     and reason of each removed query
    The script refuses to overwrite an existing output.

USAGE
    python 175_requetes_run18.py --repo . \
        --queries configs/descriptor_queries_v2.jsonl \
        --out configs/descriptor_queries_run18.jsonl
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REF26 = "configs/lb_reference_edges_original26.json"
EXPECT = {"input": 287, "kept": 210}


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(s):
    return " ".join(str(s or "").replace("_", " ").split()).strip().lower()


def mentions(term, text):
    return bool(term) and re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text) is not None


def anchor_text(focus):
    f = str(focus or "")
    for p in ("combo:", "gapfill:", "lb:", "LB:"):
        if f.lower().startswith(p.lower()):
            f = f[len(p):]
    return " ".join(norm(p) for p in re.split(r"->|\+|:", f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-expect", action="store_true", help="ne pas exiger 287 -> 210")
    args = ap.parse_args()

    out = Path(args.out)
    meta = Path(str(out) + ".meta.json")
    if out.exists() or meta.exists():
        die("sortie existante, rien n'est ecrase : " + str(out))
    repo = Path(args.repo)
    edges = [(norm(e["subject"]), norm(e["object"])) for e in
             json.load(open(repo / REF26, encoding="utf-8"))["edges"]]
    lines = [l for l in open(args.queries, encoding="utf-8") if l.strip()]
    Q = [json.loads(l) for l in lines]

    kept, removed, why = [], [], Counter()
    for i, (line, q) in enumerate(zip(lines, Q)):
        focus = str(q.get("focus") or "")
        reasons = []
        if focus.startswith("LB:") or focus.startswith("gapfill:"):
            reasons.append("R1_" + focus.split(":")[0])
        a, t = anchor_text(focus), norm(q.get("query"))
        if any((mentions(s, a) and mentions(o, a)) or (mentions(s, t) and mentions(o, t)) for s, o in edges):
            reasons.append("R2_names_both_arguments")
        if reasons:
            removed.append({"index": i, "focus": focus, "query": q.get("query"), "reasons": reasons})
            for r in reasons:
                why[r] += 1
        else:
            kept.append(line.rstrip("\n"))

    if not args.no_expect and (len(Q), len(kept)) != (EXPECT["input"], EXPECT["kept"]):
        die("attendu %d -> %d, obtenu %d -> %d" % (EXPECT["input"], EXPECT["kept"], len(Q), len(kept)))
    # check: no kept query names an edge under the same whole-term rule
    for line in kept:
        q = json.loads(line)
        a, t = anchor_text(q.get("focus")), norm(q.get("query"))
        assert not any((mentions(s, a) and mentions(o, a)) or (mentions(s, t) and mentions(o, t)) for s, o in edges)

    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    strat = Counter(json.loads(l).get("strategy") for l in kept)
    doc = {"produced_by": "175_requetes_run18.py",
           "rule": {"R1": "focus starts with LB: or gapfill:",
                    "R2": "focus or query text names both arguments of a reference edge (whole terms)"},
           "reference_edges": REF26, "reference_sha256": sha(repo / REF26),
           "input": args.queries, "input_sha256": sha(args.queries), "n_input": len(Q),
           "output": str(out), "output_sha256": sha(out), "n_kept": len(kept), "n_removed": len(removed),
           "removed_by_reason": dict(why), "kept_by_strategy": dict(strat), "removed": removed}
    meta.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print("requetes : %d -> %d gardees, %d retirees %s" % (len(Q), len(kept), len(removed), dict(why)))
    print("gardees par strategie : " + json.dumps(dict(strat)))
    print("sha256 sortie : " + doc["output_sha256"][:12])
    print("[fait] " + str(out))


if __name__ == "__main__":
    main()