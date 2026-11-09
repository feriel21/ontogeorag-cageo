#!/usr/bin/env python3
"""181 — Decision rules of the verifying-LLMs, measured on the self-audit records.

WHY
    The self-audit (stress18.sh) gives two verdicts per verifying-LLM for each
    record: with the evidence sentence (p) and blind (v).
    A RULE turns these verdicts into a decision.
    Three rules are measured on the same records:
      admission    rule of the pipeline: the relation is set aside only if
                   p is not SUPPORTED AND v is not PLAUSIBLE
      sentence_first the sentence comes first: set aside if p = NOT_SUPPORTED, or if
                   p = PARTIALLY_SUPPORTED and v is not PLAUSIBLE
      sentence_only  set aside if p = NOT_SUPPORTED (verdict with the sentence only)
    For each rule and each verifying-LLM, then for the panel (set aside if either
    verifier sets it aside): share of corrupted copies set aside, per corruption
    type, and share of the graph relations set aside.
    Same definitions as scripts/evaluation/verifier_decision_rules.py.
    Only sentence_only is used in the manuscript: 191 recomputed the three rules
    from the joined verdicts and did not reproduce the counts of the other two.

USAGE
    python 181_decision_rules.py --stress output/run18/stress_test
    (the folder holds controls.jsonl, llama/m4_verdicts.jsonl, mistral/m4_verdicts.jsonl)
    Outputs: decision_rules.json and decision_rules.txt in the --stress folder.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def p_of(v):
    e = v.get("evidence")
    return e.get("verdict") if isinstance(e, dict) else e


def v_of(v):
    b = v.get("blind")
    return b.get("verdict") if isinstance(b, dict) else b


RULES = {
    "admission":    lambda v: p_of(v) != "SUPPORTED" and v_of(v) != "PLAUSIBLE",
    "sentence_first": lambda v: p_of(v) == "NOT_SUPPORTED"
                              or (p_of(v) == "PARTIALLY_SUPPORTED" and v_of(v) != "PLAUSIBLE"),
    "sentence_only": lambda v: p_of(v) == "NOT_SUPPORTED",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", required=True)
    ap.add_argument("--judges", nargs="+", default=["llama", "mistral"])
    args = ap.parse_args()
    d = Path(args.stress)
    controls = jsonl(d / "controls.jsonl")
    V = {}
    for j in args.judges:
        V[j] = jsonl(d / j / "m4_verdicts.jsonl")
        if len(V[j]) != len(controls):
            sys.exit("STOP: %s has %d verdicts for %d records" % (j, len(V[j]), len(controls)))
        for i, v in enumerate(V[j]):
            if "m4_index" in v and v["m4_index"] != i:
                sys.exit("STOP: %s line %d is not aligned" % (j, i))

    cls = [c.get("corruption_class", "none") for c in controls]
    corrupted = [i for i, c in enumerate(controls) if c.get("label") == "corrupted"]
    originals = [i for i, c in enumerate(controls) if c.get("label") != "corrupted"]
    classes = sorted({cls[i] for i in corrupted})
    n_cls = Counter(cls[i] for i in corrupted)

    def flags(rule, who):
        if who == "panel":
            return [any(RULES[rule](V[j][i]) for j in args.judges) for i in range(len(controls))]
        return [RULES[rule](V[who][i]) for i in range(len(controls))]

    out = {}
    L = ["=== 181 decision rules on %d records (%d originals, %d corrupted) ==="
         % (len(controls), len(originals), len(corrupted)),
         "per corruption type: " + ", ".join("%s %d" % (k, n_cls[k]) for k in classes), ""]
    header = "%-13s %-8s %16s %18s  " % ("rule", "verifier", "corrupted_set_aside", "originals_set_aside") + \
             "  ".join("%-14s" % k for k in classes)
    L.append(header)
    for rule in RULES:
        for who in args.judges + ["panel"]:
            f = flags(rule, who)
            tp = sum(f[i] for i in corrupted)
            fp = sum(f[i] for i in originals)
            per = {k: sum(f[i] for i in corrupted if cls[i] == k) for k in classes}
            out["%s|%s" % (rule, who)] = {"corrupted_flagged": tp, "n_corrupted": len(corrupted),
                                          "originals_flagged": fp, "n_originals": len(originals),
                                          "per_class": per}
            L.append("%-13s %-8s %8d/%-3d %3.0f%% %8d/%-3d %3.0f%%  " % (
                rule, who, tp, len(corrupted), 100 * tp / len(corrupted),
                fp, len(originals), 100 * fp / len(originals)) +
                "  ".join("%-14s" % ("%d/%d" % (per[k], n_cls[k])) for k in classes))
        L.append("")
    L.append("admission = rule of the pipeline; sentence_first = the sentence comes first; sentence_only = verdict with the sentence only")
    L.append("panel = set aside if either verifier sets it aside (the most cautious, as in the pipeline)")
    txt = "\n".join(L)
    (d / "decision_rules.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (d / "decision_rules.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()