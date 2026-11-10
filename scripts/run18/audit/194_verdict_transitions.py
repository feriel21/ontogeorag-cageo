#!/usr/bin/env python3
"""194 — Evidence check on each corrupted copy, crossed with the evidence check on its original.

READ-ONLY. CPU. No model. Prints its results and writes nothing.

The stress test judges 728 records: the 364 delivered relations and one corrupted copy of
each. The rejection rate of the manuscript counts the flagged copies among all copies of a
type. Some originals are themselves flagged by the evidence check, so a flagged copy does not
always mean that the corruption changed the outcome. For each type of corruption, this script
therefore counts the four transitions from the original to its copy:
    original not flagged -> copy not flagged
    original not flagged -> copy flagged        (the corruption changed the outcome)
    original flagged     -> copy not flagged
    original flagged     -> copy flagged
and gives the share of flagged copies among the copies whose original is not flagged, with a
95 % Wilson score interval.

A record is flagged by one verifying-LLM when its verdict on the evidence is NOT_SUPPORTED,
and by the evidence check of the panel when at least one verifying-LLM flags it (the rule of
the manuscript).

Join. The verdict files hold one line per record of controls.jsonl, in the same order: the
field m4_index is the line number. The script checks this, and checks that subject, relation
and object are the same on both sides, before using the order. A copy is joined to its
original by the field orig_index of controls.jsonl.

The script stops with exit code 1 if the counts of flagged copies and of unflagged originals
differ from the rates of the manuscript (tab:stress, column Evidence check).

Usage (from the repository root)
    python scripts/run18/audit/194_verdict_transitions.py \
        --stress results/run18/self_audit/stress_test
"""
import argparse
import json
import math
import os
import sys

VERIFIERS = ["llama", "mistral"]
TYPES = [("entity_sub", "replaced entity"), ("passage_perm", "swapped evidence"),
         ("relation_sub", "changed relation type"), ("inversion", "inverted direction")]
# flagged copies per type and unflagged originals behind tab:stress (evidence check of the panel):
# 0.824 x 159, 0.923 x 91, 0.549 x 91, 0.435 x 23 and 0.783 x 364
EXPECTED_FLAGGED = {"entity_sub": 131, "passage_perm": 84, "relation_sub": 50, "inversion": 10}
EXPECTED_UNFLAGGED_ORIGINALS = 285


def die(msg):
    sys.exit("STOP: " + msg)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stress", required=True,
                    help="stress-test folder with controls.jsonl, llama/ and mistral/")
    args = ap.parse_args()

    controls = read_jsonl(os.path.join(args.stress, "controls.jsonl"))
    verdicts = {v: read_jsonl(os.path.join(args.stress, v, "m4_verdicts.jsonl")) for v in VERIFIERS}
    for v in VERIFIERS:
        if len(verdicts[v]) != len(controls):
            die("%s: %d verdicts for %d records" % (v, len(verdicts[v]), len(controls)))
        for i, (c, r) in enumerate(zip(controls, verdicts[v])):
            if r.get("m4_index") != i:
                die("%s: m4_index %r on line %d" % (v, r.get("m4_index"), i))
            if any(c[k] != r[k] for k in ("subject", "relation", "object")):
                die("%s: line %d does not hold the record of controls.jsonl" % (v, i))

    def flagged(i, who):
        names = VERIFIERS if who == "panel" else [who]
        return any(verdicts[v][i]["evidence"]["verdict"] == "NOT_SUPPORTED" for v in names)

    original_line = {}
    for i, c in enumerate(controls):
        if c["label"] == "original":
            if c["orig_index"] in original_line:
                die("two originals with orig_index %s" % c["orig_index"])
            original_line[c["orig_index"]] = i
    copies = [(i, c) for i, c in enumerate(controls) if c["label"] == "corrupted"]
    for i, c in copies:
        if c["orig_index"] not in original_line:
            die("copy on line %d has no original" % i)
    print("records: %d (originals %d, corrupted copies %d)" % (len(controls), len(original_line), len(copies)))

    ok = True
    for who in VERIFIERS + ["panel"]:
        print("")
        print("== evidence check: %s ==" % who)
        print("%-22s %5s | %9s %9s %9s %9s | flagged among copies whose original is not flagged"
              % ("corruption", "n", "ok->ok", "ok->flag", "flag->ok", "flag->flag"))
        total = [0, 0, 0, 0]
        for key, name in TYPES:
            t = [0, 0, 0, 0]
            for i, c in copies:
                if c["corruption_class"] == key:
                    t[2 * flagged(original_line[c["orig_index"]], who) + flagged(i, who)] += 1
            total = [a + b for a, b in zip(total, t)]
            n0 = t[0] + t[1]
            lo, hi = wilson(t[1], n0)
            print("%-22s %5d | %9d %9d %9d %9d | %3d/%-3d = %.3f [%.3f, %.3f]"
                  % (name, sum(t), t[0], t[1], t[2], t[3], t[1], n0, t[1] / n0, lo, hi))
            if who == "panel" and t[1] + t[3] != EXPECTED_FLAGGED[key]:
                ok = False
                print("   DIFFERS from the manuscript: %d flagged copies expected, %d found"
                      % (EXPECTED_FLAGGED[key], t[1] + t[3]))
        n0 = total[0] + total[1]
        lo, hi = wilson(total[1], n0)
        print("%-22s %5d | %9d %9d %9d %9d | %3d/%-3d = %.3f [%.3f, %.3f]"
              % ("all copies", sum(total), total[0], total[1], total[2], total[3],
                 total[1], n0, total[1] / n0, lo, hi))
        unflagged = sum(1 for i in original_line.values() if not flagged(i, who))
        print("originals not flagged: %d/%d" % (unflagged, len(original_line)))
        if who == "panel" and unflagged != EXPECTED_UNFLAGGED_ORIGINALS:
            ok = False
            print("   DIFFERS from the manuscript: %d unflagged originals expected" % EXPECTED_UNFLAGGED_ORIGINALS)

    print("")
    print("[done] flagged copies and unflagged originals match tab:stress" if ok
          else "STOP: a count differs from the manuscript")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
