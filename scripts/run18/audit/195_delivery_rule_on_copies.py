#!/usr/bin/env python3
"""195 — Delivery rule of the pipeline applied to the originals and to their corrupted copies.

READ-ONLY. CPU. No model. Prints its results and writes nothing.

The rejection rates of the manuscript (tab:stress) use the evidence check: a record is
flagged when at least one verifying-LLM finds its evidence NOT_SUPPORTED. The pipeline does
not deliver its relations with this rule. Each verifying-LLM gives an anchoring verdict p on
the evidence and a plausibility verdict v without the evidence, and decides:
    accept     if p = SUPPORTED, or p = PARTIALLY_SUPPORTED and v = PLAUSIBLE
    reject     if p = NOT_SUPPORTED and v is not PLAUSIBLE
    uncertain  otherwise
The panel keeps the more cautious of the two decisions. Accepted and uncertain relations are
delivered, and rejected relations are set aside (decision equation of the manuscript, Step 4).

This script applies that rule to the 728 records of the stress test and counts, for the
originals and for each type of corruption, the records that the panel accepts, leaves
uncertain and rejects.

Two controls stop the script with exit code 1:
  - on the 364 originals the rule must return the decisions of the run, 238 accepted,
    126 uncertain and none rejected (tab:graphs);
  - the counts on the copies must be those reported in the manuscript and in
    Supplementary Material S2.

Join. The verdict files hold one line per record of controls.jsonl, in the same order: the
field m4_index is the line number. The script checks this, and checks that subject, relation
and object are the same on both sides, before using the order.

Usage (from the repository root)
    python scripts/run18/audit/195_delivery_rule_on_copies.py \
        --stress results/run18/self_audit/stress_test
"""
import argparse
import json
import os
import sys

VERIFIERS = ["llama", "mistral"]
TYPES = [("entity_sub", "replaced entity"), ("passage_perm", "swapped evidence"),
         ("relation_sub", "changed relation type"), ("inversion", "inverted direction")]
DECISIONS = ["accept", "uncertain", "reject"]
# decisions of the run on the delivered graph (tab:graphs)
EXPECTED_ORIGINALS = {"accept": 238, "uncertain": 126, "reject": 0}
# copies accepted / uncertain / rejected reported in the manuscript and in S2
EXPECTED_COPIES = {"entity_sub": (26, 101, 32), "passage_perm": (6, 64, 21),
                   "relation_sub": (36, 42, 13), "inversion": (12, 9, 2)}


def die(msg):
    sys.exit("STOP: " + msg)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def decision(record):
    p = record["evidence"]["verdict"]
    v = record["blind"]["verdict"]
    if p == "SUPPORTED" or (p == "PARTIALLY_SUPPORTED" and v == "PLAUSIBLE"):
        return "accept"
    if p == "NOT_SUPPORTED" and v != "PLAUSIBLE":
        return "reject"
    return "uncertain"


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

    def decide(i, who):
        names = VERIFIERS if who == "panel" else [who]
        # the more cautious decision is the last one in the order accept, uncertain, reject
        return max((decision(verdicts[v][i]) for v in names), key=DECISIONS.index)

    originals = [i for i, c in enumerate(controls) if c["label"] == "original"]
    copies = [(i, c["corruption_class"]) for i, c in enumerate(controls) if c["label"] == "corrupted"]
    print("records: %d (originals %d, corrupted copies %d)" % (len(controls), len(originals), len(copies)))

    ok = True
    for who in VERIFIERS + ["panel"]:
        print("")
        print("== delivery rule: %s ==" % who)
        print("%-22s %5s | %8s %9s %8s" % ("records", "n", "accepted", "uncertain", "rejected"))
        total = [0, 0, 0]
        for key, name in TYPES:
            t = [0, 0, 0]
            for i, k in copies:
                if k == key:
                    t[DECISIONS.index(decide(i, who))] += 1
            total = [a + b for a, b in zip(total, t)]
            print("%-22s %5d | %8d %9d %8d" % (name, sum(t), t[0], t[1], t[2]))
            if who == "panel" and tuple(t) != EXPECTED_COPIES[key]:
                ok = False
                print("   DIFFERS from the manuscript: %s expected" % (EXPECTED_COPIES[key],))
        print("%-22s %5d | %8d %9d %8d" % ("all copies", sum(total), total[0], total[1], total[2]))
        o = [0, 0, 0]
        for i in originals:
            o[DECISIONS.index(decide(i, who))] += 1
        print("%-22s %5d | %8d %9d %8d" % ("originals", sum(o), o[0], o[1], o[2]))
        if who == "panel":
            n = len(copies)
            print("share of copies: accepted %.3f, uncertain %.3f, rejected %.3f"
                  % (total[0] / n, total[1] / n, total[2] / n))
            print("share of originals accepted: %.3f" % (o[0] / len(originals)))
            if dict(zip(DECISIONS, o)) != EXPECTED_ORIGINALS:
                ok = False
                print("   DIFFERS from the run: %s expected on the originals" % EXPECTED_ORIGINALS)

    print("")
    print("[done] the rule returns the decisions of the run on the originals, and the counts "
          "on the copies match the manuscript" if ok else "STOP: a count differs")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
