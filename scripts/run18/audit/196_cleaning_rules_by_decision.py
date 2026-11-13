#!/usr/bin/env python3
"""196 — Relations removed by the written cleaning rules, by decision of the panel.

READ-ONLY. CPU. No model. Prints its results and writes nothing.

The written rules of 188_nettoyage_familles.py turn the delivered graph (364 relations) into
the cleaned graph (281 relations). The manuscript compares the share of removed relations
between the relations that the panel accepted and the relations that it left uncertain, with
Fisher's exact test (two-sided).

The script counts the decisions of the panel in the delivered graph and in the cleaned graph;
the removed relations are the difference. No join between the two files is needed, so two
relations with the same subject, relation and object cannot be confused.

Fisher's exact test is computed here from the hypergeometric distribution, without any
library: the two-sided p-value is the sum of the probabilities of all tables with the same
margins that are at most as probable as the observed table.

The script stops with exit code 1 if the counts differ from the manuscript: 39 of the 126
uncertain relations and 44 of the 238 accepted relations removed, p = 0.009.

Usage (from the repository root)
    python scripts/run18/audit/196_cleaning_rules_by_decision.py \
        --delivered results/run18/postprocess/187_provenance/graphe_avec_provenance.json \
        --cleaned   results/run18/postprocess/188_nettoyage/graphe_nettoye.json
"""
import argparse
import json
import sys
from collections import Counter
from math import comb

EXPECTED = {"ACCEPT": (238, 44), "UNCERTAIN": (126, 39)}   # delivered, removed
EXPECTED_P = "0.009"


def die(msg):
    sys.exit("STOP: " + msg)


def relations(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["triples"]
    return data


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact test on the table [[a, b], [c, d]]."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def prob(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    observed = prob(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--delivered", required=True, help="delivered graph with located provenance (187)")
    ap.add_argument("--cleaned", required=True, help="cleaned graph (188)")
    args = ap.parse_args()

    delivered = Counter(t["m4"]["decision"] for t in relations(args.delivered))
    cleaned = Counter(t["m4"]["decision"] for t in relations(args.cleaned))
    if set(delivered) != {"ACCEPT", "UNCERTAIN"}:
        die("unexpected decisions in the delivered graph: %s" % dict(delivered))

    ok = True
    print("%-10s %9s %8s %8s %7s" % ("decision", "delivered", "cleaned", "removed", "share"))
    table = {}
    for dec in ("UNCERTAIN", "ACCEPT"):
        removed = delivered[dec] - cleaned[dec]
        table[dec] = (removed, cleaned[dec])
        print("%-10s %9d %8d %8d %6.1f%%"
              % (dec, delivered[dec], cleaned[dec], removed, 100.0 * removed / delivered[dec]))
        if (delivered[dec], removed) != EXPECTED[dec]:
            ok = False
            print("   DIFFERS from the manuscript: %s (delivered, removed) expected" % (EXPECTED[dec],))
    total_removed = sum(delivered.values()) - sum(cleaned.values())
    print("%-10s %9d %8d %8d" % ("all", sum(delivered.values()), sum(cleaned.values()), total_removed))

    p = fisher_two_sided(table["UNCERTAIN"][0], table["UNCERTAIN"][1],
                         table["ACCEPT"][0], table["ACCEPT"][1])
    print("")
    print("Fisher's exact test, two-sided: p = %.4f" % p)
    if "%.3f" % p != EXPECTED_P:
        ok = False
        print("   DIFFERS from the manuscript: p = %s expected" % EXPECTED_P)

    print("")
    print("[done] counts and p-value match the manuscript" if ok else "STOP: a value differs from the manuscript")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
