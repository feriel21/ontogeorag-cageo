#!/usr/bin/env python3
"""186 — Three marker lists for the qualified evidence sentences, and their chi-squared tests.

READ-ONLY. CPU. No model. Prints its results and writes nothing.

The marker table of 186_scope_markers.py lists, for each delivered relation, the markers of
modality, partial scope, condition, value, particular case and part of an object found in its
evidence sentence. A marker such as "where" or "can" may be neutral in a sentence, so the
manuscript reports the share of qualified sentences under three lists:
    full       every marker of the six categories;
    reduced    without the words that can be neutral (NEUTRAL below);
    strict     without these words and without the references to figures (FIGURES below).
For each list, the script counts the qualified sentences and tests the independence between
"qualified or not" and the verdicts of the two verifying-LLMs on the evidence, grouped in four
classes (both SUPPORTED, mixed, both PARTIALLY_SUPPORTED, at least one NOT_SUPPORTED), with a
chi-squared test on the 2 x 4 table.

The recorded table (results/run18/self_audit/audit/186_portee/marqueurs_par_relation.tsv) has
French column names; the table of the translated 186_scope_markers.py has English ones. Both
are read.

Usage (from the repository root)
    python scripts/run18/audit/186_marker_list_variants.py \
        --markers results/run18/self_audit/audit/186_portee/marqueurs_par_relation.tsv
"""
import argparse
import csv
import math
import sys

NEUTRAL = {"where", "when", "can", "internal", "during", "only", "under", "most", "here", "survey", "if"}
FIGURES = {"fig.", "figure"}
EXPECTED = {"full": (269, 0.17), "reduced": (237, 0.32), "strict": (200, 0.35)}

# recorded French columns -> English columns of the translated 186
COLUMNS = {"ensemble": "set", "juges": "verifiers", "modalite": "modality", "partiel": "partial",
           "condition": "condition", "valeur": "value", "cas": "case", "partie": "part"}
VALUES = {"livre": "delivered", "tous SUPPORTED": "all SUPPORTED", "mixte": "mixed",
          "tous PARTIALLY": "all PARTIALLY", "au moins un NOT/autre": "at least one NOT/other"}
CATEGORIES = ["modality", "partial", "condition", "value", "case", "part"]
CLASSES = ["all SUPPORTED", "mixed", "all PARTIALLY", "at least one NOT/other"]


def chi2_sf_3(x):
    """Survival function of the chi-squared distribution with 3 degrees of freedom."""
    return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)


def chi2_test(table):
    rows, cols = [sum(r) for r in table], [sum(c) for c in zip(*table)]
    n = sum(rows)
    x = sum((table[i][j] - rows[i] * cols[j] / n) ** 2 / (rows[i] * cols[j] / n)
            for i in range(len(rows)) for j in range(len(cols)))
    if (len(rows) - 1) * (len(cols) - 1) != 3:
        sys.exit("STOP: the test is written for a 2 x 4 table")
    return x, chi2_sf_3(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--markers", required=True, help="marker table written by 186_scope_markers.py")
    args = ap.parse_args()

    with open(args.markers, encoding="utf-8") as f:
        raw = list(csv.DictReader(f, delimiter="\t"))
    rows = [{COLUMNS.get(k, k): VALUES.get(v, v) for k, v in r.items()} for r in raw]
    rows = [r for r in rows if r["set"] == "delivered"]
    words = [{w.strip() for c in CATEGORIES for w in (r.get(c) or "").split(";") if w.strip()} for r in rows]
    print("delivered relations: %d" % len(rows))

    ok = True
    for name, excluded in [("full", set()), ("reduced", NEUTRAL), ("strict", NEUTRAL | FIGURES)]:
        qualified = [bool(w - excluded) for w in words]
        table = [[sum(1 for r, q in zip(rows, qualified) if q == flag and r["verifiers"] == c) for c in CLASSES]
                 for flag in (True, False)]
        x, p = chi2_test(table)
        n = sum(qualified)
        print("%-8s qualified %3d/%d (%2.0f %%)  qualified %s  not qualified %s  chi2 = %.2f, df = 3, p = %.2f"
              % (name, n, len(rows), 100 * n / len(rows), table[0], table[1], x, p))
        if (n, round(p, 2)) != EXPECTED[name]:
            ok = False
            print("   DIFFERS from the manuscript: %s" % (EXPECTED[name],))
    print("neutral words: " + ", ".join(sorted(NEUTRAL)))
    print("references to figures: " + ", ".join(sorted(FIGURES)))
    print("[done] all three lists match the manuscript" if ok else "STOP: a value differs from the manuscript")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
