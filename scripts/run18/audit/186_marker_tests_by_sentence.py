#!/usr/bin/env python3
"""186 — Tests between markers and verdicts that keep the relations of one sentence together.

READ-ONLY. CPU. No model. Prints its results and writes nothing.

186_marker_list_variants.py tests, for three marker lists, the independence between "the
evidence sentence carries a marker" and the verdicts of the two verifying-LLMs, with a
chi-squared test on a 2 x 4 table of the delivered relations. That test treats the relations
as independent observations. Several relations share one evidence sentence, and the markers
are read in the sentence, so these relations are not independent.

This script keeps the same 2 x 4 table and the same chi-squared statistic, and replaces the
chi-squared distribution by a permutation test in which the unit is the evidence sentence:
    1. the delivered relations are grouped by evidence sentence (same lowercase letters and
       digits after Unicode normalisation NFKC, the comparison used for the provenance);
    2. at each permutation, the marker status of whole sentences is exchanged between
       sentences that support the same number of relations, so the relations of one sentence
       always move together and the verdicts never move;
    3. the p-value is the share of permutations whose statistic is at least the observed one,
       counted as (b + 1) / (B + 1).
Sentences are exchanged within groups of equal size because the marker status is recorded per
relation: a part-of-object word that is already in the subject of a relation is not counted for
that relation, so two relations of one sentence can differ.

The random generator is seeded (--seed, default 13, the seed of the stress test), so a rerun
prints the same values. The marker lists are imported from 186_marker_list_variants.py.

The recorded table (results/run18/self_audit/audit/186_portee/marqueurs_par_relation.tsv) has
French column names; the table of the translated 186_scope_markers.py has English ones. Both
are read; the evidence sentence is the column "preuve" or "evidence".

Usage (from the repository root)
    python scripts/run18/audit/186_marker_tests_by_sentence.py \
        --markers results/run18/self_audit/audit/186_portee/marqueurs_par_relation.tsv
"""
import argparse
import csv
import importlib.util
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("variants", os.path.join(HERE, "186_marker_list_variants.py"))
variants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variants)

EVIDENCE_COLUMNS = ("evidence", "preuve")


def sentence_key(text):
    return "".join(c for c in unicodedata.normalize("NFKC", str(text or "")).lower()
                   if c.isascii() and c.isalnum())


def statistic(qualified, classes, n_classes):
    """Chi-squared statistic of the 2 x n_classes table (marker or not, verdict class)."""
    table = [[0] * n_classes, [0] * n_classes]
    for q, c in zip(qualified, classes):
        table[0 if q else 1][c] += 1
    rows, cols = [sum(r) for r in table], [sum(c) for c in zip(*table)]
    n = sum(rows)
    x = 0.0
    for i in range(2):
        for j in range(n_classes):
            e = rows[i] * cols[j] / n
            if e:
                x += (table[i][j] - e) ** 2 / e
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--markers", required=True, help="marker table written by 186_scope_markers.py")
    ap.add_argument("--permutations", type=int, default=50000, help="number of permutations (default 50000)")
    ap.add_argument("--seed", type=int, default=13, help="seed of the random generator (default 13)")
    args = ap.parse_args()

    with open(args.markers, encoding="utf-8") as f:
        raw = list(csv.DictReader(f, delimiter="\t"))
    rows = [{variants.COLUMNS.get(k, k): variants.VALUES.get(v, v) for k, v in r.items()} for r in raw]
    rows = [r for r in rows if r["set"] == "delivered"]
    column = next((c for c in EVIDENCE_COLUMNS if rows and c in rows[0]), None)
    if column is None:
        sys.exit("STOP: no evidence column (%s) in %s" % (" or ".join(EVIDENCE_COLUMNS), args.markers))
    words = [{w.strip() for c in variants.CATEGORIES for w in (r.get(c) or "").split(";") if w.strip()}
             for r in rows]
    classes = [variants.CLASSES.index(r["verifiers"]) for r in rows]

    groups = defaultdict(list)                      # sentence -> indices of its relations
    for i, r in enumerate(rows):
        key = sentence_key(r[column])
        if not key:
            sys.exit("STOP: relation %d has an empty evidence sentence" % i)
        groups[key].append(i)
    sentences = list(groups.values())
    by_size = defaultdict(list)                     # number of relations -> sentences
    for s in sentences:
        by_size[len(s)].append(s)
    sizes = Counter(len(s) for s in sentences)
    print("delivered relations: %d" % len(rows))
    print("distinct evidence sentences: %d" % len(sentences))
    print("relations per sentence: " + ", ".join("%d sentence(s) with %d" % (n, k) for k, n in sorted(sizes.items())))
    shared = [s for s in sentences if len(s) > 1]
    print("sentences shared by several relations: %d, holding %d relations"
          % (len(shared), sum(len(s) for s in shared)))
    print("permutations: %d, seed %d" % (args.permutations, args.seed))
    print("")

    for name, excluded in [("full", set()), ("reduced", variants.NEUTRAL),
                           ("strict", variants.NEUTRAL | variants.FIGURES)]:
        qualified = [bool(w - excluded) for w in words]
        n = sum(qualified)
        if n != variants.EXPECTED[name][0]:
            sys.exit("STOP: list %s gives %d qualified relations, the manuscript %d"
                     % (name, n, variants.EXPECTED[name][0]))
        mixed = sum(1 for s in shared if len({qualified[i] for i in s}) > 1)
        observed = statistic(qualified, classes, len(variants.CLASSES))
        rng = random.Random(args.seed)
        at_least = 0
        for _ in range(args.permutations):
            permuted = list(qualified)
            for group in by_size.values():
                order = list(range(len(group)))
                rng.shuffle(order)
                for target, source in zip(group, order):
                    for a, b in zip(target, group[source]):
                        permuted[a] = qualified[b]
            if statistic(permuted, classes, len(variants.CLASSES)) >= observed - 1e-12:
                at_least += 1
        p = (at_least + 1) / (args.permutations + 1)
        print("%-8s qualified %3d/%d  chi2 = %.2f  p by sentence = %.2f (%d of %d permutations)"
              "  [independent relations: p = %.2f]  sentences whose relations differ: %d"
              % (name, n, len(rows), observed, p, at_least, args.permutations,
                 variants.chi2_sf_3(observed), mixed))
    print("")
    print("[done]")


if __name__ == "__main__":
    main()
