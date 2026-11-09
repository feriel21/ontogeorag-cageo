#!/usr/bin/env python3
"""Figure fig:assessments — the 364 delivered relations of run18 under seven assessments.

Each row partitions the same 364 relations; no row is a verdict on its own.
    Provenance      evidence sentence found in the passage index exactly / partially / not found
                    (187, column phrase_affichee_statut)
    Extraction      relation extracted in both passes / at temperature 0 only / at 0.3 only
                    (field origin: both, iter_a, iter_b)
    Self-check      verdict of the extracting-LLM, merged over the two passes (field verdict)
    Panel           decision of the panel of two verifying-LLMs (field m4.decision)
    Tier            Tier 1 / Tier 2 (field tier)
    Evidence check  flagged when at least one verifying-LLM returns NOT_SUPPORTED on the
                    evidence sentence (field m4.evidence_verdict)
    Written rules   kept in / removed from the cleaned graph (188)

Checks (nothing is written if one fails): 364 relations; 316/47/1; 147/122/95; 266/98;
238/126; 197/167; 285/79; 281/83, and the 281 + 83 relations are exactly the 364.

Usage
    python scripts/figures/run18/fig_assessments_run18.py --out output/figures/fig_assessments_run18
"""
import argparse
import csv
from collections import Counter

import matplotlib.pyplot as plt

import run18_style as S

EXPECTED = {"n": 364, "provenance": (316, 47, 1), "passes": (147, 122, 95), "self_check": (266, 98),
            "panel": (238, 126), "tier": (197, 167), "evidence": (285, 79), "rules": (281, 83),
            "rules_cover_graph": True}


def key(s, r, o):
    return (S.norm(s), r, S.norm(o))


def measure():
    delivered = S.triples(S.DELIVERED)
    table = list(csv.DictReader(open(S.PROVENANCE_TABLE, encoding="utf-8"), delimiter="\t"))
    cleaned = S.triples(S.CLEANED)
    removed = list(csv.DictReader(open(S.REMOVED, encoding="utf-8"), delimiter="\t"))
    located = Counter(r["phrase_affichee_statut"] for r in table)
    origin = Counter(t["origin"] for t in delivered)
    verdict = Counter(t["verdict"] for t in delivered)
    decision = Counter(t["m4"]["decision"] for t in delivered)
    tier = Counter(t["tier"] for t in delivered)
    n_flag = sum(S.flagged(t) for t in delivered)
    all_keys = Counter(key(t["subject"], t["relation"], t["object"]) for t in delivered)
    split = Counter(key(t["subject"], t["relation"], t["object"]) for t in cleaned) \
        + Counter(key(r["sujet"], r["relation"], r["objet"]) for r in removed)
    return {
        "n": len(delivered),
        "provenance": (located["exact"], located["partial"], located["not_found"]),
        "passes": (origin["both"], origin["iter_a"], origin["iter_b"]),
        "self_check": (verdict["STRONG_SUPPORT"], verdict["WEAK_SUPPORT"]),
        "panel": (decision["ACCEPT"], decision["UNCERTAIN"]),
        "tier": (tier[1], tier[2]),
        "evidence": (len(delivered) - n_flag, n_flag),
        "rules": (len(cleaned), len(removed)),
        "rules_cover_graph": split == all_keys,
    }


def rows(m):
    p = m["provenance"]
    return [
        ("Provenance", [p[0], p[1], p[2]],
         "evidence sentence found exactly %d, partially %d, not found %d" % p),
        ("Extraction", list(m["passes"]),
         "both passes %d, temperature 0 only %d, temperature 0.3 only %d" % m["passes"]),
        ("Self-check", list(m["self_check"]),
         "strong support %d, weak support %d (extracting-LLM, both passes merged)" % m["self_check"]),
        ("Panel", list(m["panel"]), "accepted %d, uncertain %d" % m["panel"]),
        ("Tier", list(m["tier"]), "Tier 1 %d, Tier 2 %d" % m["tier"]),
        ("Evidence check", list(m["evidence"]),
         "not flagged %d, flagged by at least one verifying-LLM %d" % m["evidence"]),
        ("Written rules", list(m["rules"]), "kept %d, removed %d" % m["rules"]),
    ]


def draw(m, out, width, height):
    S.setup()
    data = rows(m)
    colours = [S.DARK, S.MID, S.LIGHT]
    fig, ax = plt.subplots(figsize=(width, height))
    n = m["n"]
    step = 1.25
    for i, (name, values, sub) in enumerate(data):
        y = (len(data) - 1 - i) * step
        left = 0
        for value, colour in zip(values, colours):
            if value:
                ax.barh(y, value, left=left, height=0.62, color=colour, edgecolor=S.INK, linewidth=0.6)
                if value >= 12:
                    ax.text(left + value / 2, y, str(value), ha="center", va="center", fontsize=7.5,
                            color="white" if colour == S.DARK else S.INK)
            left += value
        ax.text(-n * 0.012, y, name, ha="right", va="center", fontsize=8.2, fontweight="bold", color=S.INK)
        ax.text(0, y - 0.47, sub, ha="left", va="center", fontsize=6.6, color=S.MUTED)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.85, (len(data) - 1) * step + 0.45)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
    ax.set_xlabel("Delivered relations (%d)" % n)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    S.save(fig, out)


def main():
    ap = argparse.ArgumentParser(description="Figure fig:assessments of the run18 manuscript")
    ap.add_argument("--out", default="output/figures/fig_assessments_run18")
    ap.add_argument("--width", type=float, default=6.3)
    ap.add_argument("--height", type=float, default=4.1)
    args = ap.parse_args()
    m = measure()
    print(m)
    S.check(m, EXPECTED)
    draw(m, args.out, args.width, args.height)


if __name__ == "__main__":
    main()
