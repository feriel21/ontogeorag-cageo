#!/usr/bin/env python3
"""Figure 7: corroboration of the delivered relations across the corpus.

(a) Number of relations by the number of papers in which the two arguments of a
    relation co-occur in the same indexed passage, counted at concept-family level.
(b) The same count by extraction tier: individual relations, box (quartiles, whiskers
    to the minimum and maximum), mean and median.

Inputs
    --corroboration  per-relation CSV written by scripts/analysis/family_corroboration.py
                     run with --families configs/entity_families.csv
                     (columns subject, relation, object, papers_family)
    --graph          delivered graph (flat_run16E_enforced_dedup.json); the tier of each
                     relation is read from it and joined by content key
                     (subject, relation, object, lower-cased), never by position

Checks (the figure is not written if any fails)
    232 relations, all joined to a tier; bins 0, 1, 2-3, 4-7, 8-15, 16+ hold
    17, 42, 26, 24, 62, 61 relations; Tier 1 n = 160, mean 9.4, median 8;
    Tier 2 n = 72, mean 10.1, median 8 (means rounded to one decimal).
    --no-check draws whatever the inputs give, for use on other runs.

Point jitter in panel (b) uses a fixed seed, so the figure is reproducible.

Usage
    python scripts/figures/fig_corroboration.py --corroboration output/family_corroboration.csv --graph /path/to/run16E/kg_final/flat_run16E_enforced_dedup.json --out output/figures/fig_corroboration
"""
import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINS = [("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16+", 16, None)]
EXPECTED_BINS = [17, 42, 26, 24, 62, 61]
EXPECTED_TIERS = {"1": (160, 9.4, 8), "2": (72, 10.1, 8)}
COLOURS = ["#d3d9e2", "#d3d9e2", "#8497b0", "#8497b0", "#24456d", "#24456d"]
INK, MUTED, BOX_FILL, POINT = "#1b1b1b", "#555555", "#dfe5ee", "#3b5a80"


def fail(msg):
    print("ERROR: " + msg, flush=True)
    sys.exit(1)


def key(s, r, o):
    return (str(s).strip().lower(), str(r).strip(), str(o).strip().lower())


def tier_label(value):
    text = str(value).strip().lower().replace("tier", "").strip()
    return text


def load(corroboration, graph):
    obj = json.loads(Path(graph).read_text(encoding="utf-8"))
    triples = obj["triples"] if isinstance(obj, dict) else obj
    tiers = {key(t.get("subject"), t.get("relation"), t.get("object")): tier_label(t.get("tier"))
             for t in triples}
    with open(corroboration, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    papers, by_tier, unjoined = [], {}, 0
    for r in rows:
        k = key(r["subject"], r["relation"], r["object"])
        n = int(r["papers_family"])
        papers.append(n)
        if k not in tiers:
            unjoined += 1
            continue
        by_tier.setdefault(tiers[k], []).append(n)
    return papers, by_tier, unjoined


def bin_counts(papers):
    return [sum(1 for n in papers if n >= lo and (hi is None or n <= hi)) for _, lo, hi in BINS]


def check(papers, by_tier, unjoined):
    problems = []
    if len(papers) != 232:
        problems.append("relations: %d, expected 232" % len(papers))
    if unjoined:
        problems.append("%d relations without a tier in the graph" % unjoined)
    counts = bin_counts(papers)
    if counts != EXPECTED_BINS:
        problems.append("bins %s, expected %s" % (counts, EXPECTED_BINS))
    for label, (n, mean, median) in EXPECTED_TIERS.items():
        values = by_tier.get(label, [])
        got = (len(values), round(sum(values) / len(values), 1) if values else None,
               statistics.median(values) if values else None)
        if got != (n, mean, median):
            problems.append("tier %s: (n, mean, median) %s, expected %s" % (label, got, (n, mean, median)))
    return problems


def draw(papers, by_tier, out, width, height):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height),
                                   gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.35})

    counts = bin_counts(papers)
    labels = [b[0] for b in BINS]
    bars = ax1.bar(labels, counts, color=COLOURS, edgecolor="#6b7280", linewidth=0.6, width=0.72)
    for b, v in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.2, str(v), ha="center", va="bottom",
                 fontsize=7, color=INK)
    at_most_one = counts[0] + counts[1]
    ax1.text(0.16, 0.93, "%d of %d rest on\nat most one paper" % (at_most_one, len(papers)),
             transform=ax1.transAxes, ha="left", va="top", fontsize=6.8, color=MUTED, linespacing=1.3)
    ax1.set_ylim(0, max(counts) * 1.16)
    ax1.set_ylabel("Relations")
    ax1.set_xlabel("Papers in which the relation's two concepts co-occur")
    ax1.set_title("(a) corroboration across the %d papers" % 41, loc="left", fontsize=7.8)

    order = sorted(by_tier)
    rng = random.Random(0)
    positions = list(range(1, len(order) + 1))
    data = [by_tier[t] for t in order]
    ax2.boxplot(data, positions=positions, widths=0.5, whis=(0, 100), patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=BOX_FILL, edgecolor="#24456d", linewidth=0.8),
                medianprops=dict(color="#1b1b1b", linewidth=1.3),
                whiskerprops=dict(color="#8a8a8a", linewidth=0.7),
                capprops=dict(color="#8a8a8a", linewidth=0.7))
    for pos, values in zip(positions, data):
        xs = [pos + rng.uniform(-0.17, 0.17) for _ in values]
        ax2.scatter(xs, values, s=3.5, color=POINT, alpha=0.65, linewidths=0, zorder=3)
        mean = sum(values) / len(values)
        ax2.text(pos + 0.08, max(values) + 0.8, "mean %.1f\nmedian %g" % (mean, statistics.median(values)),
                 ha="left", va="bottom", fontsize=6.6, color=MUTED, linespacing=1.25)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(["Tier %s\n(n = %d)" % (t, len(by_tier[t])) for t in order])
    ax2.set_xlim(0.4, len(order) + 0.9)
    ax2.set_ylim(-1, max(max(v) for v in data) * 1.18)
    ax2.set_title("(b) the same, by tier", loc="left", fontsize=7.8)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out) + "." + ext, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written: %s.pdf and %s.png" % (out, out))


def main():
    ap = argparse.ArgumentParser(description="Figure 7: corroboration of the delivered relations")
    ap.add_argument("--corroboration", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", default="fig_corroboration")
    ap.add_argument("--width", type=float, default=6.3)
    ap.add_argument("--height", type=float, default=2.8)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    papers, by_tier, unjoined = load(args.corroboration, args.graph)
    if not args.no_check:
        problems = check(papers, by_tier, unjoined)
        if problems:
            fail("inputs do not reproduce the published figure; nothing written\n  - " + "\n  - ".join(problems))
    print("bins " + str(bin_counts(papers)) + "; tiers " + ", ".join(
        "%s: n=%d mean=%.1f median=%g" % (t, len(v), sum(v) / len(v), statistics.median(v))
        for t, v in sorted(by_tier.items())))
    draw(papers, by_tier, args.out, args.width, args.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())