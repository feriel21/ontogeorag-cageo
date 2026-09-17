#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3: the five ablation configurations in the plane of coverage against benchmark recovery.

x axis: distinct descriptors recovered; y axis: benchmark edges recovered (of 34).
The chain A -> B -> C -> E, in which each step changes one factor, is drawn with
arrows; configuration D, which changes two factors, is drawn apart with another marker.

Data. The five pairs are those of Table 5 and are written in CONFIGS below; this
script reads no file. If Table 5 changes, change CONFIGS and nothing else
(scripts/paper/table5_ablation.py produces the table).

Size. 6.3 inches wide, the text width of the manuscript class; include with
width equal to the line width, without further scaling.

Historical name: 95y_fig_tradeoff.py

Usage
    python scripts/figures/fig_tradeoff.py --out output/figures/fig_tradeoff
    python scripts/figures/fig_tradeoff.py --out output/figures/fig_tradeoff --greyscale
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (label, prompt vocabulary, queries, distinct descriptors, benchmark edges of 34)
CONFIGS = [
    ("A", "13 imposed",   249, 21, 25),
    ("B", "13 imposed",   287, 23, 25),
    ("C", "38 imposed",   287, 30, 22),
    ("D", "none",         249, 23, 21),
    ("E", "38 suggested", 287, 31, 18),
]

CHAIN = ["A", "B", "C", "E"]


def main():
    ap = argparse.ArgumentParser(
        description="Coverage / benchmark trade-off figure (Figure 3)")
    ap.add_argument("--out", default="fig_tradeoff")
    ap.add_argument("--width", type=float, default=6.3)
    ap.add_argument("--height", type=float, default=4.2)
    ap.add_argument("--fontsize", type=float, default=9.0)
    ap.add_argument("--greyscale", action="store_true")
    args = ap.parse_args()

    by = {c[0]: c for c in CONFIGS}

    col_chain = "#555555" if args.greyscale else "#1f4e79"
    col_d = "#888888" if args.greyscale else "#b4472f"

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    # chain A -> B -> C -> E, one factor at a time
    xs = [by[k][3] for k in CHAIN]
    ys = [by[k][4] for k in CHAIN]
    for i in range(len(CHAIN) - 1):
        ax.annotate(
            "", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="-|>", color=col_chain,
                            linewidth=0.9, shrinkA=7, shrinkB=9,
                            alpha=0.75))
    ax.scatter(xs, ys, s=70, color=col_chain, marker="o",
               zorder=3, edgecolors="white", linewidths=0.8)

    # configuration D, drawn apart
    d = by["D"]
    ax.scatter([d[3]], [d[4]], s=70, color=col_d, marker="s",
               zorder=3, edgecolors="white", linewidths=0.8)

    # labels: letter and what the configuration does
    offsets = {"A": (0.35, 0.30), "B": (0.35, -0.55), "C": (0.35, 0.30),
               "D": (0.35, 0.30), "E": (-0.45, 0.45)}
    for key, vocab, nq, ndesc, nedges in CONFIGS:
        dx, dy = offsets[key]
        ha = "right" if dx < 0 else "left"
        ax.text(ndesc + dx, nedges + dy,
                "{0}: {1}, {2} queries".format(key, vocab, nq),
                fontsize=args.fontsize - 1, ha=ha, va="bottom",
                color="#222222")

    ax.set_xlabel("Distinct descriptors recovered", fontsize=args.fontsize)
    ax.set_ylabel("Benchmark edges recovered (of 34)",
                  fontsize=args.fontsize)
    ax.tick_params(labelsize=args.fontsize - 1)
    ax.set_xlim(19, 34)
    ax.set_ylim(16, 27.5)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)

    handles = [
        Line2D([], [], marker="o", linestyle="-", color=col_chain,
               markersize=6, linewidth=0.9,
               label="single-factor chain A to B to C to E"),
        Line2D([], [], marker="s", linestyle="none", color=col_d,
               markersize=6,
               label="configuration D, two factors, measuring instrument"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              fontsize=args.fontsize - 1, handletextpad=0.6)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out) + "." + ext, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("written: {0}.pdf and {0}.png".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())