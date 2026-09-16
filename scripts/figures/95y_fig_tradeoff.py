#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
95y_fig_tradeoff.py — La contrainte, en une figure

CE QUE LA FIGURE MONTRE
    Les cinq configurations d'ablation dans le plan (descripteurs
    distincts recuperes, aretes de benchmark retrouvees). La chaine
    A -> B -> C -> E est reliee par une fleche : elle varie un facteur a
    la fois. La configuration D varie deux facteurs et est dessinee
    detachee, avec un marqueur different, comme instrument de mesure et
    non comme terme de la comparaison.

    Le message est la pente : ouvrir le vocabulaire gagne des
    descripteurs et perd des aretes, et le benchmark ne peut pas
    arbitrer puisqu'il est construit sur le vocabulaire ferme qu'il sert
    a noter.

DONNEES
    Les cinq couples sont ceux du tableau d'ablation du manuscrit
    (Table 5). Aucune donnee nouvelle n'est lue : si le tableau change,
    changer CONFIGS ci-dessous et rien d'autre.

USAGE
    python3 95y_fig_tradeoff.py --out ~/ontogeorag/output/figures/fig_tradeoff
    Ajouter --greyscale pour le controle d'impression.

TAILLE
    6.3 pouces de large = \\linewidth en a4paper 12pt. Inclure avec
    width=\\linewidth et ne pas agrandir.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (etiquette, vocabulaire, requetes, descripteurs distincts, aretes / 34)
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
        description="Figure du compromis couverture / benchmark")
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

    # chaine A -> B -> C -> E, un facteur a la fois
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

    # configuration D, detachee
    d = by["D"]
    ax.scatter([d[3]], [d[4]], s=70, color=col_d, marker="s",
               zorder=3, edgecolors="white", linewidths=0.8)

    # etiquettes : lettre + ce que la configuration fait
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
    print("ecrit : {0}.pdf et {0}.png".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())