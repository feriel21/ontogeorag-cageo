#!/usr/bin/env python3
"""
98_make_figures.py - Manuscript figures from the verified session results

ADDITIVE. Writes PDF + PNG into --outdir. Reads nothing, computes nothing:
every number below was measured on the cluster and is hard-coded here with
the script that produced it, so a figure can always be traced to its source.

PROVENANCE OF EACH NUMBER
  fig_threshold_curve   : build_canonical_map re-run at 5 thresholds on the
                          216 delivered entities
  fig_redundancy        : 91_classify_redundancy_groups.py + manual
                          adjudication of the 8 UNCLASSIFIED groups. Panel (b)
                          third bar is an upper bound: family_auto grouped 23
                          heterogeneous entities under "depositional setting",
                          so the 54.2% is not a validated measurement
  fig_verifier_rules    : 94_blind_pass_characterization.py and
                          95_nli_verifier_substitution.py, 464 controls
  fig_corruption_class  : same two scripts, per-class sensitivity
  fig_corroboration     : 08_rebuild_provenance.py on run16E

STYLE
  Colour, but every series also differs by hatch or marker so the figures
  survive greyscale printing and the common forms of colour blindness.
  Annotations are placed in empty regions with an opaque background, never
  across bars. Type-42 fonts for Elsevier submission.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "grid.color": "#d3d9e0",
    "grid.linewidth": 0.45,
    "axes.axisbelow": True,
})


def grid_on(ax, axis="y"):
    """Light reference grid behind the data, so values can be read off."""
    ax.grid(axis=axis, which="major", zorder=0)
    ax.grid(axis=axis, which="minor", lw=0.3, color="#e8ecf0", zorder=0)
    ax.set_axisbelow(True)

# Palette: colour-blind safe, distinguishable in greyscale by lightness.
BLUE = "#2166ac"
ORANGE = "#d6604d"
TEAL = "#01665e"
GOLD = "#dfc27d"
GREY = "#9aa5b1"
DARK = "#1a1a1a"
RED = "#b2182b"

# Opaque box for annotations placed over a busy area.
BOX = dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.92)

W1 = 3.54   # single column, inches
W2 = 7.28   # double column


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(Path(outdir) / (name + "." + ext), dpi=400,
                    bbox_inches="tight")
    plt.close(fig)
    print("  " + name + ".pdf / .png")


# ---------------------------------------------------------------- fig 1
def fig_threshold_curve(outdir):
    thr = [0.06, 0.10, 0.15, 0.20, 0.30]
    merges = [5, 38, 83, 116, 1]
    blocked = [9, 15, 24, 39, 207]

    fig, ax = plt.subplots(figsize=(W1 * 1.25, 2.7))
    ax.plot(thr, merges, "-o", color=BLUE, lw=1.5, ms=5,
            label="merges applied")
    ax.plot(thr, blocked, "--s", color=ORANGE, lw=1.5, ms=5, mfc="white",
            mec=ORANGE, mew=1.2, label="merges blocked by the guard")

    ax.axvline(0.06, color=TEAL, lw=1.0, ls=":", zorder=0)
    ax.text(0.066, 243, "shipped threshold (0.06)", fontsize=6.8,
            color=TEAL, ha="left", va="top")

    # Placed in the empty upper-left band; the arrow runs below the curves.
    ax.annotate("guard saturates: all 216 entities\nfall into a single cluster,\n"
                "so almost nothing can merge",
                xy=(0.293, 10), xytext=(0.072, 196), fontsize=6.8, color=DARK,
                ha="left", va="top", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=DARK,
                                connectionstyle="arc3,rad=-0.28",
                                shrinkB=6))

    ax.set_xlabel("agglomerative distance threshold")
    ax.set_ylabel("entity pairs")
    ax.set_xticks(thr)
    ax.set_xticklabels(["0.06", "0.10", "0.15", "0.20", "0.30"])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, columnspacing=1.4)
    ax.set_ylim(-12, 262)
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.set_yticks([25, 75, 125, 175, 225], minor=True)
    grid_on(ax, "y")
    save(fig, outdir, "fig_threshold_curve")


# ---------------------------------------------------------------- fig 2
def fig_redundancy(outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 2.8),
                                    gridspec_kw={"width_ratios": [1.4, 1]})

    labels = ["morphological variants",
              "adjudicated: redundant",
              "adjudicated: distinct",
              "instance leakage",
              "antonym false positive",
              "type conflict"]
    vals = [25, 11, 7, 18, 8, 29]
    hatches = ["", "", "..", "//", "xx", "\\\\"]
    colours = [BLUE, BLUE, GREY, GOLD, ORANGE, TEAL]

    left = 0
    for v, h, c in zip(vals, hatches, colours):
        ax1.barh(0, v, left=left, height=0.42, color=c, hatch=h,
                 edgecolor="white", lw=0.9)
        ax1.text(left + v / 2, 0, str(v), ha="center", va="center",
                 fontsize=6.8, color="white")
        left += v

    ax1.set_xlim(0, 104)
    ax1.set_ylim(-1.15, 0.62)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    grid_on(ax1, "x")
    ax1.set_yticks([])
    ax1.set_xlabel("entities (98 of 216 sit in a lexical group)")
    ax1.set_title("(a)  what the raw 45.4% figure contains", loc="left")

    ax1.annotate("", xy=(0, -0.34), xytext=(36, -0.34),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color=BLUE))
    ax1.text(18, -0.46, "36 = genuine redundancy\n(16.7% of 216)",
             fontsize=6.6, color=BLUE, ha="center", va="top", linespacing=1.35)
    ax1.annotate("", xy=(36, -0.34), xytext=(98, -0.34),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color=GREY))
    ax1.text(67, -0.46, "62 = false positives\nof the measure",
             fontsize=6.6, color="#5b6670", ha="center", va="top",
             linespacing=1.35)

    ax1.legend(handles=[Patch(facecolor=c, hatch=h, edgecolor="white",
                              label=lab)
                        for c, h, lab in zip(colours, hatches, labels)],
               frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.34), handlelength=1.5,
               columnspacing=1.2, labelspacing=0.35)

    # The conceptual-family level is an UPPER BOUND, not a measurement: it
    # rests on a grouping in which one family lumped 23 heterogeneous
    # entities together (named basins with generic bathymetric positions).
    # It is drawn hatched and labelled as such rather than dropped, because
    # the gap it opens is the point of the panel.
    lv = ["lexical\n(audited)", "nominal\nhead", "conceptual\nfamily"]
    pc = [16.7, 33.3, 54.2]
    bars = ax2.bar(lv, pc, color=[GOLD, ORANGE, TEAL], width=0.6,
                   edgecolor="white", lw=0.9,
                   hatch=["..", "//", "xx"])
    bars[2].set_alpha(0.55)
    for i, (b, v) in enumerate(zip(bars, pc)):
        lab = str(v) + "%" + (" *" if i == 2 else "")
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.6, lab,
                 ha="center", fontsize=7.5)
    ax2.annotate("", xy=(0.05, 19.5), xytext=(1.95, 51.5),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color=RED))
    ax2.text(0.95, 44, "the gap is invisible to\nany surface measure",
             fontsize=6.6, color=RED, ha="center", va="top",
             linespacing=1.35, bbox=BOX)
    ax2.text(0.5, -0.20, "* upper bound: one family over-grouped 23 entities",
             transform=ax2.transAxes, fontsize=6.3, color="#5b6670",
             ha="center", va="top")
    ax2.set_ylabel("entities in a redundant group (%)")
    ax2.set_ylim(0, 66)
    ax2.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax2.set_yticks([5, 15, 25, 35, 45, 55], minor=True)
    grid_on(ax2, "y")
    ax2.set_title("(b)  three measurement levels", loc="left")
    save(fig, outdir, "fig_redundancy")


# ---------------------------------------------------------------- fig 3
def fig_verifier_rules(outdir):
    rules = ["blind pass\nalone", "shipped\nmatrix", "evidence\nalone",
             "NLI\nalone", "evidence-\ndominant", "NLI +\nevidence"]
    sens = [0.086, 0.086, 0.974, 0.772, 0.634, 0.767]
    spec = [0.953, 0.974, 0.328, 0.711, 0.853, 0.737]
    bal = [0.519, 0.530, 0.651, 0.741, 0.744, 0.752]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 3.1),
                                    gridspec_kw={"width_ratios": [2.0, 1]})

    x = list(range(len(rules)))
    w = 0.27
    ax1.bar([i - w for i in x], sens, w, color=BLUE, edgecolor="white",
            lw=0.7, label="sensitivity")
    ax1.bar(x, spec, w, color=GOLD, hatch="//", edgecolor="white",
            lw=0.7, label="specificity")
    ax1.bar([i + w for i in x], bal, w, color=TEAL, edgecolor="white",
            lw=0.7, label="balanced accuracy")

    for i, v in enumerate(sens):
        ax1.text(i - w, v + 0.025, f"{v:.2f}", ha="center", fontsize=6.3,
                 color=RED if v < 0.2 else DARK,
                 fontweight="bold" if v < 0.2 else "normal")
    for i, v in enumerate(bal):
        ax1.text(i + w, v + 0.025, f"{v:.3f}", ha="center", fontsize=6.3)

    ax1.axhline(0.5, color=GREY, lw=0.8, ls=":", zorder=0)
    ax1.text(5.48, 0.515, "chance", fontsize=6.4, color="#5b6670", ha="right")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rules, fontsize=7.0)
    ax1.set_ylabel("score")
    ax1.set_ylim(0, 1.52)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    grid_on(ax1, "y")
    ax1.legend(frameon=False, loc="upper right", ncol=3, columnspacing=1.1,
               bbox_to_anchor=(1.0, 1.03))
    ax1.set_title("(a)  aggregation rules on 464 adversarial controls",
                  loc="left", pad=16)

    # No arrow: it would have to cross the bars. The two sensitivity values
    # are printed in bold red on the bars themselves, which carries the link.
    ax1.text(-0.42, 1.34, "the shipped matrix inherits the blind pass\n"
                          "sensitivity unchanged (0.086, in red)",
             fontsize=6.6, color=RED, ha="left", va="top", linespacing=1.4)

    bars = ax2.bar(["clean\ncontrols", "corrupted\ncontrols"], [95.3, 91.4],
                   color=[TEAL, ORANGE], width=0.52, edgecolor="white",
                   lw=0.9, hatch=["", "xx"])
    for b, v in zip(bars, [95.3, 91.4]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v}%",
                 ha="center", fontsize=7.5)
    ax2.set_ylabel("verdict PLAUSIBLE (%)")
    ax2.set_ylim(0, 138)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_yticks([10, 30, 50, 70, 90], minor=True)
    grid_on(ax2, "y")
    ax2.set_title("(b)  the blind pass carries no signal", loc="left", pad=16)
    ax2.annotate("", xy=(0, 108), xytext=(1, 108),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color=RED))
    ax2.text(0.5, 128, "3.9-point gap\nbalanced accuracy 0.519",
             ha="center", va="top", fontsize=6.8, color=RED, linespacing=1.35)
    save(fig, outdir, "fig_verifier_rules")


# ---------------------------------------------------------------- fig 4
def fig_corruption_class(outdir):
    classes = ["passage\npermutation", "entity\nsubstitution",
               "relation\nsubstitution", "direction\ninversion"]
    n = [58, 90, 58, 26]
    evidence = [1.000, 0.989, 0.983, 0.846]
    matrix = [0.034, 0.100, 0.121, 0.077]
    dominant = [0.828, 0.722, 0.397, 0.423]

    fig, ax = plt.subplots(figsize=(W2, 2.9))
    x = list(range(len(classes)))
    w = 0.26

    ax.axvspan(1.5, 3.5, color=RED, alpha=0.05, zorder=0)

    ax.bar([i - w for i in x], evidence, w, color=BLUE, edgecolor="white",
           lw=0.7, label="evidence alone")
    ax.bar(x, matrix, w, color=GOLD, hatch="//", edgecolor="white",
           lw=0.7, label="shipped matrix")
    ax.bar([i + w for i in x], dominant, w, color=TEAL, edgecolor="white",
           lw=0.7, label="evidence-dominant")

    for i, (a, b, c) in enumerate(zip(evidence, matrix, dominant)):
        ax.text(i - w, a + 0.022, f"{a:.2f}", ha="center", fontsize=6.3)
        ax.text(i, b + 0.022, f"{b:.2f}", ha="center", fontsize=6.3, color=RED)
        ax.text(i + w, c + 0.022, f"{c:.2f}", ha="center", fontsize=6.3)

    ax.set_xticks(x)
    ax.set_xticklabels([c + "\n(n=" + str(k) + ")" for c, k in zip(classes, n)])
    ax.set_ylabel("sensitivity")
    ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    grid_on(ax, "y")
    ax.legend(frameon=False, ncol=3, loc="upper left", columnspacing=1.2,
              bbox_to_anchor=(0.0, 1.02))
    ax.set_title("sensitivity by corruption class, 464 controls",
                 loc="left", pad=16)

    ax.text(2.5, 1.30, "semantic corruption: recovery stays below 0.45",
            fontsize=6.8, color=RED, ha="center", va="top", bbox=BOX)
    ax.text(2.5, 1.20, "NLI fails here too (mean entailment 0.38 and 0.73)",
            fontsize=6.3, color="#5b6670", ha="center", va="top", bbox=BOX)
    save(fig, outdir, "fig_corruption_class")


# ---------------------------------------------------------------- fig 5
def fig_corroboration(outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 2.9),
                                    gridspec_kw={"width_ratios": [1.25, 1]})

    bins = ["1", "2-3", "4-7", "8-15", "16+"]
    counts = [128, 40, 24, 24, 16]
    cols = [ORANGE, GOLD, GREY, TEAL, BLUE]
    bars = ax1.bar(bins, counts, color=cols, width=0.62, edgecolor="white",
                   lw=0.9)
    for b, v in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, v + 3.5, str(v),
                 ha="center", fontsize=7)
    ax1.set_xlabel("distinct papers co-attesting the pair")
    ax1.set_ylabel("relations")
    ax1.set_ylim(0, 178)
    ax1.set_yticks([0, 25, 50, 75, 100, 125])
    ax1.set_yticks([12.5, 37.5, 62.5, 87.5, 112.5], minor=True)
    grid_on(ax1, "y")
    ax1.set_title("(a)  corroboration across 41 papers", loc="left")
    # "single paper" means no OTHER paper in the corpus contains both surface
    # forms in one chunk - not that the extraction missed something. Counted
    # at concept level the share falls to 23%, so half of these are attested
    # elsewhere under another form of the same concept.
    ax1.annotate("55% of relations: no other paper\n"
                 "pairs these two surface forms\n"
                 "(23% once counted at concept level)",
                 xy=(0.32, 126), xytext=(1.05, 172), fontsize=6.6,
                 color=ORANGE, ha="left", va="top", linespacing=1.35,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=ORANGE))

    lbl = ["Tier-1\n(n=160)", "Tier-2\n(n=72)"]
    mean = [3.41, 4.15]
    pct2 = [43.8, 47.2]
    xx = [0, 1]
    ax2.bar([i - 0.17 for i in xx], mean, 0.32, color=BLUE,
            edgecolor="white", lw=0.8, label="mean papers")
    ax2b = ax2.twinx()
    ax2b.bar([i + 0.17 for i in xx], pct2, 0.32, color=GOLD, hatch="//",
             edgecolor="white", lw=0.8, label="% with $\\geq$2 papers")
    ax2b.spines["top"].set_visible(False)
    for i, (m, p) in enumerate(zip(mean, pct2)):
        ax2.text(i - 0.17, m + 0.14, f"{m:.2f}", ha="center", fontsize=7)
        ax2b.text(i + 0.17, p + 1.6, f"{p:.1f}%", ha="center", fontsize=7)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(lbl)
    ax2.set_ylabel("mean supporting papers")
    ax2b.set_ylabel("relations with $\\geq$2 papers (%)")
    ax2.set_ylim(0, 6.6)
    ax2.set_yticks([0, 1, 2, 3, 4, 5, 6])
    grid_on(ax2, "y")
    ax2b.set_ylim(0, 73)
    ax2b.grid(False)
    ax2.set_title("(b)  tiering vs corroboration", loc="left", pad=16)
    ax2.text(0.5, 6.3, "$r = -0.072$ : statistically independent",
             ha="center", va="top", fontsize=7, color=RED, bbox=BOX)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.5)
    save(fig, outdir, "fig_corroboration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures_new")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print("Writing to " + str(out))
    fig_threshold_curve(out)
    fig_redundancy(out)
    fig_verifier_rules(out)
    fig_corruption_class(out)
    fig_corroboration(out)
    print("done")


if __name__ == "__main__":
    main()