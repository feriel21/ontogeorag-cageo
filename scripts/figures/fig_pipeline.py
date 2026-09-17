#!/usr/bin/env python3
"""Figure 1: pipeline overview.

Construction chain from the corpus to the delivered graph, and, in a dashed frame, the
evaluation procedures that take the delivered graph as input.

Print constraints
    Size: the figure is authored narrower than the text block, so that including it at
    the line width enlarges it rather than shrinking it.
    Fit: every label is measured against its box before saving; a label that does not fit
    is wrapped onto two or three lines, then shrunk, and any remaining overflow is
    reported and makes the script exit with code 1.
Colour is decorative only (one accent for the delivered graph), so the figure survives
greyscale printing.

Numbers drawn, with their source
    41 papers              --expect-docs 41 (index stage)
    1758 chunks            line count of the delivered index
    287, seven strategies  configs/descriptor_queries_v2.jsonl
    BM25 top 20, top 5     --bm25-topn 20, --top-k 5
    T = 0 and 0.3          --temperature, extraction passes a and b
    0.06                   --cluster-threshold 0.06
    160 / 72               tiers counted on the delivered graph
    189 / 43 / 3           m4_panel_report.json, 235 relations examined
    232 / 216              delivered graph
    26 / 34, 464           frozen benchmarks, controls file
    95                     panel-accepted relations with a directional predicate
    232, one assessor      expert review of the delivered graph

Historical name: 95q_fig_pipeline.py (corrected: expert-review box, directional-check
box, and two labels in which a line break was drawn as a literal backslash-n).

Usage
    python scripts/figures/fig_pipeline.py --outdir output/figures
    python scripts/figures/fig_pipeline.py --outdir output/figures --greyscale
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK = "#1b1b1b"
SUBC = "#4f4f4f"
RULE = "#9a9a9a"
FILL = "#ffffff"
ACCENT = "#25456d"
ACCENT_FILL = "#e9edf3"
BAND = "#6b6b6b"
FLOW = "#39424e"

TITLE_SIZE = 9.0
SUB_SIZE = 7.6
MIN_SIZE = 6.2
PAD_X = 0.016

_boxes = []


def set_scale(s):
    global TITLE_SIZE, SUB_SIZE, MIN_SIZE
    TITLE_SIZE *= s
    SUB_SIZE *= s
    MIN_SIZE *= s


def greyscale(on):
    global ACCENT, ACCENT_FILL
    if on:
        ACCENT, ACCENT_FILL = "#2b2b2b", "#ededed"


def _extent_of(fig, ax, artist):
    fig.canvas.draw()
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    bb = bb.transformed(ax.transData.inverted())
    return bb.width, bb.height


def _width_of(fig, ax, artist):
    return _extent_of(fig, ax, artist)[0]


def _height_of(fig, ax, artist):
    return _extent_of(fig, ax, artist)[1]


def _stack(fig, ax, box_xywh, t1, t2):
    """Place a title and a subtitle as a vertical stack that fits the box.
    Wrapping a label onto two lines changes its height, so the vertical
    positions cannot be fixed fractions of the box."""
    x, y, w, h = box_xywh
    pad_y, gap = 0.012, 0.010
    for _ in range(24):
        h1 = _height_of(fig, ax, t1)
        h2 = _height_of(fig, ax, t2) if t2 is not None else 0.0
        total = h1 + (gap + h2 if t2 is not None else 0.0)
        if total <= h - 2 * pad_y:
            break
        if t1.get_fontsize() <= MIN_SIZE:
            break
        t1.set_fontsize(t1.get_fontsize() - 0.2)
        if t2 is not None:
            t2.set_fontsize(max(MIN_SIZE, t2.get_fontsize() - 0.2))
    cy = y + h / 2
    if t2 is None:
        t1.set_position((x + w / 2, cy))
        return
    t1.set_position((x + w / 2, cy + (total / 2 - h1 / 2)))
    t2.set_position((x + w / 2, cy - (total / 2 - h2 / 2)))


def _wrap(words, n):
    if n <= 1:
        return [" ".join(words)]
    total = sum(len(w) for w in words) + len(words) - 1
    target = total / n
    lines, cur, cur_len = [], [], 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > target and len(lines) < n - 1:
            lines.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += add
    lines.append(" ".join(cur))
    return lines


def _fit(fig, ax, artist, text, max_w, size):
    w = _width_of(fig, ax, artist)
    if w <= max_w:
        return w
    words = text.replace("\n", " ").split()
    for n in (2, 3):
        if len(words) < n:
            break
        artist.set_text("\n".join(_wrap(words, n)))
        w = _width_of(fig, ax, artist)
        if w <= max_w:
            return w
    while w > max_w and size > MIN_SIZE:
        size -= 0.2
        artist.set_fontsize(size)
        w = _width_of(fig, ax, artist)
    return w


def box(fig, ax, x, y, w, h, title, sub=None, accent=False, dashed=False,
        muted=False):
    edge = ACCENT if accent else (BAND if muted else RULE)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.010",
        linewidth=1.4 if accent else 0.8,
        facecolor=ACCENT_FILL if accent else FILL, edgecolor=edge,
        linestyle=(0, (3, 2)) if dashed else "-"))
    max_w, cx = w - 2 * PAD_X, x + w / 2
    if sub:
        # a multi-line subtitle is taller: raise the title and lower the
        # subtitle so that the two do not overlap.
        n_lines = sub.count("\n") + 1
        y_title = y + h * (0.74 if n_lines > 1 else 0.66)
        y_sub = y + h * (0.32 if n_lines > 1 else 0.29)
        t1 = ax.text(cx, y_title, title, ha="center", va="center",
                     fontsize=TITLE_SIZE, color=INK, fontweight="bold")
        t2 = ax.text(cx, y_sub, sub, ha="center", va="center",
                     fontsize=SUB_SIZE, color=SUBC, linespacing=1.15)
        pairs = [(t1, title, TITLE_SIZE), (t2, sub, SUB_SIZE)]
    else:
        t1 = ax.text(cx, y + h / 2, title, ha="center", va="center",
                     fontsize=TITLE_SIZE, color=INK, fontweight="bold")
        pairs = [(t1, title, TITLE_SIZE)]
    for artist, raw, size in pairs:
        _fit(fig, ax, artist, raw, max_w, size)
    _stack(fig, ax, (x, y, w, h), pairs[0][0],
           pairs[1][0] if len(pairs) > 1 else None)
    _boxes.append((title, (x, y, w, h), pairs))
    return (x, y, w, h)


def arrow(fig, ax, p0, p1, color=FLOW, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color,
                                 linewidth=lw, linestyle=ls,
                                 mutation_scale=9, shrinkA=0, shrinkB=0))


def line(ax, xs, ys, color=FLOW, lw=1.0, ls="-"):
    ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, solid_capstyle="butt")


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def build(fig, ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    b_query = box(fig, ax, 0.395, 0.905, 0.200, 0.075,
                  "Queries", "287, seven strategies", dashed=True)

    y1, h1 = 0.745, 0.115
    b_corpus = box(fig, ax, 0.020, y1, 0.165, h1, "Corpus", "41 papers")
    b_index = box(fig, ax, 0.210, y1, 0.180, h1, "Chunking",
                  "1758 chunks")
    b_bm25 = box(fig, ax, 0.415, y1, 0.180, h1, "Lexical retrieval",
                 "BM25, top 20")
    b_rank = box(fig, ax, 0.620, y1, 0.175, h1, "Re-ranking",
                 "cross-encoder, top 5")

    arrow(fig, ax, right(b_corpus), left(b_index))
    arrow(fig, ax, right(b_index), left(b_bm25))
    arrow(fig, ax, right(b_bm25), left(b_rank))
    arrow(fig, ax, bottom(b_query), top(b_bm25), ls=(0, (3, 2)))

    ya, yb, h2 = 0.590, 0.450, 0.105
    b_pa = box(fig, ax, 0.205, ya, 0.240, h2, "Extraction A",
               "Qwen 2.5-7B, T = 0")
    b_pb = box(fig, ax, 0.205, yb, 0.240, h2, "Extraction B",
               "Qwen 2.5-7B, T = 0.3")
    b_va = box(fig, ax, 0.470, ya, 0.215, h2, "Verification",
               "four verdicts")
    b_vb = box(fig, ax, 0.470, yb, 0.215, h2, "Verification",
               "four verdicts")
    b_canon = box(fig, ax, 0.715, 0.520, 0.265, h2, "Canonicalisation",
                  "SciBERT, threshold 0.06")

    bus1, xin = 0.700, 0.145
    line(ax, [bottom(b_rank)[0]] * 2, [y1, bus1])
    line(ax, [bottom(b_rank)[0], xin], [bus1, bus1])
    line(ax, [xin, xin], [bus1, yb + h2 / 2])
    arrow(fig, ax, (xin, ya + h2 / 2), left(b_pa))
    arrow(fig, ax, (xin, yb + h2 / 2), left(b_pb))
    arrow(fig, ax, right(b_pa), left(b_va))
    arrow(fig, ax, right(b_pb), left(b_vb))
    arrow(fig, ax, right(b_va), (0.715, 0.600))
    arrow(fig, ax, right(b_vb), (0.715, 0.545))

    y3, h3 = 0.275, 0.125
    b_fuse = box(fig, ax, 0.020, y3, 0.235, h3, "Fusion and tiers",
                 "Tier 1: 160   Tier 2: 72\nboth retained")
    b_panel = box(fig, ax, 0.295, y3, 0.265, h3, "Cross-family panel",
                  "Llama + Mistral, 189 / 43 / 3")
    b_schema = box(fig, ax, 0.600, y3, 0.175, h3, "Schema",
                   "typing, dedup")
    b_graph = box(fig, ax, 0.815, y3 - 0.014, 0.165, h3 + 0.028,
                  "Delivered graph", "232 relations\n216 entities",
                  accent=True)

    bus2, xcan, xfu = 0.420, 0.735, bottom(b_fuse)[0]
    line(ax, [xcan, xcan], [0.520, bus2])
    line(ax, [xcan, xfu], [bus2, bus2])
    arrow(fig, ax, (xfu, bus2), (xfu, y3 + h3))
    arrow(fig, ax, right(b_fuse), left(b_panel))
    arrow(fig, ax, right(b_panel), left(b_schema))
    arrow(fig, ax, right(b_schema), left(b_graph))

    ax.add_patch(Rectangle((0.020, 0.022), 0.960, 0.200, facecolor="none",
                           edgecolor=BAND, linewidth=0.9,
                           linestyle=(0, (4, 3))))
    ax.text(0.038, 0.196, "EVALUATION, CONSUMING THE DELIVERED GRAPH",
            fontsize=SUB_SIZE * 0.86, color=BAND, fontweight="bold")

    ye, he = 0.045, 0.110
    b_bench = box(fig, ax, 0.040, ye, 0.215, he, "Benchmark",
                  "26 primary, 34 extended", muted=True)
    b_ctrl = box(fig, ax, 0.275, ye, 0.215, he, "Adversarial controls",
                 "464, verified by Llama", muted=True)
    b_dir = box(fig, ax, 0.510, ye, 0.215, he, "Directional check",
                "95 panel-accepted relations", muted=True)
    b_exp = box(fig, ax, 0.745, ye, 0.215, he, "Expert review",
                "232 relations, one assessor", muted=True)

    bus3 = 0.172
    for b in (b_bench, b_ctrl, b_dir, b_exp):
        line(ax, [top(b)[0]] * 2, [top(b)[1], bus3], color=BAND, lw=0.9,
             ls=(0, (2, 2)))
    line(ax, [top(b_bench)[0], 0.898], [bus3, bus3], color=BAND, lw=0.9,
         ls=(0, (2, 2)))
    arrow(fig, ax, (0.898, bus3), (0.898, y3 - 0.014), color=BAND, lw=1.0,
          ls=(0, (2, 2)))


def fit_report(fig, ax):
    worst, bad = 0.0, []
    for title, (x, y, w, h), pairs in _boxes:
        avail = w - 2 * PAD_X
        for artist, raw, _ in pairs:
            r = _width_of(fig, ax, artist) / avail
            worst = max(worst, r)
            if r > 1.0:
                bad.append((title, artist.get_text()[:36], round(r, 3)))
    print("label fit check:")
    if bad:
        print("  OVERFLOW, widen the box or shorten the text:")
        for t, s, r in bad:
            print("    " + t.ljust(22) + s.ljust(40) + "x" + str(r))
    else:
        print("  every label fits; widest occupies "
              + str(round(worst * 100, 1)) + " per cent of its box")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",
                    default=str(Path.home() / "ontogeorag" / "output" / "figures"))
    ap.add_argument("--stem", default="fig_pipeline")
    ap.add_argument("--width", type=float, default=5.8,
                    help="authored width in inches; keep it below the text "
                         "block width so that width=\\linewidth enlarges it")
    ap.add_argument("--height", type=float, default=5.0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier on all font sizes")
    ap.add_argument("--greyscale", action="store_true")
    args = ap.parse_args()

    set_scale(args.scale)
    greyscale(args.greyscale)
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    build(fig, ax)
    ok = fit_report(fig, ax)

    stem = args.stem + ("_grey" if args.greyscale else "")
    pdf, png = outdir / (stem + ".pdf"), outdir / (stem + ".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png, format="png", dpi=400, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print()
    print("authored at " + str(args.width) + " x " + str(args.height) + " inches")
    print("written: " + str(pdf))
    print("         " + str(png))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()