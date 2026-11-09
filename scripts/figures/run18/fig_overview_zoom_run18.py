#!/usr/bin/env python3
"""Figure fig:overview — the cleaned run18 graph after grouping entity names by family.

Families are read from the cleaned graph (fields famille_sujet and famille_objet; an entity
outside the table of name families is its own family). Relations inside one family are left out.
The type of a family is the most frequent entity type among its relations.

Panels
    (a) largest connected part of the family graph; node size increases with the number of
        relations of the family; the 12 families with the most relations are labelled; the
        layout is a spring layout with a fixed seed;
    (b) mass transport deposits and the families linked to them by an arrow that holds at least
        two relations, an arrow being one (family, relation type, direction); the number on an
        arrow gives its number of relations, and an arrow is dashed when all its relations are in
        Tier 2.
Nodes are coloured by entity type and arrows by relation type.

Checks (nothing is written if one fails): 281 relations, 275 between distinct families,
139 families; largest part 128 families and 269 relations; five small parts with 11 families;
mass transport deposits in 81 relations and slope failure in 36; panel (b) 13 arrows holding
39 of the 81 relations of mass transport deposits.

Usage
    python scripts/figures/run18/fig_overview_zoom_run18.py --out output/figures/fig_overview_zoom_run18
"""
import argparse
import math
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import run18_style as S

FOCUS = "mass transport deposit"
SEED = 7
EXPECTED = {"relations": 281, "between_families": 275, "families": 139, "main_families": 128,
            "main_relations": 269, "small_parts": 5, "small_families": 11, "mtd_relations": 81,
            "slope_failure_relations": 36, "zoom_arrows": 13, "zoom_relations": 39}


def measure():
    cleaned = S.triples(S.CLEANED)
    between = [t for t in cleaned if S.family_of_subject(t) != S.family_of_object(t)]
    types = defaultdict(Counter)
    for t in cleaned:
        types[S.family_of_subject(t)][t["subject_type"]] += 1
        types[S.family_of_object(t)][t["object_type"]] += 1
    family_type = {f: c.most_common(1)[0][0] for f, c in types.items()}
    graph = nx.Graph()
    graph.add_nodes_from(family_type)
    for t in between:
        graph.add_edge(S.family_of_subject(t), S.family_of_object(t))
    parts = sorted(nx.connected_components(graph), key=len, reverse=True)
    main = parts[0]
    degree = Counter()
    for t in between:
        degree[S.family_of_subject(t)] += 1
        degree[S.family_of_object(t)] += 1
    arrows = defaultdict(list)
    for t in between:
        s, o = S.family_of_subject(t), S.family_of_object(t)
        if FOCUS in (s, o):
            arrows[(s, t["relation"], o)].append(t)
    zoom = {k: v for k, v in arrows.items() if len(v) >= 2}
    m = {"relations": len(cleaned), "between_families": len(between), "families": len(family_type),
         "main_families": len(main), "main_relations": sum(1 for t in between if S.family_of_subject(t) in main),
         "small_parts": len(parts) - 1, "small_families": sum(len(p) for p in parts[1:]),
         "mtd_relations": degree[FOCUS], "slope_failure_relations": degree["slope failure"],
         "zoom_arrows": len(zoom), "zoom_relations": sum(len(v) for v in zoom.values())}
    return m, between, family_type, main, degree, zoom


def panel_a(ax, between, family_type, main, degree):
    g = nx.Graph()
    for t in between:
        s, o = S.family_of_subject(t), S.family_of_object(t)
        if s in main:
            w = g.get_edge_data(s, o, {"w": 0})["w"]
            g.add_edge(s, o, w=w + 1, r=t["relation"])
    pos = nx.kamada_kawai_layout(g)
    pos = nx.spring_layout(g, pos=pos, seed=SEED, k=1.4 / math.sqrt(len(g)), iterations=60)
    for s, o, d in g.edges(data=True):
        (x1, y1), (x2, y2) = pos[s], pos[o]
        ax.plot([x1, x2], [y1, y2], color=S.RELATION_COLOURS.get(d["r"], "#999999"), linewidth=0.4 + 0.25 * d["w"],
                alpha=0.6, zorder=1)
    for f in g.nodes():
        x, y = pos[f]
        ax.scatter([x], [y], s=6 + 4.5 * degree[f], color=S.ENTITY_TYPES[family_type[f]][1], edgecolor=S.INK,
                   linewidth=0.3, zorder=2)
    # labels of the 12 families with the most relations, set in two columns beside the graph
    # and joined to their node by a thin line, so that no label covers another
    top = [f for f, _ in sorted(((f, degree[f]) for f in g.nodes()), key=lambda x: (-x[1], x[0]))[:12]]
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx = (min(xs) + max(xs)) / 2
    gap = (max(ys) - min(ys)) / 13.0
    for side in (-1, 1):
        names = sorted([f for f in top if (pos[f][0] - cx) * side >= 0], key=lambda f: -pos[f][1])
        y_prev = None
        x_lab = (min(xs) - 0.08 * (max(xs) - min(xs))) if side < 0 else (max(xs) + 0.08 * (max(xs) - min(xs)))
        for f in names:
            y = pos[f][1] if y_prev is None else min(pos[f][1], y_prev - gap)
            y_prev = y
            ax.plot([pos[f][0], x_lab], [pos[f][1], y], color="#777777", linewidth=0.4, zorder=3)
            ax.text(x_lab + (-0.01 if side < 0 else 0.01), y, S.label(f), fontsize=5.9, va="center",
                    ha="right" if side < 0 else "left", zorder=4)
    n_rel = sum(1 for t in between if S.family_of_subject(t) in main)
    ax.set_title("(a) Largest connected part: %d families, %d relations" % (len(main), n_rel),
                 fontsize=7.4, fontweight="bold", loc="left")
    ax.axis("off")
    ax.margins(x=0.32, y=0.04)


def panel_b(ax, family_type, zoom):
    neighbours = sorted({k[2] if k[0] == FOCUS else k[0] for k in zoom},
                        key=lambda f: (S.ENTITY_TYPES[family_type[f]][0], f))
    pos = {FOCUS: (0.0, 0.0)}
    for i, f in enumerate(neighbours):
        a = math.pi / 2 - 2 * math.pi * i / len(neighbours)
        pos[f] = (1.45 * math.cos(a), 1.45 * math.sin(a))
    by_pair = defaultdict(list)
    for (s, r, o), ts in zoom.items():
        by_pair[(s, o) if s != FOCUS else (o, s)].append((s, r, o, ts))
    for _, items in by_pair.items():
        for i, (s, r, o, ts) in enumerate(items):
            rad = 0.0 if len(items) == 1 else (i - 0.5) * 0.5
            tier2 = all(t["tier"] == 2 for t in ts)
            colour = S.RELATION_COLOURS[r]
            ax.add_patch(FancyArrowPatch(pos[s], pos[o], arrowstyle="-|>", mutation_scale=8,
                                         linewidth=0.8 + 0.35 * len(ts), color=colour,
                                         linestyle="--" if tier2 else "-", shrinkA=8 if s != FOCUS else 17, shrinkB=8 if o != FOCUS else 17,
                                         connectionstyle="arc3,rad=%.2f" % rad, zorder=2))
            (x1, y1), (x2, y2) = pos[s], pos[o]
            far = pos[o] if s == FOCUS else pos[s]
            f = 0.62
            mx, my = f * far[0], f * far[1]
            mx, my = mx + rad * 0.35 * (-far[1]), my + rad * 0.35 * far[0]
            ax.text(mx, my, str(len(ts)), fontsize=6.4, ha="center", va="center", color=colour, zorder=4,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=colour, lw=0.5))
    for f, (x, y) in pos.items():
        focus = f == FOCUS
        ax.scatter([x], [y], s=900 if focus else 120, color=S.ENTITY_TYPES[family_type[f]][1],
                   edgecolor=S.INK, linewidth=0.5, zorder=3)
        if focus:
            ax.text(x, y, "mass\ntransport\ndeposit", fontsize=5.6, ha="center", va="center", color="white",
                    fontweight="bold", zorder=5, linespacing=0.95)
        else:
            ax.text(x * 1.13, y * 1.13, S.label(f), fontsize=6.0, zorder=5,
                    ha="left" if x > 0.15 else ("right" if x < -0.15 else "center"),
                    va="bottom" if y > 1.3 else ("top" if y < -1.3 else "center"))
    ax.set_title("(b) Families linked to mass transport deposits\nby at least two relations (%d of 81 relations)"
                 % sum(len(v) for v in zoom.values()), fontsize=7.4, fontweight="bold", loc="left")
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-1.85, 1.85)
    ax.set_aspect("equal")
    ax.axis("off")


def draw(between, family_type, main, degree, zoom, out, width, height):
    S.setup()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(width, height), gridspec_kw={"width_ratios": [1.15, 1.0]})
    panel_a(axa, between, family_type, main, degree)
    panel_b(axb, family_type, zoom)
    used_rel = sorted({t["relation"] for t in between} & set(S.RELATION_COLOURS))
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=c, markeredgecolor=S.INK,
                      label=lab) for lab, c in S.ENTITY_TYPES.values()]
    handles += [Line2D([0], [0], color=S.RELATION_COLOURS[r], linewidth=1.5, label=r) for r in used_rel]
    handles += [Line2D([0], [0], color=S.INK, linewidth=1.0, linestyle="--", label="Tier 2 only (b)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=7, frameon=False,
               fontsize=6.0, handlelength=1.8, columnspacing=1.0)
    S.save(fig, out)


def main():
    ap = argparse.ArgumentParser(description="Figure fig:overview of the run18 manuscript")
    ap.add_argument("--out", default="output/figures/fig_overview_zoom_run18")
    ap.add_argument("--width", type=float, default=7.2)
    ap.add_argument("--height", type=float, default=3.9)
    args = ap.parse_args()
    m, between, family_type, main_part, degree, zoom = measure()
    print(m)
    S.check(m, EXPECTED)
    draw(between, family_type, main_part, degree, zoom, args.out, args.width, args.height)


if __name__ == "__main__":
    main()
