#!/usr/bin/env python3
"""Figure fig:pressurefamilies — causal relations around pore pressure and slope failure in the
cleaned run18 graph, after grouping entity names by family.

Selection. The figure shows eleven families, fixed below: pore pressure / overpressure, the eight
families it is linked to by a causal relation (causes, triggers, controls, affects), slope failure
among them, seismicity, which sends as many relations to slope failure as pore pressure (5 each),
and poor drainage, which causes a change of effective stress. Every causal relation of the cleaned graph between two of these families is
drawn; relations inside one family are not drawn.

Each node gives the number of distinct entity names grouped in the family (fields famille_sujet
and famille_objet of the cleaned graph). Arrows are grouped by (family, relation type, family);
the number on an arrow gives the number of relations when there are several, and the arrow colour
gives the relation type.

Checks (nothing is written if one fails): 29 relations between the 11 families; pore pressure
gathers 9 names, sedimentation rate 8 and seismicity 4; sedimentation rate causes pore pressure in
5 relations; pore pressure triggers slope failure in 4 relations; seismicity triggers slope failure
in 5 relations from 3 papers.

Usage
    python scripts/figures/run18/fig_pressure_families_run18.py --out output/figures/fig_pressure_families_run18
"""
import argparse
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import run18_style as S

PP, SF = "pore pressure / overpressure", "slope failure"
POS = {
    "poor drainage": (0.0, 5.4), "seismicity": (0.0, 4.2), "sedimentation rate": (0.0, 3.0),
    "gas hydrate": (0.0, 1.8), "low permeability": (0.0, 0.6), "sea level": (0.0, -0.6),
    PP: (3.0, 0.9), "effective stress": (5.8, 3.9), "failure plane generation": (5.8, -0.2),
    "gravity spreading": (5.8, -1.3), SF: (8.8, 1.6),
}
EXPECTED = {"relations": 29, "families": 11, "names_pp": 9, "names_sedimentation": 8, "names_seismicity": 4,
            "sedimentation_causes_pp": 5, "pp_triggers_sf": 4, "seismicity_triggers_sf": 5,
            "seismicity_papers": 3}


def measure():
    cleaned = S.triples(S.CLEANED)
    names = defaultdict(set)
    for t in cleaned:
        names[S.family_of_subject(t)].add(t["subject"])
        names[S.family_of_object(t)].add(t["object"])
    rel = [t for t in cleaned if t["relation"] in S.CAUSAL and S.family_of_subject(t) in POS
           and S.family_of_object(t) in POS and S.family_of_subject(t) != S.family_of_object(t)]
    groups = Counter((S.family_of_subject(t), t["relation"], S.family_of_object(t)) for t in rel)
    seis = [t for t in rel if (S.family_of_subject(t), t["relation"], S.family_of_object(t)) == ("seismicity", "triggers", SF)]
    papers = {s["ref_n"] for t in seis for s in t["provenance"]["sources"]}
    m = {"relations": len(rel), "families": len({x for g in groups for x in (g[0], g[2])}),
         "names_pp": len(names[PP]), "names_sedimentation": len(names["sedimentation rate"]),
         "names_seismicity": len(names["seismicity"]),
         "sedimentation_causes_pp": groups[("sedimentation rate", "causes", PP)],
         "pp_triggers_sf": groups[(PP, "triggers", SF)],
         "seismicity_triggers_sf": groups[("seismicity", "triggers", SF)], "seismicity_papers": len(papers)}
    return m, groups, names


def draw(groups, names, out, width, height):
    S.setup()
    fig, ax = plt.subplots(figsize=(width, height))
    pairs = defaultdict(list)
    for (a, r, b), n in sorted(groups.items()):
        pairs[(a, b)].append((r, n))
    for (a, b), rels in pairs.items():
        for i, (r, n) in enumerate(rels):
            rad = (i - (len(rels) - 1) / 2) * 0.22
            if POS[a][0] == 0 and b == SF:
                rad -= 0.12
            colour = S.RELATION_COLOURS[r]
            ax.add_patch(FancyArrowPatch(POS[a], POS[b], arrowstyle="-|>", mutation_scale=9,
                                         linewidth=0.9 + 0.35 * n, color=colour, shrinkA=12, shrinkB=15,
                                         connectionstyle="arc3,rad=%.2f" % rad, zorder=2))
            if n > 1:
                (x1, y1), (x2, y2) = POS[a], POS[b]
                t = (0.62 if b == SF else 0.3) if POS[a][0] == 0 else 0.5
                cx, cy = (x1 + x2) / 2 + rad * (y2 - y1), (y1 + y2) / 2 - rad * (x2 - x1)
                px = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
                py = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
                ax.text(px, py, str(n), fontsize=8, ha="center", va="center", zorder=4, color=colour,
                        fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colour, lw=0.6))
    for fam, (x, y) in POS.items():
        hub = fam in (PP, SF)
        ax.scatter([x], [y], s=520 if hub else 260, color="#e15759", edgecolor=S.INK, linewidth=0.6, zorder=3)
        ax.text(x, y, str(len(names[fam])), fontsize=7, ha="center", va="center", color="white",
                fontweight="bold", zorder=5)
        if x == 0:
            kw = dict(x=x - 0.32, y=y, ha="right", va="center")
        elif fam == PP:
            kw = dict(x=x - 0.15, y=y - 0.5, ha="center", va="top", linespacing=1.0,
                      bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        elif hub:
            kw = dict(x=x, y=y - 0.5, ha="center", va="top")
        else:
            kw = dict(x=x + 0.32, y=y, ha="left", va="center")
        ax.text(s=S.label(fam).replace(" / ", " /\n") if fam == PP else S.label(fam), fontsize=7.6, fontweight="bold" if hub else "normal", zorder=5, **kw)
    used = sorted({r for (_, r, _) in groups})
    handles = [Line2D([0], [0], color=S.RELATION_COLOURS[r], linewidth=1.6, label=r) for r in used]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="#e15759",
                          markeredgecolor=S.INK, label="family (number of grouped names)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(handles),
              frameon=False, fontsize=6.6, handlelength=2.0, columnspacing=1.2)
    ax.set_xlim(-3.4, 11.2)
    ax.set_ylim(-1.9, 6.0)
    ax.axis("off")
    S.save(fig, out)


def main():
    ap = argparse.ArgumentParser(description="Figure fig:pressurefamilies of the run18 manuscript")
    ap.add_argument("--out", default="output/figures/fig_pressure_families_run18")
    ap.add_argument("--width", type=float, default=6.3)
    ap.add_argument("--height", type=float, default=3.6)
    args = ap.parse_args()
    m, groups, names = measure()
    print(m)
    S.check(m, EXPECTED)
    draw(groups, names, args.out, args.width, args.height)


if __name__ == "__main__":
    main()
