#!/usr/bin/env python3
"""Figure fig:lb2019 — the 26 retained expert relations of Le Bouteiller et al. (2019),
drawn as a graph and coloured by their status in the cleaned run18 graph.

Status of an expert relation (subject, relation, object), names lowercased:
    found exactly            the cleaned graph holds the same subject, relation type and object
    same families            otherwise, the cleaned graph holds a relation of the same type between
                             the family of the subject and the family of the object; expert names
                             are mapped to families with configs/entity_families_run18.csv, and the
                             cleaned graph gives the family of each of its entities (fields
                             famille_sujet, famille_objet)
    not extracted            otherwise, both names occur as entities of the cleaned graph
    name never used          otherwise: at least one name never occurs in the cleaned graph
                             (drawn as a hollow node)

Node colour gives the kind of name: objects of hasDescriptor are seismic descriptors, objects of
occursIn are settings, slope failure, earthquake and pore pressure are processes, and the other
names are geological objects. Node positions are fixed below for legibility.

Checks (nothing is written if one fails): 26 relations; 9 found exactly, 6 of them in Tier 1 and
3 in Tier 2, 8 of the 9 with hasDescriptor; 2 between the same families; 8 not extracted;
7 with a name never used, the unused names being hemipelagite, pore pressure, abyssal plain and
basin floor.

Usage
    python scripts/figures/run18/fig_lb2019_vs_graph_run18.py --out output/figures/fig_lb2019_vs_graph_run18
"""
import argparse
import csv
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import run18_style as S

PROCESSES = {"slope failure", "earthquake", "pore pressure"}
KIND_COLOURS = {"process": "#e15759", "geological object": "#4e79a7", "setting": "#59a14f",
                "seismic descriptor": "#b07aa1"}
STATUS = {"exact": ("found exactly", "#1b7837", "-"),
          "family": ("found between the same families", "#f1a340", "-"),
          "missing": ("not extracted, both names in the graph", "#8c8c8c", "-"),
          "unused": ("a name never used in the graph", "#8c8c8c", "--")}
POS = {
    "hemipelagite": (3.3, 9.3), "turbidite": (4.8, 7.7), "slide": (4.8, 5.9),
    "mass transport deposit": (4.8, 3.6), "debris flow": (4.8, 1.3),
    "basin floor": (2.2, 7.7), "earthquake": (0.0, 5.1), "pore pressure": (0.0, 2.2),
    "slope failure": (2.1, 3.6), "turbidity current": (6.6, -0.5), "abyssal plain": (0.6, 0.9),
    "continental slope": (2.4, -0.7),
    "low-amplitude": (8.8, 9.6), "parallel": (8.8, 8.8), "continuous": (8.8, 8.0), "layered": (8.8, 7.2),
    "high-amplitude": (8.8, 6.4), "undeformed": (8.8, 5.6), "blocky": (8.8, 4.8), "chaotic": (8.8, 4.0),
    "transparent": (8.8, 3.2), "hummocky": (8.8, 2.4), "discontinuous": (8.8, 1.6), "massive": (8.8, 0.8),
}
# label positions chosen by hand where the midpoint of an arrow falls on another label
LABEL_AT = {("slide", "hemipelagite"): (3.55, 6.7), ("earthquake", "slope failure"): (1.3, 4.6),
            ("pore pressure", "slope failure"): (0.7, 2.85),
            ("mass transport deposit", "transparent"): (5.9, 2.6)}
EXPECTED = {"n": 26, "exact": 9, "exact_tiers": (6, 3), "exact_descriptor": 8, "family": 2,
            "missing": 8, "unused": 7,
            "unused_names": ["abyssal plain", "basin floor", "hemipelagite", "pore pressure"]}


def measure():
    cleaned = S.triples(S.CLEANED)
    edges = json.loads(S.EXPERT_EDGES.read_text(encoding="utf-8"))["edges"]
    table = {S.norm(r["entity"]): r["family"].strip()
             for r in csv.DictReader(open(S.FAMILIES, encoding="utf-8")) if r["family"].strip()}
    exact = {(S.norm(t["subject"]), t["relation"], S.norm(t["object"])): t["tier"] for t in cleaned}
    entities = {S.norm(t["subject"]) for t in cleaned} | {S.norm(t["object"]) for t in cleaned}
    between = defaultdict(set)
    for t in cleaned:
        between[(S.norm(S.family_of_subject(t)), t["relation"], S.norm(S.family_of_object(t)))].add(
            (S.norm(t["subject"]), S.norm(t["object"])))
    rows = []
    for e in edges:
        s, r, o = S.norm(e["subject"]), e["relation"], S.norm(e["object"])
        fs, fo = table.get(s, s), table.get(o, o)
        if (s, r, o) in exact:
            status, note = "exact", "T%d" % exact[(s, r, o)]
        elif (fs, r, fo) in between:
            pairs = between[(fs, r, fo)]
            side = sorted({x for x, _ in pairs}) if {y for _, y in pairs} == {o} or len({x for x, _ in pairs}) > 1 \
                else sorted({y for _, y in pairs})
            status = "family"
            note = "via " + (side[0] if len(side) == 1 else "%d names of the family" % len(side))
        elif s in entities and o in entities:
            status, note = "missing", ""
        else:
            status, note = "unused", ""
        rows.append((s, r, o, status, note))
    count = Counter(x[3] for x in rows)
    tiers = Counter(x[4] for x in rows if x[3] == "exact")
    unused = sorted({n for s, r, o, st, _ in rows for n in (s, o) if n not in entities})
    m = {"n": len(rows), "exact": count["exact"], "exact_tiers": (tiers["T1"], tiers["T2"]),
         "exact_descriptor": sum(1 for x in rows if x[3] == "exact" and x[1] == "hasDescriptor"),
         "family": count["family"], "missing": count["missing"], "unused": count["unused"],
         "unused_names": unused}
    return m, rows, entities


def kind(name, rows):
    if name in PROCESSES:
        return "process"
    if any(o == name and r == "hasDescriptor" for _, r, o, _, _ in rows):
        return "seismic descriptor"
    if any(o == name and r == "occursIn" for _, r, o, _, _ in rows):
        return "setting"
    return "geological object"


def draw(rows, entities, out, width, height):
    S.setup()
    fig, ax = plt.subplots(figsize=(width, height))
    missing_pos = {n for s, _, o, _, _ in rows for n in (s, o)} - set(POS)
    if missing_pos:
        raise SystemExit("no position for: " + ", ".join(sorted(missing_pos)))
    for s, r, o, status, note in rows:
        a, b = POS[s], POS[o]
        _, colour, style = STATUS[status]
        rad = 0.08 if r == "hasDescriptor" else (-0.5 if (s, o) == ("slide", "hemipelagite") else 0.15)
        arrow = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, linewidth=1.6 if status == "exact" else 1.0,
                                color=colour, linestyle=style, connectionstyle="arc3,rad=%.2f" % rad,
                                shrinkA=8, shrinkB=8, zorder=2)
        ax.add_patch(arrow)
        text = ""
        if r != "hasDescriptor":
            text = r + (" (%s)" % note if note.startswith("T") else "")
        if note.startswith("via"):
            text = (text + "\n" if text else "") + note
        if note.startswith("T") and r == "hasDescriptor":
            text = note
        if text:
            f = 0.74 if r == "hasDescriptor" else 0.5
            mx, my = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            if (s, o) in LABEL_AT:
                mx, my = LABEL_AT[(s, o)]
            elif note.startswith("via") and r == "occursIn":
                mx -= 0.55
            ax.text(mx, my + (0.0 if r == "hasDescriptor" else 0.2), text, fontsize=5.6, ha="center", va="center",
                    color=STATUS[status][1] if r == "hasDescriptor" else S.INK, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92))
    for name, (x, y) in POS.items():
        colour = KIND_COLOURS[kind(name, rows)]
        hollow = name not in entities
        ax.scatter([x], [y], s=150, zorder=3, facecolor="white" if hollow else colour,
                   edgecolor=colour, linewidth=1.4)
        right = x > 8
        above = name in ("mass transport deposit", "debris flow", "turbidite", "slide")
        ax.text(x + (0.28 if right else 0), y + (0 if right else (0.34 if above else -0.36)), name, fontsize=6.6,
                ha="left" if right else "center", va="center" if right else ("bottom" if above else "top"),
                zorder=5, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8) if above else None)
    handles = [Line2D([0], [0], color=c, linestyle=ls, linewidth=1.4, label=lab) for lab, c, ls in STATUS.values()]
    handles += [Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=c, markeredgecolor=c,
                       label=k) for k, c in KIND_COLOURS.items()]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor="white",
                          markeredgecolor="#555555", label="name never used in the graph"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
              fontsize=6.2, handlelength=2.4, columnspacing=1.2)
    ax.set_xlim(-0.9, 10.8)
    ax.set_ylim(-1.6, 10.0)
    ax.axis("off")
    S.save(fig, out)


def main():
    ap = argparse.ArgumentParser(description="Figure fig:lb2019 of the run18 manuscript")
    ap.add_argument("--out", default="output/figures/fig_lb2019_vs_graph_run18")
    ap.add_argument("--width", type=float, default=6.3)
    ap.add_argument("--height", type=float, default=5.6)
    args = ap.parse_args()
    m, rows, entities = measure()
    print(m)
    S.check(m, EXPECTED)
    draw(rows, entities, args.out, args.width, args.height)


if __name__ == "__main__":
    main()
