#!/usr/bin/env python3
"""Figure fig:texttograph — from sentences to graph for one paper, Alves et al. (2014).

Selection. A delivered relation belongs to the figure if its evidence sentence is located in the
paper Alves et al. (2014) (file Alves_AAPG_MTDs_14 in the graph with located provenance, 187) and
if the first 160 characters of its evidence, lowercased with "- " read as "-", contain the opening
words of one of the four sentences listed in SENTENCES.

Panels
    (a) the four sentences, as stored in the passage index (lowercase);
    (b) the delivered relations of each sentence, with tier, panel decision and the verdict of each
        verifying-LLM on the evidence sentence (S supported, P partially supported, N not
        supported; Llama-3.1-8B-Instruct then Mistral-7B-Instruct-v0.2); grey rows are removed by
        the written rules of 188, with the rule;
    (c) the relations kept in the cleaned graph: nodes are entities coloured by type, arrows are
        relations coloured by type, solid for Tier 1 and dashed for Tier 2; a cross marks a
        relation flagged by the evidence check.

Checks (nothing is written if one fails): 16 relations, 8 in Tier 1 and 8 in Tier 2; 5 removed,
3 by the local-label rule (R2) and 2 by the missing-subject rule (R3, subject seismic trace);
6 flagged, all in Tier 2; 5 flagged relations kept; 11 relations kept.

Usage
    python scripts/figures/run18/fig_text_to_graph_run18.py --out output/figures/fig_text_to_graph_run18
"""
import argparse
import csv
import textwrap
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import run18_style as S

PAPER_FILE = "Alves_AAPG_MTDs_14"
SENTENCES = [
    ("rafted blocks are shown as individual blocks",),
    ("mass transport deposits 5 and 6 are shown as chaotic", "turbidites in unit ib are revealed"),
    ("the mass transport deposit 6 divides the slope sediments",),
    ("above this, mass transport deposit 5",),
]
RULES = {"R1": "reference list", "R2": "local label", "R3": "subject absent"}
SHORT = {"SUPPORTED": "S", "PARTIALLY_SUPPORTED": "P", "NOT_SUPPORTED": "N"}
POS = {
    "debrite": (0.0, 5.0), "raft": (0.0, 4.0), "debris flow": (0.0, 3.0), "mass transport deposit": (0.0, 1.6),
    "megaslide": (0.0, 0.2), "extensional block": (0.0, -0.8), "turbidite": (3.4, 5.6),
    "low-amplitude": (6.6, 4.0), "moderate-amplitude": (6.6, 5.8), "high-amplitude": (6.6, 5.0),
    "slope sediments": (6.6, 3.0), "turbidite sequence": (6.6, 1.6), "chaotic": (6.6, 0.4),
    "disrupted": (6.6, -0.8),
}
EXPECTED = {"n": 16, "tiers": (8, 8), "removed": 5, "removed_rules": {"R2": 3, "R3": 2},
            "r3_subjects": ["seismic trace"], "flagged": 6, "flagged_tier2": 6, "flagged_kept": 5, "kept": 11}


def key(s, r, o):
    return (S.norm(s), r, S.norm(o))


def sentence_of(t):
    e = t["evidence"].lower().replace("- ", "-")[:160]
    for i, keys in enumerate(SENTENCES):
        if any(k in e for k in keys):
            return i
    return None


def measure():
    graph = S.triples(S.WITH_PROVENANCE)
    removed = {key(r["sujet"], r["relation"], r["objet"]): r["regles"]
               for r in csv.DictReader(open(S.REMOVED, encoding="utf-8"), delimiter="\t")}
    rows = []
    for t in graph:
        if not any(s["doc_id"] == PAPER_FILE for s in t["provenance"]["sources"]):
            continue
        i = sentence_of(t)
        if i is None:
            continue
        rule = removed.get(key(t["subject"], t["relation"], t["object"]), "")
        rows.append({"sentence": i, "t": t, "rule": rule.split(" ")[0] if rule else "", "flag": S.flagged(t)})
    rows.sort(key=lambda r: (r["sentence"], r["t"]["tier"], r["t"]["subject"], r["t"]["object"]))
    tiers = Counter(r["t"]["tier"] for r in rows)
    rules = Counter(r["rule"] for r in rows if r["rule"])
    m = {"n": len(rows), "tiers": (tiers[1], tiers[2]), "removed": sum(1 for r in rows if r["rule"]),
         "removed_rules": dict(rules),
         "r3_subjects": sorted({r["t"]["subject"] for r in rows if r["rule"] == "R3"}),
         "flagged": sum(r["flag"] for r in rows), "flagged_tier2": sum(r["flag"] and r["t"]["tier"] == 2 for r in rows),
         "flagged_kept": sum(r["flag"] and not r["rule"] for r in rows), "kept": sum(1 for r in rows if not r["rule"])}
    texts = []
    for i in range(len(SENTENCES)):
        ev = Counter(r["t"]["evidence"].strip().strip('"').strip() for r in rows if r["sentence"] == i)
        texts.append(ev.most_common(1)[0][0] if ev else "")
    return m, rows, texts


def verdicts(t):
    parts = str(t["m4"].get("evidence_verdict", "")).split("|")
    return " ".join(SHORT.get(p, "?") for p in parts)


def panel_a(ax, texts):
    ax.axis("off")
    ax.text(0.0, 1.0, "(a) Sentences of Alves et al. (2014)", fontsize=8, fontweight="bold", va="top")
    y = 0.92
    for i, text in enumerate(texts):
        lines = textwrap.wrap(text, 52)
        ax.text(0.0, y, "S%d" % (i + 1), fontsize=7, fontweight="bold", va="top", color=S.DARK)
        ax.text(0.09, y, "\n".join(lines), fontsize=6.3, va="top", linespacing=1.15, style="italic")
        y -= 0.034 * len(lines) + 0.045


def panel_b(ax, rows):
    ax.axis("off")
    ax.text(0.0, 1.0, "(b) Delivered relations of these sentences", fontsize=8, fontweight="bold", va="top")
    head = ["", "relation", "tier", "panel", "L M", "rule"]
    xs = [0.0, 0.06, 0.70, 0.77, 0.86, 0.94]
    y = 0.92
    for x, h in zip(xs, head):
        ax.text(x, y, h, fontsize=6.4, fontweight="bold", va="top")
    step = 0.86 / (len(rows) + 1)
    for r in rows:
        y -= step
        t = r["t"]
        grey = "#9a9a9a" if r["rule"] else S.INK
        cells = ["S%d" % (r["sentence"] + 1), "%s  %s  %s" % (t["subject"], t["relation"], t["object"]),
                 "T%d" % t["tier"], "A" if t["m4"]["decision"] == "ACCEPT" else "U", verdicts(t),
                 RULES.get(r["rule"], "")]
        for x, c in zip(xs, cells):
            ax.text(x, y, c, fontsize=6.2, va="top", color=grey,
                    fontweight="bold" if (c.startswith("N") or " N" in c) and x == xs[4] else "normal")


def panel_c(ax, rows):
    ax.axis("off")
    ax.set_title("(c) Relations kept in the cleaned graph", fontsize=8, fontweight="bold", loc="left")
    kept = [r for r in rows if not r["rule"]]
    types = {}
    for r in kept:
        t = r["t"]
        types[t["subject"]] = t["subject_type"]
        types[t["object"]] = t["object_type"]
    missing = set(types) - set(POS)
    if missing:
        raise SystemExit("no position for: " + ", ".join(sorted(missing)))
    for r in kept:
        t = r["t"]
        a, b = POS[t["subject"]], POS[t["object"]]
        colour = S.RELATION_COLOURS[t["relation"]]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, linewidth=1.3, color=colour,
                                     linestyle="-" if t["tier"] == 1 else "--", shrinkA=9, shrinkB=9,
                                     connectionstyle="arc3,rad=0.06", zorder=2))
        if r["flag"]:
            f = 0.62
            ax.text(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), "✖", fontsize=8, color="#c00000",
                    ha="center", va="center", zorder=4)
    for name, typ in types.items():
        x, y = POS[name]
        lab, colour = S.ENTITY_TYPES[typ]
        ax.scatter([x], [y], s=150, color=colour, edgecolor=S.INK, linewidth=0.5, zorder=3)
        left = x < 1
        mid = 1 < x < 5
        ax.text(x + (-0.25 if left else 0.25) if not mid else x, y + (0.32 if mid else 0), name, fontsize=6.6,
                ha="right" if left else ("center" if mid else "left"), va="bottom" if mid else "center", zorder=5)
    ax.set_xlim(-3.2, 9.6)
    ax.set_ylim(-1.3, 6.5)
    used_rel = sorted({r["t"]["relation"] for r in kept})
    used_typ = [k for k in S.ENTITY_TYPES if k in set(types.values())]
    handles = [Line2D([0], [0], color=S.RELATION_COLOURS[x], linewidth=1.4, label=x) for x in used_rel]
    handles += [Line2D([0], [0], color=S.INK, linewidth=1.1, linestyle="-", label="Tier 1"),
                Line2D([0], [0], color=S.INK, linewidth=1.1, linestyle="--", label="Tier 2"),
                Line2D([0], [0], marker="$✖$", color="#c00000", linestyle="", markersize=7,
                       label="flagged by the evidence check")]
    handles += [Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=S.ENTITY_TYPES[k][1],
                       markeredgecolor=S.INK, label=S.ENTITY_TYPES[k][0]) for k in used_typ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False,
              fontsize=6.2, handlelength=2.2, columnspacing=1.0)


def draw(rows, texts, out, width, height):
    S.setup()
    fig = plt.figure(figsize=(width, height))
    panel_a(fig.add_axes([0.0, 0.52, 0.37, 0.46]), texts)
    panel_b(fig.add_axes([0.39, 0.52, 0.61, 0.46]), rows)
    panel_c(fig.add_axes([0.04, 0.07, 0.92, 0.40]), rows)
    S.save(fig, out)


def main():
    ap = argparse.ArgumentParser(description="Figure fig:texttograph of the run18 manuscript")
    ap.add_argument("--out", default="output/figures/fig_text_to_graph_run18")
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=6.6)
    args = ap.parse_args()
    m, rows, texts = measure()
    print(m)
    S.check(m, EXPECTED)
    draw(rows, texts, args.out, args.width, args.height)


if __name__ == "__main__":
    main()
