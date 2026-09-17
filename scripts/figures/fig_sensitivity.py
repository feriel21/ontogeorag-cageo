#!/usr/bin/env python3
"""Figure 6: share of corrupted records the verifier rejects, by corruption rule.

A corrupted record counts as rejected when its evidence verdict is NOT_SUPPORTED (the
single-verdict rule of Section 2.3.2). Bars show the sensitivity of each corruption
rule with a Wilson 95 per cent interval; labels give the count of rejected records.
The two rules that change what the verifier reads (another relation's evidence
sentence, or one argument) are drawn dark; the two that leave the evidence sentence
and both arguments unchanged are drawn light, below a dashed separator.

Input
    --report  negatives_report.json written by m4/m4_negatives.py for the delivered run
              (key per_corruption_class: one entry per rule with tp and fn)

Corruption classes are recognised from their names:
    contains "passage" or "perm"                 -> evidence sentence swapped
    contains "inver", "direction" or "reverse"   -> subject and object swapped
    contains "relation"                          -> relation type replaced
    contains "entity", "object" or "argument"    -> one argument replaced
Each rule must be recognised exactly once.

Checks (the figure is not written if any fails): rejected / total per rule equal
48 / 58 (evidence sentence swapped), 64 / 90 (argument replaced),
11 / 26 (subject and object swapped), 19 / 58 (relation type replaced).
--no-check draws whatever the report gives.

Usage
    python scripts/figures/fig_sensitivity.py --report /path/to/run16E/m4/negatives/negatives_report.json --out output/figures/fig_sensitivity
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = [
    ("evidence", "Evidence sentence taken from another relation"),
    ("argument", "One argument replaced by another of the same type"),
    ("inversion", "Subject and object swapped"),
    ("relation", "Relation type replaced by another"),
]
EXPECTED = {"evidence": (48, 58), "argument": (64, 90), "inversion": (11, 26), "relation": (19, 58)}
DARK, LIGHT, INK, MUTED = "#24456d", "#d3d9e2", "#1b1b1b", "#555555"


def fail(msg):
    print("ERROR: " + msg, flush=True)
    sys.exit(1)


def recognise(name):
    n = name.lower()
    if "passage" in n or "perm" in n:
        return "evidence"
    if "inver" in n or "direction" in n or "reverse" in n:
        return "inversion"
    if "relation" in n:
        return "relation"
    if "entity" in n or "object" in n or "argument" in n:
        return "argument"
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def load(report_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    classes = report.get("per_corruption_class")
    if not isinstance(classes, dict):
        fail("no per_corruption_class in " + str(report_path))
    found, problems = {}, []
    for name, v in classes.items():
        label = recognise(name)
        if label is None:
            problems.append("unrecognised class name: " + name)
        elif label in found:
            problems.append("two classes recognised as " + label + ": " + found[label][0] + ", " + name)
        else:
            found[label] = (name, int(v["tp"]), int(v["tp"]) + int(v["fn"]))
    missing = [k for k, _ in ORDER if k not in found]
    if missing:
        problems.append("rules not found: " + ", ".join(missing))
    if problems:
        fail("\n  - " + "\n  - ".join(problems))
    return found


def draw(found, out, width, height):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(width, height))
    ys = [3.0, 2.0, 0.8, -0.2]
    for (key, label), y in zip(ORDER, ys):
        _, k, n = found[key]
        p = k / n
        lo, hi = wilson(k, n)
        colour = DARK if key in ("evidence", "argument") else LIGHT
        ax.barh(y, p, height=0.5, color=colour, edgecolor=INK, linewidth=0.6)
        ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, elinewidth=1.0,
                    capsize=3.5, capthick=1.0)
        ax.text(hi + 0.012, y, "%.2f  (%d of %d)" % (p, k, n), va="center", ha="left", fontsize=8, color=INK)
        ax.text(-0.02, y, label, va="center", ha="right", fontsize=8, color=INK)
    sep = 1.4
    ax.plot([0, 1], [sep, sep], color="#9a9a9a", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(1.0, sep + 0.07, "the corruption changes what the verifier reads", ha="right", va="bottom",
            fontsize=7.4, color=MUTED, style="italic")
    ax.text(1.0, sep - 0.07, "evidence sentence and both arguments left unchanged", ha="right", va="top",
            fontsize=7.4, color=MUTED, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.75, 3.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Share of corrupted records the verifier rejects, with 95 per cent interval")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out) + "." + ext, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written: %s.pdf and %s.png" % (out, out))


def main():
    ap = argparse.ArgumentParser(description="Figure 6: sensitivity by corruption rule")
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", default="fig_sensitivity")
    ap.add_argument("--width", type=float, default=5.4)
    ap.add_argument("--height", type=float, default=2.3)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    found = load(args.report)
    for key, _ in ORDER:
        name, k, n = found[key]
        lo, hi = wilson(k, n)
        print("%-10s %-24s %d / %d = %.3f  Wilson [%.3f, %.3f]" % (key, name, k, n, k / n, lo, hi))
    if not args.no_check:
        diffs = ["%s: %d / %d, expected %d / %d" % ((key, found[key][1], found[key][2]) + EXPECTED[key])
                 for key in EXPECTED if (found[key][1], found[key][2]) != EXPECTED[key]]
        if diffs:
            fail("report does not reproduce the published figure; nothing written\n  - " + "\n  - ".join(diffs))
    draw(found, args.out, args.width, args.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())