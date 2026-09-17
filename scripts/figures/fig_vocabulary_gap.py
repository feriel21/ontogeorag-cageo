#!/usr/bin/env python3
"""Figure 5: registers of description found outside the 40-descriptor extraction vocabulary.

For each register (family of terms): occurrences outside the vocabulary when the
extractor is given no descriptor vocabulary (configuration D) and when it is given the
38-term extraction list as a suggestion (configuration E), with the number of distinct
terms; and whether the 40-descriptor extraction vocabulary provides a descriptor for
that register. Registers are ordered by their occurrences under configuration D.

Inputs
    --open       vocabulary_classification.json of configuration D
    --suggested  vocabulary_classification.json of configuration E
    (both written by scripts/paper/classify_open_vocabulary.py)

The right-hand column is editorial: which descriptors of the extraction vocabulary
cover each register, and examples of terms found outside it. It is written in
REGISTERS below.

Checks (the figure is not written if any fails): (terms, occurrences) per register
    D: deformation 37/102, lithology 11/32, basal 7/25, section geometry 13/21,
       amplitude 5/18, other seismic attributes 5/17, stratigraphic terminations 6/10,
       morphometry 7/10
    E: 15/59, 6/19, 4/14, 5/5, 5/15, 2/12, 2/2, and no morphometric term.
--no-check draws whatever the inputs give.

Usage
    python scripts/figures/fig_vocabulary_gap.py --open output/vocabulary_D/vocabulary_classification.json --suggested output/vocabulary_E/vocabulary_classification.json --out output/figures/fig_vocabulary_gap
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (family key, register, what it describes, status, status is a gap, examples)
REGISTERS = [
    ("deformation_structure", "Deformation structures", "internal architecture",
     "generic terms only (deformed, folded, faulted, contorted)", False,
     "e.g. rotated blocks, listric faults, frontal ramps, duplex structures"),
    ("lithology_texture", "Lithology and texture", "rock composition",
     "no descriptor", True, "e.g. sandy, muddy, matrix-supported, clast-rich"),
    ("basal_surface_marking", "Basal-surface markings", "emplacement direction",
     "no descriptor", True, "e.g. longitudinal lineations, grooves, striations, flow lines"),
    ("section_geometry", "Section geometry", "external form",
     "generic terms only (wedge-shaped, sheet-like, tabular, mounded)", False,
     "e.g. lobate, elongate, tapering, concave-up"),
    ("amplitude_variant", "Amplitude variants", "reflection strength",
     "generic terms only (high-, low-, moderate-amplitude)", False,
     "e.g. very high, medium, mixed amplitude"),
    ("seismic_attribute", "Other seismic attributes", "reflection character",
     "no descriptor", True, "e.g. high-frequency, highly reflective, low-velocity"),
    ("stratigraphic_termination", "Stratigraphic terminations", "stratal relation",
     "generic term only (onlapping)", False, "e.g. onlap, downlap, unconformity"),
    ("morphometric", "Quantitative morphometry", "measured dimension",
     "no numeric attribute in the ontology", True, "e.g. run-out distance of 80 km, thickness of 200 m"),
]
EXPECTED_D = {"deformation_structure": (37, 102), "lithology_texture": (11, 32),
              "basal_surface_marking": (7, 25), "section_geometry": (13, 21),
              "amplitude_variant": (5, 18), "seismic_attribute": (5, 17),
              "stratigraphic_termination": (6, 10), "morphometric": (7, 10)}
EXPECTED_E = {"deformation_structure": (15, 59), "lithology_texture": (6, 19),
              "basal_surface_marking": (4, 14), "section_geometry": (5, 5),
              "amplitude_variant": (5, 15), "seismic_attribute": (2, 12),
              "stratigraphic_termination": (2, 2), "morphometric": (0, 0)}
DARK, LIGHT, INK, MUTED, GAP = "#24456d", "#d3d9e2", "#1b1b1b", "#555555", "#9b1c1c"


def fail(msg):
    print("ERROR: " + msg, flush=True)
    sys.exit(1)


def family_counts(path):
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for key, *_ in REGISTERS:
        f = j["by_family"].get(key) or {}
        out[key] = (f.get("n_terms", 0), f.get("n_occurrences", 0))
    return out


def draw(d, e, out, width, height):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.25, 0.14, 0.2, 0.74])
    n = len(REGISTERS)
    xmax = max(v[1] for v in d.values()) * 1.18
    for i, (key, name, what, status, gap, examples) in enumerate(REGISTERS):
        y = n - 1 - i
        (td, od), (te, oe) = d[key], e[key]
        ax.barh(y + 0.17, od, height=0.32, color=DARK, edgecolor=INK, linewidth=0.5)
        ax.barh(y - 0.17, oe, height=0.32, color=LIGHT, edgecolor=INK, linewidth=0.5)
        ax.text(od + xmax * 0.012, y + 0.17, "%d (%d terms)" % (od, td), va="center", fontsize=6.2, color=INK)
        ax.text((oe if oe else 0) + xmax * 0.012, y - 0.17,
                "%d (%d terms)" % (oe, te) if oe else "none retained",
                va="center", fontsize=6.2, color=MUTED, style="normal" if oe else "italic")
        ax.text(-xmax * 0.04, y + 0.12, name, ha="right", va="center", fontsize=6.8, fontweight="bold", color=INK)
        ax.text(-xmax * 0.04, y - 0.2, what, ha="right", va="center", fontsize=6.0, style="italic", color=MUTED)
        ax.text(xmax * 1.55, y + 0.12, status, ha="left", va="center", fontsize=6.1,
                color=GAP if gap else MUTED, fontweight="bold" if gap else "normal")
        ax.text(xmax * 1.55, y - 0.2, examples, ha="left", va="center", fontsize=5.9, color=MUTED)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Occurrences in the extraction output")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.text(-xmax * 0.04, n - 0.05, "Register of description\nfound in the papers", ha="right", va="bottom",
            fontsize=6.8, fontweight="bold", color=INK)
    ax.text(xmax * 0.02, n - 0.05, "Occurrences outside\nthe 40-descriptor vocabulary", ha="left",
            va="bottom", fontsize=6.8, fontweight="bold", color=INK)
    ax.text(xmax * 1.55, n - 0.05, "Does the 40-descriptor vocabulary have a descriptor for it?",
            ha="left", va="bottom", fontsize=6.8, fontweight="bold", color=INK)
    ax.legend(handles=[Patch(facecolor=DARK, edgecolor=INK, label="extractor given no descriptor vocabulary"),
                       Patch(facecolor=LIGHT, edgecolor=INK,
                             label="extractor given the 38-term extraction list as a suggestion")],
              loc="upper left", bbox_to_anchor=(-0.05, -0.16), frameon=False, fontsize=6.2)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out) + "." + ext, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written: %s.pdf and %s.png" % (out, out))


def main():
    ap = argparse.ArgumentParser(description="Figure 5: registers outside the extraction vocabulary")
    ap.add_argument("--open", required=True, help="classification JSON of configuration D")
    ap.add_argument("--suggested", required=True, help="classification JSON of configuration E")
    ap.add_argument("--out", default="fig_vocabulary_gap")
    ap.add_argument("--width", type=float, default=6.3)
    ap.add_argument("--height", type=float, default=3.9)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    d, e = family_counts(args.open), family_counts(args.suggested)
    for key, name, *_ in REGISTERS:
        print("%-28s D %3d terms / %3d occ.   E %3d terms / %3d occ." % ((name,) + d[key] + e[key]))
    if not args.no_check:
        diffs = ["D %s: %s, expected %s" % (k, d[k], v) for k, v in EXPECTED_D.items() if d[k] != v]
        diffs += ["E %s: %s, expected %s" % (k, e[k], v) for k, v in EXPECTED_E.items() if e[k] != v]
        if diffs:
            fail("inputs do not reproduce the published figure; nothing written\n  - " + "\n  - ".join(diffs))
    draw(d, e, args.out, args.width, args.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())