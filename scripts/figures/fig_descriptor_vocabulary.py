#!/usr/bin/env python3
"""Figure 4: the 40 descriptors of the extraction vocabulary and their occurrences in the corpus.

Occurrences are counted in the indexed corpus (run16E index, 1,758 chunks): case-insensitive,
whole-word, every internal hyphen of a descriptor matching a hyphen or whitespace
(rule R2 of scripts/analysis/descriptor_partition.py). Each descriptor is classed as
    recovered            object of at least one hasDescriptor relation of the delivered graph
    attested, never recovered
    absent from the corpus
Dots mark the 13 descriptors of the reference vocabulary (LB2019_BENCHMARK_DESCRIPTORS or LB2019_DESCRIPTORS). The dashed
line marks the frequency of the most frequent unrecovered descriptor.

Inputs
    --chunks     run16E index/chunks.jsonl
    --graph      delivered graph (flat_run16E_enforced_dedup.json)
    --constants  pipeline/rag/constants.py (KNOWN_DESCRIPTORS and the 13-term reference set); default
                 the file of this repository

Checks (the figure is not written if any fails): the 40 occurrence counts of the published
figure, the 8 unrecovered and 1 absent descriptors, the 13 reference descriptors, and 15
recovered descriptors rarer than the most frequent unrecovered one. --no-check draws
whatever the inputs give.

Usage
    python scripts/figures/fig_descriptor_vocabulary.py --chunks /path/to/run16E/index/chunks.jsonl --graph /path/to/run16E/kg_final/flat_run16E_enforced_dedup.json --out output/figures/fig_descriptor_vocabulary
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PUBLISHED = {
    "thick": 271, "parallel": 244, "chaotic": 182, "erosional": 171, "continuous": 164,
    "deformed": 158, "thin": 136, "undeformed": 113, "high-amplitude": 101, "rough": 71,
    "irregular": 69, "folded": 63, "faulted": 54, "low-amplitude": 51, "transparent": 49,
    "discontinuous": 47, "massive": 46, "smooth": 39, "blocky": 38, "disrupted": 35,
    "layered": 30, "hummocky": 25, "convergent": 21, "wedge-shaped": 20, "divergent": 16,
    "aggradational": 14, "onlapping": 13, "mounded": 11, "contorted": 9, "progradational": 9,
    "wavy": 8, "moderate-amplitude": 7, "tabular": 7, "sheet-like": 6, "stratified": 5,
    "retrogradational": 4, "channelised": 3, "variable-amplitude": 3, "draping": 1, "lenticular": 0,
}
UNRECOVERED = {"massive", "smooth", "convergent", "divergent", "aggradational",
               "progradational", "retrogradational", "variable-amplitude"}
ABSENT = {"lenticular"}
REFERENCE_13 = {"parallel", "chaotic", "continuous", "undeformed", "high-amplitude", "low-amplitude",
                "transparent", "discontinuous", "massive", "blocky", "layered", "hummocky", "stratified"}
RECOVERED_C, UNRECOVERED_C, INK, MUTED, LINE = "#cfd6df", "#24456d", "#1b1b1b", "#555555", "#24456d"


def fail(msg):
    print("ERROR: " + msg, flush=True)
    sys.exit(1)


def pattern(term):
    parts = [re.escape(p) for p in term.lower().split("-")]
    return re.compile(r"(?<![a-z0-9])" + r"[-\s]+".join(parts) + r"(?![a-z0-9])", re.IGNORECASE)


def load_constants(path):
    spec = importlib.util.spec_from_file_location("constants_for_figure", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(chunks_path, graph_path, constants_path):
    c = load_constants(constants_path)
    vocab = sorted({d.lower() for d in c.KNOWN_DESCRIPTORS})
    ref = getattr(c, "LB2019_BENCHMARK_DESCRIPTORS", None) or getattr(c, "LB2019_DESCRIPTORS", None)
    if ref is None:
        fail("no 13-term reference set (LB2019_BENCHMARK_DESCRIPTORS or LB2019_DESCRIPTORS) in " + str(constants_path))
    reference = {d.lower() for d in ref}
    texts = [json.loads(line).get("text", "") for line in open(chunks_path, encoding="utf-8") if line.strip()]
    obj = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    triples = obj["triples"] if isinstance(obj, dict) else obj
    in_graph = {str(t.get("object") or "").strip().lower() for t in triples if t.get("relation") == "hasDescriptor"}
    counts = {d: sum(len(pattern(d).findall(t)) for t in texts) for d in vocab}
    status = {d: ("absent" if counts[d] == 0 else ("recovered" if d in in_graph else "unrecovered")) for d in vocab}
    return counts, status, reference, len(texts)


def check(counts, status, reference):
    problems = []
    if set(counts) != set(PUBLISHED):
        problems.append("vocabulary differs: missing %s, extra %s"
                        % (sorted(set(PUBLISHED) - set(counts)), sorted(set(counts) - set(PUBLISHED))))
    diffs = {d: (counts.get(d), v) for d, v in PUBLISHED.items() if counts.get(d) != v}
    if diffs:
        problems.append("occurrence counts (obtained, published): " + json.dumps(diffs))
    unrec = {d for d, s in status.items() if s == "unrecovered"}
    absent = {d for d, s in status.items() if s == "absent"}
    if unrec != UNRECOVERED:
        problems.append("unrecovered %s, published %s" % (sorted(unrec), sorted(UNRECOVERED)))
    if absent != ABSENT:
        problems.append("absent %s, published %s" % (sorted(absent), sorted(ABSENT)))
    if reference != REFERENCE_13:
        problems.append("reference descriptors %s" % sorted(reference))
    if unrec:
        top = max(counts[d] for d in unrec)
        rarer = sum(1 for d, s in status.items() if s == "recovered" and counts[d] < top)
        if rarer != 15:
            problems.append("recovered descriptors rarer than the most frequent unrecovered: %d, published 15" % rarer)
    return problems


def draw(counts, status, reference, n_chunks, out, width, height):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    order = sorted(counts, key=lambda d: (-counts[d], d))
    fig, ax = plt.subplots(figsize=(width, height))
    ys = list(range(len(order)))[::-1]
    xmax = max(counts.values()) * 1.12
    for d, y in zip(order, ys):
        s, v = status[d], counts[d]
        colour = UNRECOVERED_C if s == "unrecovered" else RECOVERED_C
        ax.barh(y, v, height=0.72, color="white" if s == "absent" else colour, edgecolor="#6b7280",
                linewidth=0.5, hatch="////" if s == "absent" else None)
        ax.text(v + xmax * 0.008, y, str(v), va="center", fontsize=6.2, color=INK)
        label = d + ("  \u25cf" if d in reference else "")
        ax.text(-xmax * 0.012, y, label, ha="right", va="center", fontsize=6.6, color=INK,
                fontweight="bold" if s != "recovered" else "normal")
    unrec = [d for d in order if status[d] == "unrecovered"]
    if unrec:
        top = max(counts[d] for d in unrec)
        rarer = sum(1 for d in order if status[d] == "recovered" and counts[d] < top)
        ax.axvline(top, color=LINE, linestyle=(0, (2, 2)), linewidth=0.8)
        ax.text(top + xmax * 0.02, len(order) * 0.42,
                "frequency does not predict recovery:\n%d recovered descriptors are rarer\nthan the most frequent unrecovered one" % rarer,
                fontsize=6.3, color=LINE, va="center", ha="left", linespacing=1.3)
    n_rec = sum(1 for s in status.values() if s == "recovered")
    n_unrec = len(unrec)
    n_abs = sum(1 for s in status.values() if s == "absent")
    ax.legend(handles=[Patch(facecolor=RECOVERED_C, edgecolor="#6b7280", label="recovered in the delivered graph (%d)" % n_rec),
                       Patch(facecolor=UNRECOVERED_C, edgecolor="#6b7280", label="attested, never recovered (%d)" % n_unrec),
                       Patch(facecolor="white", edgecolor="#6b7280", hatch="////", label="absent from the corpus (%d)" % n_abs),
                       Line2D([], [], marker="o", linestyle="none", color=INK, markersize=3.5,
                              label="one of the 13 descriptors of the reference vocabulary")],
              loc="lower right", frameon=False, fontsize=6.3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("Occurrences in the indexed corpus (%s chunks)" % format(n_chunks, ","))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out) + "." + ext, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written: %s.pdf and %s.png" % (out, out))


def main():
    ap = argparse.ArgumentParser(description="Figure 4: descriptors of the extraction vocabulary")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--constants", default=str(Path(__file__).resolve().parents[2] / "pipeline" / "rag" / "constants.py"))
    ap.add_argument("--out", default="fig_descriptor_vocabulary")
    ap.add_argument("--width", type=float, default=5.2)
    ap.add_argument("--height", type=float, default=6.6)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    counts, status, reference, n_chunks = measure(args.chunks, args.graph, args.constants)
    print("%d descriptors, %d chunks; recovered %d, unrecovered %d, absent %d" % (
        len(counts), n_chunks, sum(s == "recovered" for s in status.values()),
        sum(s == "unrecovered" for s in status.values()), sum(s == "absent" for s in status.values())))
    if not args.no_check:
        problems = check(counts, status, reference)
        if problems:
            fail("inputs do not reproduce the published figure; nothing written\n  - " + "\n  - ".join(problems))
    draw(counts, status, reference, n_chunks, args.out, args.width, args.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())