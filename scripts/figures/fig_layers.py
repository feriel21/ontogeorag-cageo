#!/usr/bin/env python3
"""Figure 2: the 232 delivered relations under six assessments.

Each row partitions the same 232 relations; no row is a verdict on its own.
    Provenance          evidence sentence found in the index verbatim / in part / not found
    Primary verifier    strong / weak support
    Extraction tier     retained in both passes (Tier 1) / in one pass (Tier 2)
    Cross-family panel  accepted / uncertain and retained
    Directional check   forward / reversed / undirected, among panel-accepted relations with a
                        directional predicate; directional but uncertain (not checked);
                        no directional predicate
    Expert review       keep / merge / remove / uncertain

Inputs
    --graph      delivered graph with located provenance
                 (output of scripts/evaluation/graph_with_located_provenance.py):
                 fields tier, verdict, m4.decision, located_provenance.match
    --direction  m4_direction_verdicts.jsonl of the delivered run, joined by content key
                 (subject, relation, object), never by position

Expert review. The four counts come from the reviewer's spreadsheet, which contains an
identified person's judgements and is not released; they are written in EXPERT below.
The three relations rejected by the panel before delivery are not among the 232 and are
reported in the sublabel only (m4_panel_report.json).

Checks (the figure is not written if any fails): 232 relations; located 164 / 64 / 4;
verdicts 177 / 55; tiers 160 / 72; panel 189 / 43; directional predicates: 95 accepted,
12 uncertain, 125 none; direction verdicts 80 / 12 / 3, every one joined to a delivered
relation, and every accepted directional relation checked.

Usage
    python scripts/figures/fig_layers.py --graph output/audit/145_provenance/flat_run16E_with_located_provenance.json --direction /path/to/run16E/m4/direction_check/m4_direction_verdicts.jsonl --out output/figures/fig_layers
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIRECTIONAL = {"causes", "triggers", "controls", "affects", "overlies", "underlies",
               "evolvesTo", "indicates", "formedBy"}
EXPERT = [("keep", 144), ("merge", 18), ("remove", 69), ("uncertain", 1)]
PANEL_REJECTED_BEFORE_DELIVERY = 3
DARK, MID, LIGHT, NA, INK, MUTED = "#24456d", "#8497b0", "#d3d9e2", "#f0f0f0", "#1b1b1b", "#555555"


def fail(msg):
    print("ERROR: " + msg, flush=True)
    sys.exit(1)


def key(s, r, o):
    return (str(s).strip().lower(), str(r).strip(), str(o).strip().lower())


def measure(graph_path, direction_path):
    obj = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    triples = obj["triples"] if isinstance(obj, dict) else obj
    located = Counter((t.get("located_provenance") or {}).get("match") for t in triples)
    verdict = Counter(t.get("verdict") for t in triples)
    tier = Counter(str(t.get("tier")) for t in triples)
    decision = Counter((t.get("m4") or {}).get("decision") for t in triples)
    accepted_dir = {key(t["subject"], t["relation"], t["object"]) for t in triples
                    if t["relation"] in DIRECTIONAL and (t.get("m4") or {}).get("decision") == "ACCEPT"}
    uncertain_dir = sum(1 for t in triples if t["relation"] in DIRECTIONAL
                        and (t.get("m4") or {}).get("decision") != "ACCEPT")
    no_dir = sum(1 for t in triples if t["relation"] not in DIRECTIONAL)
    verdicts = [json.loads(line) for line in open(direction_path, encoding="utf-8") if line.strip()]
    checked = {key(v["subject"], v["relation"], v["object"]): v["direction_verdict"] for v in verdicts}
    direction = Counter(checked.values())
    return {
        "n": len(triples),
        "located": (located.get("exact", 0), located.get("partial", 0), located.get("not_found", 0)),
        "verdict": (verdict.get("STRONG_SUPPORT", 0), verdict.get("WEAK_SUPPORT", 0)),
        "tier": (tier.get("1", 0), tier.get("2", 0)),
        "panel": (decision.get("ACCEPT", 0), decision.get("UNCERTAIN", 0)),
        "directional": (len(accepted_dir), uncertain_dir, no_dir),
        "direction": (direction.get("FORWARD", 0), direction.get("REVERSE", 0), direction.get("UNDIRECTED", 0)),
        "n_verdicts": len(verdicts),
        "verdicts_not_in_graph": len(set(checked) - accepted_dir),
        "accepted_not_checked": len(accepted_dir - set(checked)),
    }


EXPECTED = {"n": 232, "located": (164, 64, 4), "verdict": (177, 55), "tier": (160, 72),
            "panel": (189, 43), "directional": (95, 12, 125), "direction": (80, 12, 3),
            "n_verdicts": 95, "verdicts_not_in_graph": 0, "accepted_not_checked": 0}


def rows(m):
    fwd, rev, und = m["direction"]
    acc, unc, none = m["directional"]
    exact, partial, nf = m["located"]
    return [
        ("Provenance", [(exact, DARK), (partial, MID), (nf, LIGHT)],
         "evidence sentence found verbatim %d, in part %d, not found %d" % (exact, partial, nf)),
        ("Primary verifier", [(m["verdict"][0], DARK), (m["verdict"][1], MID)],
         "%d strong, %d weak support" % m["verdict"]),
        ("Extraction tier", [(m["tier"][0], DARK), (m["tier"][1], MID)],
         "%d in both passes, %d in one" % m["tier"]),
        ("Cross-family panel", [(m["panel"][0], DARK), (m["panel"][1], MID)],
         "%d accepted, %d uncertain and retained; %d rejected before delivery"
         % (m["panel"][0], m["panel"][1], PANEL_REJECTED_BEFORE_DELIVERY)),
        ("Directional check", [(fwd, DARK), (rev, MID), (und, LIGHT), (unc, "hatch"), (none, NA)],
         "%d forward, %d reversed, %d undirected; %d uncertain, not checked; %d not directional"
         % (fwd, rev, und, unc, none)),
        ("Expert review", [(EXPERT[0][1], DARK), (EXPERT[1][1], MID), (EXPERT[2][1], LIGHT), (EXPERT[3][1], NA)],
         "%d keep, %d merge, %d remove, %d uncertain; single assessor, ordered table"
         % tuple(v for _, v in EXPERT)),
    ]


def draw(m, out, width, height):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.5})
    data = rows(m)
    fig, ax = plt.subplots(figsize=(width, height))
    n = m["n"]
    for i, (name, segments, sub) in enumerate(data):
        y = (len(data) - 1 - i) * 1.25
        left = 0
        for value, colour in segments:
            if value == 0:
                continue
            hatch = None
            face = colour
            if colour == "hatch":
                face, hatch = "white", "////"
            ax.barh(y, value, left=left, height=0.62, color=face, edgecolor=INK, linewidth=0.6, hatch=hatch)
            if value >= 9:
                ax.text(left + value / 2, y, str(value), ha="center", va="center", fontsize=7.5,
                        color="white" if colour == DARK else INK,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none") if hatch else None)
            left += value
        ax.text(-n * 0.012, y, name, ha="right", va="center", fontsize=8.2, fontweight="bold", color=INK)
        ax.text(0, y - 0.47, sub, ha="left", va="center", fontsize=6.6, color=MUTED)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.85, (len(data) - 1) * 1.25 + 0.45)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Delivered relations (%d)" % n)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out) + "." + ext, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written: %s.pdf and %s.png" % (out, out))


def main():
    ap = argparse.ArgumentParser(description="Figure 2: six assessments of the delivered relations")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--out", default="fig_layers")
    ap.add_argument("--width", type=float, default=6.3)
    ap.add_argument("--height", type=float, default=3.6)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    m = measure(args.graph, args.direction)
    print(json.dumps(m))
    if not args.no_check:
        diffs = ["%s: %s, expected %s" % (k, m[k], v) for k, v in EXPECTED.items() if m[k] != v]
        if diffs:
            fail("inputs do not reproduce the expected counts; nothing written\n  - " + "\n  - ".join(diffs))
    draw(m, args.out, args.width, args.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())