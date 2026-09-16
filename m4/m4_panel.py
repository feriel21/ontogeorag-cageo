#!/usr/bin/env python3
"""
m4_panel.py — V3: Multi-judge panel aggregation
================================================

Combines the verdicts of N independent cross-family judges (Llama,
Mistral, ... — the Qwen extractor is excluded from the jury) into:

  1. Inter-judge agreement: exact 3-class and binary agreement on the
     evidence pass, Cohen's kappa (unweighted + linear-weighted) on both
     passes — directly comparable to the inter-expert kappa reported in
     Section 4.4 (pass --human-kappa to embed that reference in the
     report).
  2. Panel decisions: each judge's verdicts go through the SAME decision
     matrix (m4_aggregate.decide); the panel decision is the majority
     vote, with a CONSERVATIVE tie-break (REJECT > UNCERTAIN > ACCEPT:
     on disagreement without majority, the most cautious decision wins).
     With two judges, "majority" reduces to consensus-or-downgrade —
     stated explicitly in the report.
  3. Output m4_panel_decisions.jsonl uses the SAME schema as
     m4_decisions.jsonl, so m4_integrate_tiers.py and m4_figures.py work
     on the panel output unchanged.
  4. Consensus flags: parametric_risk flagged by ALL judges = the
     highest-priority residual set.

Usage:
    python m4_panel.py \
        --judges llama=~/ontogeorag/output/m4/m4_verdicts.jsonl \
                 mistral=~/ontogeorag/output/m4_mistral/m4_verdicts.jsonl \
        --output ~/ontogeorag/output/m4_panel \
        --human-kappa 0.30 0.37
"""

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from m4_aggregate import confidence, decide
from m4_metrics import cohens_kappa

EV_ORDER = ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]
BLIND_ORDER = ["PLAUSIBLE", "UNCERTAIN", "IMPLAUSIBLE"]
DEC_ORDER = ["ACCEPT", "UNCERTAIN", "REJECT"]
EV_POS = {"SUPPORTED", "PARTIALLY_SUPPORTED"}
CAUTION = {"REJECT": 2, "UNCERTAIN": 1, "ACCEPT": 0}


def load_jsonl(path):
    """Read `path` as one JSON object per line; no side effects."""
    out = []
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def panel_vote(decisions: list) -> str:
    """Majority vote; conservative tie-break (most cautious wins)."""
    counts = Counter(decisions)
    top = counts.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    # tie -> most cautious among the tied decisions
    tied = [d for d, c in top if c == top[0][1]]
    return max(tied, key=lambda d: CAUTION[d])


def main():
    """CLI entry point: loads each --judges verdict file, computes pairwise inter-judge agreement/kappa, takes a conservative majority vote per triple, and writes m4_panel_decisions.jsonl + m4_panel_report.json to --output."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--judges",
        nargs="+",
        required=True,
        help="name=path pairs of m4_verdicts.jsonl files",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--human-kappa",
        nargs="*",
        type=float,
        default=None,
        help="Inter-expert kappa value(s) from Section 4.4, "
        "embedded in the report for comparison",
    )
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    judges = {}
    for spec in args.judges:
        name, path = spec.split("=", 1)
        judges[name] = {v["m4_index"]: v for v in load_jsonl(path)}
    names = sorted(judges)
    if len(names) < 2:
        raise SystemExit("Panel needs at least 2 judges.")

    # common indices, with identity sanity-check
    common = set.intersection(*(set(j.keys()) for j in judges.values()))
    common = sorted(common)
    mismatches = 0
    for i in common:
        keys = {
            (
                judges[n][i]["subject"],
                judges[n][i]["relation"],
                judges[n][i]["object"],
            )
            for n in names
        }
        if len(keys) > 1:
            mismatches += 1
    if mismatches:
        print(
            f"WARNING: {mismatches} index alignment mismatches — "
            f"judges may have run on different KG files."
        )

    report = {
        "judges": names,
        "n_common_triples": len(common),
        "note_two_judges": (
            "with two judges, majority vote reduces "
            "to consensus; on disagreement the more "
            "cautious decision is taken"
        )
        if len(names) == 2
        else None,
    }

    # ── per-judge distributions and decisions ──────────────────────────
    per_judge_dec = {n: {} for n in names}
    report["evidence_distributions"] = {}
    report["blind_distributions"] = {}
    for n in names:
        ev_c, bl_c = Counter(), Counter()
        for i in common:
            v = judges[n][i]
            bl, ev = v["blind"]["verdict"], v["evidence"]["verdict"]
            ev_c[ev] += 1
            bl_c[bl] += 1
            d, flags = decide(bl, ev)
            per_judge_dec[n][i] = {
                "decision": d,
                "flags": flags,
                "conf": confidence(bl, ev),
                "blind": bl,
                "evidence": ev,
            }
        report["evidence_distributions"][n] = dict(ev_c)
        report["blind_distributions"][n] = dict(bl_c)

    # ── pairwise inter-judge agreement ─────────────────────────────────
    report["inter_judge"] = {}
    for a, b in combinations(names, 2):
        ev_a, ev_b, bl_a, bl_b = [], [], [], []
        for i in common:
            va, vb = judges[a][i], judges[b][i]
            if (
                va["evidence"]["verdict"] in EV_ORDER
                and vb["evidence"]["verdict"] in EV_ORDER
            ):
                ev_a.append(va["evidence"]["verdict"])
                ev_b.append(vb["evidence"]["verdict"])
            if (
                va["blind"]["verdict"] in BLIND_ORDER
                and vb["blind"]["verdict"] in BLIND_ORDER
            ):
                bl_a.append(va["blind"]["verdict"])
                bl_b.append(vb["blind"]["verdict"])
        pair = {}
        if ev_a:
            pair["evidence"] = {
                "n": len(ev_a),
                "exact_agreement_3class": round(
                    sum(x == y for x, y in zip(ev_a, ev_b)) / len(ev_a), 4
                ),
                "binary_agreement": round(
                    sum(
                        (x in EV_POS) == (y in EV_POS)
                        for x, y in zip(ev_a, ev_b)
                    )
                    / len(ev_a),
                    4,
                ),
                "kappa_unweighted": cohens_kappa(ev_a, ev_b, EV_ORDER),
                "kappa_linear": cohens_kappa(
                    ev_a, ev_b, EV_ORDER, weighted=True
                ),
            }
        if bl_a:
            pair["blind"] = {
                "n": len(bl_a),
                "exact_agreement": round(
                    sum(x == y for x, y in zip(bl_a, bl_b)) / len(bl_a), 4
                ),
                "kappa_unweighted": cohens_kappa(bl_a, bl_b, BLIND_ORDER),
                "kappa_linear": cohens_kappa(
                    bl_a, bl_b, BLIND_ORDER, weighted=True
                ),
            }
        # agreement on final decisions
        d_a = [per_judge_dec[a][i]["decision"] for i in common]
        d_b = [per_judge_dec[b][i]["decision"] for i in common]
        pair["decisions"] = {
            "exact_agreement": round(
                sum(x == y for x, y in zip(d_a, d_b)) / len(d_a), 4
            ),
            "kappa_unweighted": cohens_kappa(d_a, d_b, DEC_ORDER),
            "kappa_linear": cohens_kappa(d_a, d_b, DEC_ORDER, weighted=True),
        }
        report["inter_judge"][f"{a}_x_{b}"] = pair

    if args.human_kappa:
        report["human_reference"] = {
            "inter_expert_kappa_section_4_4": args.human_kappa,
            "note": (
                "Machine inter-judge kappa above vs human "
                "inter-expert kappa: comparable levels indicate the "
                "task itself has an agreement ceiling; higher machine "
                "agreement indicates the judges are more consistent "
                "than human annotators on this taxonomy."
            ),
        }

    # ── panel decisions ────────────────────────────────────────────────
    panel_counts = Counter()
    disagreement = Counter()
    consensus_parametric = []
    out_path = out_dir / "m4_panel_decisions.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for i in common:
            ref = judges[names[0]][i]
            decs = [per_judge_dec[n][i]["decision"] for n in names]
            vote = panel_vote(decs)
            panel_counts[vote] += 1
            if len(set(decs)) > 1:
                disagreement["_".join(sorted(decs))] += 1

            flags = sorted(
                set.intersection(
                    *(set(per_judge_dec[n][i]["flags"]) for n in names)
                )
            )
            union_flags = sorted(
                set.union(*(set(per_judge_dec[n][i]["flags"]) for n in names))
            )
            if "parametric_risk" in flags:
                consensus_parametric.append(i)

            conf = round(
                sum(per_judge_dec[n][i]["conf"] for n in names) / len(names), 3
            )

            fout.write(
                json.dumps(
                    {
                        "m4_index": i,
                        "subject": ref["subject"],
                        "relation": ref["relation"],
                        "object": ref["object"],
                        "tier": ref.get("tier"),
                        "qwen_verdict": ref.get("qwen_verdict"),
                        # panel-level fields, schema-compatible with m4_decisions
                        "blind_verdict": "|".join(
                            per_judge_dec[n][i]["blind"] for n in names
                        ),
                        "evidence_verdict": "|".join(
                            per_judge_dec[n][i]["evidence"] for n in names
                        ),
                        "m4_decision": vote,
                        "m4_confidence": conf,
                        "flags": flags,  # consensus flags (all judges)
                        "flags_any_judge": union_flags,
                        "per_judge": {
                            n: per_judge_dec[n][i]["decision"] for n in names
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report["panel_decisions"] = dict(panel_counts)
    report["panel_decision_rates"] = {
        k: round(v / len(common), 4) for k, v in panel_counts.items()
    }
    report["disagreement_patterns"] = dict(disagreement)
    report["consensus_parametric_risk"] = {
        "n": len(consensus_parametric),
        "indices": consensus_parametric,
        "note": "flagged parametric_risk by ALL judges — highest-priority "
        "residual set",
    }

    (out_dir / "m4_panel_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"\nPanel decisions: {out_path}")
    print(
        "Compatible with m4_integrate_tiers.py --decisions and "
        "m4_figures.py --decisions."
    )


if __name__ == "__main__":
    main()
