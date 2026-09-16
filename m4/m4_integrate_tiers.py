#!/usr/bin/env python3
"""
m4_integrate_tiers.py — Rebuild the tiered KG with M4 verdicts
===============================================================

Implements the Tier-1 redefinition from the revised architecture:

  OLD Tier-1: verified in both extraction passes by the SAME model
              that extracted (Qwen self-verification).
  NEW Tier-1: consistent across extraction passes AND confirmed by the
              independent cross-family verifier (M4 decision = ACCEPT).

Tier assignment (non-destructive: original tier kept as `tier_original`):

  original tier | M4 decision | new tier
  --------------+-------------+---------------------------
  1             | ACCEPT      | 1        (confirmed core)
  1             | UNCERTAIN   | 2        (demoted, flagged)
  1             | REJECT      | quarantine (removed from graph,
                |             |  kept in m4_quarantine.jsonl)
  2             | ACCEPT      | 2        (confirmed periphery)
  2             | UNCERTAIN   | 2        (flagged)
  2             | REJECT      | quarantine

Nothing is deleted: quarantined triples are written to a separate file
with full M4 diagnostics, so the demotion/removal is fully auditable and
reversible — consistent with the additive, non-destructive workflow.

Usage:
    python m4_integrate_tiers.py \
        --kg        ~/ontogeorag/output/run11_kg/tiered_kg_run11.json \
        --decisions ~/ontogeorag/output/m4/m4_decisions.jsonl \
        --output    ~/ontogeorag/output/m4
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# reuse the loader from m4_verify to handle both KG formats
from m4_verify import load_triples, triple_fields


def norm_key(subject, relation, obj):
    """Build a lowercase/stripped (subject, relation, object) tuple key for joining against m4 decisions/direction verdicts; no side effects."""
    return (subject.strip().lower(), relation.strip(), obj.strip().lower())


def main():
    """CLI entry point: reassigns each --kg triple's tier from its --decisions M4 verdict (and optional --direction check), quarantines REJECTed triples, and writes tiered_kg_m4.json + m4_quarantine.jsonl to --output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument(
        "--direction",
        default=None,
        help="Optional m4_direction_verdicts.jsonl. Applies the "
        "conservative directional rule: REVERSE-flagged "
        "triples whose evidence verdict was "
        "PARTIALLY_SUPPORTED are demoted to Tier 2; "
        "UNDIRECTED is recorded as a documented flag.",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = load_triples(Path(args.kg).expanduser())

    decisions = {}
    with open(Path(args.decisions).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            decisions[norm_key(d["subject"], d["relation"], d["object"])] = d

    direction = {}
    if args.direction:
        with open(Path(args.direction).expanduser(), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dv = json.loads(line)
                direction[
                    norm_key(dv["subject"], dv["relation"], dv["object"])
                ] = dv

    kept, quarantined = [], []
    moves = Counter()

    for t in triples:
        subj, rel, obj = triple_fields(t)
        d = decisions.get(norm_key(subj, rel, obj))
        orig_tier = t.get("tier", 2)
        t["tier_original"] = orig_tier

        if d is None:
            # no M4 verdict (should not happen if M4 ran on the same KG)
            t["tier"] = orig_tier
            t["m4"] = {"decision": "MISSING"}
            moves["missing_m4_verdict"] += 1
            kept.append(t)
            continue

        t["m4"] = {
            "decision": d["m4_decision"],
            "confidence": d["m4_confidence"],
            "blind_verdict": d["blind_verdict"],
            "evidence_verdict": d["evidence_verdict"],
            "flags": d.get("flags", []),
        }

        if d["m4_decision"] == "REJECT":
            moves[f"tier{orig_tier}_to_quarantine"] += 1
            quarantined.append(t)
            continue

        if d["m4_decision"] == "ACCEPT" and orig_tier == 1:
            t["tier"] = 1
            moves["tier1_confirmed"] += 1
        elif d["m4_decision"] == "ACCEPT":
            t["tier"] = 2
            moves["tier2_confirmed"] += 1
        else:  # UNCERTAIN
            t["tier"] = 2
            moves[f"tier{orig_tier}_uncertain_to_tier2"] += 1

        # ── directional rule (conservative, documented) ────────────
        dv = direction.get(norm_key(subj, rel, obj))
        if dv is not None:
            t["m4"]["direction"] = {
                "verdict": dv["direction_verdict"],
                "evidence_verdict_at_check": dv.get("evidence_verdict"),
            }
            if dv["direction_verdict"] == "REVERSE":
                if (
                    dv.get("evidence_verdict") == "PARTIALLY_SUPPORTED"
                    and t["tier"] == 1
                ):
                    t["tier"] = 2
                    t["m4"]["direction"]["action"] = (
                        "demoted_to_tier2 (REVERSE + PARTIALLY_SUPPORTED)"
                    )
                    moves["direction_reverse_demoted"] += 1
                else:
                    t["m4"]["direction"]["action"] = (
                        "flagged (REVERSE, kept: SUPPORTED evidence or "
                        "already Tier 2)"
                    )
                    moves["direction_reverse_flagged"] += 1
            elif dv["direction_verdict"] == "UNDIRECTED":
                t["m4"]["direction"]["action"] = (
                    "flagged (direction unstated in passage)"
                )
                moves["direction_undirected_flagged"] += 1
        kept.append(t)

    tier1 = [t for t in kept if t.get("tier") == 1]
    tier2 = [t for t in kept if t.get("tier") == 2]

    kg_out = {
        "meta": {
            "source_kg": str(args.kg),
            "m4_decisions": str(args.decisions),
            "direction_verdicts": str(args.direction)
            if args.direction
            else None,
            "tier1_definition": (
                "cross-pass consistency AND independent "
                "cross-family verifier ACCEPT"
                + (
                    " AND no REVERSE direction flag with "
                    "only partial evidence support"
                    if args.direction
                    else ""
                )
            ),
            "n_tier1": len(tier1),
            "n_tier2": len(tier2),
            "n_quarantined": len(quarantined),
            "moves": dict(moves),
        },
        "tier1": tier1,
        "tier2": tier2,
    }

    kg_path = out_dir / "tiered_kg_m4.json"
    kg_path.write_text(
        json.dumps(kg_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    q_path = out_dir / "m4_quarantine.jsonl"
    with open(q_path, "w", encoding="utf-8") as f:
        for t in quarantined:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(json.dumps(kg_out["meta"], indent=2))
    print(f"\nNew tiered KG : {kg_path}")
    print(f"Quarantine    : {q_path}")


if __name__ == "__main__":
    main()
