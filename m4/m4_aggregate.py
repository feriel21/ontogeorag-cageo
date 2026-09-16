#!/usr/bin/env python3
"""
m4_aggregate.py — Verdict aggregation and final decisions
==========================================================

Consumes m4_verdicts.jsonl (from m4_verify.py) and produces, per triple:

  * confidence score in [0, 1]  =  0.7 * evidence_score + 0.3 * blind_score
  * decision: ACCEPT / REJECT / UNCERTAIN
  * diagnostic flags, in particular "parametric_risk"
    (blind PLAUSIBLE + evidence NOT_SUPPORTED: the triple sounds right but
     is not in the text — the Exp-B contamination signature).

Decision logic (matrix first, score as continuous companion)
-------------------------------------------------------------
The categorical matrix is the PRIMARY decision rule; the continuous score
is reported alongside for ranking and threshold-sensitivity analysis.

  evidence \\ blind |  PLAUSIBLE      UNCERTAIN       IMPLAUSIBLE
  -----------------+------------------------------------------------
  SUPPORTED        |  ACCEPT         ACCEPT          ACCEPT(flag:contested)
  PARTIALLY_SUPP.  |  ACCEPT(flag)   UNCERTAIN       UNCERTAIN
  NOT_SUPPORTED    |  UNCERTAIN      REJECT          REJECT
                   |  (flag:parametric_risk)
  NO_PASSAGE       |  UNCERTAIN      UNCERTAIN       REJECT

Rationale: textual support dominates (the pipeline's claim is literature
grounding). A SUPPORTED verdict is accepted even when the blind judge is
sceptical — but flagged, because that disagreement is exactly what the
expert validation should inspect first. A NOT_SUPPORTED verdict is never
promoted to ACCEPT by plausibility alone: plausibility without text is the
definition of parametric contamination.

Usage:
    python m4_aggregate.py \
        --verdicts ~/ontogeorag/output/m4/m4_verdicts.jsonl \
        --output   ~/ontogeorag/output/m4
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from m4_config import (
    ACCEPT_THRESHOLD,
    BLIND_SCORE,
    EVIDENCE_SCORE,
    REJECT_THRESHOLD,
    W_BLIND,
    W_EVIDENCE,
)


def decide(blind_v: str, ev_v: str) -> tuple:
    """Return (decision, flags) from the categorical matrix."""
    flags = []

    if ev_v == "SUPPORTED":
        if blind_v == "IMPLAUSIBLE":
            flags.append("contested_plausibility")
        return "ACCEPT", flags

    if ev_v == "PARTIALLY_SUPPORTED":
        if blind_v == "PLAUSIBLE":
            flags.append("partial_support")
            return "ACCEPT", flags
        return "UNCERTAIN", flags

    if ev_v == "NOT_SUPPORTED":
        if blind_v == "PLAUSIBLE":
            flags.append("parametric_risk")
            return "UNCERTAIN", flags
        return "REJECT", flags

    # NO_PASSAGE / UNPARSEABLE
    flags.append(f"evidence_{ev_v.lower()}")
    if blind_v == "IMPLAUSIBLE":
        return "REJECT", flags
    return "UNCERTAIN", flags


def confidence(blind_v: str, ev_v: str) -> float:
    """Continuous companion score. NO_PASSAGE treated as 0 evidence."""
    ev_s = EVIDENCE_SCORE.get(ev_v, 0.0)
    bl_s = BLIND_SCORE.get(blind_v, 0.5)  # UNPARSEABLE blind -> neutral
    return round(W_EVIDENCE * ev_s + W_BLIND * bl_s, 3)


def main():
    """CLI entry point: loads --verdicts, applies decide()/confidence() to each record, and writes per-triple decisions (m4_decisions.jsonl) plus a decision-rate summary (m4_decision_summary.json) to --output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(Path(args.verdicts).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    decisions = Counter()
    flag_counts = Counter()
    score_agree = Counter()  # matrix vs threshold-score agreement check

    out_path = out_dir / "m4_decisions.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for r in records:
            blind_v = r["blind"]["verdict"]
            ev_v = r["evidence"]["verdict"]

            decision, flags = decide(blind_v, ev_v)
            conf = confidence(blind_v, ev_v)

            # threshold-based decision, reported for sensitivity analysis
            if conf >= ACCEPT_THRESHOLD:
                score_decision = "ACCEPT"
            elif conf <= REJECT_THRESHOLD:
                score_decision = "REJECT"
            else:
                score_decision = "UNCERTAIN"
            score_agree[
                "agree" if score_decision == decision else "disagree"
            ] += 1

            decisions[decision] += 1
            for fl in flags:
                flag_counts[fl] += 1

            fout.write(
                json.dumps(
                    {
                        "m4_index": r["m4_index"],
                        "subject": r["subject"],
                        "relation": r["relation"],
                        "object": r["object"],
                        "tier": r.get("tier"),
                        "qwen_verdict": r.get("qwen_verdict"),
                        "blind_verdict": blind_v,
                        "evidence_verdict": ev_v,
                        "m4_decision": decision,
                        "m4_confidence": conf,
                        "score_decision": score_decision,
                        "flags": flags,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "n_triples": len(records),
        "decisions": dict(decisions),
        "decision_rates": {
            k: round(v / len(records), 4) for k, v in decisions.items()
        }
        if records
        else {},
        "flags": dict(flag_counts),
        "matrix_vs_score_agreement": dict(score_agree),
        "weights": {"evidence": W_EVIDENCE, "blind": W_BLIND},
        "thresholds": {"accept": ACCEPT_THRESHOLD, "reject": REJECT_THRESHOLD},
    }
    (out_dir / "m4_decision_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nDecisions written to {out_path}")


if __name__ == "__main__":
    main()
