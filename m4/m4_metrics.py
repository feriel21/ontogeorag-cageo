#!/usr/bin/env python3
"""
m4_metrics.py — M4 report: agreement rates, Cohen's kappa, diagnostics
=======================================================================

Consumes m4_decisions.jsonl (from m4_aggregate.py) and produces the final
M4 report with the metrics intended for the manuscript:

  1. Verdict distributions (blind, evidence, decision) — overall and per
     tier / per relation.
  2. Blind-vs-evidence agreement:
       * exact agreement on the collapsed positive/negative axis
       * over-interpretation index:
             P(blind = PLAUSIBLE | evidence = NOT_SUPPORTED)
         = how often the judge finds a triple plausible that the text does
           not support: the parametric-risk rate.
  3. Cross-verifier agreement with the original Qwen verdicts, when
     present (SUPPORTED~STRONG_SUPPORT, PARTIALLY~WEAK, NOT~NOT), with
     Cohen's kappa — the direct successor of Exp D, now on all triples.
  4. Optional: kappa between M4 decisions and expert labels (--experts
     CSV with columns: subject,relation,object,expert_verdict where
     expert_verdict in {Y,P,N}). Mapping: Y->ACCEPT, P->UNCERTAIN,
     N->REJECT. This is the meta-evaluation that calibrates the LLM judge
     against the human gold standard (Elia/Alain, Sara, Antoine).

Cohen's kappa is implemented locally (no sklearn dependency) with the
standard unweighted formula; linear-weighted kappa is also reported for
ordinal comparisons, matching the convention already used for the
inter-expert agreement in Section 4.4.

Usage:
    python m4_metrics.py \
        --decisions ~/ontogeorag/output/m4/m4_decisions.jsonl \
        --output    ~/ontogeorag/output/m4 \
        [--experts  ~/ontogeorag/reference/expert_labels.csv]
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

# ── Cohen's kappa (unweighted + linear-weighted) ───────────────────────


def cohens_kappa(labels_a, labels_b, categories=None, weighted=False):
    """Compute Cohen's kappa between paired `labels_a`/`labels_b` (unweighted, or linear-weighted by category distance if `weighted`); returns the rounded kappa, no side effects."""
    assert len(labels_a) == len(labels_b) and labels_a
    cats = categories or sorted(set(labels_a) | set(labels_b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = len(labels_a)

    # confusion matrix
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        cm[idx[a]][idx[b]] += 1

    # weights: 0 on diagonal; unweighted = 1 off-diagonal;
    # linear = |i-j| / (k-1)
    def w(i, j):
        if not weighted:
            return 0.0 if i == j else 1.0
        return abs(i - j) / (k - 1) if k > 1 else 0.0

    row = [sum(cm[i]) for i in range(k)]
    col = [sum(cm[i][j] for i in range(k)) for j in range(k)]

    obs = sum(w(i, j) * cm[i][j] for i in range(k) for j in range(k)) / n
    exp = sum(
        w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)
    ) / (n * n)
    if exp == 0:
        return 1.0
    return round(1.0 - obs / exp, 4)


# ── Verdict mappings ───────────────────────────────────────────────────

# collapse to a binary "supported" axis for cross-instrument comparison
EV_POS = {"SUPPORTED", "PARTIALLY_SUPPORTED"}
QWEN_POS = {"STRONG_SUPPORT", "WEAK_SUPPORT"}

QWEN_TO_M4 = {
    "STRONG_SUPPORT": "SUPPORTED",
    "WEAK_SUPPORT": "PARTIALLY_SUPPORTED",
    "NOT_SUPPORTED": "NOT_SUPPORTED",
}

EXPERT_TO_DECISION = {"Y": "ACCEPT", "P": "UNCERTAIN", "N": "REJECT"}


def norm_key(subject, relation, obj):
    """Build a lowercase/stripped (subject, relation, object) tuple key for joining against expert-label CSV rows; no side effects."""
    return (subject.strip().lower(), relation.strip(), obj.strip().lower())


def main():
    """CLI entry point: loads --decisions, computes verdict distributions, blind-vs-evidence and M4-vs-Qwen agreement (with kappa), optionally M4-vs-experts kappa from --experts, and writes m4_report.json to --output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--experts",
        default=None,
        help="Optional CSV: subject,relation,object,expert_verdict",
    )
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(Path(args.decisions).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n = len(records)

    report = {"n_triples": n}

    # ── 1. distributions ───────────────────────────────────────────────
    report["blind_distribution"] = dict(
        Counter(r["blind_verdict"] for r in records)
    )
    report["evidence_distribution"] = dict(
        Counter(r["evidence_verdict"] for r in records)
    )
    report["decision_distribution"] = dict(
        Counter(r["m4_decision"] for r in records)
    )

    by_tier = defaultdict(Counter)
    for r in records:
        by_tier[str(r.get("tier"))][r["m4_decision"]] += 1
    report["decisions_by_tier"] = {k: dict(v) for k, v in by_tier.items()}

    by_rel = defaultdict(Counter)
    for r in records:
        by_rel[r["relation"]][r["m4_decision"]] += 1
    report["decisions_by_relation"] = {k: dict(v) for k, v in by_rel.items()}

    # ── 2. blind vs evidence ───────────────────────────────────────────
    valid = [
        r
        for r in records
        if r["blind_verdict"] in ("PLAUSIBLE", "UNCERTAIN", "IMPLAUSIBLE")
        and r["evidence_verdict"]
        in ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED")
    ]
    if valid:
        bin_agree = sum(
            (
                (r["blind_verdict"] == "PLAUSIBLE")
                == (r["evidence_verdict"] in EV_POS)
            )
            for r in valid
        )
        report["blind_vs_evidence"] = {
            "n_comparable": len(valid),
            "binary_agreement": round(bin_agree / len(valid), 4),
        }
        ns = [r for r in valid if r["evidence_verdict"] == "NOT_SUPPORTED"]
        if ns:
            risk = sum(r["blind_verdict"] == "PLAUSIBLE" for r in ns)
            report["blind_vs_evidence"]["over_interpretation_index"] = {
                "definition": "P(blind=PLAUSIBLE | evidence=NOT_SUPPORTED)",
                "n_not_supported": len(ns),
                "n_plausible_among_them": risk,
                "rate": round(risk / len(ns), 4),
            }

    # ── 3. M4 vs Qwen (cross-verifier, successor of Exp D) ─────────────
    both = [
        r
        for r in records
        if r.get("qwen_verdict") in QWEN_TO_M4
        and r["evidence_verdict"]
        in ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED")
    ]
    if both:
        qwen_bin = [
            ("POS" if r["qwen_verdict"] in QWEN_POS else "NEG") for r in both
        ]
        m4_bin = [
            ("POS" if r["evidence_verdict"] in EV_POS else "NEG") for r in both
        ]
        exact_q = [QWEN_TO_M4[r["qwen_verdict"]] for r in both]
        exact_m = [r["evidence_verdict"] for r in both]
        order = ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]
        report["m4_vs_qwen"] = {
            "n_comparable": len(both),
            "binary_agreement": round(
                sum(a == b for a, b in zip(qwen_bin, m4_bin)) / len(both), 4
            ),
            "exact_agreement_3class": round(
                sum(a == b for a, b in zip(exact_q, exact_m)) / len(both), 4
            ),
            "kappa_unweighted": cohens_kappa(exact_q, exact_m, order),
            "kappa_linear": cohens_kappa(
                exact_q, exact_m, order, weighted=True
            ),
        }

    # ── 4. M4 vs experts (meta-evaluation, optional) ───────────────────
    if args.experts:
        expert = {}
        with open(Path(args.experts).expanduser(), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                v = row.get("expert_verdict", "").strip().upper()
                if v in EXPERT_TO_DECISION:
                    expert[
                        norm_key(
                            row["subject"], row["relation"], row["object"]
                        )
                    ] = EXPERT_TO_DECISION[v]
        paired_m4, paired_ex = [], []
        for r in records:
            key = norm_key(r["subject"], r["relation"], r["object"])
            if key in expert:
                paired_m4.append(r["m4_decision"])
                paired_ex.append(expert[key])
        if paired_m4:
            order = ["ACCEPT", "UNCERTAIN", "REJECT"]
            report["m4_vs_experts"] = {
                "n_paired": len(paired_m4),
                "exact_agreement": round(
                    sum(a == b for a, b in zip(paired_m4, paired_ex))
                    / len(paired_m4),
                    4,
                ),
                "kappa_unweighted": cohens_kappa(paired_m4, paired_ex, order),
                "kappa_linear": cohens_kappa(
                    paired_m4, paired_ex, order, weighted=True
                ),
                "note": (
                    "Compare with inter-expert kappa (Section 4.4) "
                    "to position the LLM judge relative to human "
                    "agreement levels."
                ),
            }
        else:
            report["m4_vs_experts"] = {
                "n_paired": 0,
                "warning": "No triples matched expert CSV keys "
                "(check entity normalization).",
            }

    out_path = out_dir / "m4_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
