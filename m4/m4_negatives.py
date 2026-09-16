#!/usr/bin/env python3
"""
m4_negatives.py — V2: Adversarial negative controls
=====================================================

Characterizes the verification instrument on ground truth that is known
BY CONSTRUCTION (no human annotation): triples corrupted by deterministic
rules must be judged NOT_SUPPORTED by a competent evidence judge.

Two subcommands:

  generate  Build the negative-control set from the KG. Four balanced
            corruption classes, seeded RNG, fully reproducible:
              inversion   swap subject/object of a directional relation
                          (causes, triggers, controls, affects, overlies,
                          underlies, evolvesTo, indicates, formedBy)
              entity_sub  replace the object with the object of another
                          triple sharing the same relation type
              relation_sub replace the relation with another allowed
                          relation valid for a different context
              passage_perm keep the triple, pair it with the passage of
                          another triple (grounding mismatch)
            Positives = the original triples themselves (their passages
            untouched). Output is a .jsonl directly consumable by
            m4_verify.py (passage embedded as `supporting_passage`).

  report    After running m4_verify.py on the generated file, compute
            the instrument-characterization metrics:
              sensitivity (recall on corrupted), specificity (on
              originals), precision, F1, balanced accuracy — overall and
              per corruption class — plus ROC-AUC on the graded evidence
              score (SUPPORTED=0, PARTIAL=0.5, NOT_SUPPORTED=1 as the
              "corruption detector" score; Mann-Whitney AUC with tie
              correction, no sklearn dependency).

These are classification metrics used for METROLOGY of the verifier —
there is no training phase anywhere; ground truth exists by construction.

Usage:
    # 1. generate (CPU, instant)
    python m4_negatives.py generate \
        --kg     ~/ontogeorag/output/run11_kg/tiered_kg_run11.json \
        --output ~/ontogeorag/output/m4/negatives \
        --seed 13

    # 2. verify positives+negatives with the SAME frozen judge
    python m4_verify.py \
        --kg     ~/ontogeorag/output/m4/negatives/controls.jsonl \
        --output ~/ontogeorag/output/m4/negatives \
        --model  meta-llama/Llama-3.1-8B-Instruct

    # 3. report (CPU, instant)
    python m4_negatives.py report \
        --controls ~/ontogeorag/output/m4/negatives/controls.jsonl \
        --verdicts ~/ontogeorag/output/m4/negatives/m4_verdicts.jsonl \
        --output   ~/ontogeorag/output/m4/negatives
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from m4_verify import evidence_from_provenance, load_triples, triple_fields

DIRECTIONAL = {
    "causes",
    "triggers",
    "controls",
    "affects",
    "overlies",
    "underlies",
    "evolvesTo",
    "indicates",
    "formedBy",
}
ALL_RELATIONS = sorted(DIRECTIONAL | {"hasDescriptor", "occursIn", "partOf"})

EV_SCORE_AS_DETECTOR = {
    "SUPPORTED": 0.0,
    "PARTIALLY_SUPPORTED": 0.5,
    "NOT_SUPPORTED": 1.0,
}


# ── generate ───────────────────────────────────────────────────────────


def cmd_generate(args):
    """`generate` subcommand: build the balanced positive/negative control set from --kg (4 corruption classes, seeded by --seed) and write controls.jsonl + controls_meta.json to --output."""
    rng = random.Random(args.seed)
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = load_triples(Path(args.kg).expanduser())
    base = []
    for i, t in enumerate(triples):
        s, r, o = triple_fields(t)
        p = evidence_from_provenance(t)
        if s and r and o and p:
            base.append(
                {
                    "orig_index": i,
                    "subject": s,
                    "relation": r,
                    "object": o,
                    "passage": p,
                }
            )
    print(f"Usable base triples (with passage): {len(base)}/{len(triples)}")

    objects_by_rel = defaultdict(list)
    for b in base:
        objects_by_rel[b["relation"]].append(b["object"])

    controls = []

    def emit(b, corruption, s, r, o, passage):
        """Append one control record (original or corrupted variant of base triple `b`) to the outer `controls` list."""
        controls.append(
            {
                "subject": s,
                "relation": r,
                "object": o,
                "supporting_passage": passage,
                "label": "corrupted" if corruption else "original",
                "corruption_class": corruption or "none",
                "orig_index": b["orig_index"],
            }
        )

    # positives: all originals
    for b in base:
        emit(b, None, b["subject"], b["relation"], b["object"], b["passage"])

    # negatives: one corruption per base triple, class round-robin for
    # balance, with per-class applicability checks
    classes = ["inversion", "entity_sub", "relation_sub", "passage_perm"]
    ci = 0
    for b in base:
        made = False
        for attempt in range(len(classes)):
            cls = classes[(ci + attempt) % len(classes)]
            if cls == "inversion":
                if (
                    b["relation"] in DIRECTIONAL
                    and b["subject"] != b["object"]
                ):
                    emit(
                        b,
                        cls,
                        b["object"],
                        b["relation"],
                        b["subject"],
                        b["passage"],
                    )
                    made = True
            elif cls == "entity_sub":
                pool = [
                    o
                    for o in objects_by_rel[b["relation"]]
                    if o != b["object"]
                ]
                if pool:
                    emit(
                        b,
                        cls,
                        b["subject"],
                        b["relation"],
                        rng.choice(pool),
                        b["passage"],
                    )
                    made = True
            elif cls == "relation_sub":
                pool = [r for r in ALL_RELATIONS if r != b["relation"]]
                emit(
                    b,
                    cls,
                    b["subject"],
                    rng.choice(pool),
                    b["object"],
                    b["passage"],
                )
                made = True
            elif cls == "passage_perm":
                pool = [
                    x
                    for x in base
                    if x["orig_index"] != b["orig_index"]
                    and x["passage"] != b["passage"]
                ]
                if pool:
                    emit(
                        b,
                        cls,
                        b["subject"],
                        b["relation"],
                        b["object"],
                        rng.choice(pool)["passage"],
                    )
                    made = True
            if made:
                ci += 1
                break
        if not made:
            print(
                f"  WARN: no corruption applicable for triple "
                f"{b['orig_index']}"
            )

    rng.shuffle(controls)
    out_path = out_dir / "controls.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in controls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    dist = Counter(c["corruption_class"] for c in controls)
    meta = {
        "seed": args.seed,
        "n_controls": len(controls),
        "class_distribution": dict(dist),
    }
    (out_dir / "controls_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(f"Controls: {out_path}")
    print("Next: run m4_verify.py on this file (see module docstring).")


# ── report ─────────────────────────────────────────────────────────────


def auc_mann_whitney(scores_pos, scores_neg):
    """AUC with tie correction: P(score_pos > score_neg) + 0.5*P(equal).
    'pos' = corrupted (should score high on the detector scale)."""
    if not scores_pos or not scores_neg:
        return None
    wins = ties = 0
    for sp in scores_pos:
        for sn in scores_neg:
            if sp > sn:
                wins += 1
            elif sp == sn:
                ties += 1
    return round((wins + 0.5 * ties) / (len(scores_pos) * len(scores_neg)), 4)


def prf(tp, fp, tn, fn):
    """Compute sensitivity/specificity/precision/F1/balanced-accuracy from a 2x2 confusion count; returns a dict with rounded metrics (None where undefined) plus the raw counts, no side effects."""
    sens = tp / (tp + fn) if tp + fn else None  # recall on corrupted
    spec = tn / (tn + fp) if tn + fp else None  # on originals
    prec = tp / (tp + fp) if tp + fp else None
    f1 = (
        2 * prec * sens / (prec + sens)
        if prec and sens and (prec + sens)
        else None
    )
    bal = (sens + spec) / 2 if sens is not None and spec is not None else None
    r4 = lambda x: round(x, 4) if x is not None else None
    return {
        "sensitivity": r4(sens),
        "specificity": r4(spec),
        "precision": r4(prec),
        "f1": r4(f1),
        "balanced_accuracy": r4(bal),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def cmd_report(args):
    """`report` subcommand: joins --controls with --verdicts by index, computes overall/per-class detection metrics (prf + ROC-AUC), and writes negatives_report.json to --output."""
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    controls = []
    with open(Path(args.controls).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                controls.append(json.loads(line))
    verdicts = {}
    with open(Path(args.verdicts).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v = json.loads(line)
                verdicts[v["m4_index"]] = v

    # join by input order: m4_index == line index of controls.jsonl
    rows = []
    for i, c in enumerate(controls):
        v = verdicts.get(i)
        if v is None:
            continue
        ev = v["evidence"]["verdict"]
        if ev not in EV_SCORE_AS_DETECTOR:
            continue
        rows.append(
            {
                "label": c["label"],
                "cls": c["corruption_class"],
                "verdict": ev,
                "score": EV_SCORE_AS_DETECTOR[ev],
            }
        )

    # decision rule: corrupted detected iff NOT_SUPPORTED
    def counts(subset):
        tp = sum(
            1
            for r in subset
            if r["label"] == "corrupted" and r["verdict"] == "NOT_SUPPORTED"
        )
        fn = sum(
            1
            for r in subset
            if r["label"] == "corrupted" and r["verdict"] != "NOT_SUPPORTED"
        )
        fp = sum(
            1
            for r in subset
            if r["label"] == "original" and r["verdict"] == "NOT_SUPPORTED"
        )
        tn = sum(
            1
            for r in subset
            if r["label"] == "original" and r["verdict"] != "NOT_SUPPORTED"
        )
        return tp, fp, tn, fn

    report = {
        "n_scored": len(rows),
        "decision_rule": "corrupted detected iff evidence verdict "
        "== NOT_SUPPORTED",
        "overall": prf(*counts(rows)),
    }

    pos = [r["score"] for r in rows if r["label"] == "corrupted"]
    neg = [r["score"] for r in rows if r["label"] == "original"]
    report["roc_auc_graded_score"] = auc_mann_whitney(pos, neg)

    per_class = {}
    originals = [r for r in rows if r["label"] == "original"]
    for cls in sorted({r["cls"] for r in rows if r["cls"] != "none"}):
        subset = [r for r in rows if r["cls"] == cls] + originals
        cls_pos = [r["score"] for r in rows if r["cls"] == cls]
        m = prf(*counts(subset))
        m["roc_auc"] = auc_mann_whitney(cls_pos, neg)
        m["verdict_distribution"] = dict(
            Counter(r["verdict"] for r in rows if r["cls"] == cls)
        )
        per_class[cls] = m
    report["per_corruption_class"] = per_class
    report["original_verdict_distribution"] = dict(
        Counter(r["verdict"] for r in originals)
    )

    out_path = out_dir / "negatives_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")


def main():
    """CLI entry point: dispatches to the `generate` or `report` subcommand based on args.cmd."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--kg", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--seed", type=int, default=13)
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("report")
    r.add_argument("--controls", required=True)
    r.add_argument("--verdicts", required=True)
    r.add_argument("--output", required=True)
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
