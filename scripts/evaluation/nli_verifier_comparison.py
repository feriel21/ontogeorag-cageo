#!/usr/bin/env python3
"""Task-specific entailment model on the adversarial controls (Sections 2.3.3 and 3.8).

Scores the 464 adversarial controls of the delivered run with an off-the-shelf natural
language inference model (default roberta-large-mnli) and reports its ROC-AUC, to be
compared with the ROC-AUC of the LLM verifier on the same controls (0.794,
negatives_report.json). The published value is 0.813.

Inputs (--dir, default the run16E m4/negatives directory)
    controls.jsonl     the 464 controls; the text scored is the field
                       supporting_passage, which holds the evidence sentence of the
                       relation (at most 400 characters), not a retrieved passage
    m4_verdicts.jsonl  verifier verdicts on the same controls

Method
    1. Verbalise each relation into a hypothesis sentence (one template per relation type).
    2. Split the evidence text into sentences (it is usually a single sentence).
    3. Score entailment for every (sentence, hypothesis) pair; keep the maximum per
       control, and the maximum contradiction.
    4. ROC-AUC with average ranks for ties, entailment low = corrupted.
    5. The same controls are also scored under the verifier decision rules and under
       NLI thresholds, for comparison. Rule names describe acceptance:
       NLI_OR_EVIDENCE accepts when either accepts, NLI_AND_EVIDENCE only when both do.

Join. Verdict rows are matched to controls by m4_index; the content key
(subject, relation, object) of both is compared and mismatches are reported.

Scope. Entailment measures whether the text states the claim, not whether the claim
is geologically true. The model revision is not fixed by this script: record the
revision of the local snapshot when reporting.

The cluster runs offline (HF_HUB_OFFLINE=1): the model must be in the local cache.

Historical name: 95_nli_verifier_substitution.py

Usage
    python scripts/evaluation/nli_verifier_comparison.py --dir /path/to/run16E/m4/negatives
    python scripts/evaluation/nli_verifier_comparison.py --limit 40
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DIR = str(
    Path.home() / "ontogeorag_D" / "output" / "run16E" / "m4" / "negatives"
)
DEFAULT_MODEL = "roberta-large-mnli"

CLEAN_LABELS = {"clean", "original", "positive", "true", "uncorrupted", "none"}

# Verbalization templates. The hypothesis must read as a natural claim, not
# as a triple: NLI models are trained on prose, and a bare "s r o" string
# scores unpredictably.
TEMPLATES = {
    "hasDescriptor": "The {s} is described as {o}.",
    "causes": "{S} causes {o}.",
    "triggers": "{S} triggers {o}.",
    "controls": "{S} controls {o}.",
    "affects": "{S} affects {o}.",
    "formedBy": "The {s} is formed by {o}.",
    "occursIn": "The {s} occurs in the {o}.",
    "partOf": "The {s} is part of the {o}.",
    "overlies": "The {s} overlies the {o}.",
    "underlies": "The {s} underlies the {o}.",
}
FALLBACK_TEMPLATE = "{S} is related to {o} by {r}."


def verbalize(subject: str, relation: str, obj: str) -> str:
    s = subject.strip()
    o = obj.strip()
    tpl = TEMPLATES.get(relation, FALLBACK_TEMPLATE)
    return tpl.format(s=s, S=s[:1].upper() + s[1:], o=o, r=relation)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str, max_sentences: int = 40) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT.split(text) if len(p.strip()) > 15]
    if not parts:
        parts = [text]
    return parts[:max_sentences]


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def infer_corruption(rec: dict) -> str:
    label = rec.get("label")
    if label is not None:
        if str(label).strip().lower() in CLEAN_LABELS:
            return "CLEAN"
        cls = rec.get("corruption_class")
        return str(cls).strip().upper() if cls else "CORRUPT"
    cls = rec.get("corruption_class")
    if cls:
        return str(cls).strip().upper()
    return "UNKNOWN"


def norm_key(rec: dict) -> tuple:
    return (
        str(rec.get("subject", "")).strip().lower(),
        str(rec.get("relation", "")).strip(),
        str(rec.get("object", "")).strip().lower(),
    )


def split_pair(field) -> list[str]:
    if field is None:
        return []
    return [p.strip().upper() for p in str(field).split("|") if p.strip()]


def worst_evidence(v: list[str]) -> str:
    for level in ("NO_PASSAGE", "NOT_SUPPORTED", "PARTIALLY_SUPPORTED", "SUPPORTED"):
        if level in v:
            return level
    return "UNKNOWN"


def worst_blind(v: list[str]) -> str:
    for level in ("IMPLAUSIBLE", "UNCERTAIN", "PLAUSIBLE"):
        if level in v:
            return level
    return "UNKNOWN"


def score_rule(rows, predicate):
    tp = fp = tn = fn = 0
    for r in rows:
        rejected = predicate(r)
        if r["corrupt"]:
            tp += rejected
            fn += not rejected
        else:
            fp += rejected
            tn += not rejected
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    bal = (sens + spec) / 2
    return {"sens": sens, "spec": spec, "prec": prec, "bal": bal,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def roc_auc(rows, key, higher_means_corrupt=True):
    """Rank-based AUC, ties handled by average rank."""
    scored = [(r[key], r["corrupt"]) for r in rows if r[key] == r[key]]
    if not scored:
        return float("nan")
    scored.sort(key=lambda x: x[0])
    ranks = {}
    i = 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = [ranks[k] for k, (_, c) in enumerate(scored) if c]
    n_pos = len(pos)
    n_neg = len(scored) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    auc = (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # `auc` is computed with scores sorted ascending and positives = corrupt.
    # If a HIGH score means corrupt, that ranking is already correct.
    # If a LOW score means corrupt (the entailment case), it must be flipped.
    return auc if higher_means_corrupt else 1 - auc


def fmt(v):
    return "  n/a" if v != v else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="nli_substitution.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="Only process N controls.")
    ap.add_argument("--threshold", type=float, default=0.5,
                     help="Entailment probability below which the triple is rejected.")
    args = ap.parse_args()

    d = Path(args.dir)
    f_controls, f_verdicts = d / "controls.jsonl", d / "m4_verdicts.jsonl"
    for f in (f_controls, f_verdicts):
        if not f.exists():
            print(f"ERROR: not found: {f}", file=sys.stderr)
            return 1

    # ---- resolve the model locally, never hit the network -----------------
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        print(f"ERROR: torch/transformers unavailable: {e}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(args.model, local_files_only=True)
    except Exception:
        model_path = args.model  # may still resolve from a local dir/cache

    print(f"Loading NLI model: {args.model}")
    try:
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True)
    except Exception as e:
        print(f"\nERROR: could not load '{args.model}' from the local cache.",
              file=sys.stderr)
        print(f"  {e}\n", file=sys.stderr)
        print("  The cluster runs offline. Fetch the model once from a machine",
              file=sys.stderr)
        print("  with network access, then copy the cache over:", file=sys.stderr)
        print("    python -c \"from huggingface_hub import snapshot_download; "
              "snapshot_download('" + args.model + "')\"", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"  device: {device}")

    # Identify which output index is entailment / contradiction.
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    ent_idx = next((i for i, l in id2label.items() if "entail" in l), None)
    con_idx = next((i for i, l in id2label.items() if "contra" in l), None)
    if ent_idx is None:
        print(f"ERROR: no entailment label in {id2label}", file=sys.stderr)
        return 1
    print(f"  labels: {id2label}")

    # ---- join controls and verdicts by m4_index; content keys are compared ----
    controls = read_jsonl(f_controls)
    verdicts = read_jsonl(f_verdicts)
    rows = []
    mismatch = 0
    for v in verdicts:
        idx = v.get("m4_index")
        if not (isinstance(idx, int) and 0 <= idx < len(controls)):
            continue
        c = controls[idx]
        if norm_key(c) != norm_key(v):
            mismatch += 1
        corruption = infer_corruption(c)
        ev_block, bl_block = v.get("evidence"), v.get("blind")
        ev = worst_evidence(split_pair(
            ev_block.get("verdict") if isinstance(ev_block, dict) else ev_block))
        bl = worst_blind(split_pair(
            bl_block.get("verdict") if isinstance(bl_block, dict) else bl_block))
        rows.append({
            "subject": c.get("subject", ""), "relation": c.get("relation", ""),
            "object": c.get("object", ""),
            "passage": c.get("supporting_passage", ""),
            "corruption": corruption,
            "corrupt": corruption not in ("CLEAN", "UNKNOWN"),
            "evidence": ev, "blind": bl,
        })

    if mismatch:
        print(f"  WARNING: {mismatch} controls whose content key differs from the verdict at the same index.")
    if args.limit:
        rows = rows[:args.limit]

    print(f"  {len(rows)} controls "
          f"({sum(r['corrupt'] for r in rows)} corrupted, "
          f"{sum(not r['corrupt'] for r in rows)} clean)\n")

    # ---- run NLI ----------------------------------------------------------
    pairs, owner = [], []
    for i, r in enumerate(rows):
        hyp = verbalize(r["subject"], r["relation"], r["object"])
        r["hypothesis"] = hyp
        sents = split_sentences(r["passage"])
        r["n_sentences"] = len(sents)
        for s in sents:
            pairs.append((s, hyp))
            owner.append(i)

    print(f"Scoring {len(pairs)} (sentence, hypothesis) pairs...")
    ent_max = [0.0] * len(rows)
    con_max = [0.0] * len(rows)

    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            chunk = pairs[start:start + args.batch_size]
            enc = tok([p for p, _ in chunk], [h for _, h in chunk],
                      return_tensors="pt", truncation=True,
                      max_length=256, padding=True).to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu()
            for k in range(len(chunk)):
                i = owner[start + k]
                ent_max[i] = max(ent_max[i], float(probs[k, ent_idx]))
                if con_idx is not None:
                    con_max[i] = max(con_max[i], float(probs[k, con_idx]))
            if start % (args.batch_size * 20) == 0:
                print(f"  {start}/{len(pairs)}")

    for i, r in enumerate(rows):
        r["nli_entailment"] = ent_max[i]
        r["nli_contradiction"] = con_max[i]

    # ---- compare rules ----------------------------------------------------
    thr = args.threshold
    rules = {
        "EVIDENCE_ONLY": lambda r: r["evidence"] != "SUPPORTED",
        "CURRENT_MATRIX": lambda r: (
            False if r["evidence"] == "SUPPORTED" else r["blind"] != "PLAUSIBLE"),
        "EVIDENCE_DOMINANT": lambda r: (
            True if r["evidence"] in ("NOT_SUPPORTED", "NO_PASSAGE")
            else (r["blind"] == "IMPLAUSIBLE"
                  if r["evidence"] == "PARTIALLY_SUPPORTED" else False)),
        "BLIND_ONLY": lambda r: r["blind"] != "PLAUSIBLE",
        "NLI_ONLY": lambda r: r["nli_entailment"] < thr,
        "NLI_OR_EVIDENCE": lambda r: (
            r["nli_entailment"] < thr and r["evidence"] != "SUPPORTED"),
        "NLI_AND_EVIDENCE": lambda r: (
            r["nli_entailment"] < thr or r["evidence"] != "SUPPORTED"),
    }

    print()
    print("=" * 78)
    print(f"DECISION RULES ON THE SAME CONTROLS (NLI threshold = {thr})")
    print("=" * 78)
    print(f"  {'rule':<20}{'sens':>8}{'spec':>8}{'prec':>8}{'bal.acc':>10}"
          f"{'TP':>6}{'FP':>6}{'TN':>6}{'FN':>6}")
    results = {}
    for name, pred in rules.items():
        s = score_rule(rows, pred)
        results[name] = s
        print(f"  {name:<20}{fmt(s['sens']):>8}{fmt(s['spec']):>8}"
              f"{fmt(s['prec']):>8}{fmt(s['bal']):>10}"
              f"{s['tp']:>6}{s['fp']:>6}{s['tn']:>6}{s['fn']:>6}")

    auc = roc_auc(rows, "nli_entailment", higher_means_corrupt=False)
    print(f"\n  NLI entailment ROC-AUC (threshold-free): {fmt(auc)}")
    print("  ROC-AUC does not depend on the threshold. The LLM verifier reaches 0.794")
    print("  on the same controls (negatives_report.json).")

    print("\n  THRESHOLD SWEEP (NLI_ONLY)")
    print(f"    {'thr':>6}{'sens':>9}{'spec':>9}{'bal.acc':>10}")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        s = score_rule(rows, lambda r, t=t: r["nli_entailment"] < t)
        print(f"    {t:>6.1f}{fmt(s['sens']):>9}{fmt(s['spec']):>9}{fmt(s['bal']):>10}")

    # ---- per corruption class --------------------------------------------
    print()
    print("=" * 78)
    print("MEAN NLI ENTAILMENT BY CORRUPTION CLASS")
    print("=" * 78)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["corruption"]].append(r)
    print(f"  {'class':<22}{'n':>5}{'mean ent':>11}{'mean contra':>13}"
          f"{'detected':>11}")
    for cls, subset in sorted(by_class.items()):
        me = sum(r["nli_entailment"] for r in subset) / len(subset)
        mc = sum(r["nli_contradiction"] for r in subset) / len(subset)
        det = sum(1 for r in subset if r["nli_entailment"] < thr) / len(subset)
        print(f"  {cls:<22}{len(subset):>5}{me:>11.3f}{mc:>13.3f}{det:>11.3f}")
    print("\n  A large gap in mean entailment between CLEAN and a corrupted class")
    print("  means the model separates that class from the originals.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        cols = ["subject", "relation", "object", "corruption", "corrupt",
                "hypothesis", "n_sentences", "nli_entailment",
                "nli_contradiction", "evidence", "blind"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    print(f"\nWritten: {args.out}")
    print("Entailment measures whether the text states the claim.")
    print("It does not measure whether the claim is geologically true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())