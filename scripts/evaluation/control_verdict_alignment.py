#!/usr/bin/env python3
"""Check that verifier verdicts are aligned with the adversarial controls before any metric is computed.

The verdict file and the control file of the delivered run have one row per control.
This script tests, rather than assumes, that verdict row i belongs to control row i:
for every control it compares the evidence text of the control with the quote of the
verdict at the same index, and with the best match among about 60 other verdicts.

It also prints the label space of the verdicts (evidence verdict, blind verdict,
model, tier), which the decision rules of the stratified metrics depend on.

Input   output/run16E/m4/negatives/{controls.jsonl, m4_verdicts.jsonl}, relative to the
        working directory (run it from the directory that holds output/run16E).
Output  output/run16E/m4/negatives/join_verify.csv, one row per control.
Computes no metric.

Historical name: 106c_join_verify.py

Usage
    cd /path/to/run_root && python /path/to/repo/scripts/evaluation/control_verdict_alignment.py
"""
import json, csv, random
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

NEG = Path("output/run16E/m4/negatives")
controls = [json.loads(l) for l in open(NEG / "controls.jsonl")]
verdicts = [json.loads(l) for l in open(NEG / "m4_verdicts.jsonl")]

assert len(controls) == len(verdicts), f"{len(controls)} vs {len(verdicts)} — not 1:1, stop here"
N = len(controls)

def norm(s):
    return " ".join(str(s).lower().split())

def quote_of(v):
    ev = v.get("evidence")
    if isinstance(ev, dict):
        return ev.get("quote", "") or ""
    return ""

def passage_of(c):
    p = c.get("supporting_passage", "")
    return p if isinstance(p, str) else str(p)

def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# --- label space: we must know the vocabulary before writing any rule logic ---
ev_verdicts = Counter()
blind_verdicts = Counter()
qwen_verdicts = Counter()
blind_shape = None
for v in verdicts:
    ev = v.get("evidence")
    if isinstance(ev, dict):
        ev_verdicts[ev.get("verdict")] += 1
    bl = v.get("blind")
    if blind_shape is None:
        blind_shape = (type(bl).__name__, list(bl.keys()) if isinstance(bl, dict) else bl)
    if isinstance(bl, dict):
        blind_verdicts[bl.get("verdict")] += 1
    else:
        blind_verdicts[bl] += 1
    qwen_verdicts[v.get("qwen_verdict")] += 1

print("=== label space ===")
print("blind field shape:", blind_shape)
print("evidence.verdict :", dict(ev_verdicts))
print("blind verdict    :", dict(blind_verdicts))
print("qwen_verdict     :", dict(qwen_verdicts))
print("model            :", dict(Counter(v.get("model") for v in verdicts)))
print("tier             :", dict(Counter(v.get("tier") for v in verdicts)))

# --- does m4_index equal the row index? ---
idx_mismatch = [i for i, v in enumerate(verdicts) if v.get("m4_index") != i]
print(f"\nm4_index != row index on {len(idx_mismatch)} / {N} verdict rows"
      + (f" (first few: {idx_mismatch[:10]})" if idx_mismatch else ""))

# --- alignment test: aligned similarity vs best-elsewhere similarity ---
print("\n=== alignment test (this may take a minute) ===")
rng = random.Random(7)
rows = []
aligned_wins = 0
aligned_scores = []
random_scores = []

for i in range(N):
    p = passage_of(controls[i])
    aligned = sim(p, quote_of(verdicts[i]))
    aligned_scores.append(aligned)
    # best score against a random sample of OTHER verdicts (full N^2 is too slow)
    others = [j for j in rng.sample(range(N), min(60, N)) if j != i]
    best_other, best_j = 0.0, None
    for j in others:
        s = sim(p, quote_of(verdicts[j]))
        if s > best_other:
            best_other, best_j = s, j
    random_scores.append(best_other)
    if aligned >= best_other:
        aligned_wins += 1
    rows.append({
        "control_idx": i,
        "orig_index": controls[i].get("orig_index"),
        "corruption_class": controls[i].get("corruption_class"),
        "label": controls[i].get("label"),
        "aligned_sim": round(aligned, 4),
        "best_other_sim": round(best_other, 4),
        "best_other_idx": best_j,
        "aligned_wins": aligned >= best_other,
    })

mean = lambda xs: sum(xs) / len(xs)
print(f"aligned index beats/ties best of ~60 others: {aligned_wins} / {N} "
      f"({100*aligned_wins/N:.1f}%)")
print(f"mean similarity, aligned      : {mean(aligned_scores):.3f}")
print(f"mean similarity, best-of-other: {mean(random_scores):.3f}")

low = [r for r in rows if r["aligned_sim"] < 0.3]
print(f"\ncontrols whose aligned quote-similarity is below 0.30: {len(low)}")
print("(expected: passage_perm controls, where the stored passage was deliberately swapped)")
print("their class breakdown:", dict(Counter(r["corruption_class"] for r in low)))

losers = [r for r in rows if not r["aligned_wins"]]
print(f"\ncontrols where some other verdict matched better: {len(losers)}")
print("their class breakdown:", dict(Counter(r["corruption_class"] for r in losers)))
for r in losers[:15]:
    print(f"  ctrl {r['control_idx']} ({r['corruption_class']}): "
          f"aligned={r['aligned_sim']:.3f} vs idx {r['best_other_idx']}={r['best_other_sim']:.3f}")

with open(NEG / "join_verify.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {NEG / 'join_verify.csv'}")