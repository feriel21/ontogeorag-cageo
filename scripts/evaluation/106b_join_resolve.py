#!/usr/bin/env python3
"""
106b_join_resolve.py — finish what 106_join_diagnostic.py started.

1. Sanity-checks that 'evidence' in m4_verdicts.jsonl actually looks like passage
   text (not a label or score) by printing a few matched pairs side by side.
2. Prints the 10 still-ambiguous groups in full so they can be adjudicated by eye
   instead of guessed.
3. Writes output/run16E/m4/negatives/join_map.csv — one row per control that
   joined cleanly, with an explicit 'match_type' column (unique / disambiguated).
   Rows with no safe match are NOT written here; they stay out of any metric
   until resolved by hand.

Nothing in this script computes sensitivity/specificity/BA. That comes after,
reading join_map.csv.
"""
import json, csv
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

NEG = Path("output/run16E/m4/negatives")
controls = [json.loads(l) for l in open(NEG / "controls.jsonl")]
verdicts = [json.loads(l) for l in open(NEG / "m4_verdicts.jsonl")]

def norm(s):
    return " ".join(str(s).lower().split())

def key(r):
    return (norm(r["subject"]), norm(r["relation"]), norm(r["object"]))

ctrl_groups = defaultdict(list)
for i, r in enumerate(controls):
    ctrl_groups[key(r)].append(i)

verd_groups = defaultdict(list)
for i, r in enumerate(verdicts):
    verd_groups[key(r)].append(i)

def as_text(v):
    """evidence/supporting_passage may not be plain strings — coerce safely and
    report what they actually are so we stop guessing."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        # common shapes: {"passage": "..."} or {"text": "..."} or {"evidence_text": "..."}
        for k in ("passage", "text", "evidence_text", "supporting_passage", "content"):
            if k in v and isinstance(v[k], str):
                return v[k]
        return json.dumps(v)[:300]
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)

def similarity(a, b):
    return SequenceMatcher(None, norm(as_text(a)), norm(as_text(b))).ratio()

join_rows = []       # (control_idx, verdict_idx, match_type)
ambiguous_groups = []

for k, c_idxs in ctrl_groups.items():
    v_idxs = list(verd_groups.get(k, []))
    if not v_idxs:
        continue
    if len(c_idxs) == 1 and len(v_idxs) == 1:
        join_rows.append((c_idxs[0], v_idxs[0], "unique"))
        continue
    remaining_v = list(v_idxs)
    group_ambiguous = []
    for ci in c_idxs:
        if not remaining_v:
            continue
        c_passage = controls[ci].get("supporting_passage", "")
        scores = [(similarity(c_passage, verdicts[vi].get("evidence", "")), vi)
                  for vi in remaining_v]
        scores.sort(reverse=True)
        best_score, best_vi = scores[0]
        if len(scores) > 1 and (best_score - scores[1][0]) < 0.05:
            group_ambiguous.append((ci, scores))
            continue
        join_rows.append((ci, best_vi, "disambiguated"))
        remaining_v.remove(best_vi)
    if group_ambiguous:
        ambiguous_groups.append((k, group_ambiguous))

# --- sanity check: does 'evidence' actually look like passage text? ---
print("=== raw type check: what IS 'evidence' in m4_verdicts.jsonl? ===")
sample_ev = verdicts[0].get("evidence")
print(f"  type: {type(sample_ev)}")
print(f"  raw value (first verdict): {sample_ev!r}"[:500])

print("\n=== sanity check: 5 matched pairs, supporting_passage vs evidence ===")
for ci, vi, mtype in join_rows[:5]:
    print(f"\n[{mtype}] {controls[ci]['subject']} -- {controls[ci]['relation']} --> {controls[ci]['object']}")
    print("  supporting_passage:", as_text(controls[ci].get("supporting_passage", ""))[:160])
    print("  evidence          :", as_text(verdicts[vi].get("evidence", ""))[:160])

# --- dump the still-ambiguous groups in full for manual adjudication ---
print(f"\n=== {len(ambiguous_groups)} ambiguous groups needing manual adjudication ===")
for k, group in ambiguous_groups:
    subj, rel, obj = k
    print(f"\nkey: ({subj}, {rel}, {obj})")
    for ci, scores in group:
        print(f"  control idx={ci} orig_index={controls[ci]['orig_index']} "
              f"class={controls[ci]['corruption_class']} label={controls[ci]['label']}")
        print(f"    passage: {as_text(controls[ci].get('supporting_passage',''))[:140]}")
        for score, vi in scores[:3]:
            print(f"    candidate verdict idx={vi} m4_index={verdicts[vi]['m4_index']} "
                  f"score={score:.3f} evidence={as_text(verdicts[vi].get('evidence',''))[:140]}")

# --- write the clean join map ---
with open(NEG / "join_map.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["control_idx", "verdict_idx", "match_type",
                "orig_index", "corruption_class", "label"])
    for ci, vi, mtype in join_rows:
        w.writerow([ci, vi, mtype, controls[ci]["orig_index"],
                    controls[ci]["corruption_class"], controls[ci]["label"]])

print(f"\nWrote {len(join_rows)} / {len(controls)} rows to {NEG / 'join_map.csv'} "
      f"({100*len(join_rows)/len(controls):.1f}%)")
print(f"{sum(len(g[1]) for g in ambiguous_groups)} controls excluded pending manual adjudication — "
      f"see the dump above. Do not compute metrics until these are resolved or explicitly "
      f"excluded with a stated reason.")