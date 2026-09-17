#!/usr/bin/env python3
"""Verifier decision rules scored on the 464 adversarial controls (Section 3.7).

Scores four decision rules on the verifier verdicts of the adversarial controls of the
delivered run, overall and by corruption class:
    blind_pass_alone   rejects when the passage-blind verdict is not PLAUSIBLE
    evidence_alone     rejects when the evidence verdict is not SUPPORTED
    shipped_matrix     rejects only when both of the above reject: the admission rule
                       of the pipeline
    evidence_dominant  the evidence verdict decides; the blind verdict may only demote

Published results (shipped_matrix): 20 of the 232 corrupted records rejected
(sensitivity 0.086), specificity 0.974, balanced accuracy 0.530. The single-verdict
figures of the article (142 rejected, specificity 0.871, balanced accuracy 0.742, a
record being rejected when its evidence verdict is NOT_SUPPORTED) come from
negatives_report.json of m4/m4_negatives.py, not from this script.

All verdicts in the file come from one verifier, Llama-3.1-8B-Instruct; the script
says so in its output.

Join. Verdict row i is asserted to belong to control row i (m4_index equals the row
index for every row); control_verdict_alignment.py tests that alignment on content.
Confidence intervals are bootstrap percentiles (2,000 resamples overall, 1,000 per
class, fixed seeds). The Wilson intervals of Figure 6 are not computed here.
REFERENCE holds earlier values used as a self-check column only.

Input   output/run16E/m4/negatives/{controls.jsonl, m4_verdicts.jsonl}, relative to the
        working directory (run it from the directory that holds output/run16E).
Output  metrics_overall.csv, metrics_stratified.csv, evidence_alone_fp.csv in that
        directory.

Historical name: 106d_metrics_stratified.py

Usage
    cd /path/to/run_root && python /path/to/repo/scripts/evaluation/verifier_decision_rules.py
"""
import json, csv, random
from pathlib import Path
from collections import defaultdict

NEG = Path("output/run16E/m4/negatives")
controls = [json.loads(l) for l in open(NEG / "controls.jsonl")]
verdicts = [json.loads(l) for l in open(NEG / "m4_verdicts.jsonl")]

assert len(controls) == len(verdicts), "files are not 1:1"
for i, v in enumerate(verdicts):
    assert v.get("m4_index") == i, f"row {i}: m4_index={v.get('m4_index')} breaks alignment"

models = {v.get("model") for v in verdicts}
if len(models) == 1:
    print("NOTE: a single verifier model in this file:")
    print("     ", next(iter(models)).split("/")[-3] if "/" in next(iter(models)) else models)
    print("      These metrics describe one model, NOT the two-model panel.\n")

def ev_verdict(i):
    ev = verdicts[i].get("evidence")
    return ev.get("verdict") if isinstance(ev, dict) else None

def bl_verdict(i):
    bl = verdicts[i].get("blind")
    return bl.get("verdict") if isinstance(bl, dict) else bl

def is_corrupted(i):
    return controls[i].get("label") == "corrupted"

RULES = {
    # flagged == the rule says "this relation is a problem"
    "blind_pass_alone":  lambda i: bl_verdict(i) != "PLAUSIBLE",
    "evidence_alone":    lambda i: ev_verdict(i) != "SUPPORTED",
    "shipped_matrix":    lambda i: bl_verdict(i) != "PLAUSIBLE" and ev_verdict(i) != "SUPPORTED",
    "evidence_dominant": lambda i: ev_verdict(i) == "NOT_SUPPORTED"
                                   or (ev_verdict(i) == "PARTIALLY_SUPPORTED"
                                       and bl_verdict(i) != "PLAUSIBLE"),
}

REFERENCE = {  # documented in the run16E write-up; used as a self-check only
    "blind_pass_alone": (0.086, 0.953),
    "shipped_matrix":   (0.086, 0.974),
    "evidence_alone":   (0.974, 0.328),
}

def counts(idxs, rule):
    tp = fn = tn = fp = 0
    for i in idxs:
        flagged, corrupted = RULES[rule](i), is_corrupted(i)
        if corrupted and flagged: tp += 1
        elif corrupted: fn += 1
        elif flagged: fp += 1
        else: tn += 1
    return tp, fn, tn, fp

def metrics(idxs, rule):
    tp, fn, tn, fp = counts(idxs, rule)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ba = (sens + spec) / 2 if sens == sens and spec == spec else float("nan")
    return sens, spec, ba, tp, fn, tn, fp

ALL = list(range(len(controls)))

def bootstrap(rule, idxs, n_boot=2000, seed=13):
    rng = random.Random(seed)
    n = len(idxs)
    out = {"sens": [], "spec": [], "ba": []}
    for _ in range(n_boot):
        sample = [idxs[rng.randrange(n)] for _ in range(n)]
        s, sp, ba, *_ = metrics(sample, rule)
        if s == s: out["sens"].append(s)
        if sp == sp: out["spec"].append(sp)
        if ba == ba: out["ba"].append(ba)
    def ci(xs):
        xs = sorted(xs)
        return (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))]) if xs else (float("nan"),) * 2
    return {k: ci(v) for k, v in out.items()}

print("=== overall, 464 controls ===")
with open(NEG / "metrics_overall.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rule", "sensitivity", "sens_lo", "sens_hi", "specificity", "spec_lo",
                "spec_hi", "balanced_acc", "ba_lo", "ba_hi", "tp", "fn", "tn", "fp",
                "ref_sens", "ref_spec", "reproduces_reference"])
    for rule in RULES:
        sens, spec, ba, tp, fn, tn, fp = metrics(ALL, rule)
        ci = bootstrap(rule, ALL)
        ref = REFERENCE.get(rule, (None, None))
        ok = ""
        if ref[0] is not None:
            ok = "yes" if abs(sens - ref[0]) < 0.01 and abs(spec - ref[1]) < 0.01 else "NO"
        w.writerow([rule, sens, *ci["sens"], spec, *ci["spec"], ba, *ci["ba"],
                    tp, fn, tn, fp, ref[0], ref[1], ok])
        flag = f"  [reference {ref[0]}/{ref[1]} reproduced: {ok}]" if ok else ""
        print(f"{rule:18s} sens={sens:.3f} [{ci['sens'][0]:.3f},{ci['sens'][1]:.3f}]  "
              f"spec={spec:.3f} [{ci['spec'][0]:.3f},{ci['spec'][1]:.3f}]  "
              f"BA={ba:.3f} [{ci['ba'][0]:.3f},{ci['ba'][1]:.3f}]{flag}")

# --- stratified: sensitivity per corruption class (corrupted controls only) ---
by_class = defaultdict(list)
for i in ALL:
    if is_corrupted(i):
        by_class[controls[i]["corruption_class"]].append(i)

print("\n=== sensitivity by corruption class (corrupted controls only) ===")
with open(NEG / "metrics_stratified.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rule", "corruption_class", "n", "sensitivity", "sens_lo", "sens_hi", "caught", "missed"])
    for rule in RULES:
        for cls, idxs in sorted(by_class.items()):
            caught = sum(1 for i in idxs if RULES[rule](i))
            sens = caught / len(idxs)
            ci = bootstrap(rule, idxs, n_boot=1000)
            w.writerow([rule, cls, len(idxs), sens, *ci["sens"], caught, len(idxs) - caught])
            print(f"{rule:18s} {cls:16s} n={len(idxs):3d} sens={sens:.3f} "
                  f"[{ci['sens'][0]:.3f},{ci['sens'][1]:.3f}]  ({caught} caught, {len(idxs)-caught} missed)")

# --- FP audit: corrupted controls that evidence_alone wrongly kept ---
with open(NEG / "evidence_alone_fp.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["control_idx", "orig_index", "corruption_class", "subject", "relation",
                "object", "evidence_verdict", "blind_verdict", "quote"])
    n_fp = 0
    for i in ALL:
        if is_corrupted(i) and not RULES["evidence_alone"](i):
            ev = verdicts[i].get("evidence") or {}
            w.writerow([i, controls[i].get("orig_index"), controls[i]["corruption_class"],
                        controls[i]["subject"], controls[i]["relation"], controls[i]["object"],
                        ev_verdict(i), bl_verdict(i), (ev.get("quote") or "")[:300]])
            n_fp += 1
print(f"\n{n_fp} corrupted controls slipped past evidence_alone -> evidence_alone_fp.csv")
print(f"Wrote {NEG/'metrics_overall.csv'} and {NEG/'metrics_stratified.csv'}")