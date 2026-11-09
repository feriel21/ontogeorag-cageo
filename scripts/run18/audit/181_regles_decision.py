#!/usr/bin/env python3
"""181 — Decision rules of the verifying-LLMs, measured on the self-audit records.

WHY
    The self-audit (stress18.sh) gives two verdicts per verifying-LLM for each
    record: with the evidence sentence (p) and blind (v).
    A RULE turns these verdicts into a decision.
    Three rules are measured on the same records:
      admission    rule of the pipeline: the relation is set aside only if
                   p is not SUPPORTED AND v is not PLAUSIBLE
      phrase       the sentence comes first: set aside if p = NOT_SUPPORTED, or if
                   p = PARTIALLY_SUPPORTED and v is not PLAUSIBLE
      phrase_seule set aside if p = NOT_SUPPORTED (verdict with the sentence only)
    For each rule and each verifying-LLM, then for the panel (set aside if either
    verifier sets it aside): share of corrupted copies set aside, per corruption
    type, and share of the graph relations set aside.
    Same definitions as scripts/evaluation/verifier_decision_rules.py.
    Only phrase_seule is used in the manuscript: 191 recomputed the three rules
    from the joined verdicts and did not reproduce the counts of the other two.

USAGE
    python 181_regles_decision.py --stress output/run18/stress_test
    (the folder holds controls.jsonl, llama/m4_verdicts.jsonl, mistral/m4_verdicts.jsonl)
    Messages and output files are in French.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def p_of(v):
    e = v.get("evidence")
    return e.get("verdict") if isinstance(e, dict) else e


def v_of(v):
    b = v.get("blind")
    return b.get("verdict") if isinstance(b, dict) else b


RULES = {
    "admission":    lambda v: p_of(v) != "SUPPORTED" and v_of(v) != "PLAUSIBLE",
    "phrase":       lambda v: p_of(v) == "NOT_SUPPORTED"
                              or (p_of(v) == "PARTIALLY_SUPPORTED" and v_of(v) != "PLAUSIBLE"),
    "phrase_seule": lambda v: p_of(v) == "NOT_SUPPORTED",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", required=True)
    ap.add_argument("--judges", nargs="+", default=["llama", "mistral"])
    args = ap.parse_args()
    d = Path(args.stress)
    controls = jsonl(d / "controls.jsonl")
    V = {}
    for j in args.judges:
        V[j] = jsonl(d / j / "m4_verdicts.jsonl")
        if len(V[j]) != len(controls):
            sys.exit("ARRET : %s a %d verdicts pour %d controles" % (j, len(V[j]), len(controls)))
        for i, v in enumerate(V[j]):
            if "m4_index" in v and v["m4_index"] != i:
                sys.exit("ARRET : %s ligne %d mal alignee" % (j, i))

    cls = [c.get("corruption_class", "none") for c in controls]
    corrupted = [i for i, c in enumerate(controls) if c.get("label") == "corrupted"]
    originals = [i for i, c in enumerate(controls) if c.get("label") != "corrupted"]
    classes = sorted({cls[i] for i in corrupted})
    n_cls = Counter(cls[i] for i in corrupted)

    def flags(rule, who):
        if who == "panel":
            return [any(RULES[rule](V[j][i]) for j in args.judges) for i in range(len(controls))]
        return [RULES[rule](V[who][i]) for i in range(len(controls))]

    out = {}
    L = ["=== 181 regles de decision sur %d controles (%d originaux, %d corrompus) ==="
         % (len(controls), len(originals), len(corrupted)),
         "par type de corruption : " + ", ".join("%s %d" % (k, n_cls[k]) for k in classes), ""]
    header = "%-13s %-8s %16s %18s  " % ("regle", "juge", "corrompues_ecart.", "originaux_mis_cote") + \
             "  ".join("%-14s" % k for k in classes)
    L.append(header)
    for rule in RULES:
        for who in args.judges + ["panel"]:
            f = flags(rule, who)
            tp = sum(f[i] for i in corrupted)
            fp = sum(f[i] for i in originals)
            per = {k: sum(f[i] for i in corrupted if cls[i] == k) for k in classes}
            out["%s|%s" % (rule, who)] = {"corrupted_flagged": tp, "n_corrupted": len(corrupted),
                                          "originals_flagged": fp, "n_originals": len(originals),
                                          "per_class": per}
            L.append("%-13s %-8s %8d/%-3d %3.0f%% %8d/%-3d %3.0f%%  " % (
                rule, who, tp, len(corrupted), 100 * tp / len(corrupted),
                fp, len(originals), 100 * fp / len(originals)) +
                "  ".join("%-14s" % ("%d/%d" % (per[k], n_cls[k])) for k in classes))
        L.append("")
    L.append("admission = regle du pipeline ; phrase = la phrase prime ; phrase_seule = verdict avec la phrase seul")
    L.append("panel = ecartee si l'un des deux juges l'ecarte (le plus prudent, comme dans le pipeline)")
    txt = "\n".join(L)
    (d / "regles_decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (d / "regles_decision.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()