#!/usr/bin/env python3
"""
05a_rule_canonicalize.py — Deterministic rule-merge BEFORE SciBERT.
===================================================================
WHY (R4)
    SciBERT@0.06 misses trivial surface variants (plural, hyphenation,
    spacing): run11 ships both `slope failure` and `slope failures` in its
    top-15 nodes, and `debris flow deposit` / `debris-flow deposit` coexist.
    Rules-first is free, risk-free, and provides the first row of the
    canonicalization ablation ("N merges recovered by deterministic rules
    that tau=0.06 missed").

WHAT
    Standalone pass over a triples .jsonl: entities identical under a
    deterministic normal form (lowercase, hyphen/underscore -> space,
    punctuation stripped, conservative de-pluralization sparing -ss/-is/-us,
    documented abbreviation expansion MTD/MTC) are merged onto a single
    representative surface form — the most frequent original form in the
    file (ties: shortest, then alphabetical). NOTHING beyond rule-identical
    groups is merged; SciBERT (05) then runs unchanged on the output.

OUTPUTS
    <input>.rulecanon.jsonl   — triples with merged surface forms
    <input>.rulecanon_log.csv — every applied substitution (audit trail /
                                ablation table row material)

USAGE (per pass, before 05_canonicalize.py)
    python 05a_rule_canonicalize.py --input output/run13_a/clean_triples.jsonl
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ABBREV = {
    "mtd": "mass transport deposit",
    "mtds": "mass transport deposit",
    "mtc": "mass transport complex",
}


def norm_form(e: str) -> str:
    x = str(e).lower().strip()
    x = x.replace("-", " ").replace("_", " ")
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    if x in ABBREV:
        x = ABBREV[x]
    toks = []
    for tok in x.split():
        if (
            len(tok) > 3
            and tok.endswith("s")
            and not tok.endswith(("ss", "is", "us"))
        ):
            tok = tok[:-1]
        toks.append(tok)
    return " ".join(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    inp = Path(args.input)
    triples = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]

    # collect surface forms per normal form
    surface_count = Counter()
    for t in triples:
        for k in ("subject", "object"):
            if t.get(k):
                surface_count[str(t[k]).strip()] += 1
    groups = defaultdict(list)
    for s in surface_count:
        groups[norm_form(s)].append(s)

    # representative per group
    rep = {}
    for nf, forms in groups.items():
        forms.sort(key=lambda s: (-surface_count[s], len(s), s))
        for s in forms:
            rep[s] = forms[0]

    # apply + log
    subs = Counter()
    for t in triples:
        for k in ("subject", "object"):
            if t.get(k):
                s = str(t[k]).strip()
                r = rep[s]
                if r != s:
                    subs[(s, r)] += 1
                    t[k] = r

    out = inp.with_suffix(".rulecanon.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    log = inp.with_suffix(".rulecanon_log.csv")
    with open(log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from", "to", "n_occurrences"])
        for (a, b), c in sorted(subs.items(), key=lambda x: -x[1]):
            w.writerow([a, b, c])

    merged_groups = sum(1 for forms in groups.values() if len(forms) > 1)
    print("=" * 60)
    print("RULE CANONICALIZATION (pre-SciBERT)")
    print("=" * 60)
    print(f"triples            : {len(triples)}")
    print(f"unique surfaces    : {len(surface_count)}")
    print(f"variant groups     : {merged_groups}")
    print(
        f"substitutions made : {sum(subs.values())} "
        f"({len(subs)} distinct mappings)"
    )
    for (a, b), c in sorted(subs.items(), key=lambda x: -x[1])[:15]:
        print(f"   {a!r} -> {b!r}  (x{c})")
    print(f"written: {out}")
    print(f"log    : {log}")


if __name__ == "__main__":
    main()
