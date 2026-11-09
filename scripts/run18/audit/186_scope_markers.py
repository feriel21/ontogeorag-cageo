#!/usr/bin/env python3
"""186 — Scope markers in the evidence sentences of the delivered graph (automatic indicator).

READ-ONLY. CPU. No model. Writes only into --outdir (refuses to overwrite).

QUESTION
    A triple (subject, relation, object) cannot carry the qualifications of its
    sentence: modality (may, potential...), partial quantifier (some, locally...),
    condition (where, under, down to...), numerical value, particular case
    (figure, study area), part of an object (top surface, distal...).
    The script counts the lexical markers of these qualifications in the
    evidence sentence of each delivered relation, then crosses the count with
    the tier and the verdicts of the panel.

WHAT THE COUNT MEANS
    The count is a LEXICAL INDICATOR, not a judgement.
    A marker does not prove that the triple lost a qualification ("where" can be
    a plain relative), and a qualification without a listed keyword is not counted.
    The lists below were fixed before reading the results, and each relation
    reports the words found, for review.

CATEGORIES (whole words, lowercase sentence, ligatures normalised with NFKC)
    modality      may, might, could, can, possibly, possible, potential(ly), probably,
                  probable, likely, unlikely, perhaps, presumably, suggest*, appear(s/ed) to,
                  seem*, inferred, interpreted, thought to, hypothes*, uncertain
    partial       some, several, many, most, often, commonly, typically, usually, generally,
                  locally, occasionally, sometimes, partly, partially, few, in places
    condition     if, when, where, unless, provided, depending, only, under, during,
                  down to, deeper, shallower, greater than, less than
    value         number followed by a unit (m, km, ms, %, degree) or "magnitude"
    case          fig., figure, this study, this area, study area, here, survey,
                  "the <name> slide/basin/margin/area/mass transport deposit"
    part          top surface, upper surface, basal, base of, distal, proximal, fill,
                  flank, toe, headscarp, headwall, upper part, lower part, internal, edge
                  (counted only if the word is not already in the subject of the triple)
    negation      not, no, never, without, absence, absent, lack (counted separately)
    "qualified sentence" = at least one category among modality, partial, condition,
    value, case, part.

OTHER SIGNALS READ FROM THE RECORD (no model computation)
    truncated : the sentence is 400 characters long, the cut length of 06
    verifiers : m4.evidence_verdict (Llama|Mistral), m4.decision, tier

USAGE (root of the run18 worktree)
    python 186_scope_markers.py --graph output/run18/kg_final/flat_run18_enforced_dedup.json \
        --quarantine output/run18/m4_panel/m4_quarantine.jsonl --outdir output/audit/186_scope
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CUT = 400  # cut length of the evidence in 06_tiered_fusion.get_evidence

PATTERNS = {
    "modality": [r"may", r"might", r"could", r"can", r"possibly", r"possible", r"potential(?:ly)?",
                 r"probabl[ey]", r"likely", r"unlikely", r"perhaps", r"presumably", r"suggest\w*",
                 r"appear(?:s|ed)? to", r"seem\w*", r"inferred", r"interpreted", r"thought to",
                 r"hypothes\w*", r"uncertain"],
    "partial": [r"some", r"several", r"many", r"most", r"often", r"commonly", r"typically", r"usually",
                r"generally", r"locally", r"occasionally", r"sometimes", r"partly", r"partially", r"few",
                r"in places"],
    "condition": [r"if", r"when", r"where", r"unless", r"provided", r"depending", r"only", r"under",
                  r"during", r"down to", r"deeper", r"shallower", r"greater than", r"less than"],
    "value": [r"\d+(?:[.,]\d+)?\s?(?:m|km|ms|%|°|degrees?)", r"magnitude"],
    "case": [r"fig\.", r"figure", r"this study", r"this area", r"study area", r"here", r"survey",
            r"the \w+ (?:slide|basin|margin|area|mass transport deposit|mass transport complex)"],
    "part": [r"top surface", r"upper surface", r"basal", r"base of", r"distal", r"proximal", r"fill",
               r"flank", r"toe", r"headscarp", r"headwall", r"upper part", r"lower part", r"internal",
               r"edge"],
    "negation": [r"not", r"no", r"never", r"without", r"absence", r"absent", r"lack"],
}
QUALIFYING = ["modality", "partial", "condition", "value", "case", "part"]
COMPILED = {c: [(p, re.compile(r"(?<![a-z0-9])" + p + r"(?![a-z0-9])")) for p in ps] for c, ps in PATTERNS.items()}


def die(msg):
    print("STOP: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(x):
    x = unicodedata.normalize("NFKC", str(x or "")).lower()
    return " ".join(x.replace('"', " ").split())


def markers(ev, subject):
    e, s = norm(ev), norm(subject)
    found = {}
    for c, pats in COMPILED.items():
        hits = []
        for p, rx in pats:
            for m in rx.finditer(e):
                w = m.group(0)
                if c == "part" and w in s:
                    continue
                hits.append(w)
        if hits:
            found[c] = sorted(set(hits))
    return found


def load_graph(p):
    d = json.load(open(p, encoding="utf-8"))
    if isinstance(d, list):
        return d
    for k in ("triples", "relations", "edges"):
        if isinstance(d.get(k), list):
            return d[k]
    die("no list of relations found in " + str(p))


def judges(t):
    m4 = t.get("m4") or {}
    ev = str(m4.get("evidence_verdict") or "")
    parts = ev.split("|") if ev else []
    if not parts:
        return "no verdict"
    if all(x == "SUPPORTED" for x in parts):
        return "all SUPPORTED"
    if any(x == "SUPPORTED" for x in parts):
        return "mixed"
    if all(x == "PARTIALLY_SUPPORTED" for x in parts):
        return "all PARTIALLY"
    return "at least one NOT/other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--quarantine", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    if out.exists():
        die("the folder already exists, nothing is overwritten: " + str(out))
    G = load_graph(args.graph)
    sets = [("delivered", G)]
    if args.quarantine and Path(args.quarantine).exists():
        sets.append(("quarantine", [json.loads(l) for l in open(args.quarantine, encoding="utf-8") if l.strip()]))

    rows = []
    for name, recs in sets:
        for t in recs:
            ev = t.get("evidence") or ""
            ev = ev if isinstance(ev, str) else json.dumps(ev, ensure_ascii=False)
            mk = markers(ev, t.get("subject", ""))
            rows.append({
                "set": name, "subject": t.get("subject"), "relation": t.get("relation"), "object": t.get("object"),
                "tier": t.get("tier"), "panel_decision": (t.get("m4") or {}).get("decision", ""),
                "verifier_verdicts": (t.get("m4") or {}).get("evidence_verdict", ""), "verifiers": judges(t),
                "evidence_length": len(ev), "truncated": "yes" if len(ev) >= CUT else "",
                "qualified_sentence": "yes" if any(c in mk for c in QUALIFYING) else "",
                **{c: ";".join(mk.get(c, [])) for c in PATTERNS},
                "evidence": ev,
            })

    out.mkdir(parents=True)
    cols = list(rows[0])
    with open(out / "markers_per_relation.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    def pct(a, b):
        return "%d/%d (%.0f %%)" % (a, b, 100.0 * a / b) if b else "0/0"

    L = ["=== 186 scope markers in the evidence sentences (lexical indicator) ===",
         "graph: %s (sha %s)" % (args.graph, sha(args.graph)[:12])]
    if len(sets) > 1:
        L.append("quarantine: %s (sha %s)" % (args.quarantine, sha(args.quarantine)[:12]))
    for name, _ in sets:
        R = [r for r in rows if r["set"] == name]
        n = len(R)
        L += ["", "-- %s: %d relations --" % (name, n),
              "qualified sentence (>= 1 category): " + pct(sum(r["qualified_sentence"] == "yes" for r in R), n)]
        for c in PATTERNS:
            L.append("  %-10s: %s" % (c, pct(sum(bool(r[c]) for r in R), n)))
        L.append("  sentence cut at %d characters: %s" % (CUT, pct(sum(r["truncated"] == "yes" for r in R), n)))
        top = Counter(w for r in R for c in QUALIFYING for w in r[c].split(";") if w)
        L.append("  most frequent words: " + ", ".join("%s %d" % kv for kv in top.most_common(15)))
        if name == "delivered":
            L.append("")
            L.append("  per tier:")
            for k in sorted({r["tier"] for r in R}, key=str):
                S = [r for r in R if r["tier"] == k]
                L.append("    tier %s: qualified %s" % (k, pct(sum(r["qualified_sentence"] == "yes" for r in S), len(S))))
            L.append("  per relation:")
            for k, _ in Counter(r["relation"] for r in R).most_common():
                S = [r for r in R if r["relation"] == k]
                L.append("    %-14s: qualified %s" % (k, pct(sum(r["qualified_sentence"] == "yes" for r in S), len(S))))
            L.append("  verdicts of the verifiers on the sentence (Llama|Mistral) by qualification:")
            tab = defaultdict(Counter)
            for r in R:
                tab["qualified" if r["qualified_sentence"] else "no marker"][r["verifiers"]] += 1
            cats = ["all SUPPORTED", "mixed", "all PARTIALLY", "at least one NOT/other", "no verdict"]
            L.append("    %-14s | %s" % ("", " | ".join(cats)))
            for k in ("qualified", "no marker"):
                tot = sum(tab[k].values())
                L.append("    %-14s | %s" % (k, " | ".join(pct(tab[k][c], tot) for c in cats)))
    L += ["", "Reminder: lexical indicator (possible over-detection, e.g. relative 'where'; under-detection of "
          "qualifications without a keyword). Read the marker columns before quoting a number."]
    txt = "\n".join(L)
    (out / "_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt + "\n[done] " + str(out), flush=True)


if __name__ == "__main__":
    main()