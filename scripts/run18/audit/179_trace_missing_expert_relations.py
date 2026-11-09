#!/usr/bin/env python3
"""179 — Where is each expert relation missing from the run18 graph lost?

WHY
    For each relation of the graph of Le Bouteiller et al. (26 relations, used
    for comparison only), the script follows its path through run18, step by
    step, to find why the relation is missing: no sentence in the corpus,
    passage never retrieved, retrieved but not extracted, extracted in another
    form, or extracted then removed at a later step.

STEPS FOLLOWED (files of --run-dir, read-only)
    corpus      index/chunks.jsonl: sentences that name both terms
    retrieval   pass_*/raw_triples.jsonl, field _provenance.selected_chunks:
                were these passages given to the extracting-LLM?
    extraction  pass_*/raw_triples.jsonl
    verif_qwen  pass_*/verified_triples.jsonl (verdict of Qwen)
    cleaning_rejected  pass_*/rejected_triples_v5.jsonl (_reject_reason)
    after_cleaning     pass_*/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl
    fusion      kg/tiered_kg_<run>.json
    panel       m4_panel/tiered_kg_m4.json
    panel_quarantine   m4_panel/m4_quarantine.jsonl
    delivered   kg_final/flat_<run>_enforced_dedup.json

MATCHING (fixed before looking at the results)
    A term is found in a text if it occurs as whole words, with space and
    hyphen treated alike and the plurals s/es accepted.
    level A : same relation; subject and object contain the two terms
              (e.g. "turbidite deposits" --hasDescriptor--> "parallel")
    level B : both terms are present but the relation differs or the direction
              is reversed (e.g. "mass transport deposit" --formedBy--> "slope failure")
    The reason reported is the first step at which the relation disappears at
    level A.

LIMITATION
    02 writes the provenance only for queries that produced at least one
    relation: a passage given to the extracting-LLM that yielded no relation is
    not visible. "Not retrieved" is therefore an upper bound.

USAGE
    python 179_trace_missing_expert_relations.py --repo . --run-dir output/run18 \
        --outdir output/audit/179_trace_run18
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REF26 = "configs/lb_reference_edges_original26.json"


def die(msg):
    print("STOP: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def low(x):
    return " ".join(str(x or "").lower().replace("-", " ").replace("_", " ").split())


def pattern(term):
    words = low(term).split()
    if not words:
        return None
    body = r"[\s\-]+".join(re.escape(w) for w in words[:-1]) + (r"[\s\-]+" if len(words) > 1 else "")
    body += re.escape(words[-1]) + r"(?:s|es)?"
    return re.compile(r"(?<![a-z])" + body + r"(?![a-z])")


def has(p, text):
    return p is not None and p.search(str(text or "").lower()) is not None


def nrel(r):
    return re.sub(r"[\s_\-]", "", str(r or "").lower())


def fields(t):
    s = t.get("source", t.get("subject", t.get("source_norm", "")))
    o = t.get("target", t.get("object", t.get("target_norm", "")))
    r = t.get("relation_norm", t.get("relation", ""))
    return s, r, o


def load_any(path):
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    raw = json.load(open(p, encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    for k in ("triples", "all_triples"):
        if isinstance(raw.get(k), list):
            return raw[k]
    out = []
    for k, v in raw.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            for t in v:
                t = dict(t)
                t.setdefault("_bucket", k)
                out.append(t)
    return out


def verdict_of(t):
    if "_reject_reason" in t:
        return "_reject_reason=%s" % t["_reject_reason"]
    for k in ("_verification", "verification", "m4"):
        v = t.get(k)
        if isinstance(v, dict):
            for kk in ("verdict", "decision", "label"):
                if kk in v:
                    return "%s.%s=%s" % (k, kk, v[kk])
    for k in ("verdict", "_reject_reason", "tier", "_bucket"):
        if k in t:
            return "%s=%s" % (k, t[k])
    return ""


def match(t, ps, po, rel):
    s, r, o = fields(t)
    if has(ps, s) and has(po, o):
        return "A" if nrel(r) == nrel(rel) else "B"
    if has(ps, o) and has(po, s):
        return "B"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-name", default=None, help="default: name of the folder")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    if out.exists():
        die("the folder already exists, nothing is overwritten: " + str(out))
    rd = Path(args.run_dir)
    run = args.run_name or rd.name
    edges = [(e["subject"], e["relation"], e["object"]) for e in
             json.load(open(Path(args.repo) / REF26, encoding="utf-8"))["edges"]]

    stages = []
    for P in ("a", "b"):
        stages += [("extraction_" + P, rd / ("pass_%s/raw_triples.jsonl" % P)),
                   ("verif_qwen_" + P, rd / ("pass_%s/verified_triples.jsonl" % P)),
                   ("cleaning_rejected_" + P, rd / ("pass_%s/rejected_triples_v5.jsonl" % P)),
                   ("after_cleaning_" + P, rd / ("pass_%s/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl" % P))]
    stages += [("fusion", rd / ("kg/tiered_kg_%s.json" % run)),
               ("panel", rd / "m4_panel/tiered_kg_m4.json"),
               ("panel_quarantine", rd / "m4_panel/m4_quarantine.jsonl"),
               ("delivered", rd / ("kg_final/flat_%s_enforced_dedup.json" % run))]
    data = {}
    for name, path in stages:
        d = load_any(path)
        data[name] = d if d is not None else []
        print("%-20s %6s  %s" % (name, len(d) if d is not None else "absent", path), flush=True)

    chunks = [json.loads(l) for l in open(rd / "index/chunks.jsonl", encoding="utf-8") if l.strip()]
    retrieved = defaultdict(set)   # chunk_id -> queries
    for P in ("a", "b"):
        for t in data["extraction_" + P]:
            pv = t.get("_provenance") or {}
            for cid in pv.get("selected_chunks") or []:
                retrieved[cid].add(pv.get("query", ""))

    rows, details = [], {}
    for (s, r, o) in edges:
        ps, po = pattern(s), pattern(o)
        sent, cochunks = [], []
        for c in chunks:
            txt = c.get("text", "")
            if has(ps, txt) and has(po, txt):
                ss = [x for x in re.split(r"(?<=[.!?])\s+", " ".join(txt.split())) if has(ps, x) and has(po, x)]
                if ss:
                    cochunks.append(c)
                    sent += [(c.get("doc_id"), x) for x in ss]
        seen = [c for c in cochunks if c.get("chunk_id") in retrieved]
        info = {"corpus_sentences": len(sent), "corpus_passages": len(cochunks),
                "passages_given_to_extractor": len(seen),
                "queries_that_retrieved_them": sorted({q for c in seen for q in retrieved[c.get("chunk_id")]})[:6],
                "example_sentences": [[d, x[:300]] for d, x in sent[:3]], "steps": {}}
        for name, _ in stages:
            hits = defaultdict(list)
            for t in data[name]:
                m = match(t, ps, po, r)
                if m:
                    hits[m].append(t)
            info["steps"][name] = {
                "A": len(hits["A"]), "B": len(hits["B"]),
                "examples": [[lvl, *[str(x) for x in fields(t)], verdict_of(t),
                              str(t.get("evidence") or "")[:200]]
                             for lvl in ("A", "B") for t in hits[lvl][:3]]}
        E = info["steps"]
        a_extr = E["extraction_a"]["A"] + E["extraction_b"]["A"]
        b_extr = E["extraction_a"]["B"] + E["extraction_b"]["B"]
        if E["delivered"]["A"]:
            why = "PRESENT in the delivered graph (level A)"
        elif not sent:
            why = "NO_SENTENCE: no sentence of the corpus names both terms"
        elif not seen:
            why = "NOT_RETRIEVED: the passages that hold it were never given to the extracting-LLM"
        elif not a_extr and not b_extr:
            why = "NOT_EXTRACTED: passage read by the extracting-LLM, no relation between the two terms"
        elif not a_extr:
            why = "OTHER_FORM: extracted only with another relation or in the other direction"
        else:
            order = ["verif_qwen", "after_cleaning", "fusion", "panel", "delivered"]
            why = "?"
            for st in order:
                n = (E[st]["A"] if st in E else E[st + "_a"]["A"] + E[st + "_b"]["A"])
                if n == 0:
                    rej = E["cleaning_rejected_a"]["A"] + E["cleaning_rejected_b"]["A"]
                    if st == "after_cleaning" and rej:
                        reasons = sorted({x[4] for k in ("cleaning_rejected_a", "cleaning_rejected_b")
                                          for x in E[k]["examples"] if x[0] == "A"})
                        why = "REMOVED_AT_CLEANING (%s)" % "; ".join(reasons)
                    else:
                        why = "REMOVED_AT_STEP " + st
                    break
        info["reason"] = why
        details["%s --%s--> %s" % (s, r, o)] = info
        rows.append([s, r, o, len(sent), len(cochunks), len(seen),
                     a_extr, b_extr, E["delivered"]["A"], E["delivered"]["B"], why])

    out.mkdir(parents=True)
    with open(out / "trace_26.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["subject", "relation", "object", "corpus_sentences", "corpus_passages",
                    "passages_read", "extracted_A", "extracted_B", "delivered_A", "delivered_B", "reason"])
        w.writerows(rows)
    json.dump(details, open(out / "details.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    L = ["=== 179 path of the expert relations in %s ===" % run,
         "%-48s %5s %5s %5s %4s %4s %4s %4s  %s" % ("relation", "sent", "pass", "read", "exA", "exB", "delA", "delB", "reason")]
    for rw in rows:
        L.append("%-48s %5d %5d %5d %4d %4d %4d %4d  %s" % ("%s --%s--> %s" % tuple(rw[:3]), *rw[3:10], rw[10]))
    L.append("sent = corpus sentences that name both terms; pass = passages that hold them;")
    L.append("read = those passages given to the extracting-LLM; exA/exB = relations extracted at level A/B; delA/delB = in the delivered graph")
    L.append("files: trace_26.tsv, details.json (sentences, queries, examples per step)")
    txt = "\n".join(L)
    (out / "_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt + "\n[done] " + str(out), flush=True)


if __name__ == "__main__":
    main()