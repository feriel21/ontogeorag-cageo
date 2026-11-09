#!/usr/bin/env python3
"""
183_prepare_qclaim_pilot.py  (read-only on inputs, CPU, seconds)

Builds the input of the qualified-claim pilot for the 102 causal relations of
the delivered graph:
  * locates each evidence sentence in the run16E index, in the chunk recorded by
    located_provenance, then in its neighbours (adjacent chunks merged, overlap
    removed), then in any chunk of the same paper;
  * cuts a passage = evidence sentence(s) + up to 2 sentences before and after,
    the evidence marked with ⟦ ⟧;
  * writes pilot_inputs.jsonl (for 184, 185, 186) and manual_sheet.csv (for the
    human annotation). The manual sheet does NOT show the KG predicate, tier or
    panel decision, and is ordered by a content hash, not by tier.

Run:   python -u 183_prepare_qclaim_pilot.py
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qclaim_schema import (CAUSAL, EXPECTED_CAUSAL, EXPECTED_ENTITIES, EXPECTED_TRIPLES,
                           FAMILY_CONTROL, MANUAL_FIELDS, LIGATURES, norm, rel_id, stop)

DEFAULT_GRAPH = "~/ontogeorag_release/results/run16E/graph_with_located_provenance.json"
DEFAULT_FAMILIES = "~/ontogeorag_release/configs/entity_families.csv"
DEFAULT_CHUNKS = "~/ontogeorag_D/output/run16E/index/chunks.jsonl"
DEFAULT_OUTDIR = "~/ontogeorag/output/qclaim_pilot"
NEIGHBOUR_SENTENCES = 2
KEY_CHARS = 60
NO_SPLIT_BEFORE = re.compile(r"(\bet al|\be\.g|\bi\.e|\bfig|\bfigs|\beq|\bca|\bcf|\bvs|\bno)\.$", re.I)


def clean(s):
    s = str(s or "")
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    return " ".join(s.split())


def alnum_map(text):
    """alnum string of text and, for each alnum char, its offset in text."""
    chars, pos = [], []
    for i, ch in enumerate(text.lower()):
        if ch.isascii() and ch.isalnum():
            chars.append(ch)
            pos.append(i)
    return "".join(chars), pos


def evidence_keys(ev):
    a = re.sub(r"[^a-z0-9]", "", clean(ev).lower())
    return a[:KEY_CHARS], a[-KEY_CHARS:], len(a)


def locate(text, ev):
    """Return (start, end) offsets of the evidence in text, or None."""
    a, pos = alnum_map(text)
    head, tail, n = evidence_keys(ev)
    if len(head) < 20 or not pos:
        return None
    i = a.find(head)
    if i >= 0:
        j = a.find(tail, i)
        if j >= 0 and j - i <= int(n * 1.6):
            end_a = j + len(tail) - 1
        else:
            end_a = min(len(pos) - 1, i + n - 1)
        return pos[i], pos[end_a] + 1
    j = a.find(tail)
    if j < 0:
        return None
    start_a = max(0, j + len(tail) - n)
    return pos[start_a], pos[j + len(tail) - 1] + 1


def split_sentences(text):
    spans, start = [], 0
    for m in re.finditer(r"[.!?][\"')\]]?\s+", text):
        cut = m.end()
        if NO_SPLIT_BEFORE.search(text[start:m.start() + 1]):
            continue
        spans.append((start, cut))
        start = cut
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def merge_overlap(a, b):
    for k in range(min(700, len(a), len(b)), 20, -1):
        if a[-k:] == b[:k]:
            return a + b[k:]
    return a + " " + b


def passage_from(text, span):
    s, e = span
    sents = split_sentences(text)
    idx = [k for k, (a, b) in enumerate(sents) if b > s and a < e]
    if not idx:
        return None
    lo = max(0, idx[0] - NEIGHBOUR_SENTENCES)
    hi = min(len(sents), idx[-1] + 1 + NEIGHBOUR_SENTENCES)
    p0, p1 = sents[lo][0], sents[hi - 1][1]
    return (text[p0:s] + "⟦" + text[s:e] + "⟧" + text[e:p1]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--families", default=DEFAULT_FAMILIES)
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if outdir.exists():
        stop("output directory exists, refusing to overwrite: " + str(outdir))

    gp = Path(args.graph).expanduser()
    with open(gp) as f:
        d = json.load(f)
    triples = d["triples"] if isinstance(d, dict) else d
    ents = {norm(t["subject"]) for t in triples} | {norm(t["object"]) for t in triples}
    if len(triples) != EXPECTED_TRIPLES or len(ents) != EXPECTED_ENTITIES:
        stop("graph " + str(len(triples)) + "/" + str(len(ents)) + ", expected 232/216")
    causal = [t for t in triples if t["relation"] in CAUSAL]
    if len(causal) != EXPECTED_CAUSAL:
        stop(str(len(causal)) + " causal relations, expected 102")

    with open(Path(args.families).expanduser(), newline="", encoding="utf-8-sig") as f:
        frows = list(csv.DictReader(f))
    fam = {norm(r["entity"]): norm(r["family"]) for r in frows}
    size = Counter(fam.values())
    multi = [k for k, n in size.items() if n >= 2]
    if (len(frows), len(size), len(multi), sum(size[k] for k in multi)) != FAMILY_CONTROL:
        stop("families control failed")

    chunks, by_doc = {}, defaultdict(dict)
    id_field = None
    with open(Path(args.chunks).expanduser()) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if id_field is None:
                id_field = next((k for k in ("chunk_id", "id", "uid", "cid") if k in c), None)
                if id_field is None:
                    stop("no chunk id field; keys: " + ", ".join(c))
            cid = str(c[id_field])
            txt = clean(c.get("text", ""))
            chunks[cid] = (str(c.get("doc_id", "")), txt)
            m = re.match(r"^(.*)::(\d+)$", cid)
            if m:
                by_doc[m.group(1)][int(m.group(2))] = txt
    if len(chunks) != 1758:
        stop("index has " + str(len(chunks)) + " chunks, expected 1758")

    def window(cid):
        m = re.match(r"^(.*)::(\d+)$", cid)
        if not m or m.group(1) not in by_doc:
            return chunks.get(cid, ("", ""))[1]
        doc, n = m.group(1), int(m.group(2))
        parts = [by_doc[doc][k] for k in (n - 1, n, n + 1) if k in by_doc[doc]]
        text = parts[0]
        for p in parts[1:]:
            text = merge_overlap(text, p)
        return text

    rows, seen, status_c = [], set(), Counter()
    for t in causal:
        rid = rel_id(t["subject"], t["relation"], t["object"])
        if rid in seen:
            stop("duplicate content key for " + t["subject"] + " | " + t["object"])
        seen.add(rid)
        lp = t.get("located_provenance") or {}
        docs = sorted(set(lp.get("doc_ids") or []))
        ev = clean(t.get("evidence", ""))
        passage, pstatus, where = None, "not_found", ""
        for cid in [str(c) for c in (lp.get("chunk_ids") or [])]:
            text = window(cid)
            span = locate(text, ev)
            if span:
                passage, pstatus, where = passage_from(text, span), "found_in_located_chunk", cid
                break
        if passage is None:
            for doc in docs:
                for n in sorted(by_doc.get(doc, {})):
                    text = window(doc + "::" + str(n))
                    span = locate(text, ev)
                    if span:
                        passage, pstatus, where = passage_from(text, span), "found_in_same_paper", doc + "::" + str(n)
                        break
                if passage:
                    break
        if passage is None:
            passage = "⟦" + ev + "⟧"
        status_c[pstatus] += 1
        m4 = t.get("m4") or {}
        rows.append({
            "rel_id": rid, "subject": t["subject"], "relation": t["relation"], "object": t["object"],
            "family_s": fam.get(norm(t["subject"]), norm(t["subject"])),
            "family_o": fam.get(norm(t["object"]), norm(t["object"])),
            "doc_ids": docs, "match": lp.get("match", ""), "tier": t.get("tier", ""),
            "panel": m4.get("decision", ""), "evidence": ev, "passage": passage,
            "passage_status": pstatus, "passage_chunk": where,
        })

    rows.sort(key=lambda r: r["rel_id"])
    outdir.mkdir(parents=True)
    with open(outdir / "pilot_inputs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(outdir / "manual_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["rel_id", "cause", "effect", "passage", "passage_status"] + MANUAL_FIELDS)
        for r in rows:
            w.writerow([r["rel_id"], r["subject"], r["object"], r["passage"], r["passage_status"]]
                       + [""] * len(MANUAL_FIELDS))
    summary = {"causal_relations": len(rows), "passage_status": dict(status_c),
               "chunk_id_field": id_field, "graph": str(gp)}
    with open(outdir / "183_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1))
    print("not found (evidence only, flagged in the sheet): "
          + ", ".join(r["rel_id"] for r in rows if r["passage_status"] == "not_found"))
    print("wrote " + str(outdir))
    print("OK")


if __name__ == "__main__":
    main()