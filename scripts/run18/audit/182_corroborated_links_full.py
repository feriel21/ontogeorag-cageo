#!/usr/bin/env python3
"""
182_corroborated_links_full.py

Read-only. For every causal relation (causes, triggers, controls, affects
pooled) that is supported by at least two papers after family fusion, dumps:
  * every supporting triple with its FULL evidence sentence, its doc_ids,
    its localisation type (exact / partial / not_found), tier and panel
    decision;
  * the chunk each doc_id was located in (from located_provenance.chunk_ids),
    with a flag saying whether the evidence sentence really occurs in that
    chunk, and a context window around it;
  * the list of papers involved with their full corpus_table.md row, and
    empty columns to mark reviews and same-volume chapters by hand.

Python 3.10, standard library only. Never overwrites. Stops on failed controls.

Run:
  source ~/kg_test/venv/bin/activate
  python -u 182_corroborated_links_full.py
Send back: <outdir>/182_report.md  and  <outdir>/182_papers.csv
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_TRIPLES = 232
EXPECTED_ENTITIES = 216
FAMILY_CONTROL = (128, 21, 19, 126)  # rows, families, multi-member families, entities in them
CAUSAL = ("causes", "triggers", "controls", "affects")

DEFAULT_GRAPH = "~/ontogeorag_release/results/run16E/graph_with_located_provenance.json"
DEFAULT_FAMILIES = "~/ontogeorag_release/configs/entity_families.csv"
DEFAULT_CHUNKS = "~/ontogeorag_D/output/run16E/index/chunks.jsonl"
DEFAULT_CORPUS = "~/ontogeorag_release/data/corpus_table.md"
DEFAULT_OUTDIR = "~/ontogeorag/output/audit/182_corroborated_links"
CONTEXT = 600  # characters kept on each side of the evidence in the chunk

LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


def stop(msg):
    print("STOP: " + msg, file=sys.stderr)
    sys.exit(1)


def norm(s):
    return " ".join(str(s or "").lower().split())


def norm_text(s):
    s = str(s or "")
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = re.sub(r"[\"'\u201c\u201d\u2018\u2019]", "", s.lower())
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--families", default=DEFAULT_FAMILIES)
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--min-papers", type=int, default=2)
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if outdir.exists():
        stop("output directory exists, refusing to overwrite: " + str(outdir))

    # ---- graph
    gp = Path(args.graph).expanduser()
    if not gp.is_file():
        stop("graph not found: " + str(gp))
    with open(gp) as f:
        d = json.load(f)
    triples = d["triples"] if isinstance(d, dict) else d
    ents = {norm(t["subject"]) for t in triples} | {norm(t["object"]) for t in triples}
    if len(triples) != EXPECTED_TRIPLES or len(ents) != EXPECTED_ENTITIES:
        stop("graph is " + str(len(triples)) + " triples / " + str(len(ents)) + " entities, expected 232 / 216")

    # ---- families
    fp = Path(args.families).expanduser()
    with open(fp, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fam = {norm(r["entity"]): norm(r["family"]) for r in rows}
    size = Counter(fam.values())
    multi = [k for k, n in size.items() if n >= 2]
    got = (len(rows), len(size), len(multi), sum(size[k] for k in multi))
    if got != FAMILY_CONTROL or len(fam) != len(rows):
        stop("families control " + str(got) + " != " + str(FAMILY_CONTROL))
    missing = [e for e in fam if e not in ents]
    if missing:
        stop("family entities absent from graph: " + "; ".join(missing))
    F = lambda e: fam.get(norm(e), norm(e))

    # ---- chunks
    cp = Path(args.chunks).expanduser()
    if not cp.is_file():
        stop("chunks not found: " + str(cp))
    chunks = {}
    id_field = None
    with open(cp) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if id_field is None:
                id_field = next((k for k in ("chunk_id", "id", "uid", "cid") if k in c), None)
                if id_field is None:
                    stop("no chunk id field in chunks.jsonl, keys: " + ", ".join(c))
            chunks[str(c[id_field])] = c
    if len(chunks) != 1758:
        stop("chunks.jsonl has " + str(len(chunks)) + " chunks, expected 1758")

    # ---- corpus table
    corpus_lines, corpus_header = [], ""
    kp = Path(args.corpus).expanduser()
    if kp.is_file():
        with open(kp, encoding="utf-8") as f:
            table = [l.rstrip("\n") for l in f if l.strip().startswith("|")]
        if table:
            corpus_header = table[0]
            corpus_lines = [l for l in table[1:] if not re.match(r"^\|\s*-", l)]

    def corpus_row(doc):
        hits = [l for l in corpus_lines if doc in l]
        return hits[0] if len(hits) == 1 else ("AMBIGUOUS: " + " || ".join(hits) if hits else "NOT FOUND")

    # ---- group causal relations by fused key
    groups = defaultdict(list)
    for t in triples:
        if t["relation"] in CAUSAL:
            groups[(F(t["subject"]), F(t["object"]))].append(t)

    def docs_of(t):
        return sorted(set((t.get("located_provenance") or {}).get("doc_ids") or []))

    selected = []
    for key, members in groups.items():
        docs = sorted({d for m in members for d in docs_of(m)})
        if len(docs) >= args.min_papers:
            selected.append((len(docs), key, members, docs))
    selected.sort(key=lambda x: (-x[0], x[1]))
    if not selected:
        stop("no fused causal relation with >= " + str(args.min_papers) + " papers")

    # ---- per-triple rows with chunk verification
    link_rows, rep = [], []
    paper_links = defaultdict(set)
    rep.append("# 182 corroborated causal links, full provenance")
    rep.append("graph " + str(gp) + " | families " + str(fp) + " | chunks " + str(cp) + " (id field '"
               + str(id_field) + "')")
    rep.append("selection: causal predicates pooled, families fused, >= " + str(args.min_papers)
               + " papers: " + str(len(selected)) + " relations")
    rep.append("CHUNK CHECK: 'yes' = evidence (normalised, ligatures repaired) occurs in the located chunk;"
               " 'no' = it does not, so the paper attribution of that doc_id is NOT confirmed.")
    for n, (fs, fo), members, docs in selected:
        rep.append("")
        rep.append("## " + fs + "  -->  " + fo + "   (" + str(n) + " papers after fusion)")
        for m in sorted(members, key=lambda x: (docs_of(x), norm(x["subject"]))):
            lp = m.get("located_provenance") or {}
            ev = " ".join(str(m.get("evidence", "")).split())
            m4 = m.get("m4") or {}
            chunk_ids = [str(c) for c in (lp.get("chunk_ids") or [])]
            checks = []
            for cid in chunk_ids:
                c = chunks.get(cid)
                if c is None:
                    checks.append((cid, "", "chunk id not in index", ""))
                    continue
                txt = " ".join(str(c.get("text", "")).split())
                ntxt, nev = norm_text(txt), norm_text(ev)
                pos = ntxt.find(nev[:120]) if nev else -1
                found = "yes" if pos >= 0 else "no"
                if pos >= 0:
                    ctx = ntxt[max(0, pos - CONTEXT): pos + len(nev) + CONTEXT]
                else:
                    ctx = ntxt[: 2 * CONTEXT]
                checks.append((cid, str(c.get("doc_id", "")), found, ctx))
            for dd in docs_of(m):
                paper_links[dd].add(fs + " -> " + fo)
            rep.append("")
            rep.append("- TRIPLE: " + m["subject"] + " --" + m["relation"] + "--> " + m["object"]
                       + " | tier " + str(m.get("tier", "")) + " | panel " + str(m4.get("decision", ""))
                       + " | match " + str(lp.get("match", "")))
            rep.append("  PAPERS: " + (", ".join(docs_of(m)) or "NONE"))
            rep.append("  EVIDENCE: " + ev)
            for cid, cdoc, found, ctx in checks:
                rep.append("  CHUNK " + cid + " [" + cdoc + "] evidence in chunk: " + found)
                rep.append("  CONTEXT: ..." + ctx + "...")
            if not checks:
                rep.append("  CHUNK: none recorded in located_provenance")
            link_rows.append([fs, fo, n, m["subject"], m["relation"], m["object"], " | ".join(docs_of(m)),
                              lp.get("match", ""), m.get("tier", ""), m4.get("decision", ""), ev,
                              " | ".join(c[0] for c in checks), " | ".join(c[2] for c in checks),
                              " ||| ".join(c[3] for c in checks)])

    # ---- papers
    paper_rows = []
    rep.append("")
    rep.append("## Papers involved (mark reviews and same-volume chapters in 182_papers.csv)")
    rep.append("corpus_table header: " + (corpus_header or "corpus_table.md NOT FOUND"))
    for dd in sorted(paper_links):
        row = corpus_row(dd)
        paper_rows.append([dd, len(paper_links[dd]), " ; ".join(sorted(paper_links[dd])), row, "", "", ""])
        rep.append("- " + dd + " | supports " + str(len(paper_links[dd])) + " relation(s) | " + row)

    outdir.mkdir(parents=True)
    with open(outdir / "182_links_full.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family_s", "family_o", "papers_after_fusion", "subject", "relation", "object",
                    "papers", "match", "tier", "panel", "evidence_full", "chunk_ids",
                    "evidence_in_chunk", "chunk_context"])
        w.writerows(link_rows)
    with open(outdir / "182_papers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "n_relations_supported", "relations", "corpus_table_row",
                    "is_review_or_synthesis", "same_volume_as", "note"])
        w.writerows(paper_rows)
    with open(outdir / "182_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    no_hit = sum(1 for r in link_rows if "no" in r[12].split(" | "))
    print("relations selected: " + str(len(selected)) + " | supporting triples: " + str(len(link_rows))
          + " | papers: " + str(len(paper_rows)) + " | triples with a chunk NOT containing the evidence: "
          + str(no_hit))
    print("wrote " + str(outdir))
    print("OK")


if __name__ == "__main__":
    main()