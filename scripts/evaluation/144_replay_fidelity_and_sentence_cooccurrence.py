#!/usr/bin/env python3
"""
144_replay_fidelity_and_sentence_cooccurrence.py

Follows 143. No language model is called for extraction or judgement; the
cross-encoder is used only to replay retrieval.

Q1 (provenance). What does each of the 232 delivered relations actually
    carry: evidence sentence, query, supporting papers? Can the paper be
    recovered by locating the evidence sentence in the index?
Q2 (replay fidelity). Is each relation's evidence sentence found in the
    replayed top-5 of the very query recorded on that relation?
Q3 (sentence co-occurrence). For each benchmark edge, how many sentences of
    the chunks passed to the extractor contain both arguments?

Rules fixed BEFORE running. Do not change them after seeing the results.
F1  Text comparison for evidence: lowercase, keep [a-z0-9] only, on both the
    evidence sentence and the chunk (neutralises quotes, hyphenation,
    line breaks).
F2  FOUND_EXACT if the whole normalised evidence is a substring of a chunk;
    FOUND_PARTIAL if not, but at least one of three 40-character windows
    (start, middle, end) is; NOT_FOUND otherwise. Evidence of 40 characters
    or fewer is tested whole only.
F3  Fidelity levels per relation, first that applies:
      IN_OWN_QUERY_TOP5, IN_OTHER_QUERY_TOP5, IN_INDEX_NOT_TOP5,
      NOT_IN_INDEX, NO_EVIDENCE, QUERY_NOT_IN_SET.
F4  Paper of a relation: document field of the chunks where its evidence is
    found (F2), if a document field exists in the index.
S1  Argument occurrence: rule R1 of 143 (word-bounded, separator-normalised,
    optional plural).
S2  Sentences: whitespace collapsed, split after . ! or ? followed by space.
    Abbreviations such as "fig." split sentences; this can only lower counts.
S3  Sentence co-occurrence is counted over the union of replayed top-5 chunks
    (all 287 queries), each distinct sentence once.
S4  Chance level: probability that at least one of the n co-occurring chunks
    of an edge falls in a uniformly random set of K chunks, K = size of the
    top-5 union. Reported, never used to classify.
Consistency: replay must reproduce 143 (852 chunks in top-20 union, 432 in
top-5 union) and recovery 13/26, 18/34, or the script stops.
Joins use content keys only. Refuses to overwrite its output directory.
"""

import csv
import glob
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home()
ROOTS = [HOME / "ontogeorag_D", HOME / "ontogeorag"]
OUT_DIR_PATH = HOME / "ontogeorag" / "output" / "audit" / "144_fidelity_and_sentences"
QUERIES_V2_SHA_PREFIX = "f4269b33"
EXPECTED = {"chunks": 1758, "queries_ext": 287, "graph": 232, "bench26": 26,
            "bench34": 34, "recovered26": 13, "recovered34": 18,
            "top20_union": 852, "top5_union": 432}
BM25_TOPN, MIN_BM25, TOPK = 20, 2.0, 5
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
WINDOW = 40
DOC_KEYS = ("doc_id", "paper_id", "paper", "document", "doc", "source_file",
            "filename", "file", "pdf", "source", "title")
S_KEYS = ("subject", "source", "head", "src")
R_KEYS = ("relation", "predicate", "rel")
O_KEYS = ("object", "target", "tail", "dst")
LIST_KEYS = ("triples", "relations", "edges", "records", "items")
OUT_DIR = None


def die(msg, code=1):
    print("STOP: " + msg, file=sys.stderr, flush=True)
    if OUT_DIR is not None and OUT_DIR.is_dir() and not any(OUT_DIR.iterdir()):
        OUT_DIR.rmdir()
    sys.exit(code)


def log(msg):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_one(patterns, sha_prefix=None):
    hits = set()
    for root in ROOTS:
        for pat in patterns:
            for p in glob.glob(str(root / pat), recursive=True):
                if Path(p).is_file():
                    hits.add(Path(p).resolve())
    hits = sorted(hits)
    if sha_prefix:
        hits = [h for h in hits if sha256(h).startswith(sha_prefix)]
    if not hits:
        die("no file found for " + str(patterns))
    shas = {sha256(h) for h in hits}
    if len(shas) > 1:
        die("distinct files match " + str(patterns) + ": " + str(hits))
    return hits[0], shas.pop()


def check(name, value):
    if value != EXPECTED[name]:
        die(name + " = " + str(value) + ", expected " + str(EXPECTED[name]))
    log("  ok  " + name + " = " + str(value))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_list(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        named = [k for k in LIST_KEYS if isinstance(data.get(k), list)]
        lists = [k for k, v in data.items() if isinstance(v, list)]
        if len(named) == 1:
            data = data[named[0]]
        elif len(lists) == 1:
            data = data[lists[0]]
        else:
            die(str(path) + ": cannot pick the record list among " + str(sorted(data)))
    return data


def pick(d, keys, what):
    found = [k for k in keys if k in d]
    if len(found) != 1:
        die("cannot identify the " + what + " field among " + str(sorted(d)))
    return found[0]


def norm_entity(x):
    x = str(x).lower().replace("_", " ").replace("-", " ")
    return " ".join(x.split())


def norm_rel(x):
    return re.sub(r"[^a-z]", "", str(x).lower())


def norm_query(x):
    return " ".join(str(x).lower().split())


def alnum(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def term_regex(term):
    parts = [re.escape(p) for p in re.split(r"[\s_-]+", term.lower().strip()) if p]
    body = r"(?:-\s*|\s+)".join(parts)
    return re.compile(r"(?<![a-z0-9])" + body + r"(?:e?s)?(?![a-z0-9])")


def triples(records):
    first = records[0]
    sk, rk, ok = pick(first, S_KEYS, "subject"), pick(first, R_KEYS, "relation"), pick(first, O_KEYS, "object")
    return [(norm_entity(r[sk]), norm_rel(r[rk]), norm_entity(r[ok])) for r in records]


def find_evidence(ev, chunk_alnum, candidates):
    """F2 over the given chunk ids. Returns (status, matching ids)."""
    e = alnum(ev)
    if not e:
        return "NO_EVIDENCE", []
    exact = [i for i in candidates if e in chunk_alnum[i]]
    if exact:
        return "FOUND_EXACT", exact
    if len(e) <= WINDOW:
        return "NOT_FOUND", []
    mid = (len(e) - WINDOW) // 2
    wins = [e[:WINDOW], e[mid:mid + WINDOW], e[-WINDOW:]]
    partial = [i for i in candidates if any(w in chunk_alnum[i] for w in wins)]
    if partial:
        return "FOUND_PARTIAL", partial
    return "NOT_FOUND", []


def p_chance(n, total, k):
    """S4: P(at least one of n items in a random k-subset of total)."""
    if n == 0:
        return 0.0
    p_none = 1.0
    for i in range(n):
        p_none *= (total - k - i) / (total - i)
        if p_none <= 0:
            return 1.0
    return 1.0 - p_none


def main():
    global OUT_DIR
    if OUT_DIR_PATH.exists():
        die("output directory exists, refusing to overwrite: " + str(OUT_DIR_PATH))
    OUT_DIR_PATH.mkdir(parents=True)
    OUT_DIR = OUT_DIR_PATH
    t0 = time.time()
    report = {"script": Path(__file__).name, "rules": "F1-F4, S1-S4, see docstring", "inputs": {}}

    # ------------------------------------------------------------- inputs
    log("[inputs]")
    p, s = find_one(["output/**/run16E/index/chunks.jsonl"])
    report["inputs"]["chunks"] = [str(p), s]
    chunks = read_jsonl(p)
    check("chunks", len(chunks))
    texts = [c["text"] for c in chunks]
    low = [t.lower() for t in texts]
    chunk_alnum = [alnum(t) for t in texts]
    doc_key = next((k for k in DOC_KEYS if isinstance(chunks[0].get(k), str)), None)
    report["chunk_fields"] = sorted(chunks[0])
    report["chunk_doc_field"] = doc_key
    log("  chunk fields: " + str(sorted(chunks[0])) + "  doc field: " + str(doc_key))
    if doc_key:
        log("  distinct documents in index: " + str(len({c[doc_key] for c in chunks})))

    p, s = find_one(["configs/descriptor_queries_v2.jsonl"], QUERIES_V2_SHA_PREFIX)
    report["inputs"]["queries_ext"] = [str(p), s]
    qext = read_jsonl(p)
    check("queries_ext", len(qext))
    qindex = defaultdict(list)
    for i, q in enumerate(qext):
        qindex[norm_query(q["query"])].append(i)

    p, s = find_one(["configs/lb_reference_edges_original26.json"])
    report["inputs"]["bench26"] = [str(p), s]
    b26 = triples(load_list(p))
    check("bench26", len(b26))
    p, s = find_one(["configs/lb_reference_edges.json"])
    report["inputs"]["bench34"] = [str(p), s]
    b34 = triples(load_list(p))
    check("bench34", len(b34))

    p, s = find_one(["output/run16E/kg_final/flat_run16E_enforced_dedup.json"])
    report["inputs"]["graph"] = [str(p), s]
    graph = load_list(p)
    check("graph", len(graph))
    g_keys = set(triples(graph))
    check("recovered26", sum(k in g_keys for k in b26))
    check("recovered34", sum(k in g_keys for k in b34))

    # ------------------------------------------------- Q1 provenance fields
    log("[provenance fields]")
    prov = {
        "evidence_nonempty": sum(1 for r in graph if str(r.get("evidence") or "").strip()),
        "query_nonempty": sum(1 for r in graph if str(r.get("query") or "").strip()),
        "supporting_papers_nonempty": sum(1 for r in graph if r.get("supporting_papers")),
        "support_count_positive": sum(1 for r in graph if (r.get("support_count") or 0) > 0),
        "fields_present_in_all": sorted(set.intersection(*[set(r) for r in graph])),
        "fields_present_in_some": sorted(set.union(*[set(r) for r in graph])),
    }
    report["provenance_fields"] = prov
    log("  " + json.dumps(prov))

    # --------------------------------------------------- retrieval replay
    log("[retrieval replay]")
    import torch
    from huggingface_hub import snapshot_download
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

    bm25 = BM25Okapi([t.split() for t in low])
    kept_per_q = []
    for q in qext:
        sc = bm25.get_scores(str(q["query"]).lower().split())
        order = np.argsort(-sc, kind="stable")[:BM25_TOPN]
        kept_per_q.append([int(i) for i in order if sc[i] >= MIN_BM25])
    ce_path = snapshot_download(RERANKER, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder(ce_path, device=device)
    pairs, owners = [], []
    for qi, kept in enumerate(kept_per_q):
        for ci in kept:
            pairs.append((str(qext[qi]["query"]), texts[ci]))
            owners.append((qi, ci))
    scores = ce.predict(pairs, batch_size=64, show_progress_bar=False)
    by_q = defaultdict(list)
    for (qi, ci), sc in zip(owners, scores):
        by_q[qi].append((float(sc), ci))
    top5_per_q = {qi: [ci for _, ci in sorted(by_q[qi], key=lambda x: -x[0])[:TOPK]]
                  for qi in range(len(qext))}
    check("top20_union", len({ci for kept in kept_per_q for ci in kept}))
    top5_union = sorted({ci for lst in top5_per_q.values() for ci in lst})
    check("top5_union", len(top5_union))

    # -------------------------------------------------- Q2 fidelity, Q1 papers
    log("[fidelity]")
    all_ids = list(range(len(texts)))
    rel_rows = []
    for r in graph:
        ev = r.get("evidence") or ""
        qids = qindex.get(norm_query(r.get("query") or ""), [])
        row = {"subject": r.get("subject"), "relation": r.get("relation"), "object": r.get("object"),
               "tier": r.get("tier"), "strategy": r.get("strategy"), "query": r.get("query")}
        idx_status, idx_hits = find_evidence(ev, chunk_alnum, all_ids)
        row["index_match"] = idx_status
        row["n_index_chunks"] = len(idx_hits)
        row["papers"] = " | ".join(sorted({str(chunks[i][doc_key]) for i in idx_hits})) if doc_key else ""
        if idx_status == "NO_EVIDENCE":
            level = "NO_EVIDENCE"
        elif not qids:
            level = "QUERY_NOT_IN_SET"
        else:
            own = sorted({ci for qi in qids for ci in top5_per_q[qi]})
            own_status, _ = find_evidence(ev, chunk_alnum, own)
            if own_status != "NOT_FOUND":
                level = "IN_OWN_QUERY_TOP5"
            elif find_evidence(ev, chunk_alnum, top5_union)[0] != "NOT_FOUND":
                level = "IN_OTHER_QUERY_TOP5"
            elif idx_status != "NOT_FOUND":
                level = "IN_INDEX_NOT_TOP5"
            else:
                level = "NOT_IN_INDEX"
        row["fidelity"] = level
        rel_rows.append(row)
    fid = {
        "levels": dict(Counter(r["fidelity"] for r in rel_rows)),
        "index_match": dict(Counter(r["index_match"] for r in rel_rows)),
    }
    if doc_key:
        n_papers = Counter(len(r["papers"].split(" | ")) if r["papers"] else 0 for r in rel_rows)
        fid["papers_per_relation"] = {str(k): v for k, v in sorted(n_papers.items())}
    report["fidelity"] = fid
    log("  " + json.dumps(fid))

    # -------------------------------------------- Q3 sentence co-occurrence
    log("[sentence co-occurrence]")
    sentences = set()
    for ci in top5_union:
        flat = " ".join(low[ci].split())
        for sent in re.split(r"(?<=[.!?])\s+", flat):
            if sent:
                sentences.add(sent)
    sentences = sorted(sentences)
    report["sentences_in_top5_union"] = len(sentences)
    edges = {}
    for name, bench in (("26", b26), ("34", b34)):
        for k in bench:
            row = edges.setdefault(k, {"subject": k[0], "relation": k[1], "object": k[2],
                                       "in26": False, "in34": False, "recovered": k in g_keys})
            row["in" + name] = True
    rx = {}
    for k in edges:
        for term in (k[0], k[2]):
            rx.setdefault(term, term_regex(term))
    occ = {t: {i for i, x in enumerate(low) if r.search(x)} for t, r in rx.items()}
    top5_set = set(top5_union)
    for k, row in edges.items():
        cooc = occ[k[0]] & occ[k[2]]
        row["n_cooc_chunks_R1"] = len(cooc)
        row["n_cooc_chunks_in_top5"] = len(cooc & top5_set)
        row["p_chance_chunk_in_top5"] = round(p_chance(len(cooc), len(texts), len(top5_union)), 3)
        row["n_sentences_both_args_in_top5"] = sum(
            1 for sent in sentences if rx[k[0]].search(sent) and rx[k[2]].search(sent))
    summ = {}
    for name in ("26", "34"):
        sel = [r for r in edges.values() if r["in" + name]]
        summ[name] = {
            "recovered_with_sentence": sum(1 for r in sel if r["recovered"] and r["n_sentences_both_args_in_top5"] > 0),
            "recovered_without_sentence": sum(1 for r in sel if r["recovered"] and r["n_sentences_both_args_in_top5"] == 0),
            "unrecovered_with_sentence": sum(1 for r in sel if not r["recovered"] and r["n_sentences_both_args_in_top5"] > 0),
            "unrecovered_without_sentence": sum(1 for r in sel if not r["recovered"] and r["n_sentences_both_args_in_top5"] == 0),
        }
    report["sentence_summary"] = summ
    log("  " + json.dumps(summ))

    # ------------------------------------------------------------ outputs
    with open(OUT_DIR / "relations_fidelity.csv", "w", newline="") as f:
        cols = ["subject", "relation", "object", "tier", "strategy", "query",
                "index_match", "n_index_chunks", "papers", "fidelity"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rel_rows)
    with open(OUT_DIR / "edges_sentence_cooccurrence.csv", "w", newline="") as f:
        cols = ["in26", "in34", "subject", "relation", "object", "recovered",
                "n_cooc_chunks_R1", "n_cooc_chunks_in_top5", "p_chance_chunk_in_top5",
                "n_sentences_both_args_in_top5"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(edges):
            w.writerow(edges[k])
    report["runtime_s"] = round(time.time() - t0, 1)
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    log("[done] " + str(OUT_DIR))


if __name__ == "__main__":
    main()