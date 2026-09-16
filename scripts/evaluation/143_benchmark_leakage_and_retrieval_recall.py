#!/usr/bin/env python3
"""
143_benchmark_leakage_and_retrieval_recall.py

Two questions, no language model involved.

Q1 (leakage). Which queries were written against benchmark edges, and are
    they in the base set of 249 queries (configuration A) or only in the
    extended set of 287?

Q2 (retrieval recall). For each benchmark edge the delivered graph does not
    recover, did the replayed retrieval (BM25 top-20, min score 2.0,
    cross-encoder top-5, parameters of the run16E manifest argv) ever pass
    a chunk containing both arguments to the extractor?

Rules fixed BEFORE running. Do not change them after seeing the results.
R1  Argument occurrence in a chunk: case-insensitive, word-bounded,
    hyphen / space / hyphen+whitespace treated as one separator,
    optional plural suffix (s|es). Nothing else.
R2  An edge co-occurs in a chunk iff subject and object both occur (R1).
R3  A query targets an edge iff its `focus` + `query` text contains both
    the subject and the object of the edge under R1.
    Validity: R3 must reproduce 95m (17 of 26 edges, 17 of 34, 24 queries).
    Otherwise the leakage verdict is WITHHELD.
R4  Recovery: normalised exact (subject, relation, object) match against
    the delivered graph; must reproduce 13/26 and 18/34. If exact fails,
    the substring variant is tried; if both fail, the script stops.
R5  Stage of an unrecovered edge, for three query subsets
    (all 287 / base 249 / all minus the queries targeting that edge):
      NO_EVIDENCE              no chunk of the index co-occurs
      NOT_IN_BM25_TOP20        co-occurring chunks exist, none in a BM25 top-20
      LOST_AT_RERANK           in some BM25 top-20, never in a top-5
      RETRIEVED_NOT_EXTRACTED  in some top-5 passed to the extractor
R6  Replay fidelity: share of delivered relations whose stored passage is
    found in a replayed top-5 chunk. Reported, never used to filter.

Joins between files use content keys only, never positions.
Refuses to overwrite an existing output directory.

Usage
  python -u 143_benchmark_leakage_and_retrieval_recall.py --leakage-only
  python -u 143_benchmark_leakage_and_retrieval_recall.py
"""

import argparse
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
OUT_BASE = HOME / "ontogeorag" / "output" / "audit"

QUERIES_V2_SHA_PREFIX = "f4269b33"
EXPECTED = {
    "chunks": 1758, "queries_base": 249, "queries_ext": 287, "graph": 232,
    "bench26": 26, "bench34": 34, "recovered26": 13, "recovered34": 18,
    "targeted26": 17, "targeted34": 17, "targeting_queries": 24,
}
BM25_TOPN, MIN_BM25, TOPK = 20, 2.0, 5
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DECLARED_RELATIONS = {
    "hasdescriptor", "causes", "occursin", "triggers", "affects", "controls",
    "overlies", "partof", "underlies", "formedby", "relatedto", "evidences",
    "indicates",
}
S_KEYS = ("subject", "source", "head", "src")
R_KEYS = ("relation", "predicate", "rel")
O_KEYS = ("object", "target", "tail", "dst")
LIST_KEYS = ("triples", "relations", "edges", "records", "items")


# ---------------------------------------------------------------- helpers
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


def load_records(path):
    with open(path) as f:
        txt = f.read()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in txt.splitlines() if line.strip()]
    if isinstance(data, dict):
        named = [k for k in LIST_KEYS if isinstance(data.get(k), list)]
        lists = [k for k, v in data.items() if isinstance(v, list)]
        if len(named) == 1:
            data = data[named[0]]
        elif len(lists) == 1:
            data = data[lists[0]]
        else:
            die(str(path) + ": cannot pick the record list among keys " + str(sorted(data)))
    return data


def pick(d, keys, path, what):
    found = [k for k in keys if k in d]
    if len(found) != 1:
        die(str(path) + ": cannot identify the " + what + " field among " + str(sorted(d)))
    return found[0]


def norm_entity(x):
    x = str(x).lower().replace("_", " ").replace("-", " ")
    return " ".join(x.split())


def norm_rel(x):
    return re.sub(r"[^a-z]", "", str(x).lower())


def as_triples(records, path):
    out = []
    first = records[0]
    if isinstance(first, (list, tuple)):
        for r in records:
            out.append({"s": r[0], "r": r[1], "o": r[2]})
    else:
        sk = pick(first, S_KEYS, path, "subject")
        rk = pick(first, R_KEYS, path, "relation")
        ok = pick(first, O_KEYS, path, "object")
        for r in records:
            out.append({"s": r[sk], "r": r[rk], "o": r[ok]})
    bad = sorted({norm_rel(t["r"]) for t in out} - DECLARED_RELATIONS)
    if bad:
        die(str(path) + ": relation values outside the 13 declared types: " + str(bad))
    return out


def ekey(t):
    return (norm_entity(t["s"]), norm_rel(t["r"]), norm_entity(t["o"]))


def term_regex(term):
    parts = [re.escape(p) for p in re.split(r"[\s_-]+", term.lower().strip()) if p]
    body = r"(?:-\s*|\s+)".join(parts)
    return re.compile(r"(?<![a-z0-9])" + body + r"(?:e?s)?(?![a-z0-9])")


def query_text(q):
    raw = (str(q.get("focus", "")) + " " + str(q["query"])).lower()
    return re.sub(r"[_:+>|]", " ", raw)


def query_key(q):
    return (str(q.get("strategy", "")),
            " ".join(str(q.get("focus", "")).split()),
            " ".join(str(q["query"]).split()))


def passage_getter(records):
    """Locate the stored passage field by content (long string), not by guess."""
    cands = []

    def walk(d, path):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, path + (k,))
        elif isinstance(d, str) and len(d) >= 300:
            cands.append(path)

    walk(records[0], ())
    named = [p for p in cands
             if re.search(r"passage|evidence|chunk|context", "/".join(map(str, p)), re.I)]
    exact = [p for p in named if p[-1] == "supporting_passage"]
    chosen = exact[0] if exact else (named[0] if len(named) == 1 else None)
    if chosen is None:
        return None, cands

    def get(rec):
        cur = rec
        for k in chosen:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    return get, chosen


def stage(cooc, top20_q, top5_q, allowed):
    if not cooc:
        return "NO_EVIDENCE"
    if any(top5_q.get(c, set()) & allowed for c in cooc):
        return "RETRIEVED_NOT_EXTRACTED"
    if any(top20_q.get(c, set()) & allowed for c in cooc):
        return "LOST_AT_RERANK"
    return "NOT_IN_BM25_TOP20"


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leakage-only", action="store_true",
                    help="Q1 and recovery check only: no retrieval replay, no model")
    args = ap.parse_args()
    mode = "leakage_only" if args.leakage_only else "full"
    out_dir = OUT_BASE / ("143_" + mode)
    if out_dir.exists():
        die("output directory exists, refusing to overwrite: " + str(out_dir))
    global OUT_DIR
    out_dir.mkdir(parents=True)
    OUT_DIR = out_dir
    t0 = time.time()
    report = {"script": Path(__file__).name, "mode": mode, "inputs": {},
              "rules": "R1-R6, see module docstring"}

    # ---------------------------------------------------------- inputs
    log("[inputs]")
    p, s = find_one(["output/**/run16E/index/chunks.jsonl"])
    report["inputs"]["chunks"] = [str(p), s]
    texts = [rec["text"] for rec in read_jsonl(p)]
    check("chunks", len(texts))
    low = [t.lower() for t in texts]

    p, s = find_one(["configs/descriptor_queries_v2.jsonl"], QUERIES_V2_SHA_PREFIX)
    report["inputs"]["queries_ext"] = [str(p), s]
    qext = read_jsonl(p)
    check("queries_ext", len(qext))
    p, s = find_one(["configs/descriptor_queries.jsonl"])
    report["inputs"]["queries_base"] = [str(p), s]
    qbase = read_jsonl(p)
    check("queries_base", len(qbase))

    base_keys = Counter(query_key(q) for q in qbase)
    ext_keys = Counter(query_key(q) for q in qext)
    for q in qext:
        q["in_base"] = query_key(q) in base_keys
    report["queries"] = {
        "ext_in_base": sum(q["in_base"] for q in qext),
        "base_missing_from_ext": sum((base_keys - ext_keys).values()),
        "ext_only_by_strategy": dict(Counter(q.get("strategy") for q in qext if not q["in_base"])),
    }
    log("  queries: " + json.dumps(report["queries"]))

    p, s = find_one(["configs/lb_reference_edges_original26.json"])
    report["inputs"]["bench26"] = [str(p), s]
    b26 = as_triples(load_records(p), p)
    check("bench26", len(b26))
    p, s = find_one(["configs/lb_reference_edges.json"])
    report["inputs"]["bench34"] = [str(p), s]
    b34 = as_triples(load_records(p), p)
    check("bench34", len(b34))

    p, s = find_one(["output/run16E/kg_final/flat_run16E_enforced_dedup.json"])
    report["inputs"]["graph"] = [str(p), s]
    g_recs = load_records(p)
    check("graph", len(g_recs))
    g_keys = {ekey(t) for t in as_triples(g_recs, p)}

    # -------------------------------------------------------- recovery (R4)
    log("[recovery]")

    def rec_exact(k):
        return k in g_keys

    def rec_substr(k):
        s_, r_, o_ = k
        return any(gr == r_ and (s_ in gs or gs in s_) and (o_ in go or go in o_)
                   for gs, gr, go in g_keys)

    matcher = None
    for name, fn in (("exact", rec_exact), ("substring", rec_substr)):
        n26 = sum(fn(ekey(e)) for e in b26)
        n34 = sum(fn(ekey(e)) for e in b34)
        log("  " + name + ": " + str(n26) + "/26, " + str(n34) + "/34")
        if n26 == EXPECTED["recovered26"] and n34 == EXPECTED["recovered34"]:
            matcher = (name, fn)
            break
    if matcher is None:
        die("recovery 13/26 and 18/34 not reproduced by either matcher", 2)
    report["recovery_matcher"] = matcher[0]

    edges = {}
    for bench_name, bench in (("26", b26), ("34", b34)):
        for e in bench:
            k = ekey(e)
            row = edges.setdefault(k, {"subject": k[0], "relation": e["r"], "object": k[2],
                                       "in26": False, "in34": False})
            row["in" + bench_name] = True
            row["recovered"] = matcher[1](k)

    # ----------------------------------------------------- co-occurrence (R1, R2)
    rx = {}
    for k in edges:
        for term in (k[0], k[2]):
            if term not in rx:
                rx[term] = term_regex(term)
    occ = {term: {i for i, t in enumerate(low) if r.search(t)} for term, r in rx.items()}
    for k, row in edges.items():
        row["cooc"] = occ[k[0]] & occ[k[2]]

    # ---------------------------------------------------------- targeting (R3)
    log("[targeting]")
    for q in qext:
        tx = query_text(q)
        q["targets"] = [k for k in edges if rx[k[0]].search(tx) and rx[k[2]].search(tx)]
    targeting = [i for i, q in enumerate(qext) if q["targets"]]
    t26 = sum(1 for k, r in edges.items() if r["in26"] and any(k in qext[i]["targets"] for i in targeting))
    t34 = sum(1 for k, r in edges.items() if r["in34"] and any(k in qext[i]["targets"] for i in targeting))
    reproduced = (t26 == EXPECTED["targeted26"] and t34 == EXPECTED["targeted34"]
                  and len(targeting) == EXPECTED["targeting_queries"])
    n_in_base = sum(qext[i]["in_base"] for i in targeting)
    if not reproduced:
        verdict = "WITHHELD: R3 does not reproduce 95m (17/26, 17/34, 24 queries)"
    elif n_in_base == 0:
        verdict = "NO targeting query in the base set: configuration A is untargeted"
    else:
        verdict = ("TARGETING PRESENT IN BASE SET (" + str(n_in_base) + " of "
                   + str(len(targeting)) + " queries): configuration A is also targeted")
    report["leakage"] = {
        "targeted_edges_26": t26, "targeted_edges_34": t34,
        "targeting_queries": len(targeting), "targeting_queries_in_base": n_in_base,
        "targeting_strategies": dict(Counter(qext[i].get("strategy") for i in targeting)),
        "reproduces_95m": reproduced, "verdict": verdict,
    }
    log("  " + json.dumps(report["leakage"]))

    # --------------------------------------------------- retrieval replay (R5)
    top20_q, top5_q = defaultdict(set), defaultdict(set)
    if not args.leakage_only:
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
        short = sum(1 for k in kept_per_q if len(k) < BM25_TOPN)
        log("  queries with fewer than 20 candidates after min score: " + str(short))

        ce_path = snapshot_download(RERANKER, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        report["reranker"] = {"name": RERANKER, "snapshot": ce_path, "device": device}
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
        for qi, kept in enumerate(kept_per_q):
            for ci in kept:
                top20_q[ci].add(qi)
            for _, ci in sorted(by_q[qi], key=lambda x: -x[0])[:TOPK]:
                top5_q[ci].add(qi)
        report["retrieval"] = {"queries_short_of_20": short,
                               "distinct_chunks_top20": len(top20_q),
                               "distinct_chunks_top5": len(top5_q)}

        all_ids = set(range(len(qext)))
        base_ids = {i for i, q in enumerate(qext) if q["in_base"]}
        for k, row in edges.items():
            own = {i for i in targeting if k in qext[i]["targets"]}
            row["evidence_in_top5_all"] = stage(row["cooc"], top20_q, top5_q, all_ids) == "RETRIEVED_NOT_EXTRACTED"
            if row["recovered"]:
                row["stage_all"] = row["stage_base"] = row["stage_without_own_targeting"] = "RECOVERED"
            else:
                row["stage_all"] = stage(row["cooc"], top20_q, top5_q, all_ids)
                row["stage_base"] = stage(row["cooc"], top20_q, top5_q, base_ids)
                row["stage_without_own_targeting"] = stage(row["cooc"], top20_q, top5_q, all_ids - own)

        stages = {}
        for bench in ("26", "34"):
            sel = [r for r in edges.values() if r["in" + bench]]
            stages[bench] = {col: dict(Counter(r[col] for r in sel))
                             for col in ("stage_all", "stage_base", "stage_without_own_targeting")}
            stages[bench]["recovered_without_evidence_in_top5"] = sum(
                1 for r in sel if r["recovered"] and not r["evidence_in_top5_all"])
        report["stages"] = stages
        log("  " + json.dumps(stages, indent=1))

        # ---------------------------------------------- replay fidelity (R6)
        get, chosen = passage_getter(g_recs)
        if get is None:
            report["fidelity"] = {"status": "NOT CHECKED: passage field not identified",
                                  "long_string_fields": [list(c) for c in chosen]}
        else:
            normchunks = [" ".join(t.split()) for t in low]
            top5_chunks = set(top5_q)
            n_pass = n_idx = n_top5 = 0
            for rec in g_recs:
                ptxt = get(rec)
                if not isinstance(ptxt, str) or len(ptxt) < 50:
                    continue
                n_pass += 1
                key = " ".join(ptxt.lower().split())[:200]
                hits = [i for i, c in enumerate(normchunks) if key in c]
                n_idx += bool(hits)
                n_top5 += any(h in top5_chunks for h in hits)
            report["fidelity"] = {"passage_field": list(chosen), "relations_with_passage": n_pass,
                                  "passage_found_in_index": n_idx,
                                  "passage_found_in_replayed_top5": n_top5}
        log("  fidelity: " + json.dumps(report["fidelity"]))

    # ------------------------------------------------------------- outputs
    cols = ["in26", "in34", "subject", "relation", "object", "recovered",
            "n_cooc_chunks", "n_targeting_queries", "n_targeting_in_base"]
    if not args.leakage_only:
        cols += ["evidence_in_top5_all", "stage_all", "stage_base", "stage_without_own_targeting"]
    with open(out_dir / "edges.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k, row in sorted(edges.items()):
            own = [i for i in targeting if k in qext[i]["targets"]]
            row["n_cooc_chunks"] = len(row["cooc"])
            row["n_targeting_queries"] = len(own)
            row["n_targeting_in_base"] = sum(qext[i]["in_base"] for i in own)
            w.writerow(row)
    with open(out_dir / "targeting_queries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "focus", "query", "in_base", "edges_targeted"])
        for i in targeting:
            q = qext[i]
            w.writerow([q.get("strategy"), q.get("focus"), q["query"], q["in_base"],
                        " | ".join(" ".join(k) for k in q["targets"])])
    report["runtime_s"] = round(time.time() - t0, 1)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    log("[done] " + str(out_dir))


if __name__ == "__main__":
    main()