#!/usr/bin/env python3
"""190 — Replay of the run18 retrieval (no LLM), to support Step 2 of the Method.

PURPOSE
    Replay exactly the retrieval of pipeline/02_extract_triples_run18.py
    (default BM25Okapi, tokens = lowercase text split on spaces, 20 candidates,
    cross-encoder reranking, 5 passages, threshold 2.0 on the BM25 score of the
    first passage after reranking) on the 210 questions, and measure:
      A  how many questions the 2.0 threshold dropped;
      B  how many of the 5 passages read come from the BM25 top 5, i.e. what the
         cross-encoder changes in what the extracting-LLM reads;
      C  the BM25 rank (1-20) of the passages that the cross-encoder moves up.

READ-ONLY
    No LLM; no existing file is modified. The script refuses to overwrite its output.

USAGE (from the repository root, on a GPU or CPU node)
    python 190_retrieval_replay_run18.py \
        --chunks  <run16E index>/chunks.jsonl \
        --queries configs/descriptor_queries_run18.jsonl \
        --out     output/run18/retrieval_replay_190.json
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

TOPN, TOPK, THR = 20, 5, 2.0
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit("ERROR: {} already exists. Refusing to overwrite.".format(out))

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8") if l.strip()]
    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    print("passages: {}   questions: {}".format(len(chunks), len(queries)))

    # identical to 02_extract_triples_run18.load_bm25
    bm25 = BM25Okapi([c.get("text", "").lower().split() for c in chunks])
    ce = CrossEncoder(RERANKER)

    rows, skipped, from_bm25_top5, promoted_ranks = [], [], [], Counter()
    for i, q in enumerate(queries):
        text = q.get("query") or q.get("question") or ""
        scores = bm25.get_scores(text.lower().split())
        top = scores.argsort()[-TOPN:][::-1]            # BM25 ranks 1..20
        bm25_rank = {int(j): r + 1 for r, j in enumerate(top)}
        ce_scores = ce.predict([(text, chunks[j].get("text", "")) for j in top])
        order = [int(top[k]) for k in sorted(range(len(top)),
                                             key=lambda k: -ce_scores[k])]
        best_bm25_after_rerank = float(scores[order[0]])
        gated = best_bm25_after_rerank < THR
        selected = order[:TOPK]
        n_in_top5 = sum(1 for j in selected if bm25_rank[j] <= TOPK)
        if gated:
            skipped.append({"index": i, "strategy": q.get("strategy"),
                            "query": text,
                            "best_bm25_after_rerank": round(best_bm25_after_rerank, 3)})
        else:
            from_bm25_top5.append(n_in_top5)
            for j in selected:
                if bm25_rank[j] > TOPK:
                    promoted_ranks[bm25_rank[j]] += 1
        rows.append({"index": i, "strategy": q.get("strategy"),
                     "best_bm25_after_rerank": round(best_bm25_after_rerank, 3),
                     "max_bm25": round(float(scores[top[0]]), 3),
                     "gated": gated,
                     "selected_bm25_ranks": [bm25_rank[j] for j in selected]})

    n = len(from_bm25_top5)
    dist = Counter(from_bm25_top5)
    res = {
        "parameters": {"bm25": "BM25Okapi defaults (k1=1.5, b=0.75)",
                       "tokens": "lower().split()", "top_n": TOPN, "top_k": TOPK,
                       "threshold": THR, "reranker": RERANKER},
        "A_questions_dropped_by_the_threshold": len(skipped),
        "A_detail": skipped,
        "B_questions_read": n,
        "B_passages_read_from_the_BM25_top5_mean": round(statistics.mean(from_bm25_top5), 2) if n else None,
        "B_distribution_0_to_5": {k: dist.get(k, 0) for k in range(6)},
        "B_questions_where_the_cross_encoder_changes_at_least_one_passage": sum(1 for x in from_bm25_top5 if x < TOPK),
        "C_BM25_ranks_of_the_passages_moved_up": dict(sorted(promoted_ranks.items())),
        "per_question": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\nA  questions dropped by the threshold {}: {}".format(THR, len(skipped)))
    for s in skipped:
        print("     [{}] {:.2f}  {}".format(s["strategy"], s["best_bm25_after_rerank"], s["query"][:90]))
    print("B  questions read: {}".format(n))
    print("   passages read from the BM25 top 5 (mean over 5): {}".format(res["B_passages_read_from_the_BM25_top5_mean"]))
    print("   distribution (number of BM25 top-5 passages among the 5 read): {}".format(res["B_distribution_0_to_5"]))
    print("   questions where the cross-encoder changes at least one passage: {}/{}".format(
        res["B_questions_where_the_cross_encoder_changes_at_least_one_passage"], n))
    print("C  BM25 ranks (6-20) of the passages moved up: {}".format(res["C_BM25_ranks_of_the_passages_moved_up"]))
    print("\nwritten: {}".format(out))


if __name__ == "__main__":
    main()