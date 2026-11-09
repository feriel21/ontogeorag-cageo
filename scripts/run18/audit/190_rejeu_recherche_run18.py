#!/usr/bin/env python3
"""190 — Rejeu de la recherche de run18 (sans LLM) pour argumenter la Step 2.

BUT
    Rejouer exactement la recherche de pipeline/02_extract_triples_run18.py
    (BM25Okapi par defaut, jetons = texte en minuscules coupe sur les espaces,
    20 candidats, re-classement cross-encoder, 5 passages, seuil 2.0 sur le
    score BM25 du premier passage apres re-classement) sur les 210 questions,
    et mesurer :
      A  combien de questions le seuil 2.0 a ecartees ;
      B  combien des 5 passages lus viennent du top 5 de BM25, c'est-a-dire
         ce que le cross-encoder change dans ce que lit l'extracting-LLM ;
      C  le rang BM25 (1-20) des passages que le cross-encoder fait remonter.

LECTURE SEULE
    Aucun LLM, aucun fichier existant modifie. Refuse d'ecraser la sortie.

USAGE (depuis la racine du depot, sur le noeud GPU ou CPU)
    python 190_rejeu_recherche_run18.py \
        --chunks  /home/talbi/ontogeorag_D/output/run16E/index/chunks.jsonl \
        --queries configs/descriptor_queries_run18.jsonl \
        --out     output/run18/rejeu_recherche_190.json
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
        sys.exit("ERREUR : {} existe deja. Refus d'ecraser.".format(out))

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8") if l.strip()]
    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    print("passages : {}   questions : {}".format(len(chunks), len(queries)))

    # identique a 02_extract_triples_run18.load_bm25
    bm25 = BM25Okapi([c.get("text", "").lower().split() for c in chunks])
    ce = CrossEncoder(RERANKER)

    rows, skipped, from_bm25_top5, promoted_ranks = [], [], [], Counter()
    for i, q in enumerate(queries):
        text = q.get("query") or q.get("question") or ""
        scores = bm25.get_scores(text.lower().split())
        top = scores.argsort()[-TOPN:][::-1]            # rangs BM25 1..20
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
        "parametres": {"bm25": "BM25Okapi defauts (k1=1.5, b=0.75)",
                       "tokens": "lower().split()", "top_n": TOPN, "top_k": TOPK,
                       "seuil": THR, "reranker": RERANKER},
        "A_questions_ecartees_par_le_seuil": len(skipped),
        "A_detail": skipped,
        "B_questions_lues": n,
        "B_passages_lus_venant_du_top5_BM25_moyenne": round(statistics.mean(from_bm25_top5), 2) if n else None,
        "B_distribution_0a5": {k: dist.get(k, 0) for k in range(6)},
        "B_questions_ou_le_cross_encoder_change_au_moins_un_passage": sum(1 for x in from_bm25_top5 if x < TOPK),
        "C_rangs_BM25_des_passages_remontes": dict(sorted(promoted_ranks.items())),
        "par_question": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\nA  questions ecartees par le seuil {} : {}".format(THR, len(skipped)))
    for s in skipped:
        print("     [{}] {:.2f}  {}".format(s["strategy"], s["best_bm25_after_rerank"], s["query"][:90]))
    print("B  questions lues : {}".format(n))
    print("   passages lus venant du top 5 BM25 (moyenne sur 5) : {}".format(res["B_passages_lus_venant_du_top5_BM25_moyenne"]))
    print("   distribution (nb de passages du top 5 BM25 parmi les 5 lus) : {}".format(res["B_distribution_0a5"]))
    print("   questions ou le cross-encoder change au moins un passage : {}/{}".format(
        res["B_questions_ou_le_cross_encoder_change_au_moins_un_passage"], n))
    print("C  rangs BM25 (6-20) des passages remontes : {}".format(res["C_rangs_BM25_des_passages_remontes"]))
    print("\nEcrit : {}".format(out))


if __name__ == "__main__":
    main()