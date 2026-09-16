# pipeline/rag/hybrid_retriever.py
"""
Hybrid retriever: BM25 + dense bi-encoder + CrossEncoder reranker.
Drop-in replacement for load_bm25() in 02_extract_triples.py.

Usage:
    from pipeline.rag.hybrid_retriever import load_hybrid_retriever
    retrieve = load_hybrid_retriever(index_dir, dense_model, reranker_model)
    chunks = retrieve(query, top_n=5)  # same interface as load_bm25()
"""

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def load_hybrid_retriever(
    index_dir: str,
    dense_model_name: str = "BAAI/bge-small-en-v1.5",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    fusion_alpha: float = 0.5,
    bm25_topk: int = 50,
    dense_topk: int = 50,
    device: str = "cuda",
) -> callable:
    """
    Returns retrieve(query, top_n) -> list[dict]
    Identical interface to load_bm25() so 02_extract_triples.py
    needs only one line changed.

    fusion_alpha: 0.0 = BM25 only, 1.0 = dense only, 0.5 = equal mix
    """

    index_dir = Path(index_dir)

    # ── Load chunks ───────────────────────────────────────────────────
    chunks = []
    with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"  [Hybrid] Loaded {len(chunks)} chunks")

    # ── Build BM25 index ──────────────────────────────────────────────
    corpus = [c.get("text", "").lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    print(f"  [Hybrid] BM25 index built")

    # ── Load dense embeddings (precomputed by 01_build_index.py) ──────
    emb_path = index_dir / "dense_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"dense_embeddings.npy not found in {index_dir}.\n"
            f"Run: python pipeline/01_build_index.py --dense ..."
        )
    embeddings = np.load(emb_path)  # shape [N, D], float32, normalized
    print(f"  [Hybrid] Dense embeddings loaded: {embeddings.shape}")

    # ── Load dense encoder (for query encoding only) ──────────────────
    dense_model = SentenceTransformer(dense_model_name, device=device)
    print(f"  [Hybrid] Dense encoder loaded: {dense_model_name}")

    # ── Load CrossEncoder reranker ────────────────────────────────────
    reranker = CrossEncoder(reranker_model)
    print(f"  [Hybrid] CrossEncoder loaded: {reranker_model}")

    def retrieve(query: str, top_n: int = 5) -> list[dict]:
        """
        1. BM25 top-50 scored candidates
        2. Dense top-50 scored candidates
        3. Reciprocal Rank Fusion → union candidate set
        4. CrossEncoder rerank → top_n
        """

        # ── BM25 scores ───────────────────────────────────────────────
        tokens = query.lower().split()
        bm25_scores = bm25.get_scores(tokens)
        bm25_top_idx = bm25_scores.argsort()[-bm25_topk:][::-1]
        bm25_map = {int(i): float(bm25_scores[i]) for i in bm25_top_idx}

        # ── Dense scores ──────────────────────────────────────────────
        q_emb = dense_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        cos_scores = (embeddings @ q_emb.T).squeeze()
        dense_top_idx = cos_scores.argsort()[-dense_topk:][::-1]
        dense_map = {int(i): float(cos_scores[i]) for i in dense_top_idx}

        # ── Reciprocal Rank Fusion ────────────────────────────────────
        candidate_ids = set(bm25_map) | set(dense_map)

        bm25_ranked = sorted(bm25_map, key=bm25_map.get, reverse=True)
        dense_ranked = sorted(dense_map, key=dense_map.get, reverse=True)

        fused = {}
        for cid in candidate_ids:
            bm25_rank = (
                bm25_ranked.index(cid) + 1
                if cid in bm25_map
                else bm25_topk + 1
            )
            dense_rank = (
                dense_ranked.index(cid) + 1
                if cid in dense_map
                else dense_topk + 1
            )
            fused[cid] = (1 - fusion_alpha) * (
                1 / (60 + bm25_rank)
            ) + fusion_alpha * (1 / (60 + dense_rank))

        # Top-100 by fused score → CrossEncoder rerank
        top_fused = sorted(fused, key=fused.get, reverse=True)[:100]

        # ── CrossEncoder rerank (batched) ─────────────────────────────
        pairs = [(query, chunks[i]["text"]) for i in top_fused]
        rerank_scores = reranker.predict(pairs, batch_size=32)

        # Build final result list
        results = []
        for idx, cid in enumerate(top_fused):
            c = chunks[cid]
            results.append(
                {
                    "chunk_id": c.get("chunk_id", f"chunk_{cid}"),
                    "text": c.get("text", ""),
                    "score": fused[cid],  # fused score (for gating)
                    "rerank_score": float(rerank_scores[idx]),
                    "bm25_score": bm25_map.get(cid, 0.0),
                    "dense_score": dense_map.get(cid, 0.0),
                    "source_file": c.get(
                        "source_file", c.get("doc_id", "unknown")
                    ),
                }
            )

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_n]

    return retrieve
