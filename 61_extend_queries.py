#!/usr/bin/env python3
"""
61_extend_queries.py — Add descriptor-anchored queries to the query set.

OBSERVATION
    The query generator (pipeline/generate_augmented_queries.py, not published)
    defines four templates, all oriented from an entity to its descriptors
    ("What are the descriptors of {e}?"), and sets
    TYPE_STRATEGIES["Descriptor"] = set().
    As a result, none of the 249 base queries is anchored on a descriptor.
    Twenty vocabulary descriptors that occur in the corpus were targeted by no
    query and never entered the graph (for example erosional 162 passages in
    28 documents, deformed 141/28, thin 129/30, faulted 52/12, folded 50/10).

WHAT THE SCRIPT DOES
    The script adds one reverse template for the Descriptor type:
        "Which seismic objects or deposits are described as {d}?"
    One query is generated for each vocabulary descriptor found in the passages.
    Descriptors with zero occurrences are skipped and listed.
    The occurrence test is a literal whole-word match: a hyphenated descriptor
    written with a space in the corpus (variable amplitude) counts as absent.
    The new queries are appended to the existing set, which is not modified.
    Each added query carries strategy="descriptor_reverse" and origin="ext_v2",
    so the original set can be recovered by filtering.

NOTE
    The extension covers the whole vocabulary, independently of the benchmark.
    The extension is therefore distinct from the gapfill queries, which were
    written for reference edges found missing, and is declared separately.

Additive: neither generate_augmented_queries.py nor the existing set is modified.
Messages are in French.

Usage:
    cd ~/ontogeorag && export PYTHONPATH=$PWD
    python 61_extend_queries.py \
        --queries configs/descriptor_queries.jsonl \
        --chunks  output/run14/step1/chunks.jsonl \
        --out     configs/descriptor_queries_v2.jsonl
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

TEMPLATE_REVERSE = "Which seismic objects or deposits are described as {d}?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="configs/descriptor_queries.jsonl")
    ap.add_argument("--chunks", default="output/run14/step1/chunks.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chunks", type=int, default=1,
                    help="n minimal d'occurrences dans le corpus pour generer la requete")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"ERREUR : {out} existe deja. Refus d'ecraser.")

    from pipeline.rag.constants import KNOWN_DESCRIPTORS

    base = [json.loads(l) for l in open(a.queries, encoding="utf-8") if l.strip()]
    seen = {q["query"].strip().lower() for q in base}
    chunks = [json.loads(l) for l in open(a.chunks, encoding="utf-8")]

    added, skipped_dead, skipped_dup = [], [], []
    for d in sorted(KNOWN_DESCRIPTORS):
        pat = re.compile(r"\b" + re.escape(d) + r"\b", re.I)
        hits = [c for c in chunks if pat.search(c.get("text", ""))]
        if len(hits) < a.min_chunks:
            skipped_dead.append(d)
            continue
        q = TEMPLATE_REVERSE.format(d=d)
        if q.strip().lower() in seen:
            skipped_dup.append(d)
            continue
        docs = {c.get("doc_id") or c.get("source") or c.get("paper") for c in hits}
        added.append({"query": q, "strategy": "descriptor_reverse", "focus": d,
                      "focus_type": "Descriptor", "origin": "ext_v2",
                      "corpus_chunks": len(hits), "corpus_docs": len(docs)})

    merged = base + added
    with out.open("w", encoding="utf-8") as f:
        for q in merged:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    (out.with_suffix(".meta.json")).write_text(json.dumps({
        "produced_by": "61_extend_queries.py",
        "base_queries": len(base), "added": len(added), "total": len(merged),
        "template": TEMPLATE_REVERSE,
        "skipped_zero_occurrence": skipped_dead,
        "skipped_duplicate": skipped_dup,
        "sha256_short": sha,
        "note": ("Extension structurelle du generateur (asymetrie entite->descripteur). "
                 "Distincte du gapfill. Filtrer origin != 'ext_v2' restitue le jeu d'origine."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"jeu d'origine : {len(base)}")
    print(f"ajoutees      : {len(added)}  (descriptor_reverse)")
    print(f"exclus, 0 occurrence dans le corpus : {skipped_dead}")
    if skipped_dup:
        print(f"exclus, deja presents : {skipped_dup}")
    print(f"total         : {len(merged)}   sha {sha}")
    print(f"\nEcrit : {out}")
    print(f"Ecrit : {out.with_suffix('.meta.json')}")
    print("\nTop 10 des descripteurs nouvellement cibles (par volume corpus) :")
    for q in sorted(added, key=lambda x: -x["corpus_chunks"])[:10]:
        print(f"   {q['focus']:20s} {q['corpus_chunks']:4d} chunks / {q['corpus_docs']:2d} docs")


if __name__ == "__main__":
    main()