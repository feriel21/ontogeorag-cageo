#!/usr/bin/env python3
"""
61_extend_queries.py — Combler l'asymetrie du generateur de requetes

CONSTAT MESURE
    pipeline/generate_augmented_queries.py definit quatre templates, tous
    orientes ENTITE -> descripteur ("What are the descriptors of {e}?"), et
    fixe TYPE_STRATEGIES["Descriptor"] = set(). Consequence : aucune des 249
    requetes n'est ancree sur un descripteur. 20 descripteurs du lexique
    canonique presents dans le corpus (erosional 162 chunks / 28 docs,
    deformed 141/28, thin 129/30, faulted 52/12, folded 50/10...) ne sont
    cibles par aucune requete et n'entrent jamais dans le graphe.

CE QUE FAIT CE SCRIPT
    Ajoute UN template inverse, le seul geologiquement coherent pour le type
    Descriptor :
        "Which seismic objects or deposits are described as {d}?"
    genere pour chaque descripteur canonique ATTESTE dans le corpus (les
    descripteurs a zero occurrence sont exclus et listes), et fusionne avec
    le jeu existant sans le modifier.

    Chaque requete ajoutee porte strategy="descriptor_reverse" et
    origin="ext_v2", donc l'effet est mesurable a posteriori et le jeu
    d'origine est reconstituble par filtrage.

DECLARATION POUR LE MANUSCRIT
    Cette extension corrige une asymetrie STRUCTURELLE du generateur : elle
    couvre systematiquement le lexique, independamment du benchmark. Elle
    n'est PAS du gapfill (ajout de requetes visant des aretes de reference
    constatees manquantes) et doit etre declaree separement.

Additif : ne modifie ni generate_augmented_queries.py ni le jeu existant.

Usage :
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