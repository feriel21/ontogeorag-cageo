#!/usr/bin/env python3
"""
94_failure_modes.py

Classe les aretes du benchmark 34 non recuperees par le graphe livre,
selon le modele a trois classes de la Section 5.3 :

  1. absence totale de preuve dans le corpus
  2. preuve presente mais non retrouvee par le retrieval
  3. preuve retrouvee mais non extraite

LIMITE ASSUMEE ET EXPLICITE : ce script ne peut trancher qu'entre la
classe 1 et les classes (2+3) reunies. Distinguer 2 de 3 exige de
rejouer le retrieval reel (BM25 + cross-encoder) pour chaque arete et
de comparer aux passages effectivement recuperes -- ce qui suppose
d'importer le pipeline de retrieval du depot, non reproduit ici pour
ne pas deviner son interface. Ce script calcule donc :

  - classe 1 (aucune preuve dans le corpus) : calculee par co-occurrence
    des deux arguments dans un meme chunk, sur l'index run16E complet
    (1758 chunks), jamais tronque. LIMITE : la recherche utilise une
    correspondance exacte au mot pres (limites de mot), donc un pluriel
    ("turbidity currents") ne correspond pas a son singulier
    ("turbidity current") dans l'argument du benchmark. Ceci peut faire
    apparaitre a tort en classe 1 une arete dont le corpus discute bien
    les deux arguments sous une forme flechie differente. Chaque arete
    classee en classe 1 est donc a relire a l'oeil avant d'etre citee
    comme telle --- ce n'est pas une garantie automatique, c'est un tri
    initial a verifier.
  - classes 2+3 reunies (preuve existe quelque part dans le corpus) :
    fiable au meme titre.
  - repartition 2 vs 3 : NON CALCULEE ICI. Le script indique la
    marche a suivre pour la completer avec le retrieval reel.

N'ecrit rien. Python 3.10.

Usage :
    python3 94_failure_modes.py
    python3 94_failure_modes.py --racine ~/ontogeorag --racine ~/ontogeorag_D
"""
import argparse
import json
import re
import sys
from pathlib import Path


def localiser(racines: list[Path], motifs: list[str]) -> Path | None:
    for racine in racines:
        if not racine.is_dir():
            continue
        for motif in motifs:
            trouves = sorted(racine.glob(motif))
            if trouves:
                return trouves[0]
    return None


def charger_jsonl(chemin: Path) -> list[dict]:
    lignes = []
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        for brute in fh:
            brute = brute.strip()
            if not brute:
                continue
            try:
                lignes.append(json.loads(brute))
            except Exception:
                continue
    return lignes


def champ(d: dict, *candidats):
    for c in candidats:
        if c in d and d[c] not in (None, ""):
            return d[c]
    minus = {k.lower(): v for k, v in d.items()}
    for c in candidats:
        if c.lower() in minus and minus[c.lower()] not in (None, ""):
            return minus[c.lower()]
    return None


def extraire_aretes(obj) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for cle in ("edges", "relations", "benchmark", "items"):
            if cle in obj and isinstance(obj[cle], list):
                return [x for x in obj[cle] if isinstance(x, dict)]
    return []


def extraire_triples_livres(obj) -> list[dict]:
    if isinstance(obj, dict) and "triples" in obj:
        return [t for t in obj["triples"] if isinstance(t, dict)]
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)]
    return []


def normaliser(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def edge_key(s: str, r: str, o: str) -> tuple:
    return (normaliser(s), normaliser(r), normaliser(o))


def motif_terme(terme: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + re.escape(terme.lower()) + r"(?![a-z])", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Classe les aretes non recuperees du benchmark 34.")
    ap.add_argument("--racine", action="append", default=None,
                    help="racine a explorer (repetable). Defaut : ~/ontogeorag et ~/ontogeorag_D")
    args = ap.parse_args()
    racines = [Path(r).expanduser() for r in (args.racine or ["~/ontogeorag", "~/ontogeorag_D"])]

    bench_path = localiser(racines, ["**/lb_reference_edges.json"])
    graphe_path = localiser(racines, ["**/run16E/kg_final/flat_run16E_enforced_dedup.json"])
    chunks_path = localiser(racines, ["**/run16E/index/chunks.jsonl"])

    print("benchmark 34  : " + (str(bench_path) if bench_path else "ABSENT"))
    print("graphe livre  : " + (str(graphe_path) if graphe_path else "ABSENT"))
    print("chunks run16E : " + (str(chunks_path) if chunks_path else "ABSENT"))
    print("=" * 78)

    if not (bench_path and graphe_path and chunks_path):
        print("")
        print("Un ou plusieurs fichiers necessaires sont introuvables. Arret.")
        return 1

    aretes = extraire_aretes(json.loads(bench_path.read_text(encoding="utf-8", errors="replace")))
    triples = extraire_triples_livres(json.loads(graphe_path.read_text(encoding="utf-8", errors="replace")))
    chunks = charger_jsonl(chunks_path)

    textes = []
    for c in chunks:
        t = champ(c, "text", "content", "chunk", "body")
        if isinstance(t, str):
            textes.append(t.lower())

    livrees = set()
    for t in triples:
        s = champ(t, "subject", "head", "s")
        r = champ(t, "relation", "predicate", "rel", "p")
        o = champ(t, "object", "tail", "o")
        if isinstance(s, str) and isinstance(r, str) and isinstance(o, str):
            livrees.add(edge_key(s, r, o))

    print("")
    print(str(len(aretes)) + " aretes dans le benchmark, "
          + str(len(livrees)) + " relations distinctes dans le graphe livre, "
          + str(len(textes)) + " chunks indexes.")

    non_recuperees = []
    for e in aretes:
        s = champ(e, "subject", "head", "s")
        r = champ(e, "relation", "predicate", "rel", "p")
        o = champ(e, "object", "tail", "o")
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(o, str)):
            continue
        if edge_key(s, r, o) not in livrees:
            non_recuperees.append((s, r, o))

    print(str(len(non_recuperees)) + " arete(s) du benchmark non retrouvee(s) telle(s) quelle(s)"
          + " dans le graphe livre.")
    print("(le manuscrit annonce 16 sur 34 -- si ce compte differe, verifier la")
    print(" methode d'appariement : substring vs egalite stricte, cf. Section 4.)")
    print("-" * 78)

    classe_1, classe_23 = [], []
    for s, r, o in non_recuperees:
        m_s, m_o = motif_terme(s), motif_terme(o)
        n_chunks_les_deux = sum(1 for t in textes if m_s.search(t) and m_o.search(t))
        entree = (s, r, o, n_chunks_les_deux)
        if n_chunks_les_deux == 0:
            classe_1.append(entree)
        else:
            classe_23.append(entree)

    print("")
    print("CLASSE 1 -- aucune preuve dans le corpus (" + str(len(classe_1))
          + ") -- A RELIRE A L'OEIL (voir limite plurielle dans le docstring) :")
    for s, r, o, n in classe_1:
        print("  " + s + " --[" + r + "]--> " + o)

    print("")
    print("CLASSES 2+3 reunies -- preuve existe quelque part dans le corpus ("
          + str(len(classe_23)) + ") :")
    for s, r, o, n in sorted(classe_23, key=lambda x: -x[3]):
        print("  " + s + " --[" + r + "]--> " + o + "   (" + str(n) + " chunks co-occurrents)")

    print("")
    print("=" * 78)
    print("Ce script s'arrete ici : il ne peut pas dire, parmi les "
          + str(len(classe_23)) + " aretes de la classe 2+3,")
    print("combien ont ete recuperees par le retrieval (classe 3, extraction")
    print("manquee) contre combien ne l'ont pas ete (classe 2, retrieval manque).")
    print("")
    print("Pour completer cette repartition, rejouer pour chacune des "
          + str(len(classe_23)) + " aretes")
    print("ci-dessus la requete BM25 + cross-encoder du pipeline reel, et verifier")
    print("si un chunk contenant les deux arguments figure parmi les passages")
    print("effectivement retournes a l'extracteur. Exemple d'ossature (a adapter")
    print("aux fonctions reelles du depot, non devinees ici) :")
    print("")
    print("    from pipeline.rag.retrieval import bm25_retrieve, rerank")
    print("    for s, r, o, n in classe_23:")
    print("        query = construire_requete(s, r, o)  # meme fonction que 01_generate_queries.py")
    print("        candidats = bm25_retrieve(query, k=20)")
    print("        retenus = rerank(query, candidats, k=5)")
    print("        a_les_deux = [c for c in retenus if s.lower() in c.lower() and o.lower() in c.lower()]")
    print("        print(s, o, 'retrouve' if a_les_deux else 'non retrouve')")
    return 0


if __name__ == "__main__":
    sys.exit(main())