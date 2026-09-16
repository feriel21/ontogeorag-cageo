#!/usr/bin/env python3
"""
96_descriptor_frequencies_run16E.py

Recalcule la frequence de chaque descripteur canonique DIRECTEMENT sur
l'index run16E (1758 chunks), pour remplacer la mesure faite par
erreur sur l'index run14 (2105 chunks). Lit KNOWN_DESCRIPTORS depuis
constants.py par import reel plutot que de le recopier a la main.

Produit trois listes traçables :
  - descripteurs absents du corpus (0 occurrence)
  - descripteurs attestes mais absents du graphe livre
  - descripteurs presents dans le graphe livre

et signale explicitement si le total croise (40 = absents + attestes-
non-extraits + extraits + ...) ne reboucle pas, plutot que de laisser
un ecart silencieux.

N'ecrit rien. Python 3.10.

Usage :
    python3 96_descriptor_frequencies_run16E.py
"""
import argparse
import importlib.util
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


def charger_constants(chemin: Path):
    spec = importlib.util.spec_from_file_location("cible_verif", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def charger_textes(chemin: Path) -> list[str]:
    textes = []
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        for ligne in fh:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                d = json.loads(ligne)
            except Exception:
                continue
            t = d.get("text") or d.get("content") or d.get("chunk") or ""
            if isinstance(t, str):
                textes.append(t.lower())
    return textes


def champ(d: dict, *candidats):
    for c in candidats:
        if c in d and d[c] not in (None, ""):
            return d[c]
    return None


def charger_graphe(chemin: Path) -> set:
    obj = json.loads(chemin.read_text(encoding="utf-8", errors="replace"))
    triples = obj["triples"] if isinstance(obj, dict) and "triples" in obj else obj
    dans_graphe = set()
    for t in triples:
        if not isinstance(t, dict):
            continue
        r = champ(t, "relation", "predicate")
        o = champ(t, "object", "tail")
        if r == "hasDescriptor" and isinstance(o, str):
            dans_graphe.add(o.strip().lower())
    return dans_graphe


def motif_terme(terme: str) -> re.Pattern:
    return re.compile(r"(?<![a-z])" + re.escape(terme.lower()) + r"(?![a-z])", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Recalcule les frequences de descripteurs sur run16E.")
    ap.add_argument("--racine", action="append", default=None,
                    help="racine a explorer (repetable). Defaut : ~/ontogeorag et ~/ontogeorag_D")
    ap.add_argument("--constants", default=None, help="chemin vers constants.py")
    args = ap.parse_args()

    racines = [Path(r).expanduser() for r in (args.racine or ["~/ontogeorag", "~/ontogeorag_D"])]

    chemin_constants = Path(args.constants).expanduser() if args.constants else \
        localiser(racines, ["**/pipeline/rag/constants.py"])
    chemin_chunks = localiser(racines, ["**/run16E/index/chunks.jsonl"])
    chemin_graphe = localiser(racines, ["**/run16E/kg_final/flat_run16E_enforced_dedup.json"])

    print("constants.py  : " + (str(chemin_constants) if chemin_constants else "ABSENT"))
    print("chunks run16E : " + (str(chemin_chunks) if chemin_chunks else "ABSENT"))
    print("graphe livre  : " + (str(chemin_graphe) if chemin_graphe else "ABSENT"))
    print("=" * 78)

    if not (chemin_constants and chemin_chunks and chemin_graphe):
        print("")
        print("Un ou plusieurs fichiers necessaires sont introuvables. Arret.")
        return 1

    try:
        c = charger_constants(chemin_constants)
    except Exception as err:
        print("ECHEC DE L'IMPORT de constants.py : " + type(err).__name__ + " : " + str(err),
              file=sys.stderr)
        return 1

    if not hasattr(c, "KNOWN_DESCRIPTORS"):
        print("KNOWN_DESCRIPTORS introuvable dans " + str(chemin_constants), file=sys.stderr)
        return 1

    descripteurs = sorted(c.KNOWN_DESCRIPTORS)
    textes = charger_textes(chemin_chunks)
    dans_graphe = charger_graphe(chemin_graphe)

    print("")
    print(str(len(descripteurs)) + " descripteurs dans KNOWN_DESCRIPTORS, "
          + str(len(textes)) + " chunks run16E, "
          + str(len(dans_graphe)) + " descripteurs distincts dans le graphe livre.")
    print("-" * 78)

    absents, attestes_non_extraits, extraits = [], [], []

    for d in descripteurs:
        m = motif_terme(d)
        n_occ = sum(len(m.findall(t)) for t in textes)
        n_doc = sum(1 for t in textes if m.search(t))
        if n_occ == 0:
            absents.append(d)
        elif d not in dans_graphe:
            attestes_non_extraits.append((d, n_occ, n_doc))
        else:
            extraits.append((d, n_occ, n_doc))

    attestes_non_extraits.sort(key=lambda x: -x[1])
    extraits.sort(key=lambda x: -x[1])

    print("")
    print("ABSENTS DU CORPUS (0 occurrence, toute forme flechie non testee ici) : "
          + str(len(absents)))
    for d in absents:
        print("  - " + d)

    print("")
    print("ATTESTES MAIS JAMAIS EXTRAITS : " + str(len(attestes_non_extraits)))
    for d, n, k in attestes_non_extraits:
        print("  - " + d + "  (" + str(n) + " occ / " + str(k) + " chunks)")

    print("")
    print("EXTRAITS DANS LE GRAPHE LIVRE : " + str(len(extraits)))
    for d, n, k in extraits:
        print("  - " + d + "  (" + str(n) + " occ / " + str(k) + " chunks)")

    total = len(absents) + len(attestes_non_extraits) + len(extraits)
    print("")
    print("=" * 78)
    print(str(len(absents)) + " + " + str(len(attestes_non_extraits)) + " + "
          + str(len(extraits)) + " = " + str(total)
          + ("  -- reboucle sur " + str(len(descripteurs))
             if total == len(descripteurs)
             else "  !! NE REBOUCLE PAS sur " + str(len(descripteurs)) + " -- verifier a la main"))
    return 0


if __name__ == "__main__":
    sys.exit(main())