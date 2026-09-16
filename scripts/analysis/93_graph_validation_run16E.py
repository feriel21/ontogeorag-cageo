#!/usr/bin/env python3
"""
93_graph_validation_run16E.py

Recalcule les proprietes structurelles du graphe LIVRE (run16E), sans
dependre d'un rapport de validation qui peut appartenir a la mauvaise
run. §5.9 du manuscrit citait output/run16/validation/... --- ce
script lit directement flat_run16E_enforced_dedup.json et recalcule
tout depuis les triples, pour ne plus dependre d'un fichier de rapport
dont on ne peut pas garantir la provenance.

Deux limites honnetes, marquees explicitement dans la sortie :

1. Le compte "concepts approximatifs" du manuscrit (144, donnant
   33,3 % de redondance) sort d'une deduplication semantique par
   embeddings SciBERT --- la meme technique que la canonicalisation
   d'entites (Section extraction). Ce script n'a pas acces au modele
   d'embeddings ; il fournit a la place un regroupement heuristique
   par normalisation de chaine (casse, pluriel simple, ponctuation),
   qui SOUS-ESTIME la redondance reelle (il ne fusionne pas les
   synonymes non lexicaux) et ne doit pas remplacer le chiffre
   SciBERT s'il est retrouvable ailleurs. Les deux chiffres sont
   affiches separement et etiquetes.

2. Le detail "pore pressure : 5, 11 ou 3 formes selon la cle" du
   manuscrit vient d'un rapport de regroupement automatique
   (graph_validation_report.json) dont la structure exacte n'est pas
   supposee ici. Ce script recompte les formes de surface contenant
   "pressure" directement dans les entites du graphe livre, ce qui
   donne un chiffre traçable independamment de ce rapport.

N'ecrit rien. Python 3.10.

Usage :
    python3 93_graph_validation_run16E.py
    python3 93_graph_validation_run16E.py --graphe chemin/vers/flat.json
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def localiser_graphe(racines: list[Path]) -> Path | None:
    motifs = ["**/run16E/kg_final/flat_run16E_enforced_dedup.json"]
    for racine in racines:
        if not racine.is_dir():
            continue
        for motif in motifs:
            trouves = sorted(racine.glob(motif))
            if trouves:
                return trouves[0]
    return None


def charger_triples(chemin: Path) -> list[dict]:
    obj = json.loads(chemin.read_text(encoding="utf-8", errors="replace"))
    if isinstance(obj, dict) and "triples" in obj:
        return [t for t in obj["triples"] if isinstance(t, dict)]
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)]
    raise ValueError("structure de fichier non reconnue (ni {'triples': [...]} ni [...])")


def entites(triples: list[dict]) -> set:
    vus = set()
    for t in triples:
        for champ in ("subject", "object"):
            v = t.get(champ)
            if isinstance(v, str) and v.strip():
                vus.add(v.strip().lower())
    return vus


def cle_normalisee(nom: str) -> str:
    """Normalisation heuristique de surface : casse, ponctuation, pluriel
    simple en -s/-es. Ne fusionne PAS les synonymes non lexicaux (ex.
    'overpressure' et 'excess pore pressure' restent distincts) ---
    c'est la limite assumee de cette approche, cf. docstring du module."""
    n = nom.lower().strip()
    n = re.sub(r"[-_]", " ", n)
    n = re.sub(r"[^a-z0-9 ]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    mots = n.split()
    if mots and mots[-1].endswith("es") and len(mots[-1]) > 4:
        mots[-1] = mots[-1][:-2]
    elif mots and mots[-1].endswith("s") and not mots[-1].endswith("ss") and len(mots[-1]) > 3:
        mots[-1] = mots[-1][:-1]
    return " ".join(mots)


def regrouper_par_cle(ents: set) -> dict[str, list[str]]:
    groupes: dict[str, list[str]] = defaultdict(list)
    for e in ents:
        groupes[cle_normalisee(e)].append(e)
    return groupes


def connectivite(triples: list[dict], ents: set, ancre_motifs=("mass transport deposit", "mass-transport deposit")):
    voisins: dict[str, set] = defaultdict(set)
    for t in triples:
        s, o = t.get("subject"), t.get("object")
        if isinstance(s, str) and isinstance(o, str):
            s, o = s.strip().lower(), o.strip().lower()
            voisins[s].add(o)
            voisins[o].add(s)
    departs = [e for e in ents if any(m in e for m in ancre_motifs)]
    if not departs:
        return None, [], ents
    vus = set(departs)
    pile = list(departs)
    while pile:
        cur = pile.pop()
        for v in voisins.get(cur, ()):
            if v not in vus:
                vus.add(v)
                pile.append(v)
    isoles = ents - (vus & ents)
    return len(isoles), sorted(departs), isoles


def formes_pore_pressure(ents: set) -> list[str]:
    return sorted(e for e in ents if "pressure" in e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Recalcule les proprietes structurelles du graphe run16E.")
    ap.add_argument("--racine", action="append", default=None,
                    help="racine a explorer (repetable). Defaut : ~/ontogeorag et ~/ontogeorag_D")
    ap.add_argument("--graphe", default=None,
                    help="chemin direct vers flat_run16E_enforced_dedup.json, si connu")
    args = ap.parse_args()

    if args.graphe:
        chemin = Path(args.graphe).expanduser()
    else:
        racines = [Path(r).expanduser() for r in (args.racine or ["~/ontogeorag", "~/ontogeorag_D"])]
        chemin = localiser_graphe(racines)

    if chemin is None or not chemin.is_file():
        print("ABSENT : impossible de localiser flat_run16E_enforced_dedup.json", file=sys.stderr)
        return 1

    print("Graphe : " + str(chemin))
    print("=" * 78)

    try:
        triples = charger_triples(chemin)
    except Exception as err:
        print("ECHEC DE LECTURE : " + type(err).__name__ + " : " + str(err), file=sys.stderr)
        return 1

    ents = entites(triples)
    print("")
    print("relations : " + str(len(triples)))
    print("entites distinctes (brutes, minuscules) : " + str(len(ents)))

    # --- redondance heuristique par cle normalisee -----------------------
    groupes = regrouper_par_cle(ents)
    n_concepts_heuristique = len(groupes)
    redondance_heuristique = 100 * (1 - n_concepts_heuristique / len(ents)) if ents else 0
    print("")
    print("--- Redondance de surface : HEURISTIQUE (normalisation de chaine) ---")
    print("[ATTENTION : sous-estime la redondance reelle -- ne fusionne pas les")
    print(" synonymes non lexicaux. Ne remplace pas un chiffre SciBERT s'il")
    print(" existe. Voir docstring du module.]")
    print("concepts apres regroupement heuristique : " + str(n_concepts_heuristique))
    print("redondance heuristique : " + str(round(redondance_heuristique, 1)) + " %")
    multi = sorted((g for g in groupes.values() if len(g) > 1), key=lambda g: -len(g))
    if multi:
        print("groupes non triviaux (" + str(len(multi)) + " au total), les 10 plus grands :")
        for g in multi[:10]:
            print("  " + str(len(g)) + "x : " + ", ".join(sorted(g)))

    # --- connectivite ------------------------------------------------------
    n_isoles, departs, isoles = connectivite(triples, ents)
    print("")
    print("--- Connectivite ---")
    if n_isoles is None:
        print("aucune entite 'mass transport deposit' trouvee -- non calculable")
    else:
        print("point(s) de depart : " + ", ".join(departs))
        print("entites sans chemin vers 'mass transport deposit' : " + str(n_isoles)
              + " / " + str(len(ents)))
        if isoles:
            echantillon = sorted(isoles)[:15]
            print("echantillon (15 max) : " + ", ".join(echantillon))

    # --- formes de surface autour de "pressure" -----------------------------
    formes = formes_pore_pressure(ents)
    print("")
    print("--- Formes de surface contenant 'pressure' (recompte direct) ---")
    print("nombre de formes distinctes : " + str(len(formes)))
    for f in formes:
        print("  - " + f)

    print("")
    print("=" * 78)
    print("Chiffres du manuscrit a comparer a ce qui precede :")
    print("  33,3 % de redondance / 144 concepts (source : mauvaise run, run16")
    print("  au lieu de run16E) -- comparer au chiffre heuristique ci-dessus,")
    print("  qui n'est PAS un remplacement direct (sous-estimation attendue).")
    print("  67 entites isolees -- deja recalcule et corrige a 61 en session")
    print("  precedente ; ce script devrait reproduire 61 s'il lit le meme")
    print("  fichier.")
    print("  pore pressure : 5, 11 ou 3 formes selon la cle du rapport --")
    print("  comparer a la liste ci-dessus, qui vient directement des entites")
    print("  du graphe livre plutot que d'un rapport de regroupement tiers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())