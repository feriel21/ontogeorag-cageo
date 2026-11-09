#!/usr/bin/env python3
"""182 — Familles de noms et tables de l'abstract, sur le graphe livre de run18.

DEUX ETAPES
  1. proposer : chaque entite du graphe recoit une famille.
       - « table »     : le nom est deja dans configs/entity_families.csv (table faite
                         a la main sur run16E) ;
       - « proposee »  : nom absent de la table ; famille proposee si au moins la
                         moitie de ses mots porteurs sont partages avec les noms d'une
                         famille (une seule famille gagnante) ; A VALIDER A LA MAIN ;
       - « aucune »    : rien de proche.
     Ecrit <outdir>/familles_run18_a_valider.csv (entity, family, methode, occurrences).
     Tu corriges la colonne family (vide = pas de famille), puis tu l'enregistres sous
     configs/entity_families_run18.csv.

  2. mesurer : avec la table validee, calcule ce que cite l'abstract :
       - Table 1 : par famille, le nombre de noms et les noms ;
       - composantes du graphe entier avant / apres regroupement des familles ;
       - composantes du sous-graphe causal (causes, triggers, controls, affects)
         avant / apres regroupement ;
       - voisinage de « slope failure » : noms distincts relies par une relation
         causale a un membre de la famille slope failure, avant / apres ;
       - Table 2 : relations Tier 1 « X triggers <slope failure> » avec leur phrase
         (tableau a lire pour reperer les conditions perdues).

USAGE
  python 182_familles_noms_run18.py proposer --repo . \
      --graph output/run18/kg_final/flat_run18_enforced_dedup.json \
      --outdir output/audit/182_familles
  python 182_familles_noms_run18.py mesurer --repo . \
      --graph output/run18/kg_final/flat_run18_enforced_dedup.json \
      --families configs/entity_families_run18.csv \
      --outdir output/audit/182_mesures
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CAUSAL = {"causes", "triggers", "controls", "affects"}
STOP = {"of", "the", "in", "on", "a", "an", "and", "to", "at", "by", "for", "with", "from",
        "its", "their", "high", "higher", "low", "lower", "increase", "increased", "increasing",
        "decrease", "decreased", "reduction", "development", "generation", "new"}


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def low(x):
    return " ".join(str(x or "").lower().split())


def words(x):
    w = re.findall(r"[a-z]+", low(x).replace("-", " "))
    return {re.sub(r"(es|s)$", "", t) for t in w if t not in STOP and len(t) > 2}


def load_graph(p):
    raw = json.load(open(p, encoding="utf-8"))
    return raw.get("triples", raw) if isinstance(raw, dict) else raw


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(n) for n in nodes})


def proposer(args):
    G = load_graph(args.graph)
    table = {low(r["entity"]): r["family"] for r in csv.DictReader(open(Path(args.repo) / "configs/entity_families.csv", encoding="utf-8"))}
    fam_words = defaultdict(set)
    for e, f in table.items():
        fam_words[f] |= words(e)
    occ = Counter()
    for t in G:
        occ[low(t["subject"])] += 1
        occ[low(t["object"])] += 1
    rows = []
    for e, n in sorted(occ.items(), key=lambda x: (-x[1], x[0])):
        if e in table:
            rows.append([e, table[e], "table", n])
            continue
        w = words(e)
        best = []
        for f, fw in fam_words.items():
            share = len(w & fw) / len(w) if w else 0
            if share >= 0.5:
                best.append((share, f))
        best.sort(reverse=True)
        if best and (len(best) == 1 or best[0][0] > best[1][0]):
            rows.append([e, best[0][1], "proposee", n])
        else:
            rows.append([e, "", "aucune", n])
    out = Path(args.outdir)
    out.mkdir(parents=True)
    with open(out / "familles_run18_a_valider.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entity", "family", "methode", "occurrences"])
        w.writerows(rows)
    c = Counter(r[2] for r in rows)
    print("entites du graphe : %d ; dans la table run16E : %d ; famille proposee : %d ; aucune : %d"
          % (len(rows), c["table"], c["proposee"], c["aucune"]))
    print("a valider : " + str(out / "familles_run18_a_valider.csv"))
    print("ensuite : corriger la colonne family, enregistrer sous configs/entity_families_run18.csv, lancer « mesurer »")


def mesurer(args):
    G = load_graph(args.graph)
    fam = {}
    for r in csv.DictReader(open(args.families, encoding="utf-8")):
        if (r.get("family") or "").strip():
            fam[low(r["entity"])] = r["family"].strip()
    g = lambda e: fam.get(low(e), low(e))
    out = Path(args.outdir)
    out.mkdir(parents=True)

    members = defaultdict(set)
    for t in G:
        for e in (t["subject"], t["object"]):
            if low(e) in fam:
                members[fam[low(e)]].add(low(e))

    nodes = {low(t[k]) for t in G for k in ("subject", "object")}
    E = [(low(t["subject"]), low(t["object"])) for t in G]
    c_before = components(nodes, E)
    c_after = components({g(n) for n in nodes}, [(g(a), g(b)) for a, b in E])
    Ec = [(low(t["subject"]), low(t["object"])) for t in G if t["relation"] in CAUSAL]
    nc = {x for e in Ec for x in e}
    cc_before = components(nc, Ec)
    cc_after = components({g(n) for n in nc}, [(g(a), g(b)) for a, b in Ec])

    sf = {n for n in nodes if g(n) == "slope failure"}
    neigh = {x for a, b in Ec for x in (a, b) if (a in sf or b in sf)} - sf
    neigh_fam = {g(x) for x in neigh}

    t2 = [t for t in G if t["relation"] == "triggers" and g(t["object"]) == "slope failure" and t.get("tier") == 1]
    with open(out / "table2_triggers_slope_failure.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["relation", "tier", "phrase"])
        for t in t2:
            w.writerow(["%s triggers %s" % (t["subject"], t["object"]), t.get("tier"), (t.get("evidence") or "").strip('"')])
    with open(out / "table1_familles.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["famille", "nombre_de_noms", "noms"])
        for fa, ms in sorted(members.items(), key=lambda x: -len(x[1])):
            w.writerow([fa, len(ms), ", ".join(sorted(ms))])

    L = ["=== 182 familles de noms sur le graphe livre ===",
         "relations : %d ; entites : %d ; entites dans une famille : %d" % (len(G), len(nodes), sum(len(m) for m in members.values())),
         "composantes du graphe entier : %d avant regroupement -> %d apres" % (c_before, c_after),
         "composantes du sous-graphe causal : %d avant -> %d apres" % (cc_before, cc_after),
         "voisinage causal de slope failure : %d noms distincts -> %d familles ou noms apres regroupement" % (len(neigh), len(neigh_fam)),
         "relations Tier 1 « X triggers slope failure » : %d (table2_triggers_slope_failure.tsv)" % len(t2),
         "", "Table 1 (familles d'au moins 2 noms) :"]
    for fa, ms in sorted(members.items(), key=lambda x: -len(x[1])):
        if len(ms) >= 2:
            L.append("  %-35s %3d  %s" % (fa, len(ms), ", ".join(sorted(ms))[:160]))
    txt = "\n".join(L)
    (out / "_resume.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("proposer")
    p.add_argument("--repo", default=".")
    p.add_argument("--graph", required=True)
    p.add_argument("--outdir", required=True)
    m = sp.add_parser("mesurer")
    m.add_argument("--repo", default=".")
    m.add_argument("--graph", required=True)
    m.add_argument("--families", required=True)
    m.add_argument("--outdir", required=True)
    args = ap.parse_args()
    if Path(args.outdir).exists():
        die("le dossier existe deja, rien n'est ecrase : " + args.outdir)
    proposer(args) if args.cmd == "proposer" else mesurer(args)


if __name__ == "__main__":
    main()