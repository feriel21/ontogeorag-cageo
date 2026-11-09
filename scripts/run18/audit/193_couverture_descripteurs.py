#!/usr/bin/env python3
"""193_couverture_descripteurs.py : lecture seule, aucun modele, aucun run.

Explique l'ecart sur la couverture des 13 descripteurs LB2019 :
  189 trouve 13/13 (objets hasDescriptor, NFKC, minuscules, tiret = espace) ;
  article_metrics.json (07) trouve 12/13 (Tier 1 et 2) et 11/13 (Tier 1).

Pour chaque descripteur et chaque graphe (livre 364, nettoye 281), compte les
relations hasDescriptor dont l'objet correspond :
  exact      : chaine identique
  minuscules : identique apres strip().lower()
  normalise  : NFKC, minuscules, tiret et souligne = espace (regle de 189)
et donne la repartition par tier. Affiche aussi les lignes de 07 qui calculent
la couverture, pour lire la regle de 07 sans la supposer.

Usage (depuis le worktree run18) :
  python ~/ontogeorag/193_couverture_descripteurs.py --repo . \
      --outdir output/audit/193_couverture_descripteurs
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter


def normalise(x):
    s = unicodedata.normalize("NFKC", str(x or "")).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def charger(p):
    with open(p, encoding="utf-8") as f:
        g = json.load(f)
    if isinstance(g, dict):
        g = next(v for v in g.values() if isinstance(v, list))
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--graphe-livre", default="output/run18/kg_final/flat_run18_enforced_dedup.json")
    ap.add_argument("--graphe-nettoye", default="output/audit/188_nettoyage_v2/graphe_nettoye.json")
    ap.add_argument("--script-07", default="pipeline/07_final_metrics.py")
    a = ap.parse_args()
    R = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(a.repo, x))
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} existe deja : aucune sortie n'est ecrasee.")
    os.makedirs(outdir)
    sys.path.insert(0, os.path.abspath(a.repo))
    from pipeline.rag.constants import LB2019_BENCHMARK_DESCRIPTORS as LB  # lecture seule

    out = ["193_couverture_descripteurs : lecture seule.",
           f"descripteurs LB2019 ({len(LB)}) : " + ", ".join(repr(d) for d in LB)]
    for nom, chemin in (("graphe livre", R(a.graphe_livre)), ("graphe nettoye 188", R(a.graphe_nettoye))):
        g = charger(chemin)
        desc = [r for r in g if str(r.get("relation")).strip().lower() == "hasdescriptor"]
        out.append(f"\n=== {nom} : {chemin}, {len(g)} relations, {len(desc)} hasDescriptor ===")
        out.append(f"   {'descripteur':<16} {'exact':>6} {'minusc.':>8} {'normalise':>10}   tiers (normalise)   formes trouvees")
        couv = Counter()
        for d in LB:
            ex = [r for r in desc if str(r.get("object")) == d]
            mi = [r for r in desc if str(r.get("object")).strip().lower() == str(d).strip().lower()]
            no = [r for r in desc if normalise(r.get("object")) == normalise(d)]
            for cle, l in (("exact", ex), ("minuscules", mi), ("normalise", no)):
                couv[cle] += bool(l)
                couv[cle + " tier1"] += any(str(r.get("tier")) == "1" for r in l)
            tiers = Counter(str(r.get("tier")) for r in no)
            formes = Counter(str(r.get("object")) for r in no)
            out.append(f"   {d!s:<16} {len(ex):>6} {len(mi):>8} {len(no):>10}   "
                       f"{dict(sorted(tiers.items()))!s:<18}  " + ", ".join(f"{f!r} {n}" for f, n in formes.items()))
        out.append("   couverture : " + ", ".join(f"{k} {v}/{len(LB)}" for k, v in couv.items()))

    s07 = R(a.script_07)
    out.append(f"\n=== lignes de {s07} qui parlent des descripteurs ===")
    if os.path.exists(s07):
        with open(s07, encoding="utf-8", errors="replace") as f:
            lignes = f.readlines()
        garder = set()
        for i, l in enumerate(lignes):
            if re.search(r"descriptor|DESCRIPTOR|coverage", l):
                garder.update(range(max(0, i - 2), min(len(lignes), i + 3)))
        prec = -2
        for i in sorted(garder):
            if i != prec + 1:
                out.append("   ...")
            out.append(f"   {i + 1:>5}  {lignes[i].rstrip()}")
            prec = i
    else:
        out.append("   introuvable")

    with open(os.path.join(outdir, "_resume.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()