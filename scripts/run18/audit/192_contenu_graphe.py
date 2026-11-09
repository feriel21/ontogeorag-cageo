#!/usr/bin/env python3
"""
192 - Contenu du graphe livré : ce qu'il dit des MTD (lecture seule).

Compte les relations par type, liste les descripteurs rattachés aux MTD,
les causes et déclencheurs, et le contexte (occursIn, overlies, underlies...).
Ne modifie aucun fichier existant. Écrit seulement dans un nouveau dossier de sortie.

Usage (depuis ~/ontogeorag_run18) :
    python scripts/192_contenu_graphe.py
    python scripts/192_contenu_graphe.py --graph <autre_graphe.json> --out output/audit/192_contenu_nettoye
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_GRAPH = "output/run18/kg_final/flat_run18_enforced_dedup.json"
DEFAULT_OUT = "output/audit/192_contenu"

SUBJ_KEYS = ["subject", "head", "source", "s", "subj"]
PRED_KEYS = ["predicate", "relation", "rel", "p", "pred", "relation_type"]
OBJ_KEYS = ["object", "tail", "target", "o", "obj"]
TIER_KEYS = ["tier", "tier_label", "tier_before_panel"]
LIST_KEYS = ["relations", "edges", "triples", "triplets", "kg", "data", "items"]

MTD_TERMS = ["mtd", "mass-transport", "mass transport", "masstransport"]


def load_relations(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in LIST_KEYS:
            if k in data and isinstance(data[k], list):
                print(f"[info] relations lues sous la clé '{k}'")
                return data[k]
    sys.exit(f"[erreur] structure non reconnue dans {path} : clés = {list(data)[:20]}")


def pick(rec, keys):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return k
    return None


def norm(x):
    return str(x).strip()


def is_mtd(term):
    t = term.lower()
    return any(m in t for m in MTD_TERMS)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--termes", default=",".join(MTD_TERMS),
                    help="sous-chaînes qui repèrent la cible (défaut : MTD). Ex. : --termes 'slope failure'")
    args = ap.parse_args()
    MTD_TERMS[:] = [t.strip().lower() for t in args.termes.split(",") if t.strip()]

    out = Path(args.out)
    if out.exists():
        sys.exit(f"[stop] {out} existe déjà : rien n'est écrasé. Choisis un autre --out.")

    rels = load_relations(args.graph)
    if not rels or not isinstance(rels[0], dict):
        sys.exit("[erreur] les relations ne sont pas des dictionnaires.")

    first = rels[0]
    ks, kp, ko, kt = pick(first, SUBJ_KEYS), pick(first, PRED_KEYS), pick(first, OBJ_KEYS), pick(first, TIER_KEYS)
    print(f"[info] {len(rels)} relations ; champs détectés : sujet={ks}, relation={kp}, objet={ko}, tier={kt}")
    print(f"[info] clés de la première relation : {sorted(first.keys())}")
    if not (ks and kp and ko):
        sys.exit("[stop] champs sujet/relation/objet non trouvés : colle-moi les clés affichées ci-dessus.")

    out.mkdir(parents=True)

    by_pred = Counter()
    by_pred_tier = defaultdict(Counter)
    mtd_desc = defaultdict(lambda: {"n": 0, "tiers": Counter()})
    causes = defaultdict(lambda: {"n": 0, "tiers": Counter(), "preds": Counter()})
    context = defaultdict(lambda: {"n": 0, "tiers": Counter()})
    mtd_any = 0

    CAUSAL = {"causes", "triggers", "controls", "affects"}
    CONTEXT = {"occursin", "overlies", "underlies", "partof"}

    for r in rels:
        s, p, o = norm(r.get(ks, "")), norm(r.get(kp, "")), norm(r.get(ko, ""))
        tier = norm(r.get(kt, "?")) if kt else "?"
        pl = p.lower()
        by_pred[p] += 1
        by_pred_tier[p][tier] += 1
        s_mtd, o_mtd = is_mtd(s), is_mtd(o)
        if s_mtd or o_mtd:
            mtd_any += 1
        if pl == "hasdescriptor" and s_mtd:
            mtd_desc[o]["n"] += 1
            mtd_desc[o]["tiers"][tier] += 1
        if pl in CAUSAL and o_mtd:
            causes[s]["n"] += 1
            causes[s]["tiers"][tier] += 1
            causes[s]["preds"][p] += 1
        if pl in CONTEXT and s_mtd:
            key = f"{p} -> {o}"
            context[key]["n"] += 1
            context[key]["tiers"][tier] += 1

    tiers_all = sorted({t for c in by_pred_tier.values() for t in c})

    # 1. relations par type
    rows = [[p, n] + [by_pred_tier[p][t] for t in tiers_all] for p, n in by_pred.most_common()]
    write_csv(out / "1_relations_par_type.csv", ["relation", "total"] + [f"tier_{t}" for t in tiers_all], rows)

    # 2. descripteurs des MTD (colonne à remplir : visible sur la sismique ?)
    rows = [[d, v["n"], "; ".join(f"{t}:{c}" for t, c in sorted(v["tiers"].items())), ""]
            for d, v in sorted(mtd_desc.items(), key=lambda x: -x[1]["n"])]
    write_csv(out / "2_descripteurs_MTD.csv", ["descripteur", "n_relations", "tiers", "visible_sur_sismique (à remplir)"], rows)

    # 3. causes et déclencheurs des MTD
    rows = [[c, v["n"], "; ".join(f"{p}:{n}" for p, n in v["preds"].most_common()),
             "; ".join(f"{t}:{n}" for t, n in sorted(v["tiers"].items()))]
            for c, v in sorted(causes.items(), key=lambda x: -x[1]["n"])]
    write_csv(out / "3_causes_declencheurs_MTD.csv", ["cause", "n_relations", "relations", "tiers"], rows)

    # 4. contexte des MTD
    rows = [[k, v["n"], "; ".join(f"{t}:{n}" for t, n in sorted(v["tiers"].items()))]
            for k, v in sorted(context.items(), key=lambda x: -x[1]["n"])]
    write_csv(out / "4_contexte_MTD.csv", ["relation -> objet", "n_relations", "tiers"], rows)

    # résumé
    n_desc_rel = sum(v["n"] for v in mtd_desc.values())
    n_cause_rel = sum(v["n"] for v in causes.values())
    n_ctx_rel = sum(v["n"] for v in context.values())
    lines = [
        "# 192 - Contenu du graphe",
        f"Graphe : {args.graph}",
        f"Relations : {len(rels)} ; relations qui touchent un MTD (sujet ou objet) : {mtd_any}",
        "",
        "## Relations par type",
        *[f"- {p} : {n}" for p, n in by_pred.most_common()],
        "",
        f"## Descripteurs des MTD : {len(mtd_desc)} distincts, {n_desc_rel} relations",
        *[f"- {d} ({v['n']})" for d, v in sorted(mtd_desc.items(), key=lambda x: -x[1]['n'])],
        "",
        f"## Causes et déclencheurs des MTD : {len(causes)} distincts, {n_cause_rel} relations",
        *[f"- {c} ({v['n']})" for c, v in sorted(causes.items(), key=lambda x: -x[1]['n'])],
        "",
        f"## Contexte des MTD : {len(context)} distincts, {n_ctx_rel} relations",
        *[f"- {k} ({v['n']})" for k, v in sorted(context.items(), key=lambda x: -x[1]['n'])],
        "",
        "Repérage MTD par sous-chaîne : " + ", ".join(MTD_TERMS),
    ]
    (out / "resume.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:5 + len(by_pred)]))
    print(f"\n[ok] descripteurs MTD : {len(mtd_desc)} | causes : {len(causes)} | contexte : {len(context)}")
    print(f"[ok] sorties dans {out}/ (resume.md + 4 fichiers CSV)")


if __name__ == "__main__":
    main()