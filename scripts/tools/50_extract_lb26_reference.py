#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
50_extract_lb26_reference.py  --  materialiser le benchmark a 26 aretes
================================================================
PROBLEME
    Le "18/26 = 69,2 %" cite comme chiffre defendable provient du rapport
    interne de pipeline/04_clean_validate.py, qui evalue la constante
    LB_REFERENCE_EDGES (26 aretes, codee en dur ligne 52) PAR PASSE, avant
    fusion. Ce n'est donc pas une mesure sur le KG livre.

    Par ailleurs configs/lb_reference_edges.json et
    configs/lb_extended_benchmark.json ont le MEME SHA-256 : ce sont deux
    noms pour le meme fichier de 34 aretes. Le nom promet 26, le contenu
    en donne 34.

CE QUE FAIT CE SCRIPT
    Extrait LB_REFERENCE_EDGES par analyse syntaxique (ast), sans importer
    ni executer le module de pipeline, et l'ecrit dans un fichier de
    reference explicite au MEME FORMAT que lb_extended_benchmark.json,
    utilisable par 25_metrics_with_ref.py.

    Compare ensuite les 26 aretes aux 34 du benchmark etendu et signale
    toute arete des 26 absente des 34 (ce qui invaliderait l'idee que les
    34 = 26 originales + 8 ajoutees).

Ne modifie aucun script de pipeline. Refuse d'ecraser un fichier existant.

Usage :
    python 50_extract_lb26_reference.py \
        --source pipeline/04_clean_validate.py \
        --like   configs/lb_extended_benchmark.json \
        --out    configs/lb_26_original.json
"""

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path


def extract_constant(py_path, name):
    """Recupere une constante litterale par ast, sans executer le module."""
    src = Path(py_path).read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name \
                    and node.value is not None:
                return ast.literal_eval(node.value)
    raise KeyError("{} introuvable dans {}".format(name, py_path))


def norm(s):
    return " ".join(str(s).lower().split())


def load_ref_edges(path):
    """Lit un fichier de reference et retourne (edges_normalises, structure)."""
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, list):
        items, container = data, None
    elif isinstance(data, dict):
        container = None
        for k in ("edges", "reference_edges", "triples", "items", "data"):
            if isinstance(data.get(k), list):
                items, container = data[k], k
                break
        else:
            raise ValueError("format de reference non reconnu : {}".format(path))
    else:
        raise ValueError("format de reference non reconnu : {}".format(path))

    edges, keys = [], None
    for it in items:
        if isinstance(it, dict):
            if keys is None:
                keys = list(it.keys())
            s = it.get("subject") or it.get("s") or it.get("source")
            r = it.get("relation") or it.get("r") or it.get("predicate")
            o = it.get("object") or it.get("o") or it.get("target")
            edges.append((norm(s), norm(r), norm(o)))
        elif isinstance(it, (list, tuple)) and len(it) >= 3:
            edges.append((norm(it[0]), norm(it[1]), norm(it[2])))
    return edges, {"container": container, "item_keys": keys,
                   "is_list": isinstance(data, list)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pipeline/04_clean_validate.py")
    ap.add_argument("--constant", default="LB_REFERENCE_EDGES")
    ap.add_argument("--like", default="configs/lb_extended_benchmark.json",
                    help="fichier dont on copie le format")
    ap.add_argument("--out", default="configs/lb_26_original.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    if out.exists() and not args.force:
        print("ERREUR : {} existe deja. Refus d'ecraser.".format(out),
              file=sys.stderr)
        sys.exit(2)

    print("=" * 74)
    print("50  extraction du benchmark a 26 aretes")
    print("=" * 74)

    raw = extract_constant(args.source, args.constant)
    print("source   : {} :: {}".format(args.source, args.constant))
    print("elements : {}".format(len(raw)))
    if raw:
        print("exemple  : {}".format(raw[0]))
    print()

    edges26 = []
    for it in raw:
        if isinstance(it, (list, tuple)) and len(it) >= 3:
            edges26.append((norm(it[0]), norm(it[1]), norm(it[2])))
        elif isinstance(it, dict):
            edges26.append((norm(it.get("subject")), norm(it.get("relation")),
                            norm(it.get("object"))))
    if len(edges26) != len(raw):
        print("[!] {} elements sur {} n'ont pas pu etre lus".format(
            len(raw) - len(edges26), len(raw)))

    dup = len(edges26) - len(set(edges26))
    if dup:
        print("[!] {} arete(s) dupliquee(s) dans la constante".format(dup))

    # format cible
    like = Path(os.path.expanduser(args.like))
    fmt = {"container": None, "item_keys": ["subject", "relation", "object"],
           "is_list": True}
    edges34 = []
    if like.is_file():
        edges34, fmt = load_ref_edges(like)
        print("format copie de : {}".format(like))
        print("  sha256        : {}".format(sha256(like)))
        print("  aretes        : {}".format(len(edges34)))
        print("  cles element  : {}".format(fmt["item_keys"]))
        print()
    else:
        print("[!] {} absent : format par defaut utilise".format(like))
        print()

    # comparaison 26 vs 34
    if edges34:
        s26, s34 = set(edges26), set(edges34)
        inter = s26 & s34
        print("--- COMPARAISON 26 vs 34 ---------------------------------")
        print("  26 aretes presentes dans les 34 : {}".format(len(inter)))
        print("  26 aretes ABSENTES des 34       : {}".format(len(s26 - s34)))
        for e in sorted(s26 - s34):
            print("    - {} | {} | {}".format(*e))
        print("  aretes des 34 hors des 26       : {}".format(len(s34 - s26)))
        for e in sorted(s34 - s26):
            print("    + {} | {} | {}".format(*e))
        if len(s26 - s34) == 0 and len(s34 - s26) == len(s34) - len(s26):
            print("  -> coherent : les 34 = les 26 originales + les ajoutees")
        else:
            print("  -> INCOHERENT : les 34 ne contiennent pas exactement")
            print("     les 26. Le manuscrit ne peut pas presenter les 34")
            print("     comme un sur-ensemble des 26 sans verification.")
        print()

    # ecriture au format cible
    items = []
    for s, r, o in edges26:
        keys = fmt["item_keys"] or ["subject", "relation", "object"]
        if set(["subject", "relation", "object"]).issubset(set(keys)):
            d = {"subject": s, "relation": r, "object": o}
        elif set(["s", "r", "o"]).issubset(set(keys)):
            d = {"s": s, "r": r, "o": o}
        elif set(["source", "predicate", "target"]).issubset(set(keys)):
            d = {"source": s, "predicate": r, "target": o}
        else:
            d = {"subject": s, "relation": r, "object": o}
        items.append(d)

    payload = items if fmt["is_list"] else {fmt["container"] or "edges": items}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print("Ecrit : {}  ({} aretes)".format(out, len(items)))
    print("sha256: {}".format(sha256(out)))
    print()
    print("Mesure sur le KG livre :")
    print("  python analysis_suite/25_metrics_with_ref.py \\")
    print("      --kg output/run14/kg/tiered_kg_run14.json \\")
    print("      --ref {} \\".format(out))
    print("      --output output/run14/kg/metrics_26.json")


if __name__ == "__main__":
    main()