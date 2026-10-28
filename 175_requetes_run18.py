#!/usr/bin/env python3
"""175 — Jeu de requetes de run18 : aucune requete ne nomme une arete d'evaluation.

REGLE (fixee avant run18)
    Une requete du fichier source est retiree si
      R1  son ancre (focus) appartient aux familles LB: ou gapfill:
          (ancres ecrites a partir du graphe expert ou d'une arete visee), ou
      R2  son ancre OU son texte nomme les deux arguments d'une arete de
          configs/lb_reference_edges_original26.json, en termes entiers
          (un terme contenu dans un terme plus long ne compte pas).
    Toutes les autres requetes sont gardees, dans leur ordre, sans modification.

SORTIES
    --out            fichier jsonl des requetes gardees
    --out.meta.json  regle, sha256 de l'entree et de la sortie, comptes,
                     indices retires et raison
    Le script refuse d'ecraser une sortie existante.

USAGE
    python 175_requetes_run18.py --repo . \
        --queries configs/descriptor_queries_v2.jsonl \
        --out configs/descriptor_queries_run18.jsonl
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REF26 = "configs/lb_reference_edges_original26.json"
EXPECT = {"input": 287, "kept": 210}


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(s):
    return " ".join(str(s or "").replace("_", " ").split()).strip().lower()


def mentions(term, text):
    return bool(term) and re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text) is not None


def anchor_text(focus):
    f = str(focus or "")
    for p in ("combo:", "gapfill:", "lb:", "LB:"):
        if f.lower().startswith(p.lower()):
            f = f[len(p):]
    return " ".join(norm(p) for p in re.split(r"->|\+|:", f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-expect", action="store_true", help="ne pas exiger 287 -> 210")
    args = ap.parse_args()

    out = Path(args.out)
    meta = Path(str(out) + ".meta.json")
    if out.exists() or meta.exists():
        die("sortie existante, rien n'est ecrase : " + str(out))
    repo = Path(args.repo)
    edges = [(norm(e["subject"]), norm(e["object"])) for e in
             json.load(open(repo / REF26, encoding="utf-8"))["edges"]]
    lines = [l for l in open(args.queries, encoding="utf-8") if l.strip()]
    Q = [json.loads(l) for l in lines]

    kept, removed, why = [], [], Counter()
    for i, (line, q) in enumerate(zip(lines, Q)):
        focus = str(q.get("focus") or "")
        reasons = []
        if focus.startswith("LB:") or focus.startswith("gapfill:"):
            reasons.append("R1_" + focus.split(":")[0])
        a, t = anchor_text(focus), norm(q.get("query"))
        if any((mentions(s, a) and mentions(o, a)) or (mentions(s, t) and mentions(o, t)) for s, o in edges):
            reasons.append("R2_names_both_arguments")
        if reasons:
            removed.append({"index": i, "focus": focus, "query": q.get("query"), "reasons": reasons})
            for r in reasons:
                why[r] += 1
        else:
            kept.append(line.rstrip("\n"))

    if not args.no_expect and (len(Q), len(kept)) != (EXPECT["input"], EXPECT["kept"]):
        die("attendu %d -> %d, obtenu %d -> %d" % (EXPECT["input"], EXPECT["kept"], len(Q), len(kept)))
    # controle : plus aucune requete gardee ne nomme une arete
    for line in kept:
        q = json.loads(line)
        a, t = anchor_text(q.get("focus")), norm(q.get("query"))
        assert not any((mentions(s, a) and mentions(o, a)) or (mentions(s, t) and mentions(o, t)) for s, o in edges)

    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    strat = Counter(json.loads(l).get("strategy") for l in kept)
    doc = {"produced_by": "175_requetes_run18.py",
           "rule": {"R1": "focus starts with LB: or gapfill:",
                    "R2": "focus or query text names both arguments of a reference edge (whole terms)"},
           "reference_edges": REF26, "reference_sha256": sha(repo / REF26),
           "input": args.queries, "input_sha256": sha(args.queries), "n_input": len(Q),
           "output": str(out), "output_sha256": sha(out), "n_kept": len(kept), "n_removed": len(removed),
           "removed_by_reason": dict(why), "kept_by_strategy": dict(strat), "removed": removed}
    meta.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print("requetes : %d -> %d gardees, %d retirees %s" % (len(Q), len(kept), len(removed), dict(why)))
    print("gardees par strategie : " + json.dumps(dict(strat)))
    print("sha256 sortie : " + doc["output_sha256"][:12])
    print("[fait] " + str(out))


if __name__ == "__main__":
    main()