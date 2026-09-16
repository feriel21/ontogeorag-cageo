#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
52_repair_provenance.py  --  reconstruire la provenance de run14
================================================================
CONSTAT (run14, mesure sur disque)
    02 selectionne 5 chunks par requete et archive, ligne 532,
    `context_preview = context[:1500]` -- un APERCU destine au journal.
    03 cherche `top_chunks` (absent des 962 triples), ne le trouve pas, et
    se rabat silencieusement sur cet apercu : une chaine unique de 1500
    caracteres, soit 15 % des ~10 000 caracteres recuperes, et moins d'un
    chunk complet puisque les chunks de run14 font 2 000 caracteres.
    D'ou `chunks_checked: 1` alors que 5 passages avaient ete retenus.

    L'extracteur, lui, voit 2 800 caracteres (concat_chunks). Le
    verificateur juge donc sur un contexte DIFFERENT et deux fois plus
    petit que celui qui a produit le triple.

CE QUE FAIT CE SCRIPT
    Reconstruit, pour chaque triple de run14, le texte EXACT des chunks
    listes dans `_provenance.selected_chunks`, en les relisant depuis
    chunks.jsonl, dans l'ordre, sans troncature -- et les ecrit sous la
    cle `top_chunks` que 03 attend.

    STRICT PAR CONSTRUCTION : si un identifiant de chunk ne se resout pas,
    le script s'ARRETE. C'est exactement le comportement qui a manque a
    l'interface 02 -> 03.

    Le verdict de run14 est conserve sous `_verification_run14` pour
    permettre la comparaison differentielle triple par triple.

NE MODIFIE RIEN dans run14. Refuse d'ecraser une sortie existante.

Usage :
    # controle a blanc, aucun fichier ecrit
    python 52_repair_provenance.py \
        --chunks output/run14/step1/chunks.jsonl \
        --triples output/run14/pass_a/verified_triples.jsonl \
                  output/run14/pass_b/verified_triples.jsonl \
        --outdir output/run15 --dry-run

    # ecriture
    python 52_repair_provenance.py \
        --chunks output/run14/step1/chunks.jsonl \
        --triples output/run14/pass_a/verified_triples.jsonl \
                  output/run14/pass_b/verified_triples.jsonl \
        --outdir output/run15
"""

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict
from pathlib import Path


def load_jsonl(p):
    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise ValueError("{} ligne {} : {}".format(p, i, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--triples", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--expect-chunks", type=int, default=5,
                    help="nombre de chunks attendu par triple")
    ap.add_argument("--allow-missing", action="store_true",
                    help="poursuivre malgre des identifiants non resolus "
                         "(DECONSEILLE : reproduit la degradation silencieuse)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    outdir = Path(os.path.expanduser(args.outdir))

    print("=" * 76)
    print("52  reparation de la provenance   {}".format(
        "(DRY-RUN)" if args.dry_run else ""))
    print("=" * 76)

    # ---- index des chunks ------------------------------------------
    idx = {}
    dup = 0
    for c in load_jsonl(args.chunks):
        cid = c.get("chunk_id")
        if cid is None:
            continue
        if cid in idx:
            dup += 1
        idx[cid] = c.get("text", "")
    print("chunks indexes : {}".format(len(idx)))
    if dup:
        print("  [!] {} identifiants de chunk dupliques".format(dup))
    L = sorted(len(t) for t in idx.values())
    if L:
        print("  longueur : med {} | max {} car.".format(
            L[len(L) // 2], L[-1]))
    print()

    total = resolved_all = 0
    missing_ids = Counter()
    n_chunks_hist = Counter()
    avail_chars = []
    preview_chars = []
    per_pass = []

    for tpath in args.triples:
        name = Path(tpath).parent.name or Path(tpath).stem
        triples = load_jsonl(tpath)
        out_recs = []
        n_ok = n_bad = 0
        verdicts = Counter()

        for t in triples:
            total += 1
            prov = t.get("_provenance", {}) or {}
            sel = prov.get("selected_chunks", []) or []
            n_chunks_hist[len(sel)] += 1
            preview_chars.append(len(prov.get("context_preview", "") or ""))

            top, miss = [], []
            for cid in sel:
                txt = idx.get(cid)
                if txt is None:
                    miss.append(cid)
                    missing_ids[cid] += 1
                else:
                    top.append({"chunk_id": cid, "text": txt})

            if miss:
                n_bad += 1
            else:
                n_ok += 1
                resolved_all += 1
            avail_chars.append(sum(len(c["text"]) for c in top))

            r = dict(t)
            newprov = dict(prov)
            newprov["top_chunks"] = top
            newprov["repaired_by"] = "52_repair_provenance.py"
            newprov["n_chunks_repaired"] = len(top)
            newprov["n_chunks_missing"] = len(miss)
            newprov["missing_chunk_ids"] = miss
            # l'apercu d'origine est conserve, mais 03 lira top_chunks
            r["_provenance"] = newprov
            v = t.get("_verification", {}) or {}
            r["_verification_run14"] = v
            verdicts[v.get("verdict", "MISSING")] += 1
            r.pop("_verification", None)   # 03 recalculera
            out_recs.append(r)

        per_pass.append((name, tpath, triples, out_recs, n_ok, n_bad, verdicts))
        print("--- {}".format(tpath))
        print("    triples {} | provenance complete {} | incomplete {}".format(
            len(triples), n_ok, n_bad))
        print("    verdicts run14 : {}".format(dict(verdicts)))

    print()
    print("--- INTEGRITE --------------------------------------------------")
    print("  triples totaux                 : {}".format(total))
    print("  a provenance entierement resolue: {}".format(resolved_all))
    print("  identifiants non resolus       : {}".format(len(missing_ids)))
    for cid, n in missing_ids.most_common(10):
        print("      {}  ({}x)".format(cid, n))
    print("  repartition du nombre de chunks : {}".format(
        dict(sorted(n_chunks_hist.items()))))
    print()

    if avail_chars:
        A = sorted(avail_chars)
        P = sorted(preview_chars)
        med_a = A[len(A) // 2]
        med_p = P[len(P) // 2]
        print("--- CONTEXTE ---------------------------------------------------")
        print("  disponible apres reparation : med {} car.".format(med_a))
        print("  apercu utilise par run14    : med {} car.".format(med_p))
        if med_a:
            print("  run14 voyait donc           : {:.0f} % du materiel".format(
                100.0 * med_p / med_a))
        print()

    bad = total - resolved_all
    if bad and not args.allow_missing:
        print("ECHEC : {} triple(s) a provenance incomplete. Rien n'a ete "
              "ecrit.".format(bad))
        print("  Un identifiant non resolu signifie que chunks.jsonl ne")
        print("  correspond pas au run ayant produit ces triples.")
        print("  Verifier --chunks, ou passer --allow-missing en connaissance")
        print("  de cause (cela reproduirait la degradation silencieuse que")
        print("  ce script vise precisement a corriger).")
        sys.exit(3)

    if args.dry_run:
        print("DRY-RUN : aucun fichier ecrit.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    audit = {"chunks_indexed": len(idx), "triples_total": total,
             "fully_resolved": resolved_all,
             "missing_chunk_ids": dict(missing_ids),
             "n_chunks_distribution": dict(n_chunks_hist),
             "median_available_chars": (sorted(avail_chars)[len(avail_chars)//2]
                                        if avail_chars else 0),
             "median_preview_chars": (sorted(preview_chars)[len(preview_chars)//2]
                                      if preview_chars else 0),
             "passes": []}

    for name, tpath, triples, out_recs, n_ok, n_bad, verdicts in per_pass:
        d = outdir / name
        d.mkdir(parents=True, exist_ok=True)
        f = d / "triples_repaired.jsonl"
        if f.exists() and not args.force:
            print("ERREUR : {} existe deja. Refus d'ecraser.".format(f),
                  file=sys.stderr)
            sys.exit(2)
        with open(f, "w", encoding="utf-8") as fh:
            for r in out_recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("Ecrit : {}  ({} triples)".format(f, len(out_recs)))
        audit["passes"].append({"name": name, "source": tpath,
                                "n": len(out_recs), "fully_resolved": n_ok,
                                "incomplete": n_bad,
                                "run14_verdicts": dict(verdicts)})

    fa = outdir / "provenance_audit.json"
    fa.write_text(json.dumps(audit, indent=2, ensure_ascii=False),
                  encoding="utf-8")
    print("Ecrit : {}".format(fa))
    print()
    print("Le verdict de run14 est conserve sous _verification_run14 dans")
    print("chaque enregistrement ; _verification a ete retire pour que 03")
    print("le recalcule sur le contexte repare.")


if __name__ == "__main__":
    main()