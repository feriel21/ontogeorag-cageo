#!/usr/bin/env python3
"""180 — Le panel voit-il la phrase qui soutient chaque relation ?

POURQUOI
    m4/m4_verify.py donne au verificateur `evidence_from_provenance(t)` :
    le texte du PREMIER passage retrouve (_provenance.top_chunks[0]),
    tronque a MAX_EVIDENCE_CHARS (1 500 caracteres). Dans run18, l'extracteur
    lit les 5 passages : la phrase de preuve peut etre dans un passage 2 a 5,
    ou apres le 1 500e caractere du premier. Le juge dit alors NOT_SUPPORTED
    sans avoir vu la phrase. Ce script mesure ce cas sur le graphe fusionne
    (entree du panel) et le croise avec la decision du panel.

QUOI (lecture seule, CPU)
    Pour chaque relation de kg/tiered_kg_<run>.json, dans l'ordre que lit m4_verify :
      vue       la phrase de preuve est dans top_chunks[0][:1500]
      chunk1    dans top_chunks[0] mais apres 1 500 caracteres
      chunkN    dans top_chunks[N-1], N = 2..5 (jamais montre au juge)
      absente   dans aucun des 5 passages
    (comparaison en minuscules alphanumeriques ; si la phrase complete n'est pas
    trouvee, une des trois fenetres de 40 caracteres suffit, comme dans 145)
    Decision du panel et verdicts : m4/llama et m4/mistral (m4_verdicts.jsonl,
    meme ordre que l'entree), et m4_panel/m4_panel_decisions.jsonl.

USAGE
    python 180_passage_vu_par_le_panel.py --repo . --run-dir output/run18 \
        --outdir output/audit/180_passage_panel
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

WINDOW = 40


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def alnum(x):
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower())


def contains(ev, text):
    e, t = alnum(ev), alnum(text)
    if not e or not t:
        return False
    if e in t:
        return True
    if len(e) <= WINDOW:
        return False
    mid = (len(e) - WINDOW) // 2
    return any(w in t for w in (e[:WINDOW], e[mid:mid + WINDOW], e[-WINDOW:]))


def jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    if out.exists():
        die("le dossier existe deja, rien n'est ecrase : " + str(out))
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "m4"))
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("m4v", repo / "m4" / "m4_verify.py")
    m4v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m4v)
    from m4_config import MAX_EVIDENCE_CHARS

    rd = Path(args.run_dir)
    run = args.run_name or rd.name
    kg = m4v.load_triples(rd / "kg" / ("tiered_kg_%s.json" % run))
    ver = {j: jsonl(rd / "m4" / j / "m4_verdicts.jsonl") for j in ("llama", "mistral")}
    dec = jsonl(rd / "m4_panel" / "m4_panel_decisions.jsonl")
    for j, v in ver.items():
        if len(v) != len(kg):
            die("%s : %d verdicts pour %d relations, ordre non garanti" % (j, len(v), len(kg)))
    if dec and len(dec) != len(kg):
        die("panel : %d decisions pour %d relations" % (len(dec), len(kg)))

    rows, tab = [], defaultdict(Counter)
    for i, t in enumerate(kg):
        ev = t.get("evidence") or ""
        if isinstance(ev, dict):
            ev = ev.get("quote") or ev.get("text") or ""
        tops = ((t.get("_provenance") or {}).get("top_chunks") or [])
        texts = [(c or {}).get("text", "") for c in tops]
        shown = m4v.evidence_from_provenance(t)[:MAX_EVIDENCE_CHARS]
        if contains(ev, shown):
            pos = "vue"
        elif texts and contains(ev, texts[0]):
            pos = "chunk1_apres_%d" % MAX_EVIDENCE_CHARS
        else:
            pos = "absente"
            for k, tx in enumerate(texts[1:], start=2):
                if contains(ev, tx):
                    pos = "chunk%d" % k
                    break
        d = (dec[i].get("decision") or dec[i].get("panel_decision") or "?") if dec else "?"
        evv = "|".join((ver[j][i].get("evidence") or {}).get("verdict", "?") for j in ("llama", "mistral"))
        tab["decision"][(pos, d)] += 1
        tab["evidence"][(pos, evv)] += 1
        s, r, o = m4v.triple_fields(t)
        rows.append({"index": i, "subject": s, "relation": r, "object": o, "position": pos,
                     "panel": d, "evidence_verdicts": evv, "evidence": ev[:300]})

    out.mkdir(parents=True)
    with open(out / "relations.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos_c = Counter(r["position"] for r in rows)
    L = ["=== 180 la phrase de preuve est-elle dans le texte montre au panel ? (%s) ===" % run,
         "relations : %d ; texte montre au juge : top_chunks[0][:%d]" % (len(rows), MAX_EVIDENCE_CHARS),
         "position de la phrase : " + json.dumps(dict(pos_c.most_common()), ensure_ascii=False), "",
         "decision du panel selon la position :"]
    decs = sorted({d for (_, d) in tab["decision"]})
    L.append("  %-22s" % "position" + "".join("%11s" % d for d in decs))
    for p, _ in pos_c.most_common():
        L.append("  %-22s" % p + "".join("%11d" % tab["decision"][(p, d)] for d in decs))
    L.append("")
    L.append("verdicts avec passage (llama|mistral) selon la position :")
    for p, _ in pos_c.most_common():
        c = Counter({e: n for (pp, e), n in tab["evidence"].items() if pp == p})
        L.append("  %-22s %s" % (p, ", ".join("%s %d" % kv for kv in c.most_common(4))))
    L.append("fichier : relations.jsonl")
    txt = "\n".join(L)
    (out / "_resume.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt + "\n[fait] " + str(out), flush=True)


if __name__ == "__main__":
    main()