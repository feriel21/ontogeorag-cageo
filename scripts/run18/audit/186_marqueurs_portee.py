#!/usr/bin/env python3
"""186 — Marqueurs de portee dans les phrases de preuve du graphe livre (indicateur automatique).

LECTURE SEULE. CPU. Aucun modele. N'ecrit que dans --outdir (refuse d'ecraser).

QUESTION
    Un triplet (sujet, relation, objet) ne peut pas porter les nuances de sa phrase :
    modalite (may, potential...), quantificateur partiel (some, locally...), condition
    (where, under, down to...), valeur chiffree, cas particulier (figure, zone d'etude),
    partie d'un objet (top surface, distal...). Ce script compte, dans la phrase de preuve
    de chaque relation livree, les marqueurs lexicaux de ces nuances, puis croise ce compte
    avec le tier et les verdicts des juges du panel.

CE QUE LE CHIFFRE VEUT DIRE
    C'est un INDICATEUR LEXICAL, pas un jugement : un marqueur present ne prouve pas que
    le triplet a perdu une nuance (ex. "where" peut etre un simple relatif), et une nuance
    sans mot-cle de la liste n'est pas comptee. Les listes sont fixees ci-dessous, avant
    lecture des resultats, et chaque relation reporte les mots trouves pour relecture.

CATEGORIES (mots entiers, phrase en minuscules, ligatures normalisees NFKC)
    modalite      may, might, could, can, possibly, possible, potential(ly), probably,
                  probable, likely, unlikely, perhaps, presumably, suggest*, appear(s/ed) to,
                  seem*, inferred, interpreted, thought to, hypothes*, uncertain
    partiel       some, several, many, most, often, commonly, typically, usually, generally,
                  locally, occasionally, sometimes, partly, partially, few, in places
    condition     if, when, where, unless, provided, depending, only, under, during,
                  down to, deeper, shallower, greater than, less than
    valeur        nombre suivi d'une unite (m, km, ms, %, degre) ou "magnitude"
    cas           fig., figure, this study, this area, study area, here, survey,
                  "the <nom> slide/basin/margin/area/mass transport deposit"
    partie        top surface, upper surface, basal, base of, distal, proximal, fill,
                  flank, toe, headscarp, headwall, upper part, lower part, internal, edge
                  (compte seulement si le mot n'est pas deja dans le sujet du triplet)
    negation      not, no, never, without, absence, absent, lack (compte a part)
    "phrase nuancee" = au moins une categorie parmi modalite, partiel, condition, valeur,
    cas, partie.

AUTRES SIGNAUX LUS DANS L'ENREGISTREMENT (aucun calcul de modele)
    tronquee : la phrase fait 400 caracteres, longueur de coupe de 06
    juges    : m4.evidence_verdict (Llama|Mistral), m4.decision, tier

USAGE (racine du worktree run18)
    python 186_marqueurs_portee.py --graph output/run18/kg_final/flat_run18_enforced_dedup.json \
        --quarantine output/run18/m4_panel/m4_quarantine.jsonl --outdir output/audit/186_portee
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CUT = 400  # longueur de coupe de evidence dans 06_tiered_fusion.get_evidence

PATTERNS = {
    "modalite": [r"may", r"might", r"could", r"can", r"possibly", r"possible", r"potential(?:ly)?",
                 r"probabl[ey]", r"likely", r"unlikely", r"perhaps", r"presumably", r"suggest\w*",
                 r"appear(?:s|ed)? to", r"seem\w*", r"inferred", r"interpreted", r"thought to",
                 r"hypothes\w*", r"uncertain"],
    "partiel": [r"some", r"several", r"many", r"most", r"often", r"commonly", r"typically", r"usually",
                r"generally", r"locally", r"occasionally", r"sometimes", r"partly", r"partially", r"few",
                r"in places"],
    "condition": [r"if", r"when", r"where", r"unless", r"provided", r"depending", r"only", r"under",
                  r"during", r"down to", r"deeper", r"shallower", r"greater than", r"less than"],
    "valeur": [r"\d+(?:[.,]\d+)?\s?(?:m|km|ms|%|°|degrees?)", r"magnitude"],
    "cas": [r"fig\.", r"figure", r"this study", r"this area", r"study area", r"here", r"survey",
            r"the \w+ (?:slide|basin|margin|area|mass transport deposit|mass transport complex)"],
    "partie": [r"top surface", r"upper surface", r"basal", r"base of", r"distal", r"proximal", r"fill",
               r"flank", r"toe", r"headscarp", r"headwall", r"upper part", r"lower part", r"internal",
               r"edge"],
    "negation": [r"not", r"no", r"never", r"without", r"absence", r"absent", r"lack"],
}
NUANCE = ["modalite", "partiel", "condition", "valeur", "cas", "partie"]
COMPILED = {c: [(p, re.compile(r"(?<![a-z0-9])" + p + r"(?![a-z0-9])")) for p in ps] for c, ps in PATTERNS.items()}


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(x):
    x = unicodedata.normalize("NFKC", str(x or "")).lower()
    return " ".join(x.replace('"', " ").split())


def markers(ev, subject):
    e, s = norm(ev), norm(subject)
    found = {}
    for c, pats in COMPILED.items():
        hits = []
        for p, rx in pats:
            for m in rx.finditer(e):
                w = m.group(0)
                if c == "partie" and w in s:
                    continue
                hits.append(w)
        if hits:
            found[c] = sorted(set(hits))
    return found


def load_graph(p):
    d = json.load(open(p, encoding="utf-8"))
    if isinstance(d, list):
        return d
    for k in ("triples", "relations", "edges"):
        if isinstance(d.get(k), list):
            return d[k]
    die("liste de relations introuvable dans " + str(p))


def judges(t):
    m4 = t.get("m4") or {}
    ev = str(m4.get("evidence_verdict") or "")
    parts = ev.split("|") if ev else []
    if not parts:
        return "sans verdict"
    if all(x == "SUPPORTED" for x in parts):
        return "tous SUPPORTED"
    if any(x == "SUPPORTED" for x in parts):
        return "mixte"
    if all(x == "PARTIALLY_SUPPORTED" for x in parts):
        return "tous PARTIALLY"
    return "au moins un NOT/autre"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--quarantine", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    if out.exists():
        die("le dossier existe deja, rien n'est ecrase : " + str(out))
    G = load_graph(args.graph)
    sets = [("livre", G)]
    if args.quarantine and Path(args.quarantine).exists():
        sets.append(("quarantaine", [json.loads(l) for l in open(args.quarantine, encoding="utf-8") if l.strip()]))

    rows = []
    for name, recs in sets:
        for t in recs:
            ev = t.get("evidence") or ""
            ev = ev if isinstance(ev, str) else json.dumps(ev, ensure_ascii=False)
            mk = markers(ev, t.get("subject", ""))
            rows.append({
                "ensemble": name, "sujet": t.get("subject"), "relation": t.get("relation"), "objet": t.get("object"),
                "tier": t.get("tier"), "decision_panel": (t.get("m4") or {}).get("decision", ""),
                "verdicts_juges": (t.get("m4") or {}).get("evidence_verdict", ""), "juges": judges(t),
                "longueur_preuve": len(ev), "tronquee": "oui" if len(ev) >= CUT else "",
                "phrase_nuancee": "oui" if any(c in mk for c in NUANCE) else "",
                **{c: ";".join(mk.get(c, [])) for c in PATTERNS},
                "preuve": ev,
            })

    out.mkdir(parents=True)
    cols = list(rows[0])
    with open(out / "marqueurs_par_relation.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    def pct(a, b):
        return "%d/%d (%.0f %%)" % (a, b, 100.0 * a / b) if b else "0/0"

    L = ["=== 186 marqueurs de portee dans les phrases de preuve (indicateur lexical) ===",
         "graphe : %s (sha %s)" % (args.graph, sha(args.graph)[:12])]
    if len(sets) > 1:
        L.append("quarantaine : %s (sha %s)" % (args.quarantine, sha(args.quarantine)[:12]))
    for name, _ in sets:
        R = [r for r in rows if r["ensemble"] == name]
        n = len(R)
        L += ["", "-- %s : %d relations --" % (name, n),
              "phrase nuancee (>= 1 categorie) : " + pct(sum(r["phrase_nuancee"] == "oui" for r in R), n)]
        for c in PATTERNS:
            L.append("  %-10s : %s" % (c, pct(sum(bool(r[c]) for r in R), n)))
        L.append("  phrase tronquee a %d caracteres : %s" % (CUT, pct(sum(r["tronquee"] == "oui" for r in R), n)))
        top = Counter(w for r in R for c in NUANCE for w in r[c].split(";") if w)
        L.append("  mots les plus frequents : " + ", ".join("%s %d" % kv for kv in top.most_common(15)))
        if name == "livre":
            L.append("")
            L.append("  par tier :")
            for k in sorted({r["tier"] for r in R}, key=str):
                S = [r for r in R if r["tier"] == k]
                L.append("    tier %s : nuancee %s" % (k, pct(sum(r["phrase_nuancee"] == "oui" for r in S), len(S))))
            L.append("  par relation :")
            for k, _ in Counter(r["relation"] for r in R).most_common():
                S = [r for r in R if r["relation"] == k]
                L.append("    %-14s : nuancee %s" % (k, pct(sum(r["phrase_nuancee"] == "oui" for r in S), len(S))))
            L.append("  verdicts des juges sur la phrase (Llama|Mistral) selon la nuance :")
            tab = defaultdict(Counter)
            for r in R:
                tab["nuancee" if r["phrase_nuancee"] else "sans marqueur"][r["juges"]] += 1
            cats = ["tous SUPPORTED", "mixte", "tous PARTIALLY", "au moins un NOT/autre", "sans verdict"]
            L.append("    %-14s | %s" % ("", " | ".join(cats)))
            for k in ("nuancee", "sans marqueur"):
                tot = sum(tab[k].values())
                L.append("    %-14s | %s" % (k, " | ".join(pct(tab[k][c], tot) for c in cats)))
    L += ["", "Rappel : indicateur lexical (sur-detection possible, ex. 'where' relatif ; sous-detection des "
          "nuances sans mot-cle). Relire les colonnes de marqueurs avant de citer un chiffre."]
    txt = "\n".join(L)
    (out / "_resume.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt + "\n[fait] " + str(out), flush=True)


if __name__ == "__main__":
    main()