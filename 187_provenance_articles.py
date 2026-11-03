#!/usr/bin/env python3
"""187 — Article, reference complete et section (deduite) de chaque relation du graphe livre.

LECTURE SEULE. CPU. Aucun modele. N'ecrit que dans --outdir (refuse d'ecraser).
Le graphe livre n'est pas modifie : une COPIE enrichie est ecrite dans --outdir.

ENTREES
  --graph   graphe livre (flat_run18_enforced_dedup.json)
  --run     dossier du run (index/chunks.jsonl, pass_a|b/verified_triples.jsonl)
  --suivi   suivi_extractions.tsv de 184 (sortie --nfkc), pour retrouver TOUTES les
            extractions qui soutiennent une relation, pas seulement celle affichee
  --refs    dossier qui contient pdf_to_ref.tsv et lb2019_41_refs.txt (dossier data/ du depot)

CE QUE CALCULE LE SCRIPT, POUR CHAQUE RELATION
  1. Passages ou la preuve est localisee [MESURE]
     - phrase affichee dans le graphe : cherchee dans tout l'index ;
     - phrases des autres extractions qui soutiennent la relation (verdict STRONG ou WEAK,
       gardees ou absorbees comme doublon, d'apres 184) : cherchees dans leurs passages.
     Regles de 145, avec ligatures normalisees (NFKC) : comparaison en minuscules
     alphanumeriques ; exact, sinon partial (une des trois fenetres de 40 caracteres).
  2. Article et reference [CORRESPONDANCE FICHIER]
     doc_id du passage -> pdf_to_ref.tsv (fichier -> numero) -> lb2019_41_refs.txt
     (numero -> reference complete, DOI s'il existe).
  3. Section [DEDUITE, A VERIFIER]
     Dernier titre trouve dans le texte de l'article AVANT la phrase de preuve :
     titre numerote (ex. "3.2.1. triggering mechanisms...") ou mot de titre
     (introduction, results, discussion, conclusions, references...). Les PDF ont perdu
     leur mise en forme : c'est une deduction a partir du texte, pas une donnee du fichier.
  4. Legende de figure [DEDUITE] : "fig. 5." ou "figure 10." suivi de texte, dans les 120
     caracteres avant la phrase ou a son debut (un renvoi "(fig. 5)" ne compte pas).
  5. Position dans l'article : numero du passage sur le nombre de passages de l'article.

SORTIES
  provenance_par_relation.tsv   une ligne par relation livree
  provenance_detail.tsv         une ligne par (relation, passage ou la preuve est localisee)
  graphe_avec_provenance.json   copie du graphe ; chaque relation gagne un champ "provenance"
  references_non_resolues.txt   doc_id sans reference (doit etre vide)
  _resume.txt

USAGE (racine du worktree run18)
  python 187_provenance_articles.py \
      --graph output/run18/kg_final/flat_run18_enforced_dedup.json --run output/run18 \
      --suivi output/audit/184_trace_nfkc_v2/suivi_extractions.tsv \
      --refs ~/ontogeorag_release/data --outdir output/audit/187_provenance
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

WINDOW = 40
SUPPORTED = ("STRONG_SUPPORT", "WEAK_SUPPORT")

SECTION_WORDS = (r"abstract|introduction(?: and regional setting)?|regional (?:geological )?setting|"
                 r"geological (?:setting|background)|data and methods?|data|methods?|methodology|"
                 r"results|discussion|conclusions?|summary|acknowledge?ments|references")
STOP = {"and", "to", "or", "of", "in", "on", "at", "the", "a", "an", "for", "with", "by", "is", "are", "was",
        "were", "m", "km", "ms", "cm", "mm", "ka", "ma", "kyr", "s", "yr", "years", "deg", "degrees", "to",
        "times", "x", "per", "mpa", "kpa", "hz"}
NUM_HEAD = re.compile(r"(?<![\w.(¼=–\-/])(\d{1,2}(?:\.\d{1,2}){0,5})(\.?)\s+([a-z][a-z\-']+)((?:\s+[a-z0-9\-'():,]+){0,8})")
WORD_HEAD = re.compile(r"(?:^|(?<=\.\s)|(?<=\d\s)|(?<=\d\.\s))(" + SECTION_WORDS + r")(?![a-z])")
BAD_BEFORE = re.compile(r"(?:fig|figs|figure|figures|table|tables|no|eq|section|sections|unit|units|site|"
                        r"line|lines|profile|chapter|vol|p|pp|ref)\.?\s*$")
CAPTION = re.compile(r"(?<![(\w])fig(?:ure)?\.?\s?\d+[a-z]?\.\s?[—–\-]?\s?\(?[a-z]")


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def low(x):
    return unicodedata.normalize("NFKC", str(x or "")).lower()


def alnum_map(text):
    """Chaine alphanumerique (NFKC, minuscules) et, pour chaque caractere garde, sa position d'origine."""
    chars, pos = [], []
    for i, ch in enumerate(str(text or "")):
        for c in unicodedata.normalize("NFKC", ch).lower():
            if c.isascii() and c.isalnum():
                chars.append(c)
                pos.append(i)
    return "".join(chars), pos


def locate(ev, text, amap=None):
    """(statut, position dans text) ; statut exact / partial / not_found / vide."""
    e, _ = alnum_map(ev)
    a, pos = amap if amap else alnum_map(text)
    if not e:
        return "vide", None
    k = a.find(e)
    if k >= 0:
        return "exact", pos[k]
    if len(e) <= WINDOW:
        return "not_found", None
    mid = (len(e) - WINDOW) // 2
    for off, w in ((0, e[:WINDOW]), (mid, e[mid:mid + WINDOW]), (len(e) - WINDOW, e[-WINDOW:])):
        k = a.find(w)
        if k >= 0:
            return "partial", pos[max(0, k - off)] if k - off >= 0 else pos[k]
    return "not_found", None


def chunk_no(cid):
    try:
        return int(str(cid).rsplit("::", 1)[1])
    except (IndexError, ValueError):
        return -1


def doc_of(cid):
    return str(cid).rsplit("::", 1)[0]


def last_heading(text):
    """Dernier titre (numerote ou mot de section) dans text ; renvoie (titre, type) ou ("", "")."""
    t = low(text)
    best, kind, at = "", "", -1
    for m in NUM_HEAD.finditer(t):
        num, dot, w1, rest = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        top = int(num.split(".")[0])
        if top == 0 or top > 15 or w1 in STOP or len(w1) < 3:
            continue
        prev = t[max(0, m.start() - 12):m.start()].rstrip()
        if BAD_BEFORE.search(prev) or prev.endswith(("¼", "=", "<", ">", "~", "±")) or \
                re.search(r"(?:^|\s)(?:and|or|to|of|m|km|ms|between|from|about|around|than)$", prev):
            continue
        # sans fin de phrase avant le numero, on n'accepte qu'un numero a plusieurs niveaux termine par un point
        if prev and not prev.endswith((".", ":", ")")) and not ("." in num and dot):
            continue
        if "." not in num and not dot and not re.match(SECTION_WORDS + r"|regional|geological|study|background",
                                                       w1):
            continue
        title = (num + (dot or "") + " " + (w1 + rest).split(". ")[0]).strip(" ,:;")
        if m.start() > at:
            best, kind, at = title[:90], "numerote", m.start()
    for m in WORD_HEAD.finditer(t):
        if m.start() > at:
            best, kind, at = m.group(1), "mot de section", m.start()
    return best, kind


def short_ref(ref):
    m = re.search(r"\((\d{4}[a-z]?)\)", ref)
    year = m.group(1) if m else "s.d."
    authors = ref[:m.start()] if m else ref
    first = authors.split(",")[0].strip()
    n_comma = authors.count(".,") + authors.count(". &") + ("&" in authors)
    if "&" not in authors:
        return "%s (%s)" % (first, year)
    if n_comma <= 2 and authors.count("&") == 1 and authors.count(".,") <= 1:
        second = authors.split("&")[1].split(",")[0].strip()
        return "%s & %s (%s)" % (first, second, year)
    return "%s et al. (%s)" % (first, year)


def load_refs(d):
    d = Path(d).expanduser()
    p1, p2 = d / "pdf_to_ref.tsv", d / "lb2019_41_refs.txt"
    for p in (p1, p2):
        if not p.exists():
            die("introuvable : %s (passer --refs vers le dossier data/ du depot public)" % p)
    file2n = {}
    for row in csv.reader(open(p1, encoding="utf-8"), delimiter="\t"):
        if len(row) >= 2 and row[1].strip().isdigit():
            file2n[low(re.sub(r"\.pdf$", "", row[0].strip(), flags=re.I)).strip()] = int(row[1])
    n2ref = {}
    for line in open(p2, encoding="utf-8"):
        if line.startswith("#") or "|" not in line:
            continue
        n, ref = line.split("|", 1)
        if n.strip().isdigit():
            ref = ref.strip()
            doi = re.search(r"doi:\s*(\S+?)\.?$", ref, re.I)
            n2ref[int(n)] = {"reference": ref, "doi": doi.group(1) if doi else "", "court": short_ref(ref)}
    return file2n, n2ref, (sha(p1)[:12], sha(p2)[:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--suivi", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out, run = Path(args.outdir), Path(args.run)
    if out.exists():
        die("le dossier existe deja, rien n'est ecrase : " + str(out))
    file2n, n2ref, refsha = load_refs(args.refs)

    raw = json.load(open(args.graph, encoding="utf-8"))
    key_list = "triples" if isinstance(raw, dict) else None
    G = raw["triples"] if key_list else raw

    chunks = jl(run / "index" / "chunks.jsonl")
    by_id = {c["chunk_id"]: c for c in chunks}
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c.get("doc_id", doc_of(c["chunk_id"]))].append(c)
    for d in by_doc:
        by_doc[d].sort(key=lambda c: chunk_no(c["chunk_id"]))
    amaps = {}

    def amap(cid):
        if cid not in amaps:
            amaps[cid] = alnum_map(by_id[cid]["text"])
        return amaps[cid]

    ver = {P: jl(run / ("pass_" + P) / "verified_triples.jsonl") for P in ("a", "b")}
    support = defaultdict(list)                 # cle -> [(passe, ligne, evidence, [chunk_ids])]
    for r in csv.DictReader(open(args.suivi, encoding="utf-8"), delimiter="\t"):
        if not r.get("cle_sujet") or r.get("verdict_qwen") not in SUPPORTED:
            continue
        if r.get("etape_sortie") not in ("", "doublon_04"):
            continue
        t = ver[r["passe"]][int(r["ligne"])]
        ev = (t.get("_verification") or {}).get("evidence", "")
        cids = [c.get("chunk_id") for c in (t.get("_provenance") or {}).get("top_chunks") or []]
        support[(r["cle_sujet"], r["cle_relation"], r["cle_objet"])].append((r["passe"], int(r["ligne"]), ev, cids))

    unresolved = Counter()

    def describe(cid, pos):
        c = by_id[cid]
        d = c.get("doc_id", doc_of(cid))
        n = file2n.get(low(d).strip())
        ref = n2ref.get(n) if n else None
        if not ref:
            unresolved[d] += 1
        seq = by_doc[d]
        idx = [x["chunk_id"] for x in seq].index(cid)
        before = " ".join(x["text"] for x in seq[:idx]) + " " + c["text"][:pos if pos is not None else 0]
        head, kind = last_heading(before)
        ctx = low(c["text"][max(0, (pos or 0) - 120):(pos or 0) + 60])
        after_refs = kind == "mot de section" and head == "references"
        return {"doc_id": d, "ref_n": n or "", "reference_courte": ref["court"] if ref else "",
                "reference": ref["reference"] if ref else "", "doi": ref["doi"] if ref else "",
                "chunk_id": cid, "passage": "%d/%d" % (idx + 1, len(seq)),
                "section_deduite": head or "(avant le premier titre repere)",
                "type_titre": kind or "", "apres_references": "oui" if after_refs else "",
                "legende_figure_probable": "oui" if CAPTION.search(ctx) else ""}

    rows, detail, enriched = [], [], []
    for t in G:
        key = (t.get("subject"), t.get("relation"), t.get("object"))
        ev = t.get("evidence") or ""
        # phrase affichee : dans tout l'index
        shown = []
        for c in chunks:
            st, p = locate(ev, None, amap(c["chunk_id"]))
            if st == "exact":
                shown.append((st, c["chunk_id"], p))
        if not shown:
            for c in chunks:
                st, p = locate(ev, None, amap(c["chunk_id"]))
                if st == "partial":
                    shown.append((st, c["chunk_id"], p))
        prov, seen = [], set()
        for st, cid, p in shown:
            dsc = describe(cid, p)
            dsc.update(statut=st, origine="phrase affichee")
            prov.append(dsc)
            seen.add(cid)
        # autres extractions qui soutiennent la relation
        for P, li, ev2, cids in support.get(key, []):
            for cid in cids:
                if cid not in by_id or cid in seen:
                    continue
                st, p = locate(ev2, None, amap(cid))
                if st in ("exact", "partial"):
                    dsc = describe(cid, p)
                    dsc.update(statut=st, origine="extraction %s:%d" % (P, li))
                    prov.append(dsc)
                    seen.add(cid)
                    break
        docs = sorted({x["doc_id"] for x in prov})
        refs = sorted({x["reference_courte"] or x["doc_id"] for x in prov})
        sh = [x for x in prov if x["origine"] == "phrase affichee"]
        s0 = sh[0] if sh else {}
        rows.append({"sujet": key[0], "relation": key[1], "objet": key[2], "tier": t.get("tier"),
                     "decision_panel": (t.get("m4") or {}).get("decision", ""),
                     "n_articles": len(docs), "articles": " ; ".join(refs),
                     "phrase_affichee_statut": s0.get("statut", "not_found"),
                     "phrase_affichee_article": s0.get("reference_courte", ""),
                     "phrase_affichee_ref_n": s0.get("ref_n", ""),
                     "phrase_affichee_passage": s0.get("chunk_id", ""),
                     "phrase_affichee_section_deduite": s0.get("section_deduite", ""),
                     "phrase_affichee_type_titre": s0.get("type_titre", "") or ("aucun" if s0 else "non localisee"),
                     "phrase_affichee_legende_figure": s0.get("legende_figure_probable", ""),
                     "phrase_affichee_apres_references": s0.get("apres_references", ""),
                     "preuve": ev})
        for x in prov:
            detail.append({"sujet": key[0], "relation": key[1], "objet": key[2], **x})
        e = dict(t)
        e["provenance"] = {"method": "187 : preuve localisee dans l'index (NFKC, minuscules alphanumeriques ; "
                                     "partial = une des trois fenetres de 40 caracteres) ; section deduite du "
                                     "dernier titre trouve avant la phrase, a verifier",
                           "n_articles": len(docs),
                           "sources": [{k: x[k] for k in ("ref_n", "reference", "doi", "doc_id", "chunk_id",
                                                          "passage", "section_deduite", "legende_figure_probable",
                                                          "statut", "origine")} for x in prov]}
        enriched.append(e)

    out.mkdir(parents=True)

    def tsv(p, recs):
        if not recs:
            return
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]), delimiter="\t")
            w.writeheader()
            w.writerows(recs)
    tsv(out / "provenance_par_relation.tsv", rows)
    tsv(out / "provenance_detail.tsv", detail)
    cont = dict(raw) if key_list else None
    if cont is not None:
        cont["triples"] = enriched
    json.dump(cont if cont is not None else enriched, open(out / "graphe_avec_provenance.json", "w",
                                                            encoding="utf-8"), indent=1, ensure_ascii=False)
    (out / "references_non_resolues.txt").write_text("\n".join("%s\t%d" % kv for kv in unresolved.items()) + "\n",
                                                     encoding="utf-8")

    def pct(a, b):
        return "%d/%d (%.0f %%)" % (a, b, 100.0 * a / b) if b else "0/0"
    n = len(rows)
    sec_known = sum(1 for r in rows if r["phrase_affichee_section_deduite"] not in ("", "(avant le premier titre repere)"))
    L = ["=== 187 article, reference et section de chaque relation ===",
         "graphe : %s (sha %s) ; suivi 184 : %s (sha %s)" % (args.graph, sha(args.graph)[:12], args.suivi,
                                                             sha(args.suivi)[:12]),
         "references : pdf_to_ref.tsv %s, lb2019_41_refs.txt %s" % refsha,
         "", "relations : %d" % n,
         "phrase affichee localisee : %s" % dict(Counter(r["phrase_affichee_statut"] for r in rows)),
         "relations avec au moins un article : %s" % pct(sum(r["n_articles"] > 0 for r in rows), n),
         "articles par relation : %s" % dict(sorted(Counter(r["n_articles"] for r in rows).items())),
         "references non resolues (doc_id sans reference) : %d %s" % (len(unresolved), sorted(unresolved)),
         "", "-- phrase affichee --",
         "section deduite trouvee : %s" % pct(sec_known, n),
         "type de titre : %s" % dict(Counter(r["phrase_affichee_type_titre"] for r in rows)),
         "legende de figure probable : %s" % pct(sum(r["phrase_affichee_legende_figure"] == "oui" for r in rows), n),
         "apres le titre 'references' : %s" % pct(sum(r["phrase_affichee_apres_references"] == "oui" for r in rows), n),
         "", "-- articles les plus cites (relations dont la preuve y est localisee) --"]
    art = Counter(x for r in rows for x in r["articles"].split(" ; ") if x)
    L += ["  %3d  %s" % (c, a) for a, c in art.most_common(15)]
    L += ["", "Section : deduction a partir du texte extrait des PDF (titres numerotes ou mots de section), "
          "a verifier sur l'article avant de la citer."]
    txt = "\n".join(L)
    (out / "_resume.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt + "\n[fait] " + str(out), flush=True)


if __name__ == "__main__":
    main()