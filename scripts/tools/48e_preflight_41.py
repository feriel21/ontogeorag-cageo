#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
48e_preflight_41.py  --  controle pre-vol avant relance du pipeline
================================================================
A executer APRES avoir copie les 3 chapitres manquants dans
data/full_corpus, et AVANT toute reindexation.

Repond a trois questions, dans l'ordre :

  A. Les 41 references LB2019 sont-elles toutes presentes ?
  B. Chaque PDF rend-il du texte exploitable, et avec quel extracteur ?
  C. Le texte contient-il des filigranes ou artefacts a nettoyer ?

Puis simule le decoupage en chunks pour predire le volume attendu,
et delivre un verdict GO / NO-GO par document.

Ne modifie rien. N'ecrit que dans --out.

Usage :
    python 48e_preflight_41.py \
        --corpus-dir ~/ontogeorag/data/full_corpus \
        --refs ~/ontogeorag/data/lb2019_41_refs.txt \
        --out ~/ontogeorag/output/preflight \
        --chunk-size 1888 --overlap 200

Extracteurs : pymupdf est prefere (le plus robuste sur ce corpus).
    pip install pymupdf
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

AVAIL = {}
for name, mod in (("pypdf", "pypdf"), ("pymupdf", "fitz"),
                  ("pdfplumber", "pdfplumber")):
    try:
        __import__(mod)
        AVAIL[name] = True
    except Exception:
        AVAIL[name] = False

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)
YEAR_RE = re.compile(r"\((\d{4})[a-z]?\)")

# Filigranes et lignes parasites a retirer avant chunking
WATERMARK_PAT = [
    re.compile(r"Downloaded from .*geoscienceworld\.org.*", re.I),
    re.compile(r"by .{0,40}Universite? user on \d{1,2} \w+ \d{4}", re.I),
    re.compile(r"chapter-pdf/\d+/\S+\.pdf", re.I),
    re.compile(r"This content downloaded from .*", re.I),
    re.compile(r"All use subject to https?://\S+", re.I),
    re.compile(r"Downloaded by \[.*?\] at \d", re.I),
]

STOP = {"a", "an", "and", "the", "of", "in", "on", "for", "from", "to",
        "with", "by", "at", "as", "its", "their", "this", "that", "is",
        "are", "be", "et", "al", "eds", "ed", "pp", "vol", "no", "doi"}
WEAK_SURNAME = {"lee", "li", "wu", "sun", "gao", "kim", "hogg", "gray"}


def flat(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def toks(s):
    s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
    return {t for t in s.split() if len(t) > 2 and t not in STOP}


def surnames(ref):
    head = ref.split("(")[0]
    return [m.group(1).lower() for m in
            re.finditer(r"\b([A-Z][a-zA-Z\u00C0-\u024F'\-]{2,})", head)
            if m.group(1).lower() not in STOP]


# ---------------- extraction ----------------
def extract(path, backend, max_pages=None):
    """Retourne (texte, n_pages, erreur)."""
    try:
        if backend == "pymupdf":
            import fitz  # type: ignore
            d = fitz.open(str(path))
            n = len(d)
            parts = []
            for i, p in enumerate(d):
                if max_pages and i >= max_pages:
                    break
                parts.append(p.get_text() or "")
            d.close()
            return "".join(parts), n, None
        if backend == "pypdf":
            import pypdf  # type: ignore
            r = pypdf.PdfReader(str(path))
            n = len(r.pages)
            parts = []
            for i, p in enumerate(r.pages):
                if max_pages and i >= max_pages:
                    break
                try:
                    parts.append(p.extract_text() or "")
                except Exception:
                    pass
            return "".join(parts), n, None
        if backend == "pdfplumber":
            import pdfplumber  # type: ignore
            parts = []
            with pdfplumber.open(str(path)) as pdf:
                n = len(pdf.pages)
                for i, p in enumerate(pdf.pages):
                    if max_pages and i >= max_pages:
                        break
                    parts.append(p.extract_text() or "")
            return "".join(parts), n, None
    except Exception as e:
        return "", None, "{}: {}".format(type(e).__name__, str(e)[:90])
    return "", None, "backend inconnu"


def clean_text(t):
    """Retire filigranes ; retourne (texte_nettoye, n_lignes_retirees)."""
    removed = 0
    out = []
    for line in t.splitlines():
        if any(p.search(line) for p in WATERMARK_PAT):
            removed += 1
            continue
        out.append(line)
    return "\n".join(out), removed


def simulate_chunks(t, size, overlap):
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return 0
    if len(t) <= size:
        return 1
    step = max(1, size - overlap)
    return 1 + (len(t) - size + step - 1) // step


def dup_ratio(t):
    """Fraction de lignes strictement dupliquees : detecte le texte double."""
    lines = [l.strip() for l in t.splitlines() if len(l.strip()) > 40]
    if not lines:
        return 0.0
    c = Counter(lines)
    dups = sum(v - 1 for v in c.values() if v > 1)
    return round(dups / float(len(lines)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--refs", default=None)
    ap.add_argument("--out", default="preflight")
    ap.add_argument("--chunk-size", type=int, default=1888)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=5000,
                    help="seuil sous lequel un article est juge trop pauvre")
    ap.add_argument("--dump-text", action="store_true",
                    help="ecrit le texte nettoye en .txt dans <out>/text/")
    args = ap.parse_args()

    corpus = Path(os.path.expanduser(args.corpus_dir))
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("48e  CONTROLE PRE-VOL  --  corpus 41 documents")
    print("=" * 74)
    print("corpus      : {}".format(corpus))
    print("extracteurs : " + ", ".join(
        "{}={}".format(k, "OUI" if v else "non") for k, v in AVAIL.items()))
    if not AVAIL.get("pymupdf"):
        print("  [!] pymupdf ABSENT : c'est l'extracteur le plus robuste ici.")
        print("      pip install pymupdf")
    print("chunk       : taille {} / recouvrement {}".format(
        args.chunk_size, args.overlap))
    print()

    pdfs = sorted(p for p in corpus.rglob("*.pdf")
                  if ".ipynb_checkpoints" not in str(p)
                  and "-checkpoint" not in p.name.lower()
                  and not re.match(r"^(fig|figure|kg_portrait|table)[-_0-9]",
                                   p.name, re.I))

    # ---------- A. presence ----------
    print("--- A. PRESENCE ------------------------------------------------")
    print("  PDF trouves : {}".format(len(pdfs)))
    if len(pdfs) != 41:
        print("  [!] ATTENDU 41. Ecart de {}.".format(len(pdfs) - 41))
    else:
        print("  OK : 41 documents.")
    print()

    # ---------- B + C. extraction ----------
    print("--- B/C. EXTRACTION ET NETTOYAGE -------------------------------")
    rows = []
    order = [b for b in ("pymupdf", "pypdf", "pdfplumber") if AVAIL.get(b)]
    if not order:
        print("  [!] aucun extracteur disponible. Arret.")
        return

    txtdir = out / "text"
    if args.dump_text:
        txtdir.mkdir(exist_ok=True)

    for p in pdfs:
        res = OrderedDict()
        res["fichier"] = p.name
        res["taille_ko"] = round(p.stat().st_size / 1024)
        best_b, best_t, best_n = None, "", 0
        for b in order:
            t, npg, err = extract(p, b)
            res[b] = len(t.strip()) if not err else -1
            res[b + "_err"] = err or ""
            if len(t.strip()) > best_n:
                best_b, best_t, best_n = b, t, len(t.strip())
            if npg:
                res["n_pages"] = npg
        res["meilleur_extracteur"] = best_b
        res["chars_bruts"] = best_n

        cleaned, nwm = clean_text(best_t)
        res["lignes_filigrane"] = nwm
        res["chars_nettoyes"] = len(cleaned.strip())
        res["taux_lignes_dupliquees"] = dup_ratio(best_t)
        res["chunks_estimes"] = simulate_chunks(
            cleaned, args.chunk_size, args.overlap)

        # ecart entre extracteurs : signale une extraction douteuse
        vals = [res[b] for b in order if res.get(b, -1) > 0]
        res["ecart_extracteurs"] = (round(max(vals) / float(min(vals)), 2)
                                    if len(vals) > 1 and min(vals) > 0 else 1.0)

        if best_n == 0:
            verdict = "NO-GO  aucun texte (OCR requis)"
        elif res["chars_nettoyes"] < args.min_chars:
            verdict = "NO-GO  texte trop pauvre ({} car.)".format(
                res["chars_nettoyes"])
        elif res["ecart_extracteurs"] >= 2.0:
            verdict = "ATTENTION  extracteurs divergent x{}".format(
                res["ecart_extracteurs"])
        elif res["taux_lignes_dupliquees"] > 0.15:
            verdict = "ATTENTION  {:.0%} de lignes dupliquees".format(
                res["taux_lignes_dupliquees"])
        elif nwm > 0:
            verdict = "GO  ({} lignes de filigrane a retirer)".format(nwm)
        else:
            verdict = "GO"
        res["verdict"] = verdict
        rows.append(res)

        if args.dump_text and best_n:
            safe = re.sub(r"[^\w\-.]", "_", p.stem)[:120]
            (txtdir / (safe + ".txt")).write_text(cleaned, encoding="utf-8")

        flag = "" if verdict.startswith("GO") else "  <<<"
        print("  {:<52} {:>7} car. {:>4} chunks  {}{}".format(
            p.name[:52], res["chars_nettoyes"], res["chunks_estimes"],
            verdict, flag))
    print()

    # ---------- D. appariement ----------
    if args.refs and Path(os.path.expanduser(args.refs)).is_file():
        print("--- D. APPARIEMENT LB2019 --------------------------------------")
        refs = []
        with open(os.path.expanduser(args.refs), encoding="utf-8",
                  errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                body = line.split("|", 1)[1].strip() if "|" in line else line
                m, y = DOI_RE.search(body), YEAR_RE.search(body)
                refs.append({"raw": body,
                             "doi": m.group(0).rstrip(".,;)") if m else None,
                             "year": y.group(1) if y else None,
                             "sn": surnames(body), "tok": toks(body)})
        pairs = []
        for i, r in enumerate(refs):
            for j, p in enumerate(pdfs):
                hay = flat(p.name)
                s = 0.0
                if r["doi"]:
                    suf = flat(r["doi"].split("/", 1)[-1])
                    if len(suf) >= 6 and suf in hay:
                        s = 0.97
                if s == 0:
                    for sn in r["sn"]:
                        if len(sn) >= 4 and sn not in WEAK_SURNAME \
                                and flat(sn) in hay:
                            s = 0.95 if (r["year"] and r["year"] in hay) else 0.90
                            break
                if s == 0:
                    ct = toks(p.name)
                    if ct:
                        s = min(0.85, len(r["tok"] & ct) / float(len(ct)))
                if s >= 0.45:
                    pairs.append((s, i, j))
        pairs.sort(key=lambda t: -t[0])
        ra, pa = {}, {}
        for s, i, j in pairs:
            if i in ra or j in pa:
                continue
            ra[i], pa[j] = j, i
        print("  references {} | apparies {} | non apparies {}".format(
            len(refs), len(ra), len(refs) - len(ra)))
        for i, r in enumerate(refs):
            if i not in ra:
                print("    MANQUE : {}".format(r["raw"][:95]))
        for j, p in enumerate(pdfs):
            if j not in pa:
                print("    PDF sans reference : {}".format(p.name))
        print("  (appariement sur noms de fichiers ; les titres/DOI de 48c")
        print("   restent la reference)")
        print()

    # ---------- synthese ----------
    print("=" * 74)
    print("SYNTHESE")
    print("=" * 74)
    tot = sum(r["chunks_estimes"] for r in rows)
    wm = sum(r["lignes_filigrane"] for r in rows)
    nogo = [r for r in rows if r["verdict"].startswith("NO-GO")]
    warn = [r for r in rows if r["verdict"].startswith("ATTENTION")]
    print("  documents          : {}".format(len(rows)))
    print("  chunks estimes     : {}".format(tot))
    print("  lignes filigrane   : {}  (a retirer a la normalisation)".format(wm))
    print("  NO-GO              : {}".format(len(nogo)))
    print("  ATTENTION          : {}".format(len(warn)))
    for r in nogo + warn:
        print("    - {:<50} {}".format(r["fichier"][:50], r["verdict"]))
    print()
    if not nogo:
        print("  Aucun bloquant. Le pipeline peut etre relance apres")
        print("  application du correctif d'extracteur et de filigrane.")
    else:
        print("  NE PAS RELANCER avant traitement des NO-GO ci-dessus.")

    f = out / "preflight_41.csv"
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print()
    print("Ecrit : {}".format(f))
    if args.dump_text:
        print("Textes nettoyes : {}".format(txtdir))


if __name__ == "__main__":
    main()