#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight check of a reconstituted corpus before indexing.

For a directory of PDFs obtained by the reader (the corpus cannot be redistributed):
  A  presence: 41 PDFs expected;
  B  text extraction with every available extractor (pymupdf, pypdf, pdfplumber),
     keeping the longest text;
  C  watermark and download-stamp lines, duplicated lines, estimated number of chunks;
  D  optional matching of file names against the list of the 41 references
     (data/lb2019_41_refs.txt).
Each document receives a verdict: GO, ATTENTION (warning) or NO-GO.

The delivered index used chunks of 2,000 characters with a 200-character overlap: pass
--chunk-size 2000 to estimate the same volume (the default 1888 is kept from the
original script). The index itself is built by 49_build_index_v2.py, which selects the
extractor per document and records it in ingestion_report.json.

Output: <out>/preflight_41.csv. Column names and verdict details are in French, as
produced by the original script:
    fichier = file, taille_ko = size (kB), <extractor> = characters extracted,
    <extractor>_err = extraction error, meilleur_extracteur = best extractor,
    chars_bruts = raw characters, lignes_filigrane = watermark lines,
    chars_nettoyes = characters after cleaning, taux_lignes_dupliquees = share of
    duplicated lines, chunks_estimes = estimated chunks, ecart_extracteurs = ratio of
    the longest to the shortest extraction;
    "aucun texte (OCR requis)" = no text, OCR required; "texte trop pauvre" = too little
    text; "extracteurs divergent" = extractors disagree; "de lignes dupliquees" =
    duplicated lines; "lignes de filigrane a retirer" = watermark lines to remove.

Read-only; writes only under --out.

Historical name: 48e_preflight_41.py

Usage
    python scripts/tools/corpus_preflight.py --corpus-dir /path/to/corpus --refs data/lb2019_41_refs.txt --out output/preflight --chunk-size 2000 --overlap 200
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

# Watermark and stamp lines removed before chunking
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
    """Return (text, n_pages, error)."""
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
    """Remove watermark lines; return (cleaned_text, n_removed_lines)."""
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
    """Share of exactly duplicated lines; detects doubled text."""
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
                    help="minimum number of characters after cleaning")
    ap.add_argument("--dump-text", action="store_true",
                    help="write the cleaned text of each document under <out>/text/")
    args = ap.parse_args()

    corpus = Path(os.path.expanduser(args.corpus_dir))
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("CORPUS PREFLIGHT  --  41 documents")
    print("=" * 74)
    print("corpus      : {}".format(corpus))
    print("extractors  : " + ", ".join(
        "{}={}".format(k, "yes" if v else "no") for k, v in AVAIL.items()))
    if not AVAIL.get("pymupdf"):
        print("  [!] pymupdf not installed: it is the most robust extractor on this corpus.")
        print("      pip install pymupdf")
    print("chunk       : size {} / overlap {}".format(
        args.chunk_size, args.overlap))
    print()

    pdfs = sorted(p for p in corpus.rglob("*.pdf")
                  if ".ipynb_checkpoints" not in str(p)
                  and "-checkpoint" not in p.name.lower()
                  and not re.match(r"^(fig|figure|kg_portrait|table)[-_0-9]",
                                   p.name, re.I))

    # ---------- A. presence ----------
    print("--- A. PRESENCE ------------------------------------------------")
    print("  PDFs found: {}".format(len(pdfs)))
    if len(pdfs) != 41:
        print("  [!] EXPECTED 41. Difference: {}.".format(len(pdfs) - 41))
    else:
        print("  OK: 41 documents.")
    print()

    # ---------- B + C. extraction ----------
    print("--- B/C. EXTRACTION AND CLEANING ------------------------------")
    rows = []
    order = [b for b in ("pymupdf", "pypdf", "pdfplumber") if AVAIL.get(b)]
    if not order:
        print("  [!] no extractor available. Stopping.")
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

        # ratio between extractors: a large ratio flags a doubtful extraction
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
        print("  {:<52} {:>7} chars {:>4} chunks  {}{}".format(
            p.name[:52], res["chars_nettoyes"], res["chunks_estimes"],
            verdict, flag))
    print()

    # ---------- D. appariement ----------
    if args.refs and Path(os.path.expanduser(args.refs)).is_file():
        print("--- D. MATCHING AGAINST THE LB2019 REFERENCE LIST ------------")
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
        print("  references {} | matched {} | unmatched {}".format(
            len(refs), len(ra), len(refs) - len(ra)))
        for i, r in enumerate(refs):
            if i not in ra:
                print("    MISSING: {}".format(r["raw"][:95]))
        for j, p in enumerate(pdfs):
            if j not in pa:
                print("    PDF without reference: {}".format(p.name))
        print("  (matching uses file names only)")
        print()

    # ---------- synthese ----------
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    tot = sum(r["chunks_estimes"] for r in rows)
    wm = sum(r["lignes_filigrane"] for r in rows)
    nogo = [r for r in rows if r["verdict"].startswith("NO-GO")]
    warn = [r for r in rows if r["verdict"].startswith("ATTENTION")]
    print("  documents          : {}".format(len(rows)))
    print("  estimated chunks   : {}".format(tot))
    print("  watermark lines    : {}  (removed before chunking)".format(wm))
    print("  NO-GO              : {}".format(len(nogo)))
    print("  ATTENTION          : {}".format(len(warn)))
    for r in nogo + warn:
        print("    - {:<50} {}".format(r["fichier"][:50], r["verdict"]))
    print()
    if not nogo:
        print("  No blocking document.")
    else:
        print("  Resolve the NO-GO documents above before indexing.")

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
    print("Written: {}".format(f))
    if args.dump_text:
        print("Cleaned texts: {}".format(txtdir))


if __name__ == "__main__":
    main()