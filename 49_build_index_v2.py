#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
49_build_index_v2.py  --  ingestion corrigee (additif)
================================================================
Remplace fonctionnellement pipeline/01_build_index.py SANS LE MODIFIER.
Ecrit dans un repertoire de sortie distinct (par defaut output/run14/step1).

Quatre corrections par rapport a 01 :

  1. SELECTION D'EXTRACTEUR PAR QUALITE MESUREE
     01 utilise pdfplumber puis, a defaut, pymupdf, sans jamais verifier
     ce qui sort. Ici les trois extracteurs sont essayes et celui qui
     produit le plus de caracteres alphanumeriques est retenu.

  2. DECODAGE DES GLYPHES
     Certains PDF sortent en /C65/C98/C114... ou C<n> code un caractere.
     Mulder & Alexander 2001a (36 903 glyphes) et Mangeney-Castelnau 2005
     (216) sont dans ce cas. Sans decodage, le premier est illisible et
     avait ete exclu du corpus.

  3. RETRAIT DES FILIGRANES
     Les PDF issus de GeoScienceWorld portent en pied de chaque page une
     ligne de telechargement (URL, identifiant de chapitre, nom d'usager).
     128 lignes sur 5 fichiers. Non retirees, elles injectent des tokens
     parasites dans BM25 et dans les embeddings.

  4. CONTROLE DE QUALITE BLOQUANT
     01 accepte silencieusement un document rendant du bruit. Ici tout
     document sous les seuils est REFUSE et journalise ; le script sort
     en erreur plutot que de produire un index degrade.

normalize_text et chunk_text sont des copies CONFORMES de 01, afin que
les frontieres de chunks restent comparables entre les deux runs.

Ne modifie aucun script de pipeline. Refuse d'ecraser un fichier existant.

Usage :
    python 49_build_index_v2.py \
        --pdf-dir data/full_corpus \
        --outdir  output/run14/step1 \
        --chunk-size <valeur de run13/step1/index_meta.json> \
        --overlap   <idem>

    # verification a blanc, n'ecrit rien :
    python 49_build_index_v2.py --pdf-dir data/full_corpus \
        --outdir output/run14/step1 --dry-run
"""

import argparse
import hashlib
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

# ── extracteurs disponibles ──────────────────────────────────────────
AVAIL = {}
for _n, _m in (("pypdf", "pypdf"), ("pymupdf", "fitz"),
               ("pdfplumber", "pdfplumber")):
    try:
        __import__(_m)
        AVAIL[_n] = True
    except Exception:
        AVAIL[_n] = False

# ── normalisation : COPIE CONFORME DE 01 ─────────────────────────────
SYNONYM_MAP = {
    "mtd": "mass transport deposit",
    "mtds": "mass transport deposits",
    "mass wasting": "mass transport deposit",
    "mass movement": "mass transport deposit",
    "submarine landslide": "mass transport deposit",
    "slope failure": "slope failure",
}


def normalize_text(text):
    """COPIE CONFORME de 01_build_index.py. Ne pas modifier."""
    text = text.lower()
    for abbr, full in SYNONYM_MAP.items():
        text = re.sub(r"\b" + re.escape(abbr) + r"\b", full, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text, doc_id, chunk_size=800, overlap=200):
    """Copie de 01_build_index.py AVEC UNE SEULE DEVIATION, documentee.

    DEVIATION (correction 5) : ajout de `if end >= len(text): break`.

    Dans 01, en fin de document `end` atteint len(text), l'ajustement de
    frontiere de phrase est saute (`end < len(text)` faux), puis
    `start = end - overlap` vaut len(text)-overlap. La condition de sortie
    `start >= len(text) - 50` est alors FAUSSE pour tout overlap > 50, et
    la boucle reemet indefiniment le meme dernier chunk.
    Verifie sur trois formes de texte : aucune terminaison.

    Le reste (taille de fenetre, recherche de frontiere, seuil de 50
    caracteres, indexation des chunk_id) est inchange, de sorte que les
    frontieres des chunks non terminaux restent identiques a 01.
    """
    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", end - 100, end)
            if boundary != -1:
                end = boundary + 1
        chunk_text_str = text[start:end].strip()
        if len(chunk_text_str) > 50:
            chunks.append({
                "chunk_id": "{}::{}".format(doc_id, chunk_idx),
                "doc_id": doc_id,
                "text": chunk_text_str,
                "char_start": start,
                "char_end": end,
            })
            chunk_idx += 1
        if end >= len(text):          # <-- DEVIATION : terminaison
            break
        start = end - overlap
        if start >= len(text) - 50:
            break
    return chunks


# ── correction 3 : filigranes ────────────────────────────────────────
WATERMARK_PAT = [
    re.compile(r"Downloaded from .*geoscienceworld\.org.*", re.I),
    re.compile(r"by .{0,40}Universit[ea]? user on \d{1,2} \w+ \d{4}", re.I),
    re.compile(r"chapter-pdf/\d+/\S+\.pdf", re.I),
    re.compile(r"This content downloaded from .*", re.I),
    re.compile(r"All use subject to https?://\S+", re.I),
    re.compile(r"Downloaded by \[.*?\] at \d", re.I),
]


def strip_watermarks(t):
    kept, removed = [], 0
    for line in t.splitlines():
        if any(p.search(line) for p in WATERMARK_PAT):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


# ── correction 2 : glyphes ───────────────────────────────────────────
GLYPH_RE = re.compile(r"/[A-Za-z]{1,2}(\d{1,4})")


def decode_glyphs(t):
    n = len(GLYPH_RE.findall(t))
    if n == 0:
        return t, 0

    def rep(m):
        try:
            c = int(m.group(1))
            return chr(c) if 9 <= c <= 0x10FFFF else " "
        except Exception:
            return " "

    return GLYPH_RE.sub(rep, t), n


# ── correction 4 : qualite ───────────────────────────────────────────
def quality(t):
    """Indicateurs de qualite.

    ATTENTION : le ratio ALPHANUMERIQUE est trompeur sur un texte encode
    en glyphes -- dans "/C65/C98/C114", chiffres et lettre C comptent
    comme alphanumeriques et donnent un ratio de 0,76 sur du bruit pur.
    Le ratio ALPHABETIQUE, lui, s'effondre. C'est donc lui qui sert de
    critere de selection et de declenchement du decodage.
    """
    if not t:
        return {"chars": 0, "alnum": 0, "alnum_ratio": 0.0,
                "alpha": 0, "alpha_ratio": 0.0,
                "mots": 0, "mots_uniq": 0, "glyphes": 0}
    alnum = sum(c.isalnum() for c in t)
    alpha = sum(c.isalpha() for c in t)
    w = t.split()
    return {"chars": len(t), "alnum": alnum,
            "alnum_ratio": round(alnum / float(len(t)), 4),
            "alpha": alpha,
            "alpha_ratio": round(alpha / float(len(t)), 4),
            "mots": len(w),
            "mots_uniq": len(set(x.lower() for x in w)),
            "glyphes": len(GLYPH_RE.findall(t))}


def extract_raw(path, backend):
    try:
        if backend == "pymupdf":
            import fitz  # type: ignore
            d = fitz.open(str(path))
            t = "\n".join(p.get_text() or "" for p in d)
            d.close()
            return t, None
        if backend == "pypdf":
            import pypdf  # type: ignore
            r = pypdf.PdfReader(str(path))
            parts = []
            for p in r.pages:
                try:
                    parts.append(p.extract_text() or "")
                except Exception:
                    parts.append("")
            return "\n".join(parts), None
        if backend == "pdfplumber":
            import pdfplumber  # type: ignore
            with pdfplumber.open(str(path)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages), None
    except Exception as e:
        return "", "{}: {}".format(type(e).__name__, str(e)[:120])
    return "", "backend indisponible"


# ── correction 1 : selection par qualite ─────────────────────────────
def pdf_to_text_v2(path):
    """Retourne (texte, backend, glyphes_decodes, filigranes, erreurs).

    Selection par nombre de caracteres ALPHABETIQUES (voir quality()).
    Le decodage de glyphes est tente des que le motif /C<n> est present
    en nombre, sans condition sur le ratio alphanumerique : celui-ci
    reste eleve sur du texte de glyphes et masquait le probleme.
    """
    cands, errs = [], []
    for b in ("pymupdf", "pypdf", "pdfplumber"):
        if not AVAIL.get(b):
            continue
        t, err = extract_raw(path, b)
        if err:
            errs.append("{}: {}".format(b, err))
            continue
        if not t:
            continue
        q = quality(t)
        if q["glyphes"] > 100:
            dt, n = decode_glyphs(t)
            dq = quality(dt)
            # Comparer les RATIOS, pas les comptes absolus : le texte brut
            # "/C65/C98..." contient tant de lettres C qu'il totalise plus
            # de caracteres alphabetiques que sa version decodee, tout en
            # etant illisible. Le ratio, lui, passe de ~0,21 a ~0,85.
            if dq["alpha_ratio"] > q["alpha_ratio"]:
                cands.append((dq["alpha"], dt, b, n))
                continue
        cands.append((q["alpha"], t, b, 0))
    if not cands:
        return "", None, 0, 0, errs
    cands.sort(key=lambda x: -x[0])
    _, t, b, ndec = cands[0]
    t, nwm = strip_watermarks(t)
    return t, b, ndec, nwm, errs


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--min-alpha-ratio", type=float, default=0.55,
                    help="fraction minimale de caracteres alphabetiques")
    ap.add_argument("--min-unique-words", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=5000)
    ap.add_argument("--expect-docs", type=int, default=None,
                    help="sort en erreur si le compte differe (ex. 41)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="autorise l'ecrasement (desactive par defaut)")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    outdir = Path(args.outdir)
    chunks_path = outdir / "chunks.jsonl"

    if chunks_path.exists() and not args.force and not args.dry_run:
        print("ERREUR : {} existe deja. Refus d'ecraser.".format(chunks_path),
              file=sys.stderr)
        print("         Choisir un autre --outdir, ou passer --force.",
              file=sys.stderr)
        sys.exit(2)

    pdfs = sorted(p for p in pdf_dir.glob("*.pdf")
                  if ".ipynb_checkpoints" not in str(p)
                  and "-checkpoint" not in p.name.lower())

    print("=" * 76)
    print("49_build_index_v2   {}".format("(DRY-RUN)" if args.dry_run else ""))
    print("=" * 76)
    print("pdf-dir     : {}".format(pdf_dir))
    print("outdir      : {}".format(outdir))
    print("chunk       : {} / recouvrement {}".format(
        args.chunk_size, args.overlap))
    print("extracteurs : " + ", ".join(
        "{}={}".format(k, "OUI" if v else "non") for k, v in AVAIL.items()))
    print("PDF trouves : {}".format(len(pdfs)))
    print()

    all_chunks, report, rejected = [], [], []
    for p in pdfs:
        doc_id = p.stem
        t, backend, ndec, nwm, errs = pdf_to_text_v2(p)
        q = quality(t)
        reason = None
        if q["chars"] == 0:
            reason = "aucun texte extrait"
        elif q["chars"] < args.min_chars:
            reason = "texte trop court ({} car.)".format(q["chars"])
        elif q["alpha_ratio"] < args.min_alpha_ratio:
            reason = "ratio alphabetique {:.2f} < {} (texte probablement " \
                     "corrompu ou en glyphes)".format(
                         q["alpha_ratio"], args.min_alpha_ratio)
        elif q["mots_uniq"] < args.min_unique_words:
            reason = "vocabulaire absurde ({} mots uniques)".format(
                q["mots_uniq"])

        norm = normalize_text(t)
        chunks = chunk_text(norm, doc_id, args.chunk_size, args.overlap)

        rec = OrderedDict([
            ("doc_id", doc_id), ("fichier", p.name), ("sha16", sha16(p)),
            ("backend", backend), ("glyphes_decodes", ndec),
            ("filigranes_retires", nwm),
            ("chars", q["chars"]), ("alpha_ratio", q["alpha_ratio"]),
            ("alnum_ratio", q["alnum_ratio"]),
            ("mots_uniq", q["mots_uniq"]),
            ("chars_normalises", len(norm)), ("chunks", len(chunks)),
            ("erreurs_extracteurs", errs), ("rejete", bool(reason)),
            ("motif_rejet", reason),
        ])
        report.append(rec)

        if reason:
            rejected.append(rec)
            print("  REJET  {:<46} {}".format(p.name[:46], reason))
            continue
        all_chunks.extend(chunks)
        extra = []
        if ndec:
            extra.append("{} glyphes decodes".format(ndec))
        if nwm:
            extra.append("{} filigranes".format(nwm))
        print("  ok     {:<46} {:>5} chunks  [{}]{}".format(
            p.name[:46], len(chunks), backend,
            "  " + ", ".join(extra) if extra else ""))

    print()
    print("-" * 76)
    ok = len(pdfs) - len(rejected)
    print("  documents retenus : {}".format(ok))
    print("  documents rejetes : {}".format(len(rejected)))
    print("  chunks totaux     : {}".format(len(all_chunks)))
    print("  glyphes decodes   : {}".format(
        sum(r["glyphes_decodes"] for r in report)))
    print("  filigranes retires: {}".format(
        sum(r["filigranes_retires"] for r in report)))
    print()

    if rejected:
        print("ECHEC : {} document(s) rejete(s). Rien n'a ete ecrit.".format(
            len(rejected)))
        for r in rejected:
            print("  - {} : {}".format(r["fichier"], r["motif_rejet"]))
        sys.exit(3)

    if args.expect_docs is not None and ok != args.expect_docs:
        print("ECHEC : {} documents retenus, {} attendus. Rien n'a ete "
              "ecrit.".format(ok, args.expect_docs))
        sys.exit(4)

    if args.dry_run:
        print("DRY-RUN : aucun fichier ecrit.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("Ecrit : {}  ({} chunks)".format(chunks_path, len(all_chunks)))

    meta = {
        "produced_by": "49_build_index_v2.py",
        "n_papers": len(pdfs),
        "n_retained": ok,
        "n_rejected": len(rejected),
        "n_chunks": len(all_chunks),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "extractor_policy": "meilleur ratio alphanumerique parmi "
                            "pymupdf/pypdf/pdfplumber",
        "glyph_decoding": True,
        "watermark_stripping": True,
        "quality_gate": {"min_alpha_ratio": args.min_alpha_ratio,
                         "min_unique_words": args.min_unique_words,
                         "min_chars": args.min_chars},
    }
    (outdir / "index_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    (outdir / "ingestion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Ecrit : {}".format(outdir / "index_meta.json"))
    print("Ecrit : {}".format(outdir / "ingestion_report.json"))


if __name__ == "__main__":
    main()