#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
53_verify_repaired.py  --  rejouer 03 sur la provenance reparee
================================================================
POURQUOI UN WRAPPER PLUTOT QU'UN SCRIPT AUTONOME
    pipeline/03_verify_triples.py expose deja --chunk-chars, mais
    get_chunks() porte `max_chunks: int = 3` EN DUR, sans option de ligne
    de commande. Avec la provenance reparee mais 03 inchange, seuls 3 des
    5 chunks seraient lus.

    Ce script IMPORTE 03 et ne remplace que cette valeur par defaut, puis
    appelle son main(). Les prompts, le parseur de verdict, la logique de
    reessai et le backend restent STRICTEMENT ceux de 03 : aucune
    duplication, donc aucun risque de divergence entre la description et
    l'implementation -- le defaut meme que cette session a passe son temps
    a traquer.

    pipeline/03_verify_triples.py n'est jamais modifie sur disque.

PREREQUIS
    python 52_repair_provenance.py ...   (provenance reparee, top_chunks)

Usage :
    python 53_verify_repaired.py \
        --input  output/run15/pass_a/triples_repaired.jsonl \
        --output output/run15/pass_a/verified_triples.jsonl \
        --model "$MODEL_LOCAL" --backend hf \
        --max-chunks 5 --chunk-chars 2000

    # verifier le patch sans lancer le GPU :
    python 53_verify_repaired.py --input ... --output ... --check-only
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def load_stage03(path):
    p = Path(path)
    if not p.is_file():
        print("ERREUR : {} introuvable".format(p), file=sys.stderr)
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("stage03", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--max-chunks", type=int, default=5,
                    help="03 utilise 3 en dur ; 5 = tous les passages retenus")
    ap.add_argument("--chunk-chars", type=int, default=2000,
                    help="03 utilise 1500 ; 2000 = un chunk entier de run14")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stage03", default="pipeline/03_verify_triples.py")
    ap.add_argument("--check-only", action="store_true",
                    help="verifie le patch et les entrees, ne lance pas le GPU")
    args = ap.parse_args()

    print("=" * 76)
    print("53  verification sur provenance reparee")
    print("=" * 76)

    mod = load_stage03(args.stage03)

    # --- verification du point de patch -----------------------------
    if not hasattr(mod, "get_chunks") or not hasattr(mod, "verify_triple"):
        print("ERREUR : get_chunks/verify_triple absents de {}".format(
            args.stage03), file=sys.stderr)
        sys.exit(3)
    old_default = (mod.get_chunks.__defaults__ or (None,))[0]
    print("03 charge        : {}".format(args.stage03))
    print("max_chunks       : {} (defaut de 03) -> {}".format(
        old_default, args.max_chunks))
    print("chunk_chars      : 1500 (defaut de 03) -> {}".format(
        args.chunk_chars))

    _orig = mod.get_chunks

    def get_chunks_patched(triple, max_chunks=args.max_chunks):
        return _orig(triple, max_chunks)

    mod.get_chunks = get_chunks_patched

    # --- controle des entrees ---------------------------------------
    import json
    inp = Path(os.path.expanduser(args.input))
    if not inp.is_file():
        print("ERREUR : {} introuvable".format(inp), file=sys.stderr)
        sys.exit(2)
    n = with_top = chars = 0
    ncl = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            tc = (r.get("_provenance", {}) or {}).get("top_chunks") or []
            if tc:
                with_top += 1
                ncl.append(len(tc))
                chars += sum(len(c.get("text", "")) for c in tc)
    print()
    print("entree           : {}".format(inp))
    print("  triples        : {}".format(n))
    print("  avec top_chunks: {}".format(with_top))
    if ncl:
        print("  chunks/triple  : moy {:.2f}".format(sum(ncl) / len(ncl)))
        print("  car./triple    : moy {:.0f}".format(chars / len(ncl)))
        lus = sum(min(args.max_chunks, k) for k in ncl) * args.chunk_chars
        print("  car. LUS estimes: moy {:.0f}  ({:.0f} % du disponible)".format(
            lus / len(ncl), 100.0 * lus / max(1, chars)))
    if with_top < n:
        print("  [!] {} triple(s) SANS top_chunks : 03 retomberait sur".format(
            n - with_top))
        print("      context_preview pour ceux-la. Relancer 52.")
        if not args.check_only:
            sys.exit(4)
    print()

    if args.check_only:
        print("CHECK-ONLY : patch verifie, rien n'a ete execute.")
        return

    # --- appel de main() de 03, inchange -----------------------------
    argv = ["03_verify_triples.py",
            "--input", args.input,
            "--output", args.output,
            "--model", args.model,
            "--backend", args.backend,
            "--chunk-chars", str(args.chunk_chars)]
    if args.backend == "ollama":
        argv += ["--ollama-url", args.ollama_url]
    if args.limit:
        argv += ["--limit", str(args.limit)]

    print("appel : {}".format(" ".join(argv[1:])))
    print("-" * 76)
    old = sys.argv
    sys.argv = argv
    try:
        mod.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()