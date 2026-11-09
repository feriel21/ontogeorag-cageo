#!/usr/bin/env python3
"""183 — Does a replayed deterministic pass give the same relations again?

Compares two raw_triples.jsonl files (output of 02) on the set of DISTINCT
relations (source, relation, target), in lowercase with normalised spaces.
The measure is the same as the "279/279" of run16E.
Writes the summary next to --b.

USAGE
    python 183_comparer_passes.py --a output/run18/pass_a/raw_triples.jsonl \
        --b output/run18_repro/pass_a/raw_triples.jsonl
"""
import argparse
import json
from pathlib import Path


def keys(p):
    out = []
    for l in open(p, encoding="utf-8"):
        if l.strip():
            t = json.loads(l)
            out.append(tuple(" ".join(str(t.get(k, "")).lower().split()) for k in ("source", "relation", "target")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="passe d'origine")
    ap.add_argument("--b", required=True, help="passe rejouee")
    args = ap.parse_args()
    A, B = keys(args.a), keys(args.b)
    sa, sb = set(A), set(B)
    L = ["=== 183 reproductibilite de la passe deterministe ===",
         "origine : %d relations brutes, %d distinctes" % (len(A), len(sa)),
         "rejouee : %d relations brutes, %d distinctes" % (len(B), len(sb)),
         "communes : %d ; seulement origine : %d ; seulement rejouee : %d" % (len(sa & sb), len(sa - sb), len(sb - sa)),
         "identiques (meme liste, meme ordre) : %s" % (A == B)]
    txt = "\n".join(L)
    Path(Path(args.b).parent / "_reproductibilite.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()