#!/usr/bin/env python3
"""189 — Frequency of the extraction vocabulary terms in the corpus.

PURPOSE
    Support the 27 added descriptors and the 21 settings: for each term of
    KNOWN_DESCRIPTORS and KNOWN_SETTINGS, count the papers and the passages of
    the index that contain it (whole word; the text is already lowercased by 49).

READ-ONLY
    No existing file is modified. The script refuses to overwrite its output.

USAGE (from the repository root)
    python 189_vocabulary_frequency.py \
        --chunks <run16E index>/chunks.jsonl \
        --out    output/run18/vocabulary_frequency.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.rag.constants import (  # noqa: E402
    KNOWN_DESCRIPTORS,
    KNOWN_SETTINGS,
    LB2019_BENCHMARK_DESCRIPTORS,
)


def pattern(term):
    # whole word; a hyphen or a space in the term may be written
    # as a hyphen or a space in the text (high-amplitude / high amplitude)
    body = r"[\s-]".join(re.escape(w) for w in re.split(r"[\s-]", term))
    return re.compile(r"(?<![a-z0-9-])" + body + r"(?![a-z0-9-])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit("ERROR: {} already exists. Refusing to overwrite.".format(out))

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
    n_docs = len({c["doc_id"] for c in chunks})
    print("passages: {}   papers: {}".format(len(chunks), n_docs))

    terms = [(t, "descriptor_LB2019") for t in sorted(LB2019_BENCHMARK_DESCRIPTORS)]
    terms += [(t, "descriptor_added") for t in
              sorted(KNOWN_DESCRIPTORS - LB2019_BENCHMARK_DESCRIPTORS)]
    terms += [(t, "setting") for t in sorted(KNOWN_SETTINGS)]

    rows = []
    for term, group in terms:
        rx = pattern(term)
        hit = [c for c in chunks if rx.search(c["text"])]
        docs = {c["doc_id"] for c in hit}
        rows.append({"term": term, "group": group,
                     "papers": len(docs), "passages": len(hit)})

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["term", "group", "papers", "passages"])
        w.writeheader()
        w.writerows(rows)

    for group in ("descriptor_LB2019", "descriptor_added", "setting"):
        g = [r for r in rows if r["group"] == group]
        g.sort(key=lambda r: -r["papers"])
        print("\n== {} ({} terms)".format(group, len(g)))
        for r in g:
            print("  {:<26} {:>3} papers  {:>5} passages".format(
                r["term"], r["papers"], r["passages"]))
        print("  -> terms present in >= 5 papers: {}/{}".format(
            sum(r["papers"] >= 5 for r in g), len(g)))
        print("  -> terms absent from the corpus: {}".format(
            [r["term"] for r in g if r["papers"] == 0]))
    print("\nwritten: {}".format(out))


if __name__ == "__main__":
    main()