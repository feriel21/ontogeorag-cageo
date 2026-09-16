# Corpus

The 41-document corpus cannot be distributed due to copyright.

## Corpus composition

| Subcorpus | Documents | Purpose |
|---|---|---|
| `full_corpus/` | 41 PDFs | Main pipeline (used in paper) |
| `small_corpus/` | ~10 PDFs | Development & debugging |

## Reconstruction

Collect PDFs of peer-reviewed MTD literature (Scopus/Web of Science,
query: "mass transport deposit" OR "MTD seismic" OR "submarine landslide").

Key papers cited in the manuscript:
- Alves (2014) — Marine Geology
- Posamentier & Martinsen (2011) — SEPM Special Publication
- Reis et al. (2016) — Marine and Petroleum Geology
- Le Bouteiller et al. (2019) — [thesis, ground truth]

Place PDFs in `data/full_corpus/` then rebuild the index:
```bash
python pipeline/01_build_index.py \
    --pdf-dir data/full_corpus/ \
    --outdir  output/step1/
```

The BM25 index (`output/step1/chunks.jsonl`, ~40MB) is derived from
these PDFs and is equally not distributed.
