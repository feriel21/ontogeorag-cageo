# Corpus

The corpus is the list of 41 publications in the References sheet of the supplementary table of Le Bouteiller et al. (2019). The sheet has 42 entries; entry 42 is a website and is not part of the corpus. The full references are in `lb2019_41_refs.txt`.

The publications are under publishers' copyright and are **not distributed**, nor is the index built from them (`chunks.jsonl`).

## Corpus table

`corpus_table.md` is generated from `lb2019_41_refs.txt`, the run16E index and the corpus files. For each publication it gives the reference and DOI. For each indexed file it gives the file name, sha256 checksum, number of passages, PDF extractor and matching reference. Use it to check a rebuilt corpus file by file.

## Index of the delivered run (run16E)

Built by `49_build_index_v2.py` (which replaces `pipeline/01_build_index.py`, not distributed) with `--chunk-size 2000 --overlap 200 --expect-docs 41`: 41 documents, 1,758 passages, none rejected. 27 documents were read with pypdf and 14 with PyMuPDF. The sha256 of `chunks.jsonl` is in `corpus_table.md`.

## Preflight

`scripts/tools/corpus_preflight.py` checks a PDF folder before indexing. Its CSV columns and verdicts are in French; the glossary is in its docstring.

## Rebuilding

1. Collect the 41 publications listed in `lb2019_41_refs.txt`.
2. Compare each file with `corpus_table.md` (the sha256 identifies the exact file used; another copy of the same article, for example a different publisher version, may give different passages).
3. Run `PDF_DIR=/path/to/the/pdfs bash workflows/run_pipeline.sh` from the repository root.