# OntoGeoRAG — code accompanying the *Computers & Geosciences* manuscript

This repository contains the code that built the knowledge graph on mass-transport deposits (MTDs) reported in the OntoGeoRAG manuscript, the tools that tie this code to the recorded run, and the scripts that produce every table and figure of the manuscript.

## What the pipeline does

Starting from 41 publications, the pipeline builds a BM25 index of text passages and extracts triples with Qwen2.5-7B-Instruct over hybrid retrieval (BM25 then cross-encoder reranking) in two passes, at temperature 0.0 and 0.3. It then repairs the provenance of each triple and verifies it against its source passages. Next come ontology validation, rule-based canonicalisation and lexicon enforcement. The two passes are merged into two tiers. Each relation is then judged by a panel of two models from other families (Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.2), which judge the cited evidence sentence. Finally the graph is aligned to the schema (flagging, not rejecting), deduplicated and measured.

The delivered run, **run16E**, indexed 41 documents into 1,758 passages. It produced a graph of **232 relations between 216 entities** (160 in tier 1, 72 in tier 2).

## What is not distributed

- **The 41 publications and the index built from them.** They are under publishers' copyright. `data/README.md` and `data/corpus_table.md` give the references, DOIs, file names, sha256 checksums and passage counts, so the corpus can be rebuilt and checked.
- **The commit that ran (`9c03730`).** Its tree contains the PDFs. The link between this repository and that commit is made by checksums (see below).
- **Intermediate outputs** of the run (passes, verification, panel verdicts) are not included.

`reference/reference_kg.json` is derived from the supplementary table of Le Bouteiller et al. (2019) and is reused with the permission of its authors.

## Delivered graph and controls

`results/run16E/flat_run16E_enforced_dedup.json` is the delivered graph exactly as the run wrote it; the first 16 characters of its sha256 are recorded in `results/run16E/manifest.json`. `results/run16E/graph_with_located_provenance.json` holds the same 232 relations with, for each, the location of its evidence sentence in the index (`located_provenance`: file names, passage identifiers, and whether the match is exact or partial), and with local paths removed. Each relation carries its evidence sentence (at most 400 characters), the query that retrieved its passages, its tier, the primary verdict and the panel decision. `results/run16E/controls/` holds the metrics of the 464 adversarial controls of the run (`negatives_report.json`) and the re-analysis comparing four decision rules (`metrics_overall.csv`).

These files are released under CC BY 4.0 (`LICENSE-DATA`), except the evidence sentences, which are short quotations from the source articles and remain subject to their publishers' terms.

## Layout

| Path | Content |
|---|---|
| `49_build_index_v2.py`, `52_repair_provenance.py`, `53_verify_repaired.py` | pipeline stages kept at the repository root, byte-identical to the run |
| `58v2_freeze_benchmark.py`, `61_extend_queries.py` | benchmark freeze check; query extension |
| `pipeline/` | extraction, verification, validation, fusion, metrics; `rag/` retrieval and model wrappers |
| `analysis_suite/` | schema enforcement, lexicon, rule canonicalisation, flattening |
| `m4/` | two-judge panel, tier integration, direction check, negative controls |
| `configs/` | ontology schema, extraction queries, benchmark, entity families |
| `reference/` | reference graph |
| `results/` | manifests of the runs (run16E is the delivered one); delivered graph and control metrics under `results/run16E/` |
| `scripts/analysis/`, `scripts/evaluation/` | analyses and evaluations reported in the manuscript |
| `scripts/figures/`, `scripts/paper/` | producers of the figures and tables |
| `scripts/tools/` | corpus preflight, benchmark extraction |
| `tools/` | provenance, replay and code-equivalence checks |
| `workflows/run_pipeline.sh` | public driver of the full pipeline |
| `slurm/run16_full.sh` | cluster driver used for the delivered run (kept as run) |
| `environment/` | pinned requirements (CPU and GPU) |
| `docs/provenance.md` | recorded steps, options and checksums |

## How the published code is tied to the run

`results/run16E/manifest.json` records, for each of the 22 steps, the script, its sha256, its command line and its start and end times. `PROVENANCE.tsv` records the origin and checksum of each published file. `tools/verify_provenance.py` checks both. The 15 stage scripts have the checksums recorded in the manifest and keep their names and locations.

The analysis, evaluation and figure scripts were translated into English and renamed after the run. `tools/check_code_unchanged.py` compared each translated script with its original at the level of the syntax tree. With `--allow-messages` it tolerates translated printed text, help strings and exception messages. With `--allow-renames` it tolerates consistent renaming of identifiers and command-line flags. All were found equivalent. Known limitation: expressions inside printed f-strings are not compared.

`tools/replay_cpu_steps.py` replays the final CPU steps from the recorded intermediate outputs. On the authors' copy it regenerates the delivered graph; the only differences are the absolute input paths stored in metadata fields (`flattened_from`, `input_kg`). It needs outputs that are not distributed.

`docs/provenance.md` lists all steps with their options.

## Environment

Python 3.10. Install `environment/requirements-cpu.txt`, plus `environment/requirements-gpu.txt` for the extraction, verification and panel steps. The delivered run used one A100 slice with 40 GB (`a100_3g.40gb`). Extraction and verification of both passes took about 3.5 hours.

| Model | Revision |
|---|---|
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |
| mistralai/Mistral-7B-Instruct-v0.2 | `63a8b081895390a26e140280378bc85ec8bce07a` |
| roberta-large-mnli (verifier comparison) | `2a8f12d27941090092df78e4ba6f0928eb5eac98` |
| cross-encoder/ms-marco-MiniLM-L-6-v2 (reranker) | not recorded |

Llama 3.1 requires accepting its licence on Hugging Face before download.

## Running the pipeline

```bash
PDF_DIR=/path/to/the/41/pdfs bash workflows/run_pipeline.sh
SMOKE=1 PDF_DIR=/path/to/the/41/pdfs bash workflows/run_pipeline.sh   # chain check, about 10 minutes
```

The driver runs the 22 steps with the options of run16E and writes a manifest with the same structure. A rerun is not expected to reproduce the delivered graph byte for byte, because pass b samples at temperature 0.3.

## Figures and tables

Scripts are run from the repository root: `PYTHONPATH=. python scripts/<folder>/<script>.py`. Each producer stops without writing if it does not find the values published in the manuscript, except `fig_tradeoff.py`, which plots the values of Table 5 as constants; those values are checked by `table5_ablation.py`.

| Manuscript | Producer |
|---|---|
| Figure 1 | `scripts/figures/fig_pipeline.py` |
| Figure 2 | `scripts/figures/fig_layers.py` |
| Figure 3 | `scripts/figures/fig_tradeoff.py` |
| Figure 4 | `scripts/figures/fig_descriptor_vocabulary.py` |
| Figure 5 | `scripts/figures/fig_vocabulary_gap.py` |
| Figure 6 | `scripts/figures/fig_sensitivity.py` |
| Figure 7 | `scripts/figures/fig_corroboration.py` |
| Table 5 | `scripts/paper/table5_ablation.py` |

The entity-family table used for corroboration is `configs/entity_families.csv`.

## Command-line reference

<!-- BEGIN CLI REFERENCE (generated by 167, do not edit) -->
### `scripts/figures/fig_corroboration.py`

Figure 7: corroboration of the delivered relations across the corpus.

| Option | Default | Help |
|---|---|---|
| `--corroboration` | required |  |
| `--graph` | required |  |
| `--out` | fig_corroboration |  |
| `--width` | 6.3 |  |
| `--height` | 2.8 |  |
| `--no-check` | flag |  |

### `scripts/figures/fig_descriptor_vocabulary.py`

Figure 4: the 40 descriptors of the extraction vocabulary and their occurrences in the corpus.

| Option | Default | Help |
|---|---|---|
| `--chunks` | required |  |
| `--graph` | required |  |
| `--constants` | str(Path(__file__).resolve().parents[2] / "pipeline" / "rag" / "constants.py") |  |
| `--out` | fig_descriptor_vocabulary |  |
| `--width` | 5.2 |  |
| `--height` | 6.6 |  |
| `--no-check` | flag |  |

### `scripts/figures/fig_layers.py`

Figure 2: the 232 delivered relations under six assessments.

| Option | Default | Help |
|---|---|---|
| `--graph` | required |  |
| `--direction` | required |  |
| `--out` | fig_layers |  |
| `--width` | 6.3 |  |
| `--height` | 3.6 |  |
| `--no-check` | flag |  |

### `scripts/figures/fig_pipeline.py`

Figure 1: pipeline overview.

| Option | Default | Help |
|---|---|---|
| `--outdir` | str(Path.home() / "ontogeorag" / "output" / "figures") |  |
| `--stem` | fig_pipeline |  |
| `--width` | 5.8 | authored width in inches; keep it below the text block width so that width=\linewidth enlarges it |
| `--height` | 5.0 |  |
| `--scale` | 1.0 | multiplier on all font sizes |
| `--greyscale` | flag |  |

### `scripts/figures/fig_sensitivity.py`

Figure 6: share of corrupted records the verifier rejects, by corruption rule.

| Option | Default | Help |
|---|---|---|
| `--report` | required |  |
| `--out` | fig_sensitivity |  |
| `--width` | 5.4 |  |
| `--height` | 2.3 |  |
| `--no-check` | flag |  |

### `scripts/figures/fig_tradeoff.py`

Figure 3: the five ablation configurations in the plane of coverage against benchmark recovery.

| Option | Default | Help |
|---|---|---|
| `--out` | fig_tradeoff |  |
| `--width` | 6.3 |  |
| `--height` | 4.2 |  |
| `--fontsize` | 9.0 |  |
| `--greyscale` | flag |  |

### `scripts/figures/fig_vocabulary_gap.py`

Figure 5: registers of description found outside the 40-descriptor extraction vocabulary.

| Option | Default | Help |
|---|---|---|
| `--open` | required | classification JSON of configuration D |
| `--suggested` | required | classification JSON of configuration E |
| `--out` | fig_vocabulary_gap |  |
| `--width` | 6.3 |  |
| `--height` | 3.9 |  |
| `--no-check` | flag |  |

### `scripts/paper/classify_open_vocabulary.py`

Classification of the descriptor values extracted outside the extraction vocabulary (Section 3.6).

| Option | Default | Help |
|---|---|---|
| `--run` | required |  |
| `--out` | required |  |

### `scripts/paper/table5_ablation.py`

Reproduce Table 5 (ablation across configurations A-E).

| Option | Default | Help |
|---|---|---|
| `--run` | required | run directory of one configuration (A to E) |
| `--queries-dir` | configs | directory holding the query files (default: configs) |
| `--check-previous` | None | optional paper_results.json from an earlier run, to compare against |
| `--out` | required | output directory (must not exist) |

### `tools/check_code_unchanged.py`

Check that an edited Python file does the same computation as its original.

| Option | Default | Help |
|---|---|---|
| `paths` |  |  |
| `--tree` | flag | compare every .py file of two directories |
| `--allow-messages` | flag | tolerate translated or removed text |
| `--allow-renames` | flag | tolerate consistent renaming |

### `tools/replay_cpu_steps.py`

Replay the CPU-only final stages of a recorded run and compare with its manifest.

| Option | Default | Help |
|---|---|---|
| `--repo` | . | repository root holding the scripts (default: .) |
| `--manifest` | required |  |
| `--panel-output` | required | m4_panel/tiered_kg_m4.json of the run |
| `--out` | required | output directory (must not exist) |

### `tools/verify_provenance.py`

Check that the repository files are the ones recorded in PROVENANCE.tsv and in the run manifest.

| Option | Default | Help |
|---|---|---|
| `--repo` | . | repository root (default: .) |
| `--provenance` | PROVENANCE.tsv |  |
| `--manifest` | results/run16E/manifest.json |  |

<!-- END CLI REFERENCE -->

## Citation and licence

See `CITATION.cff`. Archived on Zenodo (all versions): https://doi.org/10.5281/zenodo.22827694. Code under the Apache License 2.0 (`LICENSE`); data under CC BY 4.0 (`LICENSE-DATA`).