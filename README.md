# OntoGeoRAG 
This repository contains the code that built the knowledge graph on mass-transport deposits (MTDs) reported in the OntoGeoRAG manuscript, the tools that tie this code to the recorded run, the scripts that produce the data figures and the numbers of the tables of the manuscript, and the recorded outputs of the run and of its self-audit.

## What the pipeline does

Starting from 41 publications, the pipeline builds a BM25 index of text passages and extracts triples with Qwen2.5-7B-Instruct over hybrid retrieval (BM25 then cross-encoder reranking) in two passes, at temperature 0.0 and 0.3. It then repairs the provenance of each triple and verifies it against its source passages. Next come ontology validation, rule-based canonicalisation and lexicon enforcement. The two passes are merged into two tiers. Each relation is then judged by a panel of two models from other families (Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.2), which judge the cited evidence sentence. Finally the graph is aligned to the schema (flagging, not rejecting), deduplicated and measured.

The run of the manuscript, **run18**, indexed the 41 documents into 1,758 passages and asked 210 questions. It delivered **364 relations between 272 entities** (197 in tier 1). After rule-based cleaning, the graph holds **281 relations between 230 entities** (164 in tier 1). The earlier run, **run16E**, produced 232 relations between 216 entities (160 in tier 1, 72 in tier 2); its files are kept below.

The interactive page `docs/graph_run18.html` shows the run18 graph: https://feriel21.github.io/ontogeorag-cageo/graph_run18.html.

## What is not distributed

- **The 41 publications and the index built from them.** They are under publishers' copyright. `data/README.md` and `data/corpus_table.md` give the references, DOIs, file names, sha256 checksums and passage counts, so the corpus can be rebuilt and checked.
- **The commit that ran (`9c03730`).** Its tree contains the PDFs. The link between this repository and that commit is made by checksums (see below).
- **Intermediate outputs of run16E** (passes, verification, panel verdicts) are not included. Those of run18 are in `results/run18/` (see below).

`reference/reference_kg.json` is derived from the supplementary table of Le Bouteiller et al. (2019) and is reused with the permission of its authors.

## run16E: delivered graph and controls (earlier version)

`results/run16E/flat_run16E_enforced_dedup.json` is the delivered graph exactly as the run wrote it; the first 16 characters of its sha256 are recorded in `results/run16E/manifest.json`. `results/run16E/graph_with_located_provenance.json` holds the same 232 relations with, for each, the location of its evidence sentence in the index (`located_provenance`: file names, passage identifiers, and whether the match is exact or partial), and with local paths removed. Each relation carries its evidence sentence (at most 400 characters), the query that retrieved its passages, its tier, the primary verdict and the panel decision. `results/run16E/controls/` holds the metrics of the 464 adversarial controls of the run (`negatives_report.json`) and the re-analysis comparing four decision rules (`metrics_overall.csv`). `results/run16E/model_comparison/` holds the comparison of the three extraction models run with the run16E code (Supplementary Material S5): `tab_model_comparison.tex` and `95p_model_comparison_report.json`, written by `scripts/analysis/model_comparison.py`, whose commands are in `scripts/analysis/model_comparison_commands.sh`.

These files are released under CC BY 4.0 (`LICENSE-DATA`), except the evidence sentences, which are short quotations from the source articles and remain subject to their publishers' terms.

## run18 (current version)

The branch `main` holds the run18 settings and the tools to rerun and check them; the release `v2.1-run18` freezes this version. run18 is the run reported in the manuscript submitted to *Computers & Geosciences*. run16E, described above, is the earlier version; its files are unchanged.

- **Driver**: `workflows/run_pipeline_run18.sh` (210 queries in `configs/descriptor_queries_run18.jsonl`, full retrieved context, a gate that checks the run18 files and the leak audit `174_audit_fuite_benchmark.py` before any model is loaded). The changes with respect to run16E are listed in `pipeline/RUN18_PATCHES.md`.
- **Rule-based cleaning**: `workflows/postprocess_run18.sh` chains three read-only CPU scripts on a run output: `184_trace_requetes_et_pertes.py` (follows every extraction and checks the replay against the run's own files), `187_provenance_articles.py` (article, reference and deduced section of every relation) and `188_nettoyage_familles.py` (removes relations by fixed rules R1 bibliography, R2 local label, R3 subject absent from the source sentences; flags S1–S4; groups relations by entity families). Their messages and output columns are in French; `docs/run18_cleaning_outputs.md` describes them in English.
- **Recorded outputs**: `results/run18/` holds the delivered graph (`kg_final/`), the metrics, the panel quarantine, both passes (`pass_a/`, `pass_b/`), the manifest, and the cleaning outputs (`postprocess/`). The delivered graph has 364 relations (197 in tier 1); after cleaning, 281 relations (164 in tier 1). The figures and rules are in `results/run18/postprocess/*/_resume.txt`.
- **Self-audit outputs**: `results/run18/self_audit/` holds the recorded stress test (the 728 judged records, the verdicts of each verifying-LLM and the rates under each decision rule), produced by `slurm/stress18.sh`, and the outputs of the audit scripts behind the tables of the manuscript. `results/run18/self_audit/README.md` maps each folder to its script and to the table it supports, and explains why `kg_final/article_metrics.json` reports other descriptor and recall values than the manuscript. These outputs were written before the translation, so their file names are in French.
- **Interactive page**: `docs/graph_run18.html`, produced by `scripts/figures/run18/graph_page_run18.py`. It shows the cleaned or the delivered graph, by entity name or by name family; clicking an arrow shows the evidence sentence, the source papers, the tier, the panel decision and the verdict of each verifying-LLM. The page loads the graph library vis-network from a CDN, so it needs an internet connection.
- **Running on a SLURM cluster**: `docs/INSTALL_cluster.md` (full guide) and `docs/cluster_quickstart.md` (copy-paste version, including clusters without internet access).
- **Checking a rerun**: `scripts/tools/compare_runs.py --ref results/run18 --new <your output>` compares every stage, the delivered graph, the metrics and the cleaned graph with the recorded run.

These files are released under CC BY 4.0 (`LICENSE-DATA`), with the same exception for evidence sentences as above.

## Layout

| Path | Content |
|---|---|
| `49_build_index_v2.py`, `52_repair_provenance.py`, `53_verify_repaired.py` | pipeline stages kept at the repository root, byte-identical to the run |
| `58v2_freeze_benchmark.py`, `61_extend_queries.py` | benchmark freeze check; query extension |
| `174_audit_fuite_benchmark.py` | run18: leak audit of the queries and pipeline scripts against the evaluation edges (gate of the run18 driver) |
| `175_requetes_run18.py`, `176_generer_scripts_run18.py`, `177_candidats_exemples_hors_corpus.py` | run18: query set (210 queries), generation of `pipeline/02_extract_triples_run18.py` and `pipeline/04_clean_validate_run18.py` from 02 and 04 without modifying them, prompt examples taken from articles outside the corpus |
| `184_trace_requetes_et_pertes.py`, `187_provenance_articles.py`, `188_nettoyage_familles.py` | run18: trace of every extraction, provenance of every relation, rule-based cleaning and entity families (read-only, CPU) |
| `scripts/run18/audit/` | run18: audit scripts behind the numbers of the manuscript (comparison with the expert relations, self-audit decision rules, marker counts, vocabulary frequencies, checks on the graphs), copied from the cluster, then translated into English and renamed |
| `pipeline/` | extraction, verification, validation, fusion, metrics; `rag/` retrieval and model wrappers |
| `analysis_suite/` | schema enforcement, lexicon, rule canonicalisation, flattening |
| `m4/` | two-judge panel, tier integration, direction check, negative controls |
| `configs/` | ontology schema, extraction queries, benchmark, entity families |
| `reference/` | reference graph |
| `results/` | manifests of both runs; run16E delivered graph and control metrics under `results/run16E/`; recorded run18 outputs and cleaning under `results/run18/` |
| `scripts/analysis/`, `scripts/evaluation/` | analyses and evaluations reported in the manuscript |
| `scripts/figures/`, `scripts/paper/` | producers of the figures and tables |
| `scripts/figures/run18/` | producers of the five figures of the run18 manuscript and of the interactive page, from the files of `results/run18/` (`scripts/figures/run18/README.md`) |
| `scripts/tools/` | corpus preflight, benchmark extraction, comparison of a rerun with a recorded run (`compare_runs.py`) |
| `tools/` | provenance, replay and code-equivalence checks |
| `workflows/run_pipeline.sh` | public driver of the full pipeline (run16E settings) |
| `workflows/run_pipeline_run18.sh`, `workflows/postprocess_run18.sh` | run18 driver; rule-based cleaning of a run18 output |
| `workflows/prefetch_models.sh` | download of the models at the pinned revisions (login node with internet) |
| `slurm/run16_full.sh` | cluster driver used for run16E (kept as run) |
| `slurm/run18_clean.sh` | cluster driver used for the recorded run18 (kept as run) |
| `slurm/stress18.sh` | cluster driver used for the recorded stress test of run18 (comments translated; commands and messages kept as run) |
| `slurm/run18.sbatch` | SLURM template for the run18 driver on another cluster |
| `environment/` | pinned requirements (CPU and GPU) |
| `docs/graph_run18.html` | interactive page of the run18 graph |
| `docs/provenance.md` | recorded steps, options and checksums |
| `docs/INSTALL_cluster.md`, `docs/cluster_quickstart.md` | installing and running run18 on a SLURM cluster (full guide; copy-paste version) |

## How the published code is tied to the run

`results/run16E/manifest.json` records, for each of the 22 steps, the script, its sha256, its command line and its start and end times. `PROVENANCE.tsv` records the origin and checksum of each published file. `tools/verify_provenance.py` checks both. The 15 stage scripts have the checksums recorded in the manifest and keep their names and locations.

For run18, the manifest records 22 steps: one index copy and 21 steps run by 14 distinct scripts. The postprocess manifest records the checksums of the three cleaning scripts and of their input tables. Both are checked with:

```
python tools/verify_provenance.py --manifest results/run18/manifest.json \
    --postprocess results/run18/postprocess/postprocess_manifest.txt
```

### Language of comments

Comments and docstrings are in English, except in the files that must stay byte-identical to the code that ran, because a manifest records their checksum. These files keep their original comments, some of them in French:

- the stage scripts recorded in the run16E and run18 manifests, among them `49_build_index_v2.py`, `52_repair_provenance.py`, `53_verify_repaired.py`, `pipeline/02_extract_triples_run18.py` and `pipeline/04_clean_validate_run18.py`;
- the cleaning scripts recorded in the run18 postprocess manifest: `184_trace_requetes_et_pertes.py`, `187_provenance_articles.py` and `188_nettoyage_familles.py`. `docs/run18_cleaning_outputs.md` describes, in English, what these three scripts do and what each of their output columns means.

The modules these scripts import at run time (`pipeline/rag/`, `m4/m4_config.py`, `analysis_suite/kg_io.py`) are also left as they were at the recorded commit. In the other scripts, the comments were translated after the run, and `PROVENANCE.tsv` records both versions. For the Python scripts, `tools/check_code_unchanged.py` confirmed that the syntax tree, docstrings excluded, is unchanged. For the shell drivers, the commands are identical once the comments are removed. In these scripts, printed messages and output file names are not translated, because changing them would change the files that the scripts write.

The analysis, evaluation and figure scripts were translated into English and renamed after the run. `tools/check_code_unchanged.py` compared each translated script with its original at the level of the syntax tree. With `--allow-messages` it tolerates translated printed text, help strings and exception messages. With `--allow-renames` it tolerates consistent renaming of identifiers and command-line flags. All were found equivalent. Known limitation: expressions inside printed f-strings are not compared.

The run18 audit scripts in `scripts/run18/audit/` were translated completely, printed messages and output file names included, and renamed. With `--allow-strings`, `tools/check_code_unchanged.py` ignores the text of every string constant and compares the rest of the syntax tree; all were found equivalent. A rerun writes their outputs under the new English names; `scripts/run18/audit/README.md` gives the order in which to rerun them.

`tools/replay_cpu_steps.py` replays the final CPU steps from the recorded intermediate outputs. On the authors' copy it regenerates the delivered graph; the only differences are the absolute input paths stored in metadata fields (`flattened_from`, `input_kg`). It needs outputs that are not distributed.

`docs/provenance.md` lists all steps with their options.

## Environment

Python 3.10. Install `environment/requirements-cpu.txt`, plus `environment/requirements-gpu.txt` for the extraction, verification and panel steps. run16E and run18 each used one A100 slice with 40 GB (`a100_3g.40gb`). For run16E, extraction and verification of both passes took about 3.5 hours.

`pdfplumber` must not be installed: `49_build_index_v2.py` uses it when present, which changes the extracted text and the index. `accelerate` is required to load the language models. Both are handled by the requirements files.

| Model | Revision |
|---|---|
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |
| mistralai/Mistral-7B-Instruct-v0.2 | `63a8b081895390a26e140280378bc85ec8bce07a` |
| roberta-large-mnli (verifier comparison) | `2a8f12d27941090092df78e4ba6f0928eb5eac98` |
| cross-encoder/ms-marco-MiniLM-L-6-v2 (reranker) | not recorded |

Llama 3.1 requires accepting its licence on Hugging Face before download. For run18, `workflows/prefetch_models.sh` downloads the three language models at the revisions above and compares the reranker and SciBERT with the revisions of the recorded run. On a cluster without internet access, see step 3 of `docs/cluster_quickstart.md`.

## Running the pipeline

```bash
PDF_DIR=/path/to/the/41/pdfs bash workflows/run_pipeline.sh
SMOKE=1 PDF_DIR=/path/to/the/41/pdfs bash workflows/run_pipeline.sh   # chain check, about 10 minutes
```

The driver runs the 22 steps with the options of run16E and writes a manifest with the same structure. A rerun is not expected to reproduce the delivered graph byte for byte, because pass b samples at temperature 0.3.

run18:

```bash
PDF_DIR=/path/to/the/41/pdfs bash workflows/run_pipeline_run18.sh             # writes output/run18_rerun
SMOKE=1 PDF_DIR=/path/to/the/41/pdfs bash workflows/run_pipeline_run18.sh     # chain check, about 10 minutes
RUN_DIR=output/run18_rerun bash workflows/postprocess_run18.sh                # rule-based cleaning (CPU)
python scripts/tools/compare_runs.py --ref results/run18 --new output/run18_rerun --out output/compare_run18
```

On a SLURM cluster, follow `docs/cluster_quickstart.md`.

## Figures and tables

The five figures of the run18 manuscript are produced by the scripts of `scripts/figures/run18/` (see its README). The figures and tables below are those of the run16E version of the manuscript. Scripts are run from the repository root: `PYTHONPATH=. python scripts/<folder>/<script>.py`. Each producer stops without writing if it does not find the values published in the manuscript, except `fig_tradeoff.py`, which plots the values of Table 5 as constants; those values are checked by `table5_ablation.py`.

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
| `--allow-strings` | flag | tolerate changes to the text of any string |
| `--list-strings` | None | with --allow-strings, write the changed strings here |

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

See `CITATION.cff`. Archived on Zenodo (all versions): https://doi.org/10.5281/zenodo.22827694. Release `v1.0-cageo` holds run16E (https://doi.org/10.5281/zenodo.22827695). Release `v2.1-run18` holds run18, the run of the manuscript, with the outputs of its self-audit (https://doi.org/10.5281/zenodo.24108337); release `v2.0-run18` is the same run without these outputs. Code under the Apache License 2.0 (`LICENSE`); data under CC BY 4.0 (`LICENSE-DATA`).
