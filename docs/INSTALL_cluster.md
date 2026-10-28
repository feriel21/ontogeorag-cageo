# Installing and running the run18 pipeline on a SLURM cluster

This guide installs the OntoGeoRAG pipeline (run18 settings) on a new cluster and runs it in two stages: a 10-minute test, then the full run.

## 1. What you need

| Item | Detail |
|---|---|
| Python | 3.10 |
| GPU | The recorded run used one A100 slice with 40 GB. Smaller GPUs are untested. |
| Disk | About 50 GB for the three models in the Hugging Face cache, plus outputs. |
| Hugging Face account | Needed for Llama 3.1, which is gated: accept the Meta licence on the model page first. |
| Corpus | The 41 PDFs listed in `data/corpus_table.md`. They are under publishers' copyright and are not in this repository. |
| SLURM | A project number (`--account`) and the name of a GPU partition. |

## 2. Code and environment

```bash
git clone https://github.com/feriel21/ontogeorag-cageo.git ~/ontogeorag
cd ~/ontogeorag && git checkout run18          # or the release tag
python3.10 -m venv ~/ontogeorag_venv
source ~/ontogeorag_venv/bin/activate
pip install -r environment/requirements-cpu.txt -r environment/requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch` must match the CUDA version of the cluster. If the pinned wheel does not load, install the `torch` build recommended by the cluster administrators and keep the other versions.

## 3. Models (once, on a login node with internet)

```bash
source ~/ontogeorag_venv/bin/activate
huggingface-cli login          # or: export HF_TOKEN=...
bash workflows/prefetch_models.sh
```

The script downloads Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.2 at the revisions used by the run, plus the reranker and SciBERT. Compute nodes then read the cache offline (`HF_HUB_OFFLINE=1` in the SLURM script).

## 4. Corpus

Put the 41 PDFs in one folder, for example `~/ontogeorag_corpus_pdfs`. Check them before indexing:

```bash
PYTHONPATH=. python scripts/tools/corpus_preflight.py --corpus-dir ~/ontogeorag_corpus_pdfs \
    --chunk-size 2000 --overlap 200 --out output/preflight
```

`data/corpus_table.md` gives, for each file, its sha256 and number of passages. A different copy of the same article (another publisher version) can give different passages. The index of the paper has 1,758 passages (sha256 `03db65349faa…`); the driver prints the values of the index it builds so they can be compared.

The index can be checked on its own, on CPU, in a few minutes, before any GPU job:

```bash
python 49_build_index_v2.py --pdf-dir ~/ontogeorag_corpus_pdfs --outdir output/test_index49 \
    --chunk-size 2000 --overlap 200 --expect-docs 41
wc -l output/test_index49/chunks.jsonl                    # expected: 1758
sha256sum output/test_index49/chunks.jsonl | cut -c1-12   # expected: 03db65349faa
```

The reference index was rebuilt this way from the 41 PDFs on 28 Oct 2026 with `pypdf` 6.15.0 and `PyMuPDF` 1.23.7 installed and `pdfplumber` absent (the script prints which readers it finds). With other reader versions, the text extracted from some PDFs can change.

## 5. SLURM script

Edit the lines marked `A_REMPLIR` in `slurm/orion_run18.sbatch`: `--account` (project number), `--partition` (GPU partition), and, if the cluster uses them, the `module load` line. Keep `--gres=gpu:1`.

## 6. Test (about 10 minutes)

```bash
cd ~/ontogeorag
mkdir -p logs                  # SLURM needs this folder before the job starts
sbatch --export=ALL,SMOKE=1 slurm/orion_run18.sbatch
squeue -u $USER
tail -f logs/run18_<jobid>.out
```

The test runs the whole chain on 5 queries and 20 verified triples. Its numbers mean nothing. Check that:

- the `GATE` section passes (checksums and leak audit 174, exit code 0);
- the log ends with `manifest: … (22 steps …)` and `smoke_run18 finished` (21 steps if an existing index was copied with `INDEX_FROM`).

## 7. Full run

```bash
sbatch slurm/orion_run18.sbatch
```

On the recorded run, extraction and verification of both passes took about 3.5 hours; the panel adds to this. Outputs go to `output/run18_rerun/`:

| File | Content |
|---|---|
| `kg_final/flat_run18_rerun_enforced_dedup.json` | delivered graph |
| `m4_panel/m4_quarantine.jsonl` | relations rejected by the panel |
| `kg_final/article_metrics.json` | metrics |
| `manifest.json` | each step with its script checksum, arguments and times |

A rerun is not expected to reproduce the published graph byte for byte, because pass b samples at temperature 0.3. Pass a (temperature 0) is reproducible.

## 8. Common problems

| Message | Cause |
|---|---|
| `Invalid account` or job refused | wrong `--account` or partition |
| `401` / `403` on Llama | licence not accepted, or no token |
| model download on a compute node fails | models not prefetched; run step 3 on a login node |
| `CUDA out of memory` | GPU too small; request a larger one |
| `STOP: … exists, nothing is overwritten` | choose another `OUT` or move the previous output |
