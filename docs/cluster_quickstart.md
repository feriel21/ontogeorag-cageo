# run18 on a SLURM cluster — copy-paste quickstart

Short version of `docs/INSTALL_cluster.md` for a session where the repository is already cloned and the Python environment already created.
Fill in **block 0 only**. Every other block is copied as is, in order, in the same terminal. Each block ends with the result to check before going on.

No file needs editing: `slurm/run18.sbatch` reads `REPO`, `VENV`, `HF_HOME`, `PDF_DIR`, `SMOKE`, `INDEX_FROM` and `OUT` from the environment, and `--account` / `--partition` given on the `sbatch` command line replace the `A_REMPLIR` lines. The cluster's values therefore never enter the repository.

Use `workflows/run_pipeline_run18.sh` (called by the sbatch), not `workflows/run_pipeline.sh`.

---

## 0. Values to fill in (once per terminal session)

```bash
# --- FILL IN ---------------------------------------------------------
export CLUSTER_ACCOUNT="FILL_ME"     # project number:  sacctmgr show assoc user=$USER format=account%30
export CLUSTER_PARTITION="FILL_ME"   # GPU partition:   sinfo -o "%P %G %l"
export REPO="$HOME/ontogeorag"     # where the repository was cloned
export VENV="$HOME/ontogeorag_venv"  # the venv created for OntoGeoRAG
export PDF_DIR="FILL_ME"           # folder holding the 41 corpus PDFs
export HF_HOME="$HOME/.cache/huggingface"  # must be visible from compute nodes; ~50 GB free
# ---------------------------------------------------------------------
```

If the cluster needs modules for Python or CUDA, load them here too (`module avail python cuda` to see the names), for example `module load <python_module> <cuda_module>`. `sbatch --export=ALL` passes the loaded environment to the job.

Check: `env | grep -E "CLUSTER_|^REPO=|^VENV=|PDF_DIR|HF_HOME"` shows no `FILL_ME`.

---

## 1. Branch run18

```bash
cd "$REPO" && pwd
git fetch origin
git checkout run18
git pull origin run18
git log -1 --oneline
ls slurm/run18.sbatch workflows/prefetch_models.sh workflows/run_pipeline_run18.sh
```

Check: the three files are listed.

---

## 2. Environment

```bash
source "$VENV/bin/activate"
python --version
python -c "import pypdf, fitz; print('pypdf', pypdf.__version__); print(fitz.__doc__)"
python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
```

Check: Python 3.10, `pypdf 6.15.0`, `PyMuPDF 1.23.7` (versions of the paper index).
`torch.cuda.is_available()` is False on a login node; the job prints it again on the GPU node.

---

## 3. Models (login node with internet, once)

The Llama 3.1 licence must first be accepted on the Hugging Face account used here (model page `meta-llama/Llama-3.1-8B-Instruct`).

```bash
cd "$REPO" && pwd
huggingface-cli login
bash workflows/prefetch_models.sh 2>&1 | tee prefetch_models.log
```

Check: one `ok` line per model (three LLMs at the pinned revisions, reranker, SciBERT). A `WARNING` on the reranker or SciBERT means a different revision than the paper: the run still works, but write it down for the report.

---

## 4. Corpus and index (CPU, a few minutes, before any GPU job)

```bash
cd "$REPO" && pwd
ls "$PDF_DIR"/*.pdf | wc -l
PYTHONPATH=. python scripts/tools/corpus_preflight.py --corpus-dir "$PDF_DIR" \
    --chunk-size 2000 --overlap 200 --out output/preflight_check
python 49_build_index_v2.py --pdf-dir "$PDF_DIR" --outdir output/test_index \
    --chunk-size 2000 --overlap 200 --expect-docs 41
wc -l output/test_index/chunks.jsonl
sha256sum output/test_index/chunks.jsonl | cut -c1-12
```

Check: `41` PDFs; per-file sha256 and passage counts as in `data/corpus_table.md`; index `1758` passages and sha `03db65349faa`.
If the index differs, stop here: a PDF is a different copy of the article or the PDF readers differ.

---

## 5. Smoke test (about 10 minutes)

```bash
cd "$REPO" && pwd
mkdir -p logs
sbatch --account="$CLUSTER_ACCOUNT" --partition="$CLUSTER_PARTITION" \
       --export=ALL,SMOKE=1 slurm/run18.sbatch
squeue -u "$USER"
```

Follow the job (replace `<jobid>` with the number printed by `sbatch`):

```bash
tail -f logs/run18_<jobid>.err
tail -f logs/run18_<jobid>.out
```

Check in `logs/run18_<jobid>.out`:
- the `=== GATE ===` section passes;
- the line `index: 1758 passages, sha256 03db65349faa`;
- the end: `manifest: … (22 steps …)` then `=== smoke_run18 finished ===`.

The smoke numbers mean nothing; only the chain is tested.

---

## 6. Full run (several hours)

```bash
cd "$REPO" && pwd
sbatch --account="$CLUSTER_ACCOUNT" --partition="$CLUSTER_PARTITION" \
       --export=ALL slurm/run18.sbatch
squeue -u "$USER"
```

Outputs go to `output/run18_rerun/` (delivered graph in `kg_final/`, manifest in `manifest.json`).
To reuse the index checked in step 4 instead of rebuilding it, add `INDEX_FROM=output/test_index` to `--export` (the manifest then has 21 steps).

The driver never overwrites an output folder. If it stops with `STOP: … exists, nothing is overwritten`, add `OUT=output/<new_name>` to `--export`; do not delete the previous output.

---

## 7. If something fails

| Message | Cause |
|---|---|
| `Invalid account` or job refused | wrong `CLUSTER_ACCOUNT` or `CLUSTER_PARTITION` |
| `401` / `403` on Llama | licence not accepted, or not logged in |
| model download attempted on a compute node | step 3 not done, or `HF_HOME` differs between login and compute nodes |
| `CUDA out of memory` | GPU too small; the recorded run used an A100 slice with 40 GB |
| `STOP: set PDF_DIR …` | `PDF_DIR` not exported in block 0 |
