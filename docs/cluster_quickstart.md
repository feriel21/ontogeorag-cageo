# run18 on a SLURM cluster — copy-paste quickstart

Short version of `docs/INSTALL_cluster.md` for a session where the repository is already cloned and the Python environment already created.
Fill in **block 0 only**. Every other block is copied as is, in order, in the same terminal. Each block ends with the result to check before going on.

The cluster needs **no internet access**: the models are downloaded once on a machine that has internet (step 3a), copied to the cluster (step 3b), and the job reads them offline (`HF_HUB_OFFLINE=1`, set by the sbatch). Nothing has to be installed on the cluster for this; `huggingface-hub` is already in the venv.

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
export HF_HOME="FILL_ME"           # model cache on a shared scratch space visible from compute nodes; ~50 GB free
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
ls slurm/run18.sbatch workflows/run_pipeline_run18.sh
```

Check: the two files are listed.

---

## 2. Environment

```bash
source "$VENV/bin/activate"
python --version
python -c "import pypdf, fitz; print('pypdf', pypdf.__version__); print(fitz.__doc__)"
python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
python -c "import accelerate, transformers, sentence_transformers; print('accelerate', accelerate.__version__, '| transformers', transformers.__version__)"
python -c "import importlib.util as u; print('pdfplumber installed:', u.find_spec('pdfplumber') is not None)"
```

Check: Python 3.10, `pypdf 6.15.0`, `PyMuPDF 1.23.7` (versions of the paper index), `accelerate` importable, `pdfplumber installed: False`.
If `accelerate` is missing: `pip install -r environment/requirements-gpu.txt`. If pdfplumber is installed: `pip uninstall -y pdfplumber` (otherwise the index of step 4 differs from the paper's, or a PDF is rejected).
`torch.cuda.is_available()` is False on a login node; the job prints it again on the GPU node.

---

## 3. Models (once)

The pipeline needs five models: three LLMs at the revisions of the paper, the reranker and SciBERT.
The Llama 3.1 licence must first be accepted on the Hugging Face account used (model page `meta-llama/Llama-3.1-8B-Instruct`).

### 3a. Download — on a machine with internet (laptop or workstation, not the cluster)

`huggingface-hub` is installed at the same version as in the cluster venv; it provides the `hf` command (the former `huggingface-cli`).

```bash
python3 -m venv ~/hf_download_env
source ~/hf_download_env/bin/activate
pip install "huggingface-hub==0.36.0"
hf auth login
export HF_HOME="$HOME/hf_ontogeorag"
hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --exclude "*.bin" "*.pth" "original/*"
hf download mistralai/Mistral-7B-Instruct-v0.2 --revision 63a8b081895390a26e140280378bc85ec8bce07a --exclude "*.bin" "*.pth" "original/*"
hf download meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 --exclude "*.bin" "*.pth" "original/*"
hf download cross-encoder/ms-marco-MiniLM-L-6-v2
hf download allenai/scibert_scivocab_uncased
ls "$HF_HOME/hub"
du -sh "$HF_HOME"
```

The `--exclude` patterns skip duplicate copies of the LLM weights (`.bin`, `.pth`, `original/`). The pipeline loads the `.safetensors` files only, so they are not needed and would double the size. Do not use `--exclude` on SciBERT and the reranker: they are small, and SciBERT may only have a `.bin`.
With `huggingface-hub` 1.x, several patterns are given by repeating the flag (`--exclude "*.bin" --exclude "*.pth"`); version 0.36.0 takes them after a single `--exclude`, as above.

Check: five `models--…` folders. The reranker and SciBERT are loaded by name; compare their revision with the paper's:

```bash
ls "$HF_HOME/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/"   # paper: 233902d25c440f23af6f7d6e94d2946bac0bee0a
ls "$HF_HOME/hub/models--allenai--scibert_scivocab_uncased/snapshots/"       # paper: 24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1
```

A different revision still works; write it down for the report.
If the five models are already cached on another machine (another cluster, for example), copy that cache in step 3b instead of downloading again.

### 3b. Copy to the cluster — keep the symbolic links

In the Hugging Face cache, `snapshots/` only holds symbolic links to the files in `blobs/`. `scp -r`, WinSCP, FileZilla or a zip follow the links and copy every file twice (or more), which multiplies the size on the cluster. Use `rsync -a`, which keeps the links:

```bash
rsync -a --info=progress2 "$HOME/hf_ontogeorag/" <user>@<cluster>:<HF_HOME_on_the_cluster>/
```

If `rsync` is not available on the machine, use `tar`, which also keeps the links:

```bash
tar -cf hf_ontogeorag.tar -C "$HOME" hf_ontogeorag
scp hf_ontogeorag.tar <user>@<cluster>:<scratch_folder>/
```

then on the cluster:

```bash
cd <scratch_folder> && pwd
tar -xf hf_ontogeorag.tar
ls hf_ontogeorag/hub
```

and set `HF_HOME` in block 0 to `<scratch_folder>/hf_ontogeorag`.

### 3c. Check on the cluster (offline, as the job will read the models)

```bash
find "$HF_HOME/hub" -type l | wc -l
du -sh "$HF_HOME"
HF_HUB_OFFLINE=1 python -c "from huggingface_hub import snapshot_download as s; [print('ok', r, s(r, revision=v, local_files_only=True)) for r, v in [('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'), ('meta-llama/Llama-3.1-8B-Instruct', '0e9e39f249a16976918f6564b8830bc894c89659'), ('mistralai/Mistral-7B-Instruct-v0.2', '63a8b081895390a26e140280378bc85ec8bce07a'), ('cross-encoder/ms-marco-MiniLM-L-6-v2', None), ('allenai/scibert_scivocab_uncased', None)]]"
```

Check: the number of links is not `0`; the size is close to the size measured in 3a; five `ok` lines.
Do not run `workflows/prefetch_models.sh` on a cluster without internet: it downloads.

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

## 7. Rule-based cleaning (CPU, a few minutes)

The full run delivers the graph before cleaning. The cleaning of the paper is applied afterwards by three read-only scripts, chained by `workflows/postprocess_run18.sh`: 184 follows every extraction and checks the replay against the run's own files, 187 locates the article, reference and section of every relation, 188 removes relations by fixed rules (R1 bibliography, R2 local label, R3 subject absent from the source sentences), flags others (S1 place name, S2 model comment, S3 figure caption, S4 pronoun) and groups relations by entity families. The run folder is not modified.

```bash
cd "$REPO" && pwd
RUN_DIR=output/run18_rerun bash workflows/postprocess_run18.sh
cat output/run18_rerun/postprocess/188_nettoyage/_resume.txt
```

Check: 184 ends with `0 controle(s) en echec` (if a check fails, the script stops before 187); the last line is `=== postprocess finished ===`.
Outputs in `output/run18_rerun/postprocess/`: `188_nettoyage/graphe_nettoye.json` (cleaned graph), `relations_retirees.tsv` (each removed relation with its rule), `relations_signalees.tsv`, `relations_par_famille.tsv`, and the summaries `_resume.txt` of each step (in French).

---

## 8. Compare with the recorded run18

The outputs of the recorded run18 are distributed in `results/run18/`. Compare your run with them yourself (CPU, a few seconds, nothing is modified):

```bash
cd "$REPO" && pwd
ls results/run18
python scripts/tools/compare_runs.py --ref results/run18 --new output/run18_rerun --out output/compare_run18
```

Open:
- `output/compare_run18/compare_report.txt`: number of records at each stage, pass a, delivered graph (relations, tiers, judges' decisions, relation types, relations in common), `article_metrics.json` value by value, cleaned graph of step 7 (relations kept and removed by rule, flags, family groups, LB2019 before and after), manifests;
- `output/compare_run18/relations_only_ref.tsv` and `relations_only_new.tsv`: the delivered relations found by one run only, with their evidence sentence;
- `output/compare_run18/metrics_side_by_side.tsv`: every metric of both runs.

What to expect: counts of the same order on both sides, not identical ones. Pass b samples at temperature 0.3, and greedy decoding (pass a) in bf16 can also differ slightly on another GPU. A large gap at one stage shows where the runs diverge.

---

## 9. If something fails

| Message | Cause |
|---|---|
| `Invalid account` or job refused | wrong `CLUSTER_ACCOUNT` or `CLUSTER_PARTITION` |
| `401` / `403` on Llama (step 3a) | licence not accepted, or not logged in (`hf auth login`) |
| `hf: command not found` (step 3a) | `huggingface-hub` not installed in the active environment |
| `LocalEntryNotFoundError` or a download attempted in the job | model missing from the cache, or `HF_HOME` not the same in block 0 and in step 3b |
| cache 2–3 times larger on the cluster than in 3a | symbolic links were followed during the copy; redo 3b with `rsync -a` or `tar` |
| `CUDA out of memory` | GPU too small; the recorded run used an A100 slice with 40 GB |
| `STOP: set PDF_DIR …` | `PDF_DIR` not exported in block 0 |
| `REJET … ratio alphabetique 0.xx < 0.55` in step 4 | `pdfplumber` installed; uninstall it (step 2) |
| `STOP: a 184 check failed` in step 7 | a file of the run does not match its replay; read `postprocess/184_trace/_resume.txt` (lines `ECHEC`) |
| `requires accelerate` or `ModuleNotFoundError` when a model loads | environment installed from an older `requirements-gpu.txt`; reinstall it (step 2) |
