#!/bin/bash
# =====================================================================
# slurm/orion_run18.sbatch — template to run the run18 pipeline on a SLURM cluster
#
# Lines marked A_REMPLIR must be filled for the target cluster (Orion):
# project number, partition and GPU request. Ask the cluster documentation
# or the admins for the exact names.
#
#   Test (about 10 minutes):  sbatch --export=ALL,SMOKE=1 slurm/orion_run18.sbatch
#   Full run (several hours): sbatch slurm/orion_run18.sbatch
#
# Before the first job, on a login node with internet access:
#   bash workflows/prefetch_models.sh        (downloads the three pinned models)
# =====================================================================
#SBATCH --job-name=ontogeorag-run18
#SBATCH --account=A_REMPLIR_numero_de_projet
#SBATCH --partition=A_REMPLIR_partition_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run18_%j.out
#SBATCH --error=logs/run18_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=A_REMPLIR

set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
mkdir -p logs

# ---- environment (adapt: module load, conda or venv) ----------------
VENV="${VENV:-$HOME/ontogeorag_venv}"
# module load python/3.10 cuda      # A_REMPLIR si le cluster utilise des modules
source "$VENV/bin/activate"

# ---- data and models --------------------------------------------------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"     # compute nodes often have no internet
PDF_DIR="${PDF_DIR:-$HOME/ontogeorag_corpus_pdfs}"
export PDF_DIR SMOKE="${SMOKE:-0}"
[ -n "${INDEX_FROM:-}" ] && export INDEX_FROM

echo "host $(hostname) ; job $SLURM_JOB_ID ; repo $REPO ; smoke $SMOKE"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

bash workflows/run_pipeline_run18.sh
