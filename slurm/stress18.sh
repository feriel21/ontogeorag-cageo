#!/bin/bash
# =====================================================================
# slurm/stress18.sh — stress test des verificateurs sur le graphe livre de run18
#
# Couche 2 du protocole : controles corrompus construits par des regles fixes
# (m4/m4_negatives.py generate, graine 13 comme pour run16E), puis jugement
# par les DEUX verificateurs du panel, chacun separement :
#   Llama-3.1-8B-Instruct  (snapshot 0e9e39f) et Mistral-7B-Instruct-v0.2 (63a8b08).
# Pour chaque verificateur : sensibilite, specificite, AUC, par type de corruption
# (m4_negatives.py report). Aucun fichier de run18 n'est modifie.
#
#   sbatch --export=ALL,REPO=$HOME/ontogeorag_run18 slurm/stress18.sh
# =====================================================================
#SBATCH --job-name=stress18
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/stress18_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/stress18_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=talbiferyel@gmail.com

set -euo pipefail
REPO=${REPO:?ARRET : REPO non defini (racine du worktree run18)}
VENV=/home/talbi/kg_test/venv
KG=$REPO/output/run18/kg_final/flat_run18_enforced_dedup.json
OUT=$REPO/output/run18/stress_test
SEED=13

cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO
export HF_HUB_OFFLINE=1
unset SLURM_PROCID SLURM_NTASKS RANK WORLD_SIZE MASTER_ADDR MASTER_PORT || true

[ -f "$KG" ] || { echo "ARRET : $KG introuvable." >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "ARRET : $OUT existe deja. Rien n'est ecrase." >&2; exit 2; }
mkdir -p "$OUT"
echo "graphe : $KG  sha=$(sha256sum "$KG" | cut -c1-12)"
echo "commit : $(git rev-parse HEAD)"

resolve () { python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$1', local_files_only=True))"; }
LLAMA_LOCAL=$(resolve meta-llama/Llama-3.1-8B-Instruct)
MISTRAL_LOCAL=$(resolve mistralai/Mistral-7B-Instruct-v0.2)

echo "=== generate (graine $SEED)"
python -u m4/m4_negatives.py generate --kg "$KG" --output "$OUT" --seed $SEED

for J in llama mistral; do
    [ "$J" = "llama" ] && JM="$LLAMA_LOCAL" || JM="$MISTRAL_LOCAL"
    mkdir -p "$OUT/$J"
    echo "=== verify $J"
    python -u m4/m4_verify.py --kg "$OUT/controls.jsonl" --output "$OUT/$J" --model "$JM"
    echo "=== report $J"
    python -u m4/m4_negatives.py report --controls "$OUT/controls.jsonl" \
        --verdicts "$OUT/$J/m4_verdicts.jsonl" --output "$OUT/$J"
done

{ echo "stress18 termine sur $(hostname) le $(date)"; echo "Sorties : $OUT"; echo "";
  for J in llama mistral; do echo "--- $J ---"; cat "$OUT/$J"/negatives_report.json 2>/dev/null || ls "$OUT/$J"; echo; done; } \
  | tee "$OUT/_resume.txt" | mail -s "[ontogeorag] stress18 termine" talbiferyel@gmail.com \
  || echo "(envoi du mail impossible)"