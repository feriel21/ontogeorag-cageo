#!/bin/bash
# =====================================================================
# slurm/run16_full.sh — run16 COMPLET DEPUIS LES PDF
#
# Contrairement a run15 (qui repartait des triples verifies de run14 via
# 52/53), ce driver execute la chaine entiere : index -> extraction ->
# provenance -> verification -> validation -> fusion -> panel -> metriques.
#
# CONSEQUENCE ASSUMEE : les chiffres seront neufs. run16 n'est PAS une
# reproduction de run15 et ne doit pas etre compare arete par arete.
#
# PARAMETRES D'EXTRACTION : repris a l'identique de slurm/run11_gpu.sh, qui
# a produit run14 (donc l'amont de run15). Ne pas les changer sans le
# declarer — les defauts de 02 (top-k 3, min-bm25 0.0) ne reproduisent rien.
#     pass_a : temperature 0.0
#     pass_b : temperature 0.3
#     top-k 5, bm25-topn 20, min-bm25 2.0
#     reranker cross-encoder/ms-marco-MiniLM-L-6-v2
#
# SEULE VARIABLE A FOURNIR : PDF_DIR
#     sbatch --export=ALL,PDF_DIR=/chemin/vers/les/41/pdf slurm/run16_full.sh
#
# MODE SMOKE (fortement recommande avant le run complet, ~10 min) :
#     SMOKE=1 PDF_DIR=... bash slurm/run16_full.sh
#     -> limite l'extraction a 5 requetes et la verification a 20 triples.
#        Valide la compatibilite 49 -> 02 et l'enchainement complet sans
#        bruler 3 h de GPU pour decouvrir une incompatibilite de format.
# =====================================================================
#SBATCH --job-name=ontogeorag-run16
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/run16_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/run16_%j.err

set -euo pipefail

REPO=${REPO:-/home/talbi/ontogeorag}
VENV=/home/talbi/kg_test/venv
RUN="${RUN_TAG:-run16}"
OUT=$REPO/output/$RUN
SMOKE="${SMOKE:-0}"
PDF_DIR="${PDF_DIR:-}"

[ -n "$PDF_DIR" ] || { echo "ARRET : \$PDF_DIR non defini." >&2; exit 2; }
[ -d "$PDF_DIR" ] || { echo "ARRET : $PDF_DIR introuvable." >&2; exit 2; }

if [ "$SMOKE" = "1" ]; then
    RUN=run16_smoke; OUT=$REPO/output/$RUN
    SMOKE_EXTRACT="--max-queries 5"; SMOKE_VERIFY="--limit 20"; EXPECT_DOCS=""
    echo ">>> MODE SMOKE — resultats non scientifiques, validation d'enchainement seulement"
else
    SMOKE_EXTRACT=""; SMOKE_VERIFY=""; EXPECT_DOCS="--expect-docs 41"
fi

RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2
TOPK=5; BM25TOPN=20; MINBM25=2.0; MAXCHARS=2800

cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO
export HF_HUB_OFFLINE=1
unset SLURM_PROCID SLURM_NTASKS RANK WORLD_SIZE MASTER_ADDR MASTER_PORT || true
mkdir -p $REPO/logs $OUT

resolve () { python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$1', local_files_only=True))"; }
MODEL_LOCAL=$(resolve Qwen/Qwen2.5-7B-Instruct)
LLAMA_LOCAL=$(resolve meta-llama/Llama-3.1-8B-Instruct)
MISTRAL_LOCAL=$(resolve mistralai/Mistral-7B-Instruct-v0.2)

# =====================================================================
# PREFLIGHT
# =====================================================================
echo "=== PREFLIGHT ==="
BRANCH=$(git rev-parse --abbrev-ref HEAD)
[[ "$BRANCH" == run16* ]] || { echo "ARRET : branche '$BRANCH', attendu 'run16'." >&2; exit 2; }
if [ "$SMOKE" != "1" ]; then
    git diff --quiet || { echo "ARRET : arbre sale. Committer d'abord." >&2; exit 2; }
fi
python 58v2_freeze_benchmark.py check --benchmark configs/benchmark/benchmark_v3.json \
    || { echo "ARRET : benchmark non conforme." >&2; exit 2; }
GIT_SHA=$(git rev-parse HEAD); GIT_DESC=$(git describe --tags --always --dirty)
echo "commit $GIT_SHA / $GIT_DESC"

STEPS_LOG=$OUT/_steps.jsonl; : > $STEPS_LOG
run_step () {
    local name="$1"; shift; local script="$1"; shift
    local sha t0 t1
    sha=$(sha256sum "$script" | cut -d' ' -f1); t0=$(date -Iseconds)
    echo ""; echo "=== [$name] $script  sha=${sha:0:12}"; echo "    argv: $*"
    python -u "$script" "$@"
    t1=$(date -Iseconds)
    python - "$name" "$script" "$sha" "$t0" "$t1" "$STEPS_LOG" "$@" <<'PY'
import json, sys
n, s, sha, t0, t1, log = sys.argv[1:7]
open(log, "a", encoding="utf-8").write(json.dumps(
    {"step": n, "script": s, "sha256": sha, "started": t0,
     "ended": t1, "argv": sys.argv[7:]}, ensure_ascii=False) + "\n")
PY
}

# =====================================================================
# 1 — INDEX (49 remplace 01 : boucle de chunking non terminante, cle meta)
#     L'index de run14 n'est pas reproductible ; celui-ci sera different.
# =====================================================================
run_step index 49_build_index_v2.py \
    --pdf-dir "$PDF_DIR" --outdir "$OUT/index" \
    --chunk-size 2000 --overlap 200 $EXPECT_DOCS

# =====================================================================
# 2 — EXTRACTION, deux regimes de temperature (design de la fusion tiered)
# =====================================================================
for P in a b; do
    [ "$P" = "a" ] && TEMP=0.0 || TEMP=0.3
    mkdir -p "$OUT/pass_$P"
    run_step "extract_pass_$P" pipeline/02_extract_triples.py \
        --index-dir "$OUT/index" \
        --schema configs/ontology_schema.json \
        --queries "${QUERIES:-configs/descriptor_queries.jsonl}" \
        --output "$OUT/pass_$P/raw_triples.jsonl" \
        --model "$MODEL_LOCAL" --backend hf \
        --top-k $TOPK --bm25-topn $BM25TOPN --min-bm25 $MINBM25 \
        --max-chars $MAXCHARS --temperature $TEMP \
        --reranker $RERANKER $SMOKE_EXTRACT
done

# =====================================================================
# 3 — PROVENANCE (02 ecrit selected_chunks, 03 lit top_chunks : c'est la
#     discordance qui faisait verifier sur 15 % du contexte) puis 53
# =====================================================================
run_step repair_provenance 52_repair_provenance.py \
    --chunks "$OUT/index/chunks.jsonl" \
    --triples "$OUT/pass_a/raw_triples.jsonl" "$OUT/pass_b/raw_triples.jsonl" \
    --outdir "$OUT" --expect-chunks 5

for P in a b; do
    run_step "verify_pass_$P" 53_verify_repaired.py \
        --input "$OUT/pass_$P/triples_repaired.jsonl" \
        --output "$OUT/pass_$P/verified_triples.jsonl" \
        --model "$MODEL_LOCAL" --backend hf \
        --max-chunks 5 --chunk-chars 2000 $SMOKE_VERIFY
done

# =====================================================================
# 4 — VALIDATION / CANONICALISATION / LEXIQUE
# =====================================================================
for P in a b; do
    run_step "clean_pass_$P" pipeline/04_clean_validate.py \
        --input "$OUT/pass_$P/verified_triples.jsonl" \
        --outdir "$OUT/pass_$P" --verif-policy normal --cluster-threshold 0.06
    run_step "rulecanon_pass_$P" analysis_suite/05a_rule_canonicalize.py \
        --input "$OUT/pass_$P/canonical_triples_v5.jsonl"
    run_step "lexicon_pass_$P" analysis_suite/04c_lexicon_enforce.py \
        --input "$OUT/pass_$P/canonical_triples_v5.rulecanon.jsonl" --enforce
done

# =====================================================================
# 5 — FUSION
# =====================================================================
mkdir -p "$OUT/kg"
run_step fusion pipeline/06_tiered_fusion.py \
    --iter-a "$OUT/pass_a/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl" \
    --iter-b "$OUT/pass_b/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl" \
    --output "$OUT/kg/tiered_kg_$RUN.json"

# =====================================================================
# 6 — PANEL M4  (le panel juge la phrase d'evidence citee, pas les passages
#     sources : independance de famille de modele, pas de materiel)
# =====================================================================
for J in llama mistral; do
    [ "$J" = "llama" ] && JM="$LLAMA_LOCAL" || JM="$MISTRAL_LOCAL"
    mkdir -p "$OUT/m4/$J"
    run_step "m4_verify_$J" m4/m4_verify.py \
        --kg "$OUT/kg/tiered_kg_$RUN.json" --index "$OUT/index" \
        --output "$OUT/m4/$J" --model "$JM"
    run_step "m4_aggregate_$J" m4/m4_aggregate.py \
        --verdicts "$OUT/m4/$J/m4_verdicts.jsonl" \
        --output "$OUT/m4/$J"
done

mkdir -p "$OUT/m4_panel"
run_step m4_panel m4/m4_panel.py \
    --judges "llama=$OUT/m4/llama/m4_verdicts.jsonl" "mistral=$OUT/m4/mistral/m4_verdicts.jsonl" \
    --output "$OUT/m4_panel"
run_step m4_integrate m4/m4_integrate_tiers.py \
    --kg "$OUT/kg/tiered_kg_$RUN.json" \
    --decisions "$OUT/m4_panel/m4_panel_decisions.jsonl" \
    --output "$OUT/m4_panel"

# =====================================================================
# 7 — SCHEMA ENFORCE (flag-not-reject) + DEDUP
# =====================================================================
mkdir -p "$OUT/kg_enforced" "$OUT/kg_final"
run_step schema_enforce analysis_suite/04b_schema_enforce.py \
    --kg "$OUT/m4_panel/tiered_kg_m4.json" --outdir "$OUT/kg_enforced" \
    --default-type Geological_Object --descriptor-fallback Environmental_Control
run_step flatten analysis_suite/24_kg_to_flat.py \
    --kg "$OUT/kg_enforced/tiered_kg_enforced.json" \
    --include tier1,tier2 --dedup --out "$OUT/kg_final/flat_${RUN}_enforced_dedup.json"

# =====================================================================
# 8 — METRIQUES
# =====================================================================
run_step metrics pipeline/07_final_metrics.py \
    --kg "$OUT/kg_final/flat_${RUN}_enforced_dedup.json" \
    --output "$OUT/kg_final/article_metrics.json"

# =====================================================================
# 9 — MANIFESTE + comparaison indicative a run15
# =====================================================================
python - "$OUT/manifest.json" "$STEPS_LOG" "$GIT_SHA" "$GIT_DESC" "$OUT" "$RUN" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
man, log, sha, desc, out, run = sys.argv[1:7]
steps = [json.loads(l) for l in Path(log).read_text(encoding="utf-8").splitlines() if l.strip()]
arte = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in sorted(Path(out).rglob("*.json*"))
        if not p.name.startswith("_") and p.name != "manifest.json"}
doc = {"run": run, "git_commit": sha, "git_describe": desc,
       "git_status_clean": subprocess.run(["git", "diff", "--quiet"]).returncode == 0,
       "benchmark": "configs/benchmark/benchmark_v3.json",
       "provenance": "COMPLET depuis les PDF — n'est PAS une reproduction de run15",
       "steps": steps, "artifact_sha256": arte}
Path(man).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nManifeste : {man}  ({len(steps)} etapes, {len(arte)} artefacts)")
try:
    a = json.load(open(f"{out}/kg_final/article_metrics.json", encoding="utf-8"))
    b = json.load(open("output/run15/kg_final/article_metrics.json", encoding="utf-8"))
    print("\n-- indicatif, run16 vs run15 (runs NON comparables arete par arete) --")
    for k in sorted(set(a) & set(b)):
        if isinstance(a[k], (int, float, str)):
            print(f"   {k:34s} {a[k]}   |   run15: {b[k]}")
except Exception as e:
    print(f"(comparaison indisponible : {e})")
PY

echo ""; echo "=== $RUN termine ==="