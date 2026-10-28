#!/bin/bash
# =====================================================================
# slurm/run18_clean.sh — run18 : chaine de run17 ; le pipeline ne lit
#                        jamais les aretes d'evaluation
#
# MEMES PARAMETRES QUE run17 (contexte complet --max-chars 12000,
# --max-tokens 2048, meme index copie, memes modeles, memes temperatures,
# verification 5 x 2000, panel Llama + Mistral).
#
# SEULE INTERVENTION : separation stricte entre pipeline et evaluation
#   - requetes   : configs/descriptor_queries_run18.jsonl (175, 210 requetes)
#   - extraction : pipeline/02_extract_triples_run18.py (176, exemples lus
#                  dans $EXAMPLES, pris hors du corpus d'evaluation)
#   - nettoyage  : pipeline/04_clean_validate_run18.py (176, protection =
#                  vocabulaire complet de l'ontologie, aucune arete d'evaluation)
#   - porte      : 174_audit_fuite_benchmark.py doit sortir en code 0,
#                  sinon rien n'est lance
#   Seule l'etape 07 (metriques) lit les aretes d'evaluation, apres coup.
#
# VARIABLES
#     REPO       racine du worktree run18 (branche run18*)
#     INDEX_FROM dossier index de run16E (contient chunks.jsonl)
#     EXAMPLES   fichier JSON des exemples du prompt (celui passe a 176)
#     sbatch --export=ALL,REPO=$HOME/ontogeorag_run18,INDEX_FROM=$HOME/ontogeorag_D/output/run16E/index,EXAMPLES=configs/prompt_examples_run18.json slurm/run18_clean.sh
#
# MODE SMOKE (~10 min, a faire avant) :
#     SMOKE=1 MAIL_TO="" REPO=... INDEX_FROM=... EXAMPLES=... bash slurm/run18_clean.sh
# =====================================================================
#SBATCH --job-name=ontogeorag-run18
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/run18_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/run18_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=talbiferyel@gmail.com

set -euo pipefail

REPO=${REPO:?ARRET : REPO non defini (racine du worktree run18)}
EXAMPLES=${EXAMPLES:?ARRET : EXAMPLES non defini (exemples du prompt)}
VENV=/home/talbi/kg_test/venv
RUN="${RUN_TAG:-run18}"
SMOKE="${SMOKE:-0}"
INDEX_FROM="${INDEX_FROM:-}"
EXPECT_CHUNKS=1758

[ -n "$INDEX_FROM" ] || { echo "ARRET : \$INDEX_FROM non defini." >&2; exit 2; }
[ -f "$INDEX_FROM/chunks.jsonl" ] || { echo "ARRET : $INDEX_FROM/chunks.jsonl introuvable." >&2; exit 2; }

if [ "$SMOKE" = "1" ]; then
    RUN=run18_smoke
    SMOKE_EXTRACT="--max-queries 5"; SMOKE_VERIFY="--limit 20"
    echo ">>> MODE SMOKE — resultats non scientifiques, validation d'enchainement seulement"
else
    SMOKE_EXTRACT=""; SMOKE_VERIFY=""
fi
OUT=$REPO/output/$RUN
[ ! -e "$OUT" ] || { echo "ARRET : $OUT existe deja. Rien n'est ecrase." >&2; exit 2; }

RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2
QUERIES=configs/descriptor_queries_run18.jsonl
TOPK=5; BM25TOPN=20; MINBM25=2.0
MAXCHARS=12000          # run16E : 2800
MAXTOKENS=2048          # comme run17

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
[[ "$BRANCH" == run18* ]] || { echo "ARRET : branche '$BRANCH', attendu 'run18...'." >&2; exit 2; }
if [ "$SMOKE" != "1" ]; then
    git diff --quiet || { echo "ARRET : arbre sale. Committer d'abord." >&2; exit 2; }
fi
python 58v2_freeze_benchmark.py check --benchmark configs/benchmark/benchmark_v3.json \
    || { echo "ARRET : benchmark non conforme." >&2; exit 2; }
# Porte : scripts generes conformes a 176, puis aucune fuite (174)
for F in pipeline/02_extract_triples_run18.py pipeline/04_clean_validate_run18.py "$QUERIES" "$EXAMPLES"; do
    [ -f "$F" ] || { echo "ARRET : $F introuvable (lancer 175 et 176, puis committer)." >&2; exit 2; }
    H=$(sha256sum "$F" | cut -d' ' -f1)
    grep -q "$H" pipeline/RUN18_PATCHES.md "$QUERIES.meta.json" \
        || { echo "ARRET : sha de $F absent de RUN18_PATCHES.md et de $QUERIES.meta.json." >&2; exit 2; }
done
GATE_DIR=$REPO/output/audit/174_gate_${RUN}_$(date +%Y%m%d_%H%M%S)
python 174_audit_fuite_benchmark.py --repo . \
    --extract pipeline/02_extract_triples_run18.py --queries "$QUERIES" \
    --clean pipeline/04_clean_validate_run18.py --outdir "$GATE_DIR" \
    || { echo "ARRET : 174 a trouve une fuite (voir $GATE_DIR)." >&2; exit 2; }
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
# 1 — INDEX : COPIE de l'index de run16E (lecture seule de la source)
#     Pas de reconstruction : la recherche doit voir les memes chunks.
# =====================================================================
NCH=$(grep -c . "$INDEX_FROM/chunks.jsonl")
[ "$NCH" = "$EXPECT_CHUNKS" ] || { echo "ARRET : $NCH chunks dans $INDEX_FROM, attendu $EXPECT_CHUNKS." >&2; exit 2; }
cp -r "$INDEX_FROM" "$OUT/index"
SRC_SHA=$(sha256sum "$INDEX_FROM/chunks.jsonl" | cut -d' ' -f1)
DST_SHA=$(sha256sum "$OUT/index/chunks.jsonl" | cut -d' ' -f1)
[ "$SRC_SHA" = "$DST_SHA" ] || { echo "ARRET : copie de l'index differente de la source." >&2; exit 2; }
echo "=== [index_copy] $INDEX_FROM  chunks=$NCH  sha=${SRC_SHA:0:12}"
python - "$STEPS_LOG" "$INDEX_FROM" "$SRC_SHA" "$NCH" <<'PY'
import json, sys
log, src, sha, n = sys.argv[1:5]
open(log, "a", encoding="utf-8").write(json.dumps(
    {"step": "index_copy", "script": None, "source": src,
     "chunks_sha256": sha, "n_chunks": int(n)}, ensure_ascii=False) + "\n")
PY

# =====================================================================
# 2 — EXTRACTION, deux regimes de temperature (design de la fusion tiered)
# =====================================================================
for P in a b; do
    [ "$P" = "a" ] && TEMP=0.0 || TEMP=0.3
    mkdir -p "$OUT/pass_$P"
    run_step "extract_pass_$P" pipeline/02_extract_triples_run18.py \
        --index-dir "$OUT/index" \
        --schema configs/ontology_schema.json \
        --queries "$QUERIES" \
        --output "$OUT/pass_$P/raw_triples.jsonl" \
        --model "$MODEL_LOCAL" --backend hf \
        --top-k $TOPK --bm25-topn $BM25TOPN --min-bm25 $MINBM25 \
        --max-chars $MAXCHARS --max-tokens $MAXTOKENS --temperature $TEMP \
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
    run_step "clean_pass_$P" pipeline/04_clean_validate_run18.py \
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
# 9 — MANIFESTE + comparaison indicative a run16E
# =====================================================================
RUN16E_DIR=$(dirname "$INDEX_FROM")
python - "$OUT/manifest.json" "$STEPS_LOG" "$GIT_SHA" "$GIT_DESC" "$OUT" "$RUN" "$RUN16E_DIR" "$MAXTOKENS" <<'PY' | tee "$OUT/_resume.txt"
import hashlib, json, subprocess, sys
from pathlib import Path
man, log, sha, desc, out, run, ref, mt = sys.argv[1:9]
steps = [json.loads(l) for l in Path(log).read_text(encoding="utf-8").splitlines() if l.strip()]
arte = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in sorted(Path(out).rglob("*.json*"))
        if not p.name.startswith("_") and p.name != "manifest.json"}
doc = {"run": run, "git_commit": sha, "git_describe": desc,
       "git_status_clean": subprocess.run(["git", "diff", "--quiet"]).returncode == 0,
       "benchmark": "configs/benchmark/benchmark_v3.json",
       "provenance": "chaine de run17 (index de run16E copie, --max-chars 12000, --max-tokens " + mt + "), requetes 175, scripts 02/04 generes par 176, porte 174 passee",
       "index_from": ref + "/index",
       "steps": steps, "artifact_sha256": arte}
Path(man).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nManifeste : {man}  ({len(steps)} etapes, {len(arte)} artefacts)")
try:
    a = json.load(open(f"{out}/kg_final/article_metrics.json", encoding="utf-8"))
    b = json.load(open(f"{ref}/kg_final/article_metrics.json", encoding="utf-8"))
    print("\n-- indicatif, " + run + " vs " + ref + " (vue interne article_metrics, a ne pas citer) --")
    for k in sorted(set(a) & set(b)):
        if isinstance(a[k], (int, float, str)):
            print(f"   {k:34s} {a[k]}   |   run16E: {b[k]}")
except Exception as e:
    print(f"(comparaison indisponible : {e})")
PY

echo ""; echo "=== $RUN termine ==="

# =====================================================================
# 10 — MAIL DES RESULTATS (si la commande `mail` existe sur le noeud)
#      Echec d'envoi = sans consequence : les resultats restent sur disque.
# =====================================================================
MAIL_TO="${MAIL_TO-talbiferyel@gmail.com}"     # MAIL_TO="" = pas de mail (smoke)
if [ -z "$MAIL_TO" ]; then
    echo "(pas de mail demande)"
elif command -v mail >/dev/null 2>&1; then
    { echo "$RUN termine sur $(hostname) le $(date)"; echo "Sorties : $OUT"; echo "";
      cat "$OUT/_resume.txt"; echo ""; echo "--- kg_final/article_metrics.json ---";
      cat "$OUT/kg_final/article_metrics.json"; } \
      | mail -s "[ontogeorag] $RUN termine" "$MAIL_TO" || echo "(envoi du mail impossible)"
else
    echo "(commande mail absente : resultats seulement sur disque)"
fi