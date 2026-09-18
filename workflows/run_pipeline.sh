#!/usr/bin/env bash
# =====================================================================
# workflows/run_pipeline.sh — public driver of the OntoGeoRAG pipeline
#
# Runs the 22 steps recorded in results/run16E/manifest.json, with the
# same scripts and the same options. It is the public counterpart of
# slurm/run16_full.sh (the cluster driver used for the delivered run),
# without cluster paths.
#
# Usage (from anywhere):
#     PDF_DIR=/path/to/the/41/pdfs bash workflows/run_pipeline.sh
#
# Smoke mode (about 10 minutes: 5 queries, 20 verified triples; checks
# that the chain runs end to end, results are not meaningful):
#     SMOKE=1 PDF_DIR=/path/to/the/41/pdfs bash workflows/run_pipeline.sh
#
# Optional variables:
#     OUT      output directory      (default: output/run16E_rerun, or output/smoke)
#     QUERIES  extraction queries    (default: configs/descriptor_queries_v2.jsonl,
#                                     the file used by run16E)
#     HF_HUB_OFFLINE=1  to use only models already in the local Hugging Face cache
#
# Models are pinned to the revisions used by run16E. A rerun is not expected
# to reproduce the delivered graph byte for byte (pass b samples at
# temperature 0.3); the delivered results are tied to the recorded run
# through the manifest and PROVENANCE.tsv.
# =====================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset SLURM_PROCID SLURM_NTASKS RANK WORLD_SIZE MASTER_ADDR MASTER_PORT || true

PDF_DIR="${PDF_DIR:-}"
SMOKE="${SMOKE:-0}"
QUERIES="${QUERIES:-configs/descriptor_queries_v2.jsonl}"

[ -n "$PDF_DIR" ] || { echo "STOP: PDF_DIR is not set." >&2; exit 2; }
[ -d "$PDF_DIR" ] || { echo "STOP: $PDF_DIR not found." >&2; exit 2; }

if [ "$SMOKE" = "1" ]; then
    OUT="${OUT:-output/smoke}"
    SMOKE_EXTRACT="--max-queries 5"; SMOKE_VERIFY="--limit 20"; EXPECT_DOCS=""
    echo ">>> SMOKE MODE: checks the chain only, results are not meaningful"
else
    OUT="${OUT:-output/run16E_rerun}"
    SMOKE_EXTRACT=""; SMOKE_VERIFY=""; EXPECT_DOCS="--expect-docs 41"
fi
RUN="$(basename "$OUT")"

# ---- parameters of run16E (see results/run16E/manifest.json) --------
RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2
TOPK=5; BM25TOPN=20; MINBM25=2.0; MAXCHARS=2800
CHUNK_SIZE=2000; OVERLAP=200
TEMP_A=0.0; TEMP_B=0.3
CLUSTER_THRESHOLD=0.06

QWEN_REPO=Qwen/Qwen2.5-7B-Instruct;           QWEN_REV=a09a35458c702b33eeacc393d103063234e8bc28
LLAMA_REPO=meta-llama/Llama-3.1-8B-Instruct;  LLAMA_REV=0e9e39f249a16976918f6564b8830bc894c89659
MISTRAL_REPO=mistralai/Mistral-7B-Instruct-v0.2; MISTRAL_REV=63a8b081895390a26e140280378bc85ec8bce07a

resolve () {
    python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))
PY
}
QWEN=$(resolve "$QWEN_REPO" "$QWEN_REV")
LLAMA=$(resolve "$LLAMA_REPO" "$LLAMA_REV")
MISTRAL=$(resolve "$MISTRAL_REPO" "$MISTRAL_REV")

mkdir -p "$OUT"

# ---- preflight (as in slurm/run16_full.sh) --------------------------
echo "=== PREFLIGHT ==="
python 58v2_freeze_benchmark.py check --benchmark configs/benchmark/benchmark_v3.json \
    || { echo "STOP: benchmark file does not match its frozen checksum." >&2; exit 2; }
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
GIT_DESC=$(git describe --tags --always --dirty 2>/dev/null || echo unknown)
echo "code: $GIT_SHA ($GIT_DESC)"

STEPS_LOG="$OUT/_steps.jsonl"; : > "$STEPS_LOG"
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

# ---- 1  index (49 replaces 01) --------------------------------------
run_step index 49_build_index_v2.py \
    --pdf-dir "$PDF_DIR" --outdir "$OUT/index" \
    --chunk-size $CHUNK_SIZE --overlap $OVERLAP $EXPECT_DOCS

# ---- 2-3  extraction, two temperature regimes -----------------------
for P in a b; do
    [ "$P" = "a" ] && TEMP=$TEMP_A || TEMP=$TEMP_B
    mkdir -p "$OUT/pass_$P"
    run_step "extract_pass_$P" pipeline/02_extract_triples.py \
        --index-dir "$OUT/index" \
        --schema configs/ontology_schema.json \
        --queries "$QUERIES" \
        --output "$OUT/pass_$P/raw_triples.jsonl" \
        --model "$QWEN" --backend hf \
        --top-k $TOPK --bm25-topn $BM25TOPN --min-bm25 $MINBM25 \
        --max-chars $MAXCHARS --temperature $TEMP \
        --reranker $RERANKER $SMOKE_EXTRACT
done

# ---- 4  provenance repair; 5-6  verification ------------------------
run_step repair_provenance 52_repair_provenance.py \
    --chunks "$OUT/index/chunks.jsonl" \
    --triples "$OUT/pass_a/raw_triples.jsonl" "$OUT/pass_b/raw_triples.jsonl" \
    --outdir "$OUT" --expect-chunks 5

for P in a b; do
    run_step "verify_pass_$P" 53_verify_repaired.py \
        --input "$OUT/pass_$P/triples_repaired.jsonl" \
        --output "$OUT/pass_$P/verified_triples.jsonl" \
        --model "$QWEN" --backend hf \
        --max-chunks 5 --chunk-chars 2000 $SMOKE_VERIFY
done

# ---- 7-12  validation, rule canonicalisation, lexicon ---------------
for P in a b; do
    run_step "clean_pass_$P" pipeline/04_clean_validate.py \
        --input "$OUT/pass_$P/verified_triples.jsonl" \
        --outdir "$OUT/pass_$P" --verif-policy normal --cluster-threshold $CLUSTER_THRESHOLD
    run_step "rulecanon_pass_$P" analysis_suite/05a_rule_canonicalize.py \
        --input "$OUT/pass_$P/canonical_triples_v5.jsonl"
    run_step "lexicon_pass_$P" analysis_suite/04c_lexicon_enforce.py \
        --input "$OUT/pass_$P/canonical_triples_v5.rulecanon.jsonl" --enforce
done

# ---- 13  tiered fusion ----------------------------------------------
mkdir -p "$OUT/kg"
run_step fusion pipeline/06_tiered_fusion.py \
    --iter-a "$OUT/pass_a/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl" \
    --iter-b "$OUT/pass_b/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl" \
    --output "$OUT/kg/tiered_kg_$RUN.json"

# ---- 14-19  two-judge panel -----------------------------------------
for J in llama mistral; do
    [ "$J" = "llama" ] && JM="$LLAMA" || JM="$MISTRAL"
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

# ---- 20-21  schema enforcement (flag, not reject), flatten + dedup --
mkdir -p "$OUT/kg_enforced" "$OUT/kg_final"
run_step schema_enforce analysis_suite/04b_schema_enforce.py \
    --kg "$OUT/m4_panel/tiered_kg_m4.json" --outdir "$OUT/kg_enforced" \
    --default-type Geological_Object --descriptor-fallback Environmental_Control
run_step flatten analysis_suite/24_kg_to_flat.py \
    --kg "$OUT/kg_enforced/tiered_kg_enforced.json" \
    --include tier1,tier2 --dedup --out "$OUT/kg_final/flat_${RUN}_enforced_dedup.json"

# ---- 22  metrics ----------------------------------------------------
run_step metrics pipeline/07_final_metrics.py \
    --kg "$OUT/kg_final/flat_${RUN}_enforced_dedup.json" \
    --output "$OUT/kg_final/article_metrics.json"

# ---- manifest (same structure as results/run16E/manifest.json) ------
python - "$OUT/manifest.json" "$STEPS_LOG" "$GIT_SHA" "$GIT_DESC" "$OUT" "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
man, log, sha, desc, out, run = sys.argv[1:7]
steps = [json.loads(l) for l in Path(log).read_text(encoding="utf-8").splitlines() if l.strip()]
arte = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in sorted(Path(out).rglob("*.json*"))
        if not p.name.startswith("_") and p.name != "manifest.json"}
doc = {"run": run, "git_commit": sha, "git_describe": desc,
       "benchmark": "configs/benchmark/benchmark_v3.json",
       "provenance": "rerun with workflows/run_pipeline.sh",
       "steps": steps, "artifact_sha256": arte}
Path(man).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nmanifest: {man}  ({len(steps)} steps, {len(arte)} artefacts)")
PY

echo ""; echo "=== $RUN finished ==="