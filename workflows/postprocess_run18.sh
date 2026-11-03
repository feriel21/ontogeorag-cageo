#!/usr/bin/env bash
# =====================================================================
# workflows/postprocess_run18.sh — rule-based cleaning of a run18 output
#
# Runs, on the output folder of workflows/run_pipeline_run18.sh, the three
# read-only CPU scripts that produced the cleaned graph of the paper:
#   184_trace_requetes_et_pertes.py  follows every extraction through 04, 04c, 05a
#                                    and 06, checks the replay against the run's
#                                    own files, flags the 16 suspect queries
#   187_provenance_articles.py       article, full reference and (deduced) section
#                                    of every delivered relation
#   188_nettoyage_familles.py        removes relations by fixed rules (R1 bibliography,
#                                    R2 local label, R3 subject absent from the source
#                                    sentences), flags S1-S4, groups relations by
#                                    entity families
# The three scripts are the ones used for results/run18/postprocess/ (same files,
# same options). Their messages and output columns are in French.
#
# Usage (from anywhere; CPU, a few minutes, no model):
#     RUN_DIR=output/run18_rerun bash workflows/postprocess_run18.sh
# Optional variables:
#     POST           output folder (default $RUN_DIR/postprocess); never overwritten
#     RETIRER_LIEUX  1 = also remove relations naming a place (S1); default 0, as in the paper
#
# The run folder is not modified. 184 and 187 expect the file names of the recorded
# run (kg/tiered_kg_run18.json, kg_final/flat_run18_enforced_dedup.json); a folder of
# symbolic links ($POST/run_view) gives them those names whatever the run is called.
# =====================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_DIR="${RUN_DIR:-output/run18_rerun}"
POST="${POST:-$RUN_DIR/postprocess}"
RETIRER_LIEUX="${RETIRER_LIEUX:-0}"

[ -d "$RUN_DIR" ] || { echo "STOP: run folder $RUN_DIR not found (set RUN_DIR)." >&2; exit 2; }
[ -e "$POST" ] && { echo "STOP: $POST exists, nothing is overwritten (set POST to a new folder)." >&2; exit 2; }
RUN_ABS="$(cd "$RUN_DIR" && pwd)"

for F in pass_a/raw_triples.jsonl pass_a/verified_triples.jsonl pass_a/cleaning_stats_v5.json \
         pass_b/raw_triples.jsonl pass_b/verified_triples.jsonl pass_b/cleaning_stats_v5.json \
         index/chunks.jsonl m4_panel/m4_panel_decisions.jsonl m4_panel/m4_quarantine.jsonl; do
    [ -f "$RUN_ABS/$F" ] || { echo "STOP: $RUN_DIR/$F not found: is this a complete run18 output?" >&2; exit 2; }
done

one () {
    local hits=( $1 )
    [ "${#hits[@]}" -eq 1 ] && [ -f "${hits[0]}" ] \
        || { echo "STOP: expected exactly one file matching $1, found ${#hits[@]}." >&2; exit 2; }
    echo "${hits[0]}"
}
KG="$(one "$RUN_ABS/kg/tiered_kg_*.json")"
FLAT="$(one "$RUN_ABS/kg_final/flat_*_enforced_dedup.json")"

mkdir -p "$POST/run_view/kg" "$POST/run_view/kg_final"
POST_ABS="$(cd "$POST" && pwd)"
VIEW="$POST_ABS/run_view"
for D in pass_a pass_b index m4_panel; do
    ln -s "$RUN_ABS/$D" "$VIEW/$D"
done
ln -s "$KG" "$VIEW/kg/tiered_kg_run18.json"
ln -s "$FLAT" "$VIEW/kg_final/flat_run18_enforced_dedup.json"

LOG="$POST_ABS/postprocess_manifest.txt"
{
    echo "postprocess of $RUN_ABS"
    echo "date: $(date -Iseconds)"
    echo "code: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    for S in 184_trace_requetes_et_pertes.py 187_provenance_articles.py 188_nettoyage_familles.py \
             configs/entity_families_run18.csv configs/lb_reference_edges_original26.json \
             data/pdf_to_ref.tsv data/lb2019_41_refs.txt; do
        echo "sha256 $(sha256sum "$S" | cut -c1-12)  $S"
    done
    echo "input kg   : $KG"
    echo "input flat : $FLAT"
    echo "RETIRER_LIEUX=$RETIRER_LIEUX"
} > "$LOG"

echo "=== [184] trace of every extraction, checks against the run's own files"
python -u 184_trace_requetes_et_pertes.py --repo . --run "$VIEW" --outdir "$POST_ABS/184_trace" --nfkc \
    || { echo "STOP: a 184 check failed (ECHEC) or 184 stopped; see $POST/184_trace/_resume.txt. Steps 187-188 not run." >&2; exit 3; }

echo ""
echo "=== [187] article, reference and section of every delivered relation"
python -u 187_provenance_articles.py \
    --graph "$VIEW/kg_final/flat_run18_enforced_dedup.json" --run "$VIEW" \
    --suivi "$POST_ABS/184_trace/suivi_extractions.tsv" --refs data \
    --outdir "$POST_ABS/187_provenance"

echo ""
echo "=== [188] rule-based cleaning and entity families"
OPT=""
[ "$RETIRER_LIEUX" = "1" ] && OPT="--retirer-lieux"
python -u 188_nettoyage_familles.py \
    --graph "$POST_ABS/187_provenance/graphe_avec_provenance.json" --run "$VIEW" \
    --familles configs/entity_families_run18.csv \
    --reference configs/lb_reference_edges_original26.json \
    --outdir "$POST_ABS/188_nettoyage" $OPT

echo ""
echo "cleaned graph : $POST/188_nettoyage/graphe_nettoye.json"
echo "removed       : $POST/188_nettoyage/relations_retirees.tsv (one line per relation, with its rule)"
echo "summary       : $POST/188_nettoyage/_resume.txt"
echo "=== postprocess finished ==="
