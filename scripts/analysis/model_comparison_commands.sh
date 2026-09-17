#!/bin/bash
# Commands executed for Table 4 (controlled model comparison), as written by
# model_comparison.py --stage plan and run on the original cluster.
#
# Kept as a record: the paths are those of the original machine.
#   /tmp/run16E_code   checkout of commit 9c03730 (code of the delivered run)
#   ~/ontogeorag       configuration files; ~/ontogeorag_D  run16E index
# To rerun elsewhere, replace the paths and point --model to a local snapshot
# of each model at the listed revision.
#
# Historical name: 95p_commands.sh
set -euo pipefail

# the pipeline package is imported by name, so the repository root
# must be on PYTHONPATH; running the script by path is not enough
export PYTHONPATH=/tmp/run16E_code${PYTHONPATH:+:$PYTHONPATH}

mkdir -p /home/talbi/ontogeorag_D/output/model_comparison/qwen
/home/talbi/kg_test/venv/bin/python3 /tmp/run16E_code/pipeline/02_extract_triples.py --schema /home/talbi/ontogeorag/configs/ontology_schema.json --queries /home/talbi/ontogeorag/configs/descriptor_queries_v2.jsonl --index-dir /home/talbi/ontogeorag_D/output/run16E/index --backend hf --top-k 5 --bm25-topn 20 --min-bm25 2.0 --max-chars 2800 --temperature 0.0 --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --model /home/talbi/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --output /home/talbi/ontogeorag_D/output/model_comparison/qwen/raw_triples.jsonl

mkdir -p /home/talbi/ontogeorag_D/output/model_comparison/llama
/home/talbi/kg_test/venv/bin/python3 /tmp/run16E_code/pipeline/02_extract_triples.py --schema /home/talbi/ontogeorag/configs/ontology_schema.json --queries /home/talbi/ontogeorag/configs/descriptor_queries_v2.jsonl --index-dir /home/talbi/ontogeorag_D/output/run16E/index --backend hf --top-k 5 --bm25-topn 20 --min-bm25 2.0 --max-chars 2800 --temperature 0.0 --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --model /home/talbi/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 --output /home/talbi/ontogeorag_D/output/model_comparison/llama/raw_triples.jsonl

mkdir -p /home/talbi/ontogeorag_D/output/model_comparison/mistral
/home/talbi/kg_test/venv/bin/python3 /tmp/run16E_code/pipeline/02_extract_triples.py --schema /home/talbi/ontogeorag/configs/ontology_schema.json --queries /home/talbi/ontogeorag/configs/descriptor_queries_v2.jsonl --index-dir /home/talbi/ontogeorag_D/output/run16E/index --backend hf --top-k 5 --bm25-topn 20 --min-bm25 2.0 --max-chars 2800 --temperature 0.0 --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --model /home/talbi/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a --output /home/talbi/ontogeorag_D/output/model_comparison/mistral/raw_triples.jsonl