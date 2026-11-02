#!/usr/bin/env bash
# workflows/prefetch_models.sh — download the pinned models into the Hugging Face cache.
# Run once on a node with internet access (login node), inside the Python environment.
# Llama 3.1 is gated: the Hugging Face account must have accepted the Meta licence,
# then:  huggingface-cli login   (or export HF_TOKEN=...)
#
# The three LLMs are downloaded at the exact revisions of the paper (the driver loads
# them by revision). The reranker and SciBERT are loaded by name by the pipeline, so they
# are downloaded by name; the script then compares the revision obtained with the one
# used for the paper and prints WARNING if it differs.
set -euo pipefail
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
pinned = [
    ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "63a8b081895390a26e140280378bc85ec8bce07a"),
]
by_name = [
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
    ("allenai/scibert_scivocab_uncased", "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"),
]
for repo, rev in pinned:
    path = snapshot_download(repo, revision=rev)
    print("ok", repo, rev, "->", path)
warn = 0
for repo, paper_rev in by_name:
    path = snapshot_download(repo)
    got = Path(path).name
    if got == paper_rev:
        print("ok", repo, got, "(same revision as the paper) ->", path)
    else:
        warn += 1
        print("WARNING", repo, "revision", got, "differs from the paper's", paper_rev, "->", path)
if warn:
    print(f"{warn} model(s) differ from the paper; the run will work but record this in the report.")
PY