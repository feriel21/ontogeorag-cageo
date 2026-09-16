#!/usr/bin/env python3
"""
m4_verify.py — M4 Independent Cross-Family Verifier (main script)
==================================================================

Runs BOTH verification passes on every triple of the input graph:

  Pass 1 (BLIND):    triple only            -> parametric plausibility
  Pass 2 (EVIDENCE): triple + source passage -> textual support

The two passes are run in the SAME job (model loaded once) but their
prompts are strictly isolated: the blind pass never sees the passage.

Inputs
------
  --kg      Tiered KG JSON (e.g. output/run11_kg/tiered_kg_run11.json)
            OR a .jsonl of triples (one JSON object per line, e.g. the
            output of 05_canonicalize.py). Format auto-detected.
  --index   Optional: directory containing the chunk index (output/step1/)
            used as a fallback to locate evidence when a triple's own
            provenance does not embed the passage text.
  --model   HF model id (default: meta-llama/Llama-3.1-8B-Instruct)

Outputs (written to --output directory)
---------------------------------------
  m4_verdicts.jsonl   one record per triple:
      {
        "m4_index": int,
        "subject": str, "relation": str, "object": str,
        "tier": int|null,
        "qwen_verdict": str|null,          # original pipeline verdict, if any
        "blind":    {"verdict": ..., "reasoning": ..., "raw": ...},
        "evidence": {"verdict": ..., "quote": ..., "reasoning": ...,
                     "raw": ..., "passage_found": bool},
        "model": str
      }
  m4_run_meta.json    run metadata (model, params, timings, counts)

Aggregation (confidence + ACCEPT/REJECT/UNCERTAIN) is done separately by
m4_aggregate.py so that the expensive GPU pass never needs re-running when
aggregation rules are adjusted.

Usage (from any directory — no pipeline imports required):
    python m4_verify.py \
        --kg     ~/ontogeorag/output/run11_kg/tiered_kg_run11.json \
        --index  ~/ontogeorag/output/step1 \
        --output ~/ontogeorag/output/m4 \
        --model  meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

from m4_config import (
    BLIND_PROMPT,
    BLIND_SYSTEM,
    BLIND_VERDICTS,
    DEFAULT_MODEL,
    EVIDENCE_PROMPT,
    EVIDENCE_SYSTEM,
    EVIDENCE_VERDICTS,
    GEN_KWARGS,
    MAX_EVIDENCE_CHARS,
    get_glosses,
)

# NOTE: torch/transformers are imported lazily inside load_model() and
# generate(), so that CPU-only scripts (m4_negatives.py, m4_quote_verify,
# m4_integrate_tiers.py) can import the data helpers from this module
# without requiring GPU dependencies.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m4")


# ── Model loading / generation ─────────────────────────────────────────


def resolve_local(model_name: str) -> str:
    """Resolve a hub id to its local cache snapshot path (offline-safe).

    Works around a transformers bug where _patch_mistral_regex performs a
    network call (model_info) even under HF_HUB_OFFLINE=1, raising
    OfflineModeIsEnabled although the model is fully cached. Passing a
    local path makes _is_local True and skips that branch entirely.
    """
    if Path(model_name).expanduser().exists():
        return str(Path(model_name).expanduser())
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(model_name, local_files_only=True)
        log.info(f"Resolved to local snapshot: {local}")
        return local
    except Exception as e:
        log.warning(f"Local cache resolution failed ({e}); using hub id.")
        return model_name


def load_model(model_name: str):
    """Resolve `model_name` to a local snapshot and load its tokenizer + causal LM in bfloat16 (device_map='auto'); returns (tokenizer, model) in eval mode."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = resolve_local(model_name)
    log.info(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    mdl.eval()
    log.info("Model loaded.")
    return tok, mdl


def generate(tok, mdl, system: str, user: str) -> str:
    """Render `system`/`user` via the chat template and greedily generate a completion; returns the decoded response text, no side effects."""
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            inputs,
            max_new_tokens=GEN_KWARGS["max_new_tokens"],
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][inputs.shape[1] :], skip_special_tokens=True)


# ── Parsing ────────────────────────────────────────────────────────────


def parse_pass(response: str, allowed: tuple, fields: tuple) -> dict:
    """Parse a formatted response; robust to missing sections."""
    result = {f.lower(): "" for f in fields}
    result["verdict"] = "UNPARSEABLE"
    result["raw"] = response[:800]

    for f in fields:
        pattern = rf"{f}:\s*(.+?)(?=\n[A-Z]+:|\Z)"
        m = re.search(pattern, response, re.DOTALL | re.I)
        if m:
            result[f.lower()] = m.group(1).strip()

    m = re.search(r"VERDICT:\s*(" + "|".join(allowed) + r")", response, re.I)
    if m:
        result["verdict"] = m.group(1).upper()
        return result

    # Fallback: bare label anywhere in the response. Order matters:
    # longest / most specific first to avoid substring shadowing
    # (e.g. SUPPORTED inside NOT_SUPPORTED / PARTIALLY_SUPPORTED).
    for v in sorted(allowed, key=len, reverse=True):
        if v in response.upper():
            result["verdict"] = v
            break
    return result


# ── Input loading (handles tiered-KG JSON and triples JSONL) ───────────


def load_triples(kg_path: Path) -> list:
    """Load triples from a tiered-KG JSON (list, {"triples"/"all_triples": [...]}, or {"tierN": [...]}) or a .jsonl of triples, inferring a `tier` field from key names where absent; no side effects."""
    text = kg_path.read_text(encoding="utf-8").strip()
    triples = []
    if kg_path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                triples.append(json.loads(line))
        return triples

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # tiered KG: {"tier1": [...], "tier2": [...]} or {"triples": [...]}
        for key in ("triples", "all_triples"):
            if key in data and isinstance(data[key], list):
                return data[key]
        found = []
        for key, val in data.items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                for t in val:
                    if "tier" not in t:
                        m = re.search(r"(\d+)", key)
                        if m:
                            t["tier"] = int(m.group(1))
                    found.append(t)
        if found:
            return found
    raise ValueError(f"Unrecognised KG format: {kg_path}")


def triple_fields(t: dict) -> tuple:
    """Extract (subject, relation, object) strings from `t`, preferring normalized fields over raw ones; no side effects."""
    subj = t.get("subject") or t.get("source_norm") or t.get("source", "")
    rel = t.get("relation_norm") or t.get("relation", "")
    obj = t.get("object") or t.get("target_norm") or t.get("target", "")
    return str(subj), str(rel), str(obj)


# ── Evidence lookup ────────────────────────────────────────────────────


def evidence_from_provenance(t: dict) -> str:
    """Extract embedded passage text (same formats as 03_verify_triples)."""
    prov = t.get("_provenance", {}) or {}
    for ch in prov.get("top_chunks", [])[:1]:
        txt = (ch or {}).get("text", "")
        if txt:
            return txt
    for key in ("best_chunk_text", "chunk_text", "context_preview"):
        txt = prov.get(key, "") or ""
        if txt:
            return txt
    # tiered KG variants: supporting passage stored at top level
    for key in (
        "supporting_passage",
        "passage",
        "evidence_text",
        "source_passage",
        "chunk_text",
    ):
        txt = t.get(key, "") or ""
        if isinstance(txt, str) and txt:
            return txt
    ev = t.get("evidence", {})
    if isinstance(ev, dict):
        for key in ("passage", "text", "quote"):
            txt = ev.get(key, "") or ""
            if txt:
                return txt
    if isinstance(ev, str) and ev:
        return ev
    return ""


def load_chunk_index(index_dir: Path) -> list:
    """Load chunk texts from the BM25 index directory (fallback lookup)."""
    chunks = []
    if index_dir is None or not index_dir.exists():
        return chunks
    for name in ("chunks.jsonl", "chunks.json"):
        p = index_dir / name
        if p.exists():
            if p.suffix == ".jsonl":
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            chunks.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            else:
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        chunks = data
                except json.JSONDecodeError:
                    pass
            break
    log.info(f"Chunk index: {len(chunks)} chunks loaded from {index_dir}")
    return chunks


def evidence_from_index(t: dict, chunks: list) -> str:
    """Keyword fallback: best chunk containing both entities (as in expD)."""
    if not chunks:
        return ""
    subj, _, obj = triple_fields(t)
    s, o = subj.lower(), obj.lower()
    best, best_score = "", 0
    for ch in chunks:
        txt = ch.get("text", "") if isinstance(ch, dict) else str(ch)
        low = txt.lower()
        score = (s in low) + (o in low)
        if score > best_score:
            best_score, best = score, txt
            if score == 2:
                break
    return best if best_score == 2 else ""


# ── Main ───────────────────────────────────────────────────────────────


def main():
    """CLI entry point: loads --kg triples, runs the blind + evidence passes on each with the --model verifier, and writes per-triple verdicts (m4_verdicts.jsonl) plus run metadata (m4_run_meta.json) to --output."""
    ap = argparse.ArgumentParser(description="M4 cross-family verifier")
    ap.add_argument(
        "--kg", required=True, help="Tiered KG JSON or triples JSONL"
    )
    ap.add_argument("--index", default=None, help="Chunk index dir (fallback)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Verify only the first N triples (0 = all)",
    )
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = load_triples(Path(args.kg).expanduser())
    if args.limit:
        triples = triples[: args.limit]
    log.info(f"Loaded {len(triples)} triples from {args.kg}")

    chunks = (
        load_chunk_index(Path(args.index).expanduser()) if args.index else []
    )
    glosses = get_glosses()

    tok, mdl = load_model(args.model)

    verdict_path = out_dir / "m4_verdicts.jsonl"
    t_start = time.time()
    n_no_passage = 0

    with open(verdict_path, "w", encoding="utf-8") as fout:
        for i, t in enumerate(triples):
            subj, rel, obj = triple_fields(t)
            gloss = glosses.get(rel, rel)

            # ---- Pass 1: BLIND (no passage in prompt, by construction)
            blind_raw = generate(
                tok,
                mdl,
                BLIND_SYSTEM,
                BLIND_PROMPT.format(
                    subject=subj, relation=rel, object=obj, gloss=gloss
                ),
            )
            blind = parse_pass(blind_raw, BLIND_VERDICTS, ("REASONING",))

            # ---- Pass 2: EVIDENCE
            passage = evidence_from_provenance(t)
            if not passage:
                passage = evidence_from_index(t, chunks)
            if passage:
                ev_raw = generate(
                    tok,
                    mdl,
                    EVIDENCE_SYSTEM,
                    EVIDENCE_PROMPT.format(
                        evidence=passage[:MAX_EVIDENCE_CHARS],
                        subject=subj,
                        relation=rel,
                        object=obj,
                        gloss=gloss,
                    ),
                )
                evid = parse_pass(
                    ev_raw, EVIDENCE_VERDICTS, ("QUOTE", "REASONING")
                )
                evid["passage_found"] = True
            else:
                n_no_passage += 1
                evid = {
                    "verdict": "NO_PASSAGE",
                    "quote": "",
                    "reasoning": "",
                    "raw": "",
                    "passage_found": False,
                }

            record = {
                "m4_index": i,
                "subject": subj,
                "relation": rel,
                "object": obj,
                "tier": t.get("tier"),
                "qwen_verdict": (t.get("_verification", {}) or {}).get(
                    "verdict", t.get("verdict")
                ),
                "blind": {
                    k: blind.get(k, "")
                    for k in ("verdict", "reasoning", "raw")
                },
                "evidence": {
                    k: evid.get(k, "")
                    for k in (
                        "verdict",
                        "quote",
                        "reasoning",
                        "raw",
                        "passage_found",
                    )
                },
                "model": args.model,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == len(triples):
                elapsed = time.time() - t_start
                log.info(
                    f"[{i + 1}/{len(triples)}]  "
                    f"blind={blind['verdict']:<11s} "
                    f"evidence={evid['verdict']:<19s} "
                    f"({elapsed / (i + 1):.1f}s/triple)"
                )

    meta = {
        "model": args.model,
        "input": str(args.kg),
        "n_triples": len(triples),
        "n_no_passage": n_no_passage,
        "decoding": "greedy (do_sample=False)",
        "max_evidence_chars": MAX_EVIDENCE_CHARS,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (out_dir / "m4_run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log.info(f"Done. Verdicts: {verdict_path}")
    log.info(
        f"Triples without passage: {n_no_passage} "
        f"(check --index if this is high)"
    )


if __name__ == "__main__":
    main()
