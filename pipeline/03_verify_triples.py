#!/usr/bin/env python3
"""
pipeline/03_verify_triples.py — CoT Triple Verification (Stage 3)
====================================================================
WHY
    Stage 2's LLM extraction hallucinates; this stage cross-checks each
    candidate triple against its own source chunk(s) before it is allowed
    into the KG, so downstream tier assignment (06) has a hallucination
    signal to act on. Approach is GraphJudge-inspired.

WHAT
    For each extracted triple, retrieves the source chunk(s) and asks the
    LLM "Does the text support this triple?" with chain-of-thought
    reasoning. Outputs one of:
    STRONG_SUPPORT | WEAK_SUPPORT | NOT_SUPPORTED | NO_CHUNK | UNPARSEABLE

    Supports any chat-template HF model via --model (Qwen, Llama, Mistral, etc.)

USAGE
    python pipeline/03_verify_triples.py \\
        --input   output/step2/raw_triples.jsonl \\
        --output  output/step3/verified_triples.jsonl \\
        --model   Qwen/Qwen2.5-7B-Instruct \\
        --backend hf
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from pipeline.rag.constants import (
    ALLOWED_RELATIONS,
    RELATION_GLOSSES,
    normalize_relation,
)

# ── Prompts ────────────────────────────────────────────────────────────

COT_PROMPT = """\
You are a geological fact-checker. Verify whether a claimed knowledge-graph triple
is supported by the provided source text passage.

=== SOURCE TEXT ===
{chunk_text}
=== END SOURCE TEXT ===

=== CLAIMED TRIPLE ===
  Subject:  {source}
  Relation: {relation} (meaning: {gloss})
  Object:   {target}
=== END TRIPLE ===

Follow steps:

STEP 1 — EVIDENCE: Copy the most relevant sentence(s) from the source text
that relate to this triple. If none, write "NO EVIDENCE FOUND".

STEP 2 — REASONING: In 1-2 sentences, explain whether the evidence supports
the triple. Use ONLY the source text (no external knowledge).

STEP 3 — VERDICT: Choose exactly one:
  STRONG_SUPPORT   — text explicitly states it
  WEAK_SUPPORT     — text implies it / direct inference
  NOT_SUPPORTED    — not stated or not implied

Format EXACTLY:
EVIDENCE: <...>
REASONING: <...>
VERDICT: <STRONG_SUPPORT or WEAK_SUPPORT or NOT_SUPPORTED>
"""

FALLBACK_PROMPT = """\
Text: {chunk_text}

Does this text state or clearly imply that "{source}" {gloss} "{target}"?

Answer with ONLY one of:
STRONG_SUPPORT / WEAK_SUPPORT / NOT_SUPPORTED
"""

SYSTEM = (
    "You are a precise geological fact-checker. "
    "Use ONLY the provided source text. Follow formatting exactly."
)


def parse_cot(response: str) -> dict:
    """Extract EVIDENCE/REASONING/VERDICT fields from the LLM's CoT `response`; returns a dict with verdict defaulting to UNPARSEABLE, no side effects."""
    result = {
        "evidence": "",
        "reasoning": "",
        "verdict": "UNPARSEABLE",
        "raw": response,
    }

    m = re.search(
        r"EVIDENCE:\s*(.+?)(?=\nREASONING:|\nVERDICT:|\Z)",
        response,
        re.DOTALL | re.I,
    )
    if m:
        result["evidence"] = m.group(1).strip()

    m = re.search(
        r"REASONING:\s*(.+?)(?=\nVERDICT:|\Z)", response, re.DOTALL | re.I
    )
    if m:
        result["reasoning"] = m.group(1).strip()

    m = re.search(
        r"VERDICT:\s*(STRONG_SUPPORT|WEAK_SUPPORT|NOT_SUPPORTED)",
        response,
        re.I,
    )
    if m:
        result["verdict"] = m.group(1).upper()
        return result

    up = response.upper()
    for v in ("NOT_SUPPORTED", "STRONG_SUPPORT", "WEAK_SUPPORT"):
        if v in up or v.replace("_", " ") in up:
            result["verdict"] = v
            break

    return result


def get_chunks(triple: dict, max_chunks: int = 3) -> list[str]:
    """Extract source chunk texts from triple provenance (handles multiple formats)."""
    prov = triple.get("_provenance", {}) or {}
    texts = []

    # Format: top_chunks[] list
    for ch in prov.get("top_chunks", [])[:max_chunks]:
        t = (ch or {}).get("text", "")
        if t and t not in texts:
            texts.append(t)
    if texts:
        return texts

    # Format: best_chunk_text + second_chunk_text (old)
    for key in [
        "best_chunk_text",
        "second_chunk_text",
        "chunk_text_2",
        "chunk_text_3",
    ]:
        t = prov.get(key, "") or ""
        if t and t not in texts:
            texts.append(t)

    # Fallback: context_preview (concatenated top-k)
    if len(texts) < max_chunks:
        cp = prov.get("context_preview", "") or ""
        if cp and cp not in texts:
            texts.append(cp)

    return texts[:max_chunks]


def verify_triple(triple: dict, generate_fn, chunk_chars: int = 1500) -> dict:
    """Run CoT verification on one triple. Returns verification dict."""
    rel_raw = triple.get("relation_norm", triple.get("relation", ""))
    rel = normalize_relation(str(rel_raw))

    if rel not in ALLOWED_RELATIONS:
        return {
            "verdict": "SKIPPED_RELATION",
            "evidence": "",
            "reasoning": f"Relation '{rel}' not in allowed set.",
            "chunks_checked": 0,
        }

    chunk_texts = get_chunks(triple)
    if not chunk_texts:
        return {
            "verdict": "NO_CHUNK",
            "evidence": "",
            "reasoning": "No source chunk found in provenance.",
            "chunks_checked": 0,
        }

    source = triple.get("source_norm", triple.get("source", ""))
    target = triple.get("target_norm", triple.get("target", ""))
    gloss = RELATION_GLOSSES.get(rel, f"Subject {rel} Object")

    verdict_rank = {
        "STRONG_SUPPORT": 3,
        "WEAK_SUPPORT": 2,
        "NOT_SUPPORTED": 1,
        "UNPARSEABLE": 0,
    }
    best, best_score = None, -1

    for chunk_text in chunk_texts:
        user = COT_PROMPT.format(
            chunk_text=chunk_text[:chunk_chars],
            source=source,
            relation=rel,
            gloss=gloss,
            target=target,
        )
        try:
            resp = generate_fn(SYSTEM, user)
            parsed = parse_cot(resp)
        except Exception as e:
            return {
                "verdict": "BACKEND_ERROR",
                "evidence": "",
                "reasoning": str(e),
                "chunks_checked": 0,
            }

        # One retry with simpler prompt if unparseable
        if parsed["verdict"] == "UNPARSEABLE":
            user2 = FALLBACK_PROMPT.format(
                chunk_text=chunk_text[:1200],
                source=source,
                gloss=gloss.lower(),
                target=target,
            )
            try:
                resp2 = generate_fn(SYSTEM, user2)
                parsed2 = parse_cot(resp2)
                if parsed2["verdict"] != "UNPARSEABLE":
                    parsed = parsed2
            except Exception:
                pass

        score = verdict_rank.get(parsed["verdict"], 0)
        if score > best_score:
            best_score, best = score, parsed

    if best is None:
        return {
            "verdict": "UNPARSEABLE",
            "evidence": "",
            "reasoning": "",
            "chunks_checked": len(chunk_texts),
        }

    best["chunks_checked"] = len(chunk_texts)
    return best


def main():
    """CLI entry point: loads --input triples, runs verify_triple over each with the configured LLM backend, and writes verified triples (--output), a per-triple audit log, and verification_stats.json."""
    parser = argparse.ArgumentParser(description="CoT triple verification")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--chunk-chars", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    triples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))
    print(f"  Loaded {len(triples)} triples")
    if args.limit > 0:
        triples = triples[: args.limit]

    # Load LLM
    if args.backend == "hf":
        from pipeline.rag.llm_hf import make_hf_fn

        _gen = make_hf_fn(args.model)
        generate_fn = lambda sys, usr: _gen(sys, usr)
    else:
        import requests

        def generate_fn(sys_msg: str, user_msg: str) -> str:
            r = requests.post(
                f"{args.ollama_url}/api/chat",
                json={
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=180,
            )
            return r.json().get("message", {}).get("content", "")

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    audit_path = outpath.parent / "verification_audit.jsonl"
    stats_path = outpath.parent / "verification_stats.json"

    counts = {
        v: 0
        for v in [
            "STRONG_SUPPORT",
            "WEAK_SUPPORT",
            "NOT_SUPPORTED",
            "NO_CHUNK",
            "UNPARSEABLE",
            "BACKEND_ERROR",
            "SKIPPED_RELATION",
        ]
    }
    start = time.time()

    with (
        open(outpath, "w", encoding="utf-8") as fout,
        open(audit_path, "w", encoding="utf-8") as faudit,
    ):
        for i, triple in enumerate(triples):
            verif = verify_triple(
                triple, generate_fn, chunk_chars=args.chunk_chars
            )
            verdict = verif.get("verdict", "UNPARSEABLE")
            counts[verdict] = counts.get(verdict, 0) + 1

            rel_norm = normalize_relation(
                triple.get("relation_norm", triple.get("relation", ""))
            )
            triple["_verification"] = {
                "verdict": verdict,
                "evidence": verif.get("evidence", ""),
                "reasoning": verif.get("reasoning", ""),
                "chunks_checked": verif.get("chunks_checked", 0),
                "model": args.model,
                "backend": args.backend,
                "relation_norm": rel_norm,
            }

            fout.write(json.dumps(triple, ensure_ascii=False) + "\n")
            faudit.write(
                json.dumps(
                    {
                        "index": i,
                        "subject": triple.get(
                            "source_norm", triple.get("source", "")
                        ),
                        "relation": rel_norm,
                        "object": triple.get(
                            "target_norm", triple.get("target", "")
                        ),
                        **verif,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if (i + 1) % 25 == 0 or (i + 1) == len(triples):
                elapsed = time.time() - start
                decided = (
                    counts["STRONG_SUPPORT"]
                    + counts["WEAK_SUPPORT"]
                    + counts["NOT_SUPPORTED"]
                )
                print(
                    f"  [{i + 1}/{len(triples)}] "
                    f"S={counts['STRONG_SUPPORT']} W={counts['WEAK_SUPPORT']} "
                    f"NS={counts['NOT_SUPPORTED']} NC={counts['NO_CHUNK']} "
                    f"({(i + 1) / elapsed:.1f}/s)"
                )

    elapsed = time.time() - start
    decided = (
        counts["STRONG_SUPPORT"]
        + counts["WEAK_SUPPORT"]
        + counts["NOT_SUPPORTED"]
    )
    halluc = counts["NOT_SUPPORTED"] / decided if decided > 0 else None

    stats = {
        "model": args.model,
        "backend": args.backend,
        "total_triples": len(triples),
        "counts": counts,
        "decided_total": decided,
        "hallucination_rate_decided": halluc,
        "elapsed_seconds": round(elapsed, 1),
    }
    Path(stats_path).write_text(json.dumps(stats, indent=2))

    print(f"\n{'=' * 55}")
    print("  VERIFICATION SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Total: {len(triples)}  |  Decided: {decided}")
    print(
        f"  STRONG: {counts['STRONG_SUPPORT']}  WEAK: {counts['WEAK_SUPPORT']}  "
        f"NOT_SUPPORTED: {counts['NOT_SUPPORTED']}"
    )
    if halluc is not None:
        print(f"  Hallucination rate (decided): {halluc:.1%}")
    print(f"  Model: {args.model}")
    print(f"  Output:  {outpath}")
    print(f"  Audit:   {audit_path}")
    print(f"  Stats:   {stats_path}")


if __name__ == "__main__":
    main()
