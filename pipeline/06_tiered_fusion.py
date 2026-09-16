#!/usr/bin/env python3
"""
pipeline/06_tiered_fusion.py — Tiered KG Fusion (Stage 6)
=============================================================
WHY
    A single run under-samples the corpus; running the pipeline twice
    (e.g. iter_a + iter_b) and fusing by best-verdict-per-triple recovers
    triples either run missed, while the tier label preserves each
    triple's verification confidence for downstream reporting (07) and
    consumers that only want Tier 1 or Tier 1+2.

WHAT
    Merges two pipeline runs (e.g. iter4 + iter7) into a tiered KG:
      Tier 1 — STRONG_SUPPORT (0% hallucination, verified)
      Tier 2 — WEAK_SUPPORT   (literature-implied, ~2.9% hallucination)
      Tier 3 — NOT_SUPPORTED  (parametric-flagged; included only with --include-tier3)

    A triple present in both runs takes the better (lower-numbered) tier
    of the two ("origin": "both"); supporting-paper evidence is aggregated
    across both runs regardless of which run's tier won.

    Handles both verdict field formats:
      - Old format: _verdict (root level)
      - New format: _verification.verdict (nested)

Usage:
    python pipeline/06_tiered_fusion.py \\
        --iter-a output/step4/canonical_triples.jsonl \\
        --iter-b output/step5/canonical_triples.jsonl \\
        --output output/kg/tiered_kg.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pipeline.rag.constants import normalize_entity, normalize_relation

TIER_LABELS = {1: "verified", 2: "literature-implied", 3: "parametric-flagged"}


def get_verdict(t: dict) -> str:
    """Read the verification verdict from `t`, preferring nested `_verification.verdict` over the legacy root-level `_verdict`; returns the verdict string ("" if none), no side effects."""
    ver = t.get("_verification")
    if isinstance(ver, dict):
        v = ver.get("verdict", "")
        if v:
            return v
    return t.get("_verdict", "")


def get_evidence(t: dict) -> str:
    """Return `t`'s supporting evidence text, preferring the verification evidence (truncated to 400 chars) and falling back to the stage-2 provenance chunk preview (300 chars); no side effects."""
    ver = t.get("_verification")
    if isinstance(ver, dict):
        ev = ver.get("evidence", "")
        if ev:
            return ev[:400]
    prov = t.get("_provenance", {})
    return (prov.get("best_chunk_text", "") or "")[:300]


def verdict_to_tier(verdict: str) -> int:
    """Map a verification verdict string to its tier number (1=STRONG, 2=WEAK/UNCERTAIN, 3=everything else); no side effects."""
    v = str(verdict).upper()
    if "STRONG" in v:
        return 1
    if "WEAK" in v or "UNCERTAIN" in v:
        return 2
    return 3


def triple_key(t: dict) -> tuple:
    """Build the (subject, relation, object) dedup/fusion key for `t` using shared entity/relation normalization; no side effects."""
    return (
        normalize_entity(t.get("source", t.get("subject", ""))),
        normalize_relation(t.get("relation", "")),
        normalize_entity(t.get("target", t.get("object", ""))),
    )


def load_jsonl(path: str) -> list[dict]:
    """Read `path` as one JSON object per line, skipping (and warning on, up to 3 shown) malformed lines; returns the list of parsed dicts."""
    triples, errors = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                triples.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 3:
                    print(f"  Warning line {i + 1}: {e}")
    if errors:
        print(f"  Total malformed lines skipped: {errors}")
    return triples


def load_triples(path: str) -> list[dict]:
    """Load triples from .jsonl or .json (dict with 'triples' key).
    Detection: .json extension OR content starts with '[' = JSON array/object.
    Everything else (including .jsonl starting with '{') = JSONL.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    # Only treat as single JSON if it's a JSON array OR a .json file
    import os

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json" or content.startswith("["):
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("triples", [])
    # JSONL: one dict per line
    triples, errors = [], 0
    for i, line in enumerate(content.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            triples.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors += 1
            if errors <= 3:
                print(f"  Warning line {i + 1}: {e}")
    if errors:
        print(f"  Total malformed lines skipped: {errors}")
    return triples


def extract_doc_ids(t: dict) -> list[str]:
    """Extract unique paper doc_ids from provenance chunk ids."""
    prov = t.get("_provenance", {}) or {}
    doc_ids = set()
    # From selected_chunk_ids (list of "doc_id::chunkN")
    for cid in prov.get("selected_chunk_ids") or []:
        if "::" in str(cid):
            doc_ids.add(str(cid).split("::")[0].strip())
    # From best_chunk_id
    best = prov.get("best_chunk_id", "")
    if best and "::" in str(best):
        doc_ids.add(str(best).split("::")[0].strip())
    return sorted(doc_ids)


def to_standard(t: dict, tier: int, origin: str) -> dict:
    """Convert one raw pipeline triple `t` into the standardized fused-KG record shape (normalized subject/relation/object plus tier/origin/evidence/provenance fields); no side effects."""
    prov = t.get("_provenance", {}) or {}
    ver = (
        t.get("_verification", {})
        if isinstance(t.get("_verification"), dict)
        else {}
    )
    doc_ids = extract_doc_ids(t)
    return {
        "subject": normalize_entity(t.get("source", t.get("subject", ""))),
        "relation": normalize_relation(t.get("relation", "")),
        "object": normalize_entity(t.get("target", t.get("object", ""))),
        "subject_type": t.get("source_type", ""),
        "object_type": t.get("target_type", ""),
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "verdict": get_verdict(t),
        "origin": origin,
        "evidence": get_evidence(t),
        "query": prov.get("query", ""),
        "strategy": prov.get("strategy", ""),
        "model": ver.get("model", prov.get("model", "")),
        "reasoning": (ver.get("reasoning") or "")[:200],
        "support_count": len(doc_ids),
        "supporting_papers": doc_ids,
    }


def build_index(triples: list[dict]) -> dict:
    """Deduplicate `triples` by triple_key, keeping the best (lowest-numbered) tier seen per key; returns {key: {"tier": int, "raw": triple}}, no side effects."""
    index = {}
    for t in triples:
        k = triple_key(t)
        tier = verdict_to_tier(get_verdict(t))
        # Lower tier number = higher confidence (1=STRONG, 2=WEAK, 3=NOT
        # SUPPORTED/other), so "<" here means "replace with the more
        # trustworthy of any duplicate raw triples within this one run".
        if k not in index or tier < index[k]["tier"]:
            index[k] = {"tier": tier, "raw": t}
    return index


def main():
    """CLI entry point: loads --iter-a/--iter-b, fuses them by best-tier-per-triple with aggregated cross-run paper evidence, and writes the tiered KG (with metadata) to --output."""
    parser = argparse.ArgumentParser(description="Tiered KG fusion")
    parser.add_argument(
        "--iter-a", required=True, help="First run canonical triples JSONL"
    )
    parser.add_argument(
        "--iter-b", required=True, help="Second run canonical triples JSONL"
    )
    parser.add_argument("--output", default="output/kg/tiered_kg.json")
    parser.add_argument(
        "--include-tier3",
        action="store_true",
        help="Include NOT_SUPPORTED triples (Tier 3) in output",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TIERED KG FUSION")
    print("=" * 60)

    iter_a = load_triples(args.iter_a)
    iter_b = load_triples(args.iter_b)
    print(f"\n  Iter-A: {len(iter_a)} triples  ({args.iter_a})")
    print(f"  Iter-B: {len(iter_b)} triples  ({args.iter_b})")

    idx_a = build_index(iter_a)
    idx_b = build_index(iter_b)
    keys_b = set(idx_b.keys())

    tiered = []

    # Build per-key paper evidence index across ALL raw triples (both iters)
    paper_index: dict[tuple, set] = defaultdict(set)
    for t in iter_a + iter_b:
        k = triple_key(t)
        for doc in extract_doc_ids(t):
            paper_index[k].add(doc)

    for k, entry in idx_a.items():
        tier = entry["tier"]
        if k in keys_b:
            # A triple found by both runs takes the BETTER (lower) tier
            # of the two — if either run's verifier gave STRONG_SUPPORT,
            # that confidence carries even if the other run only reached
            # WEAK_SUPPORT. This is what makes cross-run consistency an
            # upgrade path rather than a downgrade risk.
            tier = min(tier, idx_b[k]["tier"])
            origin = "both"
        else:
            origin = "iter_a"
        if tier == 3 and not args.include_tier3:
            continue
        t_std = to_standard(entry["raw"], tier, origin)
        # Override with aggregated paper evidence
        all_papers = sorted(paper_index.get(k, set()))
        t_std["support_count"] = len(all_papers)
        t_std["supporting_papers"] = all_papers
        tiered.append(t_std)

    for k, entry in idx_b.items():
        if k not in idx_a:
            tier = entry["tier"]
            if tier == 3 and not args.include_tier3:
                continue
            t_std = to_standard(entry["raw"], tier, "iter_b")
            all_papers = sorted(paper_index.get(k, set()))
            t_std["support_count"] = len(all_papers)
            t_std["supporting_papers"] = all_papers
            tiered.append(t_std)

    tiered.sort(key=lambda x: (x["tier"], x["relation"], x["subject"]))

    tier_counts = defaultdict(int)
    rel_counts = defaultdict(int)
    origin_counts = defaultdict(int)
    for t in tiered:
        tier_counts[t["tier"]] += 1
        rel_counts[t["relation"]] += 1
        origin_counts[t["origin"]] += 1

    print(f"\n  Final KG: {len(tiered)} triples")
    print(f"  Tier 1 (verified):           {tier_counts[1]:4d}")
    print(f"  Tier 2 (literature-implied): {tier_counts[2]:4d}")
    if args.include_tier3:
        print(f"  Tier 3 (parametric):         {tier_counts[3]:4d}")
    print(f"\n  Origin:")
    for o, c in sorted(origin_counts.items()):
        print(f"    {o:20s}: {c}")
    print(f"\n  Relations:")
    for r, c in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {r:22s}: {c}")

    out = {
        "metadata": {
            "iter_a_file": str(args.iter_a),
            "iter_b_file": str(args.iter_b),
            "iter_a_count": len(iter_a),
            "iter_b_count": len(iter_b),
            "total_triples": len(tiered),
            "tier_distribution": dict(tier_counts),
            "relation_distribution": dict(rel_counts),
            "origin_distribution": dict(origin_counts),
            "include_tier3": args.include_tier3,
        },
        "triples": tiered,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
