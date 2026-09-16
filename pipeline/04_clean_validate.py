#!/usr/bin/env python3
"""
pipeline/04_clean_validate.py — Validation, Cleaning & Canonicalization (Stage 4, v5)
========================================================================================
WHY
    Stage 3's verification only says whether a triple is text-supported; it
    doesn't catch structurally bad triples (wrong relation, vague entities,
    off-ontology descriptors) or near-duplicate entity mentions. This stage
    is the ontology/lexicon gate and the entity-canonicalization step, so
    stages 5+ operate on a clean, deduplicated triple set.

    FIX #1 — relation mapping BEFORE ontology check (recover SKIPPED_RELATION)
    FIX #2 — soft lexicon: if both_not_in_lexicon but verified STRONG/WEAK => keep as novel_term=True

WHAT
    Runs the verification filter, then structural/ontology validation, then
    SciBERT-embedding-based entity clustering to merge near-duplicate
    mentions (protecting LB2019 descriptor/setting terms from being merged
    away — see build_canonical_map).

Inputs:
  - verified_triples_v5.jsonl  (from verification script)

Outputs:
  - canonical_triples_v5.jsonl
  - canonical_map_v5.json
  - rejected_triples_v5.jsonl
  - cleaning_stats_v5.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pipeline.rag.constants import (
    KNOWN_DESCRIPTORS,
)
from pipeline.rag.constants import (
    LB2019_BENCHMARK_DESCRIPTORS as LB_DESCRIPTORS,
)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

LB_REFERENCE_EDGES = [
    ("mass transport deposit", "hasDescriptor", "chaotic"),
    ("mass transport deposit", "hasDescriptor", "transparent"),
    ("mass transport deposit", "hasDescriptor", "hummocky"),
    ("mass transport deposit", "hasDescriptor", "blocky"),
    ("mass transport deposit", "hasDescriptor", "discontinuous"),
    ("mass transport deposit", "hasDescriptor", "massive"),
    ("turbidite", "hasDescriptor", "parallel"),
    ("turbidite", "hasDescriptor", "continuous"),
    ("turbidite", "hasDescriptor", "layered"),
    ("turbidite", "hasDescriptor", "high-amplitude"),
    ("debris flow", "hasDescriptor", "chaotic"),
    ("debris flow", "hasDescriptor", "hummocky"),
    ("slide", "hasDescriptor", "blocky"),
    ("slide", "hasDescriptor", "undeformed"),
    ("hemipelagite", "hasDescriptor", "parallel"),
    ("hemipelagite", "hasDescriptor", "continuous"),
    ("hemipelagite", "hasDescriptor", "low-amplitude"),
    ("slope failure", "causes", "mass transport deposit"),
    ("earthquake", "triggers", "slope failure"),
    ("pore pressure", "controls", "slope failure"),
    ("turbidity current", "formedBy", "debris flow"),
    ("mass transport deposit", "occursIn", "continental slope"),
    ("mass transport deposit", "occursIn", "abyssal plain"),
    ("debris flow", "occursIn", "continental slope"),
    ("turbidite", "occursIn", "basin floor"),
    ("slide", "overlies", "hemipelagite"),
]

VALID_RELATIONS = {
    "hasDescriptor",
    "occursIn",
    "formedBy",
    "partOf",
    "triggers",
    "causes",
    "controls",
    "affects",
    "overlies",
    "underlies",
    "associatedWith",
    "contains",
    "transports",
    "erodes",
    "deposits",
}

TYPE_CONSTRAINTS = {
    "hasDescriptor": {"object_type": "Descriptor"},
    "occursIn": {"object_type": "Setting"},
    "overlies": {
        "subject_type": "Geological_Object",
        "object_type": "Geological_Object",
    },
    "underlies": {
        "subject_type": "Geological_Object",
        "object_type": "Geological_Object",
    },
}

KNOWN_SETTINGS = {
    "continental slope",
    "continental shelf",
    "continental margin",
    "abyssal plain",
    "basin floor",
    "submarine canyon",
    "channel",
    "deep-water environment",
    "deep-water environments",
    "deep water",
    "passive margin",
    "active margin",
    "accretionary prism",
    "trench",
    "mid-ocean ridge",
    "seamount",
    "delta",
    "fan",
    "submarine fan",
    "levee",
    "overbank",
}

VAGUE_TERMS = {
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
    "something",
    "thing",
    "stuff",
    "area",
    "region",
    "feature",
    "process",
    "event",
    "result",
    "effect",
    "study",
    "analysis",
    "data",
    "figure",
    "table",
    "example",
    "case",
    "type",
    "kind",
    "form",
}

BLACKLIST_PATTERNS = [
    r"^fig(ure)?\.?\s*\d",
    r"^\d+(\.\d+)?$",
    r"^table\s+\d",
    r"^section\s+\d",
    r"^et\s+al",
    r"^\w{1,2}$",
]
BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS]


# ═══════════════════════════════════════════════════════════════════════
# FIX #1 — RELATION MAPPING
# ═══════════════════════════════════════════════════════════════════════
# Stage 2's LLM occasionally emits a close-but-nonstandard relation name
# (e.g. "hasFeature" instead of "hasDescriptor") that isn't in
# VALID_RELATIONS. Without this map, check_relation() would reject those
# triples outright as "invalid_relation" even though they carry a
# legitimate signal — this is why apply_relation_mapping() runs BEFORE
# check_relation() in main()'s validation loop, not after.

RELATION_MAP = {
    "hasfeature": "hasDescriptor",
    "has_feature": "hasDescriptor",
    "ischaracterizedby": "hasDescriptor",
    "characterizedby": "hasDescriptor",
    "describedby": "hasDescriptor",
    "locatedin": "occursIn",
    "occursinenvironment": "occursIn",
    "foundin": "occursIn",
    "composedof": "partOf",
    "madeof": "partOf",
    "consistsof": "partOf",
    "haspart": "contains",
    "include": "contains",
    "includes": "contains",
    "containing": "contains",
    "contains": "contains",
}


def normalize_relation(rel: str) -> str:
    """Collapse camelCase/spaces/hyphens/underscores in `rel` into a lowercase lookup key for RELATION_MAP; returns the key, no side effects."""
    rel = (rel or "").strip()
    if not rel:
        return ""
    rel = re.sub(r"([a-z])([A-Z])", r"\1 \2", rel)  # camelCase -> words
    rel = rel.lower()
    rel = re.sub(r"[\s\-_]+", "", rel)  # remove separators
    return rel


def apply_relation_mapping(triple: dict) -> None:
    """If `triple`'s relation is a known non-canonical alias (RELATION_MAP), rewrite its relation/relation_norm fields in place; returns None."""
    raw_rel = triple.get("relation_norm", triple.get("relation", ""))
    key = normalize_relation(raw_rel)
    if key in RELATION_MAP:
        mapped = RELATION_MAP[key]
        triple["relation"] = mapped
        triple["relation_norm"] = mapped


# ═══════════════════════════════════════════════════════════════════════
# VERIFICATION FILTER
# ═══════════════════════════════════════════════════════════════════════


def check_verification(triple: dict, policy: str) -> tuple[bool, str]:
    """Decide whether `triple` survives its stage-3 `_verification` verdict under the given `policy` (strict/normal/relaxed/off); returns (passes, reason_code), no side effects."""
    if policy == "off":
        # No stage-3 gate at all — used to inspect raw extraction output
        # (e.g. debugging stage 2) without verification noise.
        return True, "ok"

    verif = triple.get("_verification", {})
    verdict = verif.get("verdict", "MISSING")

    if verdict == "MISSING":
        # Only "strict" treats an unverified triple as untrustworthy by
        # default; the other policies assume it would likely have passed
        # (avoids silently losing triples just because stage 3 was
        # skipped for a given run, e.g. --verif-policy off upstream).
        if policy == "strict":
            return False, "verif_missing"
        return True, "ok_unverified"

    if policy == "strict":
        # Tier-1 gate: only an explicit STRONG_SUPPORT counts. This is
        # what makes Tier 1 "verified" in the tiered-fusion sense (06).
        return (verdict == "STRONG_SUPPORT"), (
            "ok"
            if verdict == "STRONG_SUPPORT"
            else f"verif_rejected_{verdict.lower()}"
        )

    if policy == "normal":
        if verdict in ("STRONG_SUPPORT", "WEAK_SUPPORT"):
            return True, "ok"
        # NO_CHUNK/UNPARSEABLE are verifier INFRASTRUCTURE failures (no
        # source chunk found, or the LLM response didn't match the
        # expected format) — not evidence against the triple itself, so
        # they are kept rather than penalized.
        if verdict in ("NO_CHUNK", "UNPARSEABLE"):
            return True, f"ok_{verdict.lower()}"
        return False, "verif_not_supported"

    if policy == "relaxed":
        # Maximally permissive: only an explicit NOT_SUPPORTED rejects.
        # Used for recall-oriented runs where false negatives are costlier
        # than false positives (e.g. corpus-gap diagnostics).
        if verdict == "NOT_SUPPORTED":
            return False, "verif_not_supported"
        return True, "ok"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION CHECKS
# ═══════════════════════════════════════════════════════════════════════


def normalize_entity(text: str) -> str:
    """Lowercase, collapse whitespace, and strip trailing punctuation from `text`; returns the normalized string, no side effects."""
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".,;:")
    return text


def check_basic(triple: dict) -> tuple[bool, str]:
    """Reject `triple` on empty/too-short/too-long fields, self-loops, blacklist patterns, or vague-term entities; returns (passes, reason_code), no side effects."""
    s = (triple.get("source_norm", triple.get("source", "")) or "").strip()
    t = (triple.get("target_norm", triple.get("target", "")) or "").strip()
    r = (triple.get("relation_norm", triple.get("relation", "")) or "").strip()

    if not s or not t or not r:
        return False, "empty_field"
    if len(s) < 3 or len(t) < 3:
        return False, "too_short"
    # Entity length filter: more than 5 words = extracted phrase, not entity
    if len(s.split()) > 5:
        return False, "entity_too_long_subject"
    if len(t.split()) > 5:
        return False, "entity_too_long_object"
    if s.lower() == t.lower():
        return False, "self_loop"

    for pattern in BLACKLIST_RE:
        if pattern.match(s):
            return False, "blacklisted_source"
        if pattern.match(t):
            return False, "blacklisted_target"

    s_vague = normalize_entity(s) in VAGUE_TERMS
    t_vague = normalize_entity(t) in VAGUE_TERMS
    if s_vague and t_vague:
        return False, "both_vague"
    if s_vague:
        return False, "vague_source"
    if t_vague:
        return False, "vague_target"

    return True, "ok"


def check_relation(triple: dict) -> tuple[bool, str]:
    """Reject `triple` if its relation is outside VALID_RELATIONS; returns (passes, reason_code), no side effects."""
    r = triple.get("relation_norm", triple.get("relation", ""))
    return (r in VALID_RELATIONS), (
        "ok" if r in VALID_RELATIONS else "invalid_relation"
    )


def check_type_constraint(triple: dict) -> tuple[bool, str]:
    """Enforce TYPE_CONSTRAINTS for hasDescriptor/occursIn objects, normalizing near-miss descriptor variants onto a KNOWN_DESCRIPTORS entry in place when possible; returns (passes, reason_code)."""
    r = triple.get("relation_norm", triple.get("relation", ""))
    t = normalize_entity(triple.get("target_norm", triple.get("target", "")))

    constraints = TYPE_CONSTRAINTS.get(r)
    if not constraints:
        return True, "ok"

    if r == "hasDescriptor":
        if t not in KNOWN_DESCRIPTORS:
            # P6: normalize descriptor variants instead of rejecting
            for d in KNOWN_DESCRIPTORS:
                if d in t or t in d:
                    triple["target"] = d
                    triple["target_norm"] = d
                    return True, "ok_descriptor_normalized"
            return False, "type_constraint"

    if r == "occursIn":
        if t not in KNOWN_SETTINGS:
            setting_kw = {
                "slope",
                "basin",
                "shelf",
                "margin",
                "canyon",
                "fan",
                "plain",
                "deep",
                "environment",
                "channel",
                "trench",
                "delta",
                "levee",
                "ridge",
            }
            if not any(kw in t for kw in setting_kw):
                return False, "type_constraint"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════
# FIX #2 — SOFT LEXICON
# ═══════════════════════════════════════════════════════════════════════


def check_lexicon_coverage_soft(
    triple: dict, lexicon: set
) -> tuple[bool, str]:
    """
    If both entities are outside lexicon:
      - keep if verified STRONG/WEAK (tag novel_term=True)
      - else reject both_not_in_lexicon
    """
    s = normalize_entity(triple.get("source_norm", triple.get("source", "")))
    t = normalize_entity(triple.get("target_norm", triple.get("target", "")))

    s_in = s in lexicon or any(term in s for term in lexicon if len(term) > 4)
    t_in = t in lexicon or any(term in t for term in lexicon if len(term) > 4)

    if not s_in and not t_in:
        verdict = (triple.get("_verification", {}) or {}).get("verdict", "")
        if verdict in ("STRONG_SUPPORT", "WEAK_SUPPORT"):
            triple["novel_term"] = True
            return True, "ok_novel_term_verified"
        return False, "both_not_in_lexicon"

    return True, "ok"


def triple_key(triple: dict) -> str:
    """Build the `subject||relation||object` dedup key for `triple` from its normalized entities; returns the key string, no side effects."""
    s = normalize_entity(triple.get("source_norm", triple.get("source", "")))
    r = triple.get("relation_norm", triple.get("relation", ""))
    t = normalize_entity(triple.get("target_norm", triple.get("target", "")))
    return f"{s}||{r}||{t}"


# ═══════════════════════════════════════════════════════════════════════
# CANONICALIZATION
# ═══════════════════════════════════════════════════════════════════════


def build_canonical_map(
    entities: list[str],
    distance_threshold: float = 0.06,
    lb_descriptors: set = LB_DESCRIPTORS,
    lexicon: set = None,
) -> dict[str, str]:
    """Cluster `entities` by SciBERT embedding similarity (agglomerative, cosine, threshold `distance_threshold`) and pick one canonical form per cluster, preferring lexicon membership then shorter phrasing; clusters containing 2+ LB2019 descriptors/settings are left unmerged. Prints progress/decisions; returns {variant: canonical} (empty if sentence-transformers/sklearn unavailable or <2 entities)."""
    if lexicon is None:
        lexicon = set()
    if len(entities) < 2:
        return {}

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        print(
            "  WARNING: sentence-transformers or sklearn not available, skipping canonicalization."
        )
        return {}

    print(f"  Computing SciBERT embeddings for {len(entities)} entities...")
    print("  Loading SciBERT for entity embeddings...")
    model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    embeddings = model.encode(entities, show_progress_bar=False)

    print(f"  Clustering (distance_threshold={distance_threshold})...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)
    for ent, label in zip(entities, labels):
        clusters[label].append(ent)

    canonical_map = {}
    blocked = []

    for _, members in clusters.items():
        if len(members) < 2:
            continue

        # SciBERT embeds domain jargon densely: distinct LB2019 descriptor
        # terms (e.g. "hummocky" and "chaotic") can land inside the same
        # 0.06-distance cluster because they co-occur in similar seismic-
        # facies contexts, not because they mean the same thing. Silently
        # merging two ground-truth descriptor categories would corrupt
        # the descriptor-coverage metric the paper reports on, so any
        # cluster containing 2+ LB descriptors is entirely skipped
        # (no merge applied to ANY member) rather than guessing a winner.
        lb_members = [m for m in members if m in lb_descriptors]
        if len(lb_members) >= 2:
            for m in members:
                blocked.append(
                    f"BLOCKED: '{m}' (cluster with {members}, both LB descriptors)"
                )
            continue
        # Same protection for LB2019 setting terms — a setting merged into
        # a co-clustered non-setting entity would silently break occursIn
        # recall against the benchmark edges. (P9 = project ticket ref.)
        lb_settings = {
            "continental slope",
            "abyssal plain",
            "basin floor",
            "hemipelagite",
            "passive margins",
            "continental shelf",
        }
        lb_setting_members = [m for m in members if m in lb_settings]
        if lb_setting_members:
            for m in members:
                blocked.append(
                    f"BLOCKED: '{m}' (cluster with {members}, contains LB2019 setting)"
                )
            continue

        # Prefer: in lexicon > shorter > alphabetical
        # (opposite of previous: long phrases were wrongly preferred)
        def canonical_score(x, _lex=lexicon):
            in_lex = 1 if x in _lex else 0
            short = -len(x.split())  # fewer words = better
            return (in_lex, short)

        canonical = max(members, key=canonical_score)
        for m in members:
            if m != canonical:
                canonical_map[m] = canonical

    if blocked:
        print(f"  Blocked {len(blocked)} bad merges:")
        for b in blocked[:10]:
            print(f"    {b}")

    if canonical_map:
        print("  Applied merges:")
        for old, new in sorted(canonical_map.items()):
            print(f"    '{old}' -> '{new}'")

    return canonical_map


def apply_canonical_map(triples: list[dict], canonical_map: dict) -> int:
    """Rewrite source/target fields of `triples` in place wherever their normalized form is a key in `canonical_map`; returns the number of field occurrences merged."""
    merged = 0
    for t in triples:
        for field in ["source_norm", "target_norm", "source", "target"]:
            val = t.get(field, "")
            norm = normalize_entity(val)
            if norm in canonical_map:
                t[field] = canonical_map[norm]
                merged += 1
    return merged


# ═══════════════════════════════════════════════════════════════════════
# LB RECALL & COVERAGE
# ═══════════════════════════════════════════════════════════════════════


def compute_lb_recall(triples: list[dict]) -> tuple[int, int, list]:
    """Count how many of the 26 LB_REFERENCE_EDGES appear (exact normalized-tuple match) in `triples`; returns (hits, total_reference_edges, missing_edges), no side effects."""
    found_keys = set()
    for t in triples:
        s = normalize_entity(t.get("source_norm", t.get("source", "")))
        r = t.get("relation_norm", t.get("relation", ""))
        o = normalize_entity(t.get("target_norm", t.get("target", "")))
        found_keys.add((s, r, o))

    hits = 0
    missing = []
    for s, r, o in LB_REFERENCE_EDGES:
        if (s, r, o) in found_keys:
            hits += 1
        else:
            missing.append((s, r, o))

    return hits, len(LB_REFERENCE_EDGES), missing


def compute_descriptor_coverage(triples: list[dict]) -> tuple[set, set]:
    """Find which of the 13 LB_DESCRIPTORS appear as hasDescriptor objects in `triples`; returns (found, missing) sets, no side effects."""
    found = set()
    for t in triples:
        r = t.get("relation_norm", t.get("relation", ""))
        if r == "hasDescriptor":
            o = normalize_entity(t.get("target_norm", t.get("target", "")))
            if o in LB_DESCRIPTORS:
                found.add(o)
    missing = LB_DESCRIPTORS - found
    return found, missing


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point: runs the verification filter, structural/ontology validation, SciBERT canonicalization and re-dedup over --input, writing canonical_triples_v5.jsonl, canonical_map_v5.json, rejected_triples_v5.jsonl and cleaning_stats_v5.json under --outdir."""
    parser = argparse.ArgumentParser(
        description="Validate, clean & canonicalize KG triples (v5)"
    )
    parser.add_argument(
        "--input", default=None, help="Input JSONL (default: $KG_INPUT)"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: $KG_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--verif-policy",
        default="normal",
        choices=["strict", "normal", "relaxed", "off"],
    )
    parser.add_argument(
        "--lexicon", default=None, help="Path to lexicon.json (optional)"
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.06,
        help="Cosine distance threshold for clustering",
    )
    args = parser.parse_args()

    input_path = args.input or os.environ.get("KG_INPUT", "")
    outdir = args.outdir or os.environ.get("KG_OUTPUT_DIR", "output/step4")
    if not input_path:
        print("ERROR: Provide --input or set $KG_INPUT")
        sys.exit(1)

    input_path = Path(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load triples
    print(f"Loading triples from {input_path}...")
    triples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))
    print(f"  Loaded {len(triples)} triples")

    # Lexicon
    lexicon = set()
    if args.lexicon and Path(args.lexicon).exists():
        with open(args.lexicon, "r", encoding="utf-8") as f:
            lex_data = json.load(f)
        if isinstance(lex_data, list):
            for entry in lex_data:
                lexicon.add(normalize_entity(entry.get("term", "")))
                for alias in entry.get("aliases", []):
                    lexicon.add(normalize_entity(alias))
        elif isinstance(lex_data, dict):
            for term in lex_data:
                lexicon.add(normalize_entity(term))
        print(f"  Loaded lexicon: {len(lexicon)} terms")
    else:
        lexicon = (
            {normalize_entity(d) for d in KNOWN_DESCRIPTORS}
            | {normalize_entity(s) for s in KNOWN_SETTINGS}
            | {
                "mass transport deposit",
                "mtd",
                "turbidite",
                "debris flow",
                "slide",
                "slump",
                "hemipelagite",
                "pelagite",
                "turbidity current",
                "slope failure",
                "submarine landslide",
                "earthquake",
                "pore pressure",
                "sedimentation",
                "erosion",
                "deposition",
                "seafloor",
                "sediment",
                "continental slope",
                "continental shelf",
            }
        )
        print(f"  Using built-in lexicon: {len(lexicon)} terms")

    # STEP 1: verification filter
    print(f"\n  Verification policy: {args.verif_policy}")
    passed_verif = []
    rejected = []
    verif_reasons = Counter()

    for t in triples:
        passes, reason = check_verification(t, args.verif_policy)
        if passes:
            passed_verif.append(t)
        else:
            t["_reject_reason"] = reason
            rejected.append(t)
            verif_reasons[reason] += 1

    print(
        f"  After verification filter: {len(passed_verif)} kept, {len(rejected)} rejected"
    )
    if verif_reasons:
        for reason, count in verif_reasons.most_common():
            print(f"    {reason:35s}: {count}")

    # STEP 2: validation
    cleaned = []
    validation_reasons = Counter()
    n_dupes = 0
    seen_keys = set()

    for t in passed_verif:
        apply_relation_mapping(t)

        ok, reason = check_basic(t)
        if not ok:
            t["_reject_reason"] = reason
            rejected.append(t)
            validation_reasons[reason] += 1
            continue

        ok, reason = check_relation(t)
        if not ok:
            t["_reject_reason"] = reason
            rejected.append(t)
            validation_reasons[reason] += 1
            continue

        ok, reason = check_type_constraint(t)
        if not ok:
            t["_reject_reason"] = reason
            rejected.append(t)
            validation_reasons[reason] += 1
            continue

        # FIX #2: soft lexicon here
        ok, reason = check_lexicon_coverage_soft(t, lexicon)
        if not ok:
            t["_reject_reason"] = reason
            rejected.append(t)
            validation_reasons[reason] += 1
            continue

        key = triple_key(t)
        if key in seen_keys:
            n_dupes += 1
            continue
        seen_keys.add(key)
        cleaned.append(t)

    print(f"\n{'=' * 60}")
    print("  VALIDATION REPORT")
    print(f"{'=' * 60}")
    print(
        f"  After verif: {len(passed_verif)} | Cleaned: {len(cleaned)} | Dupes: {n_dupes}"
    )
    if validation_reasons:
        for reason, count in validation_reasons.most_common():
            print(f"    {reason:35s}: {count}")

    rel_counts = Counter()
    for t in cleaned:
        rel_counts[t.get("relation_norm", t.get("relation", ""))] += 1
    print("  Relations:")
    for rel, count in rel_counts.most_common():
        print(f"    {rel:20s}: {count}")

    hits, total, _missing = compute_lb_recall(cleaned)
    if total > 0:
        print(f"  LB Recall: {hits}/{total} = {hits / total:.1%}")

    found_desc, missing_desc = compute_descriptor_coverage(cleaned)
    print(
        f"  Desc Coverage: {len(found_desc)}/{len(LB_DESCRIPTORS)} found={sorted(found_desc)}"
    )
    print(f"  Missing: {sorted(missing_desc)}")

    # STEP 3: canonicalization
    entities_before = set()
    for t in cleaned:
        s = normalize_entity(t.get("source_norm", t.get("source", "")))
        o = normalize_entity(t.get("target_norm", t.get("target", "")))
        entities_before.add(s)
        entities_before.add(o)

    entity_list = sorted(entities_before)
    print(f"\n  Unique entities: {len(entity_list)}")

    canonical_map = build_canonical_map(
        entity_list,
        distance_threshold=args.cluster_threshold,
        lb_descriptors=LB_DESCRIPTORS,
        lexicon=lexicon,
    )
    print(f"  Canonical map: {len(canonical_map)} merge rules")

    print(f"  Applying canonical map to {len(cleaned)} triples...")
    n_merged = apply_canonical_map(cleaned, canonical_map)
    print(f"  Entity occurrences merged: {n_merged}")

    print("  Re-deduplicating after canonicalization...")
    final = []
    seen_keys2 = set()
    n_dupes2 = 0
    n_self_loops = 0

    for t in cleaned:
        s = normalize_entity(t.get("source_norm", t.get("source", "")))
        o = normalize_entity(t.get("target_norm", t.get("target", "")))
        if s == o:
            n_self_loops += 1
            continue

        key = triple_key(t)
        if key in seen_keys2:
            n_dupes2 += 1
            continue
        seen_keys2.add(key)
        final.append(t)

    entities_after = set()
    for t in final:
        s = normalize_entity(t.get("source_norm", t.get("source", "")))
        o = normalize_entity(t.get("target_norm", t.get("target", "")))
        entities_after.add(s)
        entities_after.add(o)

    # WRITE OUTPUTS
    out_triples = outdir / "canonical_triples_v5.jsonl"
    with open(out_triples, "w", encoding="utf-8") as f:
        for t in final:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    out_map = outdir / "canonical_map_v5.json"
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump(canonical_map, f, indent=2, ensure_ascii=False)

    out_rejected = outdir / "rejected_triples_v5.jsonl"
    with open(out_rejected, "w", encoding="utf-8") as f:
        for t in rejected:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    verif_decided = 0
    verif_supported = 0
    verif_strong = 0
    for t in final:
        v = (t.get("_verification", {}) or {}).get("verdict", "")
        if v in ("STRONG_SUPPORT", "WEAK_SUPPORT", "NOT_SUPPORTED"):
            verif_decided += 1
            if v in ("STRONG_SUPPORT", "WEAK_SUPPORT"):
                verif_supported += 1
            if v == "STRONG_SUPPORT":
                verif_strong += 1

    final_halluc_rate = (
        (1 - (verif_supported / verif_decided)) if verif_decided > 0 else 0.0
    )

    stats = {
        "input_triples": len(triples),
        "after_verif_filter": len(passed_verif),
        "verif_policy": args.verif_policy,
        "verif_rejected": dict(verif_reasons),
        "after_validation": len(cleaned),
        "validation_rejected": dict(validation_reasons),
        "duplicates_pass1": n_dupes,
        "duplicates_pass2": n_dupes2,
        "self_loops_removed": n_self_loops,
        "entities_before": len(entities_before),
        "entities_after": len(entities_after),
        "merge_rules": len(canonical_map),
        "output_triples": len(final),
        "lb_recall": f"{hits}/{total}",
        "lb_recall_pct": round(hits / total, 4) if total > 0 else 0,
        "desc_coverage": f"{len(found_desc)}/{len(LB_DESCRIPTORS)}",
        "desc_found": sorted(found_desc),
        "desc_missing": sorted(missing_desc),
        "final_halluc_rate": round(final_halluc_rate, 4),
        "final_strong_support": verif_strong,
        "final_supported": verif_supported,
        "final_decided": verif_decided,
    }

    out_stats = outdir / "cleaning_stats_v5.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("  CANONICALIZATION SUMMARY (v5)")
    print(f"{'=' * 60}")
    print(f"  Output triples:      {len(final)}")
    if total > 0:
        print(f"  LB Recall:           {hits}/{total} = {hits / total:.1%}")
    print(f"  Desc Coverage:       {len(found_desc)}/{len(LB_DESCRIPTORS)}")
    print(
        f"  Final halluc rate:   {final_halluc_rate:.1%} (S={verif_strong} W={verif_supported - verif_strong} decided={verif_decided})"
    )
    print(f"{'=' * 60}")
    print(f"  Triples:   {out_triples}")
    print(f"  Rejected:  {out_rejected}")
    print(f"  Stats:     {out_stats}")


if __name__ == "__main__":
    main()
