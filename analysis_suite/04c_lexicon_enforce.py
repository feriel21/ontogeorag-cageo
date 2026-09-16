#!/usr/bin/env python3
"""
04c_lexicon_enforce.py — Closed-world enforcement of hasDescriptor targets.
===========================================================================
WHY (R5)
    run11 contains 14 observed descriptors including thin / wedge-shaped /
    erosional, while `massive` (a benchmark edge!) is absent. Hypothesis:
    triples whose relation was REMAPPED to hasDescriptor by RELATION_MAP
    (e.g. "hasthickness", "serves as", "is characterized by") bypassed the
    lexical validation, which ran on the original relation string. This
    script closes the leak WITHOUT modifying 04_clean_validate.py: it is a
    standalone post-filter to insert after cleaning (per pass) or after
    fusion.

WHAT
    --report  : diagnosis only — which hasDescriptor objects are outside
                KNOWN_DESCRIPTORS, which canonical descriptors are absent
                from the KG, and (if the field exists) the pre-mapping
                relation of each offending triple.
    --enforce : additionally writes <input>.lexicon_enforced.jsonl with
                offending triples removed, and <input>.lexicon_rejected.jsonl
                with full records for the audit trail.

    The canonical list is imported from pipeline.rag.constants
    (KNOWN_DESCRIPTORS) so the script can never diverge from the pipeline's
    own vocabulary; --lexicon FILE overrides for testing.

USAGE (from repo root, per pass, after 04/05)
    python 04c_lexicon_enforce.py --input output/run13_a/canonical_triples.jsonl --report
    python 04c_lexicon_enforce.py --input output/run13_a/canonical_triples.jsonl --enforce
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_lexicon(path_or_none):
    if path_or_none:
        data = json.load(open(path_or_none, encoding="utf-8"))
        if isinstance(data, dict):
            data = list(data)
        return {str(x).strip().lower() for x in data}
    try:
        sys.path.insert(0, ".")
        from pipeline.rag.constants import KNOWN_DESCRIPTORS

        return {str(x).strip().lower() for x in KNOWN_DESCRIPTORS}
    except Exception as e:
        raise SystemExit(
            "Could not import KNOWN_DESCRIPTORS from pipeline.rag.constants "
            f"({e}). Run from the repo root, or pass --lexicon FILE."
        )


def norm(x):
    return str(x or "").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="triples .jsonl (one JSON object per line)",
    )
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--enforce", action="store_true")
    args = ap.parse_args()
    if not (args.report or args.enforce):
        args.report = True

    lex = load_lexicon(args.lexicon)
    inp = Path(args.input)
    triples = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]

    kept, rejected = [], []
    offending_terms = Counter()
    origin_relations = Counter()
    seen_descriptors = set()

    for t in triples:
        rel = norm(t.get("relation"))
        if rel != "hasdescriptor":
            kept.append(t)
            continue
        obj = norm(t.get("object") or t.get("target"))
        seen_descriptors.add(obj)
        if obj in lex:
            kept.append(t)
        else:
            rejected.append(t)
            offending_terms[obj] += 1
            for k in (
                "raw_relation",
                "original_relation",
                "relation_raw",
                "llm_relation",
            ):
                if t.get(k):
                    origin_relations[norm(t[k])] += 1
                    break
            else:
                origin_relations["<pre-mapping relation not stored>"] += 1

    absent = sorted(lex - seen_descriptors)

    print("=" * 60)
    print("CLOSED-WORLD DESCRIPTOR ENFORCEMENT")
    print("=" * 60)
    print(f"input triples              : {len(triples)}")
    print(
        f"hasDescriptor triples      : "
        f"{sum(1 for t in triples if norm(t.get('relation')) == 'hasdescriptor')}"
    )
    print(f"canonical lexicon size     : {len(lex)}")
    print(f"lexicon: {sorted(lex)}")
    print(
        f"OFFENDING (outside lexicon): {len(rejected)} triples, "
        f"{len(offending_terms)} terms"
    )
    for term, c in offending_terms.most_common():
        print(f"   {term!r}: {c}")
    print("pre-mapping relations of offenders:")
    for r, c in origin_relations.most_common():
        print(f"   {r!r}: {c}")
    print(f"canonical descriptors ABSENT from KG: {absent}")
    if "massive" in absent:
        print(
            "   NOTE: 'massive' is a benchmark edge target — cross-check "
            "against the failure-mode analysis / rejected triples of 03."
        )

    if args.enforce:
        out_ok = inp.with_suffix(".lexicon_enforced.jsonl")
        out_rej = inp.with_suffix(".lexicon_rejected.jsonl")
        with open(out_ok, "w", encoding="utf-8") as f:
            for t in kept:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with open(out_rej, "w", encoding="utf-8") as f:
            for t in rejected:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"written: {out_ok}  ({len(kept)} triples)")
        print(f"written: {out_rej} ({len(rejected)} triples)")


if __name__ == "__main__":
    main()
