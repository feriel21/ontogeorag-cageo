#!/usr/bin/env python3
"""
analysis_suite/24_kg_to_flat.py — Emit a 07-compatible flat KG

Why this exists
---------------
pipeline/07_final_metrics.py reads its input with:

    triples = data.get("triples", data) if isinstance(data, dict) else data

The tiered/enforced KGs are shaped {meta, tier1, tier2, quarantine} — they have
no "triples" key — so 07 silently falls back to iterating the dict's KEYS and
then crashes on 'str' object has no attribute 'get'. Consequence: 07 has only
ever run on the pre-panel KGs that DO carry a "triples" key (157 records).

The published recall figures were therefore computed on the pre-M4 KG, not on
the final post-panel KG. This script does not change 07 and does not change any
KG. It writes a NEW flat file with a "triples" key so 07 can be re-run on the
final KG and the two recall numbers compared.

Selection is explicit:
  --include tier1,tier2            (default: the 151-triple final KG)
  --include tier1,tier2,quarantine (the full 157 records, for comparison)
  --include tier1                  (Tier-1 only)

Deduplication on (subject, relation, object) is OFF by default so that this
script reproduces exactly what is on disk; pass --dedup to also collapse the
known duplicate. Run it both ways to separate the two effects.

Usage:
    python analysis_suite/24_kg_to_flat.py \\
        --kg      output/run13/kg/tiered_kg_run13_enforced.json \\
        --include tier1,tier2 \\
        --out     output/run13/kg/flat_enforced_151.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.rag.constants import normalize_entity, normalize_relation


def key_of(t: dict) -> tuple:
    return (normalize_entity(t.get("subject", t.get("source", ""))),
            normalize_relation(t.get("relation", "")),
            normalize_entity(t.get("object", t.get("target", ""))))


def main():
    ap = argparse.ArgumentParser(description="Flatten a tiered KG for 07_final_metrics.py")
    ap.add_argument("--kg", required=True)
    ap.add_argument("--include", default="tier1,tier2",
                    help="Comma-separated list of top-level lists to include")
    ap.add_argument("--dedup", action="store_true",
                    help="Collapse duplicate (subject, relation, object) records")
    ap.add_argument("--out", required=True, help="New output path (refuses to overwrite)")
    args = ap.parse_args()

    kg_path = Path(args.kg)
    out_path = Path(args.out)
    if out_path.exists():
        print(f"REFUSING to overwrite existing file: {out_path}")
        return 1

    data = json.loads(kg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print(f"Expected a dict at top level in {kg_path}")
        return 1

    wanted = [s.strip() for s in args.include.split(",") if s.strip()]
    missing = [w for w in wanted if w not in data]
    if missing:
        print(f"Requested lists not present: {missing}. "
              f"Available keys: {sorted(data.keys())}")
        return 1

    records = []
    print(f"Source: {kg_path}")
    for name in wanted:
        lst = data[name]
        if not isinstance(lst, list):
            print(f"  {name}: not a list, skipping")
            continue
        print(f"  {name}: {len(lst)} records -> included")
        records.extend(lst)
    print(f"  included total: {len(records)}")

    n_before = len(records)
    if args.dedup:
        seen, kept = set(), []
        for t in records:
            k = key_of(t)
            if k in seen:
                continue
            seen.add(k)
            kept.append(t)
        records = kept
        print(f"  after dedup: {len(records)} (removed {n_before - len(records)})")
    else:
        distinct = len({key_of(t) for t in records})
        if distinct != n_before:
            print(f"  NOTE: {n_before} records but only {distinct} distinct "
                  f"(subject, relation, object) keys. Not deduplicated "
                  f"(pass --dedup to collapse). Run 07 both ways to separate "
                  f"the duplicate effect from the tier-selection effect.")

    print(f"  relations: {dict(Counter(t.get('relation') for t in records).most_common())}")
    print(f"  tiers: {dict(sorted(Counter(t.get('tier') for t in records).items(), key=lambda x: (x[0] is None, x[0])))}")

    out = {
        "meta": {
            **(data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}),
            "flattened_from": str(kg_path),
            "included_lists": wanted,
            "deduplicated": bool(args.dedup),
            "record_count": len(records),
            "note": ("Flat view emitted for 07_final_metrics.py, which requires "
                     "a 'triples' key. Not a replacement for the tiered KG."),
        },
        "triples": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    print("Now run 07 on this file and compare against the published metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())