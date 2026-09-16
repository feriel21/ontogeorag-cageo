#!/usr/bin/env python3
"""
04b_schema_enforce.py — Entity-type schema enforcement (post-M4).
=================================================================
WHY
    The extraction prompt declares 5 entity types, but the LLM is only
    *instructed*, not constrained at decoding time: run13's panel KG carries
    25 distinct type strings (Process, SeismicObject, GeologicalObject,
    Condition, GeologicalSetting, Surface, Attribute, Feature,
    StructuralDomain, Conditions, GeologicalFeature, GeologicalSurface,
    Location, Event, Property, Facies, GeologicalProcess, StructuralSurface,
    Substance, Domain, Measurement, SurfaceFeature, Structure, Criterion...).
    Any statistic broken down by entity type is meaningless until these are
    folded back onto the declared schema, and the claim "ontology-constrained
    extraction" is weakened while they persist.

WHAT (default behaviour — conservative)
    * maps every observed type onto the 5 schema types via TYPE_MAP
    * removes self-loops (subject == object after normalization)
    * re-deduplicates on (subject, relation, object)
    * FLAGS (does not delete) entities longer than --max-words and triples
      whose (subject_type, relation, object_type) signature is not allowed
    Deletion of flagged items requires --reject-long / --reject-bad-sig:
    schema enforcement must not silently drop content from a frozen run.

    The type mapping is a small ontological decision per source type, so
    every mapping is written to a change log with examples, and can be
    overridden with --type-map FILE (JSON: {"SourceType": "SchemaType"}).
    Unknown types are mapped to --default-type and listed prominently.

OUTPUTS (in --outdir)
    tiered_kg_enforced.json   same structure as the input KG
    enforced_triples.jsonl    flat list of kept triples
    rejected_triples.jsonl    removed triples with a `_reject_reason`
    type_change_log.csv       from -> to, count, up to 3 example entities
    enforcement_report.json   counts before/after, flags, unknown types

USAGE
    python analysis_suite/04b_schema_enforce.py \
        --kg output/run13/m4_panel/tiered_kg_m4.json \
        --outdir output/run13/kg --dry-run
    python analysis_suite/04b_schema_enforce.py \
        --kg output/run13/m4_panel/tiered_kg_m4.json \
        --outdir output/run13/kg
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
REPO_ROOT = Path(__file__).resolve().parent.parent
from kg_io import dump_kg, get_object, get_relation, get_subject, load_kg

SCHEMA_TYPES = ["Geological_Object", "Descriptor", "Process",
                "Environmental_Control", "Evidence"]

# Each line is an ontological decision; grouped by target type so a
# geologist can review the whole mapping at a glance.
TYPE_MAP = {
    # ── physical bodies, surfaces and features seen in seismic data ──
    "geological_object": "Geological_Object",
    "seismicobject": "Geological_Object",
    "geologicalobject": "Geological_Object",
    "object": "Geological_Object",
    "feature": "Geological_Object",
    "geologicalfeature": "Geological_Object",
    "surfacefeature": "Geological_Object",
    "surface": "Geological_Object",
    "geologicalsurface": "Geological_Object",
    "structuralsurface": "Geological_Object",
    "structure": "Geological_Object",
    "facies": "Geological_Object",
    "substance": "Geological_Object",
    "material": "Geological_Object",
    "deposit": "Geological_Object",
    # ── seismic/physical descriptors and measured properties ────────
    "descriptor": "Descriptor",
    "attribute": "Descriptor",
    "property": "Descriptor",
    "measurement": "Descriptor",
    "characteristic": "Descriptor",
    # ── dynamic processes and events ────────────────────────────────
    "process": "Process",
    "geologicalprocess": "Process",
    "event": "Process",
    "mechanism": "Process",
    "action": "Process",
    # ── settings, conditions and controlling factors ────────────────
    "environmental_control": "Environmental_Control",
    "condition": "Environmental_Control",
    "conditions": "Environmental_Control",
    "geologicalsetting": "Environmental_Control",
    "setting": "Environmental_Control",
    "location": "Environmental_Control",
    "domain": "Environmental_Control",
    "structuraldomain": "Environmental_Control",
    "criterion": "Environmental_Control",
    "factor": "Environmental_Control",
    "control": "Environmental_Control",
    # ── observational support ───────────────────────────────────────
    "evidence": "Evidence",
    "observation": "Evidence",
    "indicator": "Evidence",
}

# Allowed (subject_type, object_type) pairs per relation. Kept permissive on
# purpose: the semi-open relations accept any pair, only hasDescriptor and
# the strictly stratigraphic relations are constrained.
RELATION_SIGNATURES = {
    "hasdescriptor": {("Geological_Object", "Descriptor"),
                      ("Process", "Descriptor")},
    "overlies": {("Geological_Object", "Geological_Object")},
    "underlies": {("Geological_Object", "Geological_Object")},
    "partof": {("Geological_Object", "Geological_Object")},
}


def norm_entity(e):
    return re.sub(r"\s+", " ", str(e or "").strip().lower())


def norm_type(t):
    return re.sub(r"[^a-z_]", "", str(t or "").strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--type-map", default=None,
                    help="JSON overriding/extending the built-in TYPE_MAP")
    ap.add_argument("--default-type", default="Geological_Object")
    ap.add_argument("--descriptor-fallback", default="Environmental_Control",
                    help="type given to entities the LLM called Descriptor "
                         "but which are not in the canonical descriptor "
                         "lexicon (physical parameters, measurements)")
    ap.add_argument("--no-descriptor-rule", action="store_true",
                    help="disable the lexicon-based Descriptor rule")
    ap.add_argument("--max-words", type=int, default=6)
    ap.add_argument("--reject-long", action="store_true",
                    help="actually remove triples with over-long entities "
                         "(default: flag only)")
    ap.add_argument("--reject-bad-sig", action="store_true",
                    help="actually remove triples whose type signature is "
                         "not allowed (default: flag only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args()

    # An entity is typed Descriptor if and only if it belongs to the
    # canonical seismic-descriptor lexicon. Without this rule, TYPE_MAP
    # sends Property / Measurement / Attribute to Descriptor, and physical
    # parameters ("sea floor slope", "cohesion of the basal detachment")
    # end up typed as seismic facies adjectives — verified on run13, 6 of
    # 25 Descriptor nodes were of this kind. The rule makes the type
    # checkable against a list instead of trusting the LLM's guess.
    lexicon = set()
    if not args.no_descriptor_rule:
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from pipeline.rag.constants import KNOWN_DESCRIPTORS
            lexicon = {str(x).strip().lower() for x in KNOWN_DESCRIPTORS}
            print(f"[descriptor rule] canonical lexicon: {len(lexicon)} "
                  "terms")
        except Exception as e:
            print(f"[WARN] descriptor rule disabled ({e}); run from the "
                  "repo root to enable it")

    tmap = dict(TYPE_MAP)
    if args.type_map:
        extra = json.load(open(args.type_map, encoding="utf-8"))
        tmap.update({norm_type(k): v for k, v in extra.items()})

    kg = load_kg(args.kg)
    triples = kg["triples"]
    before_types = Counter()
    for t in triples:
        for f in ("subject_type", "object_type", "source_type",
                  "target_type"):
            if t.get(f):
                before_types[t[f]] += 1

    changes = defaultdict(lambda: {"n": 0, "examples": []})
    retyped = []
    unknown = Counter()
    kept, rejected = [], []
    seen = set()
    n_selfloop = n_dup = 0
    flag_long, flag_sig = [], []

    for t in triples:
        s, r, o = get_subject(t), get_relation(t), get_object(t)

        # 1. self-loops
        if norm_entity(s) == norm_entity(o) and norm_entity(s):
            t["_reject_reason"] = "self_loop"
            rejected.append(t)
            n_selfloop += 1
            continue

        # 2. type mapping (both naming conventions)
        for src_f, dst_f in (("subject_type", "subject_type"),
                             ("source_type", "source_type"),
                             ("object_type", "object_type"),
                             ("target_type", "target_type")):
            raw = t.get(src_f)
            if not raw:
                continue
            mapped = tmap.get(norm_type(raw))
            if mapped is None:
                unknown[raw] += 1
                mapped = args.default_type
            # lexicon-based correction of the Descriptor type
            if lexicon and mapped == "Descriptor":
                ent = s if ("subject" in src_f or "source" in src_f) else o
                if norm_entity(ent) not in lexicon:
                    mapped = args.descriptor_fallback
                    retyped.append((ent, raw))
            if mapped != raw:
                rec = changes[(raw, mapped)]
                rec["n"] += 1
                ent = s if "subject" in src_f or "source" in src_f else o
                if len(rec["examples"]) < 3 and ent not in rec["examples"]:
                    rec["examples"].append(ent)
            t[dst_f] = mapped

        # 3. entity length flag
        long_ent = [e for e in (s, o)
                    if len(str(e).split()) > args.max_words]
        if long_ent:
            t["_flag_long_entity"] = "; ".join(long_ent)
            flag_long.append((s, r, o))
            if args.reject_long:
                t["_reject_reason"] = "entity_too_long"
                rejected.append(t)
                continue

        # 4. relation signature flag
        sig_rules = RELATION_SIGNATURES.get(
            norm_type(r).replace("_", ""))
        if sig_rules:
            st = t.get("subject_type") or t.get("source_type")
            ot = t.get("object_type") or t.get("target_type")
            if st and ot and (st, ot) not in sig_rules:
                t["_flag_bad_signature"] = f"{st} -[{r}]-> {ot}"
                flag_sig.append((s, r, o, st, ot))
                if args.reject_bad_sig:
                    t["_reject_reason"] = "bad_relation_signature"
                    rejected.append(t)
                    continue

        # 5. re-deduplicate
        k = (norm_entity(s), r.strip().lower(), norm_entity(o))
        if k in seen:
            t["_reject_reason"] = "duplicate_after_enforcement"
            rejected.append(t)
            n_dup += 1
            continue
        seen.add(k)
        kept.append(t)

    after_types = Counter()
    for t in kept:
        for f in ("subject_type", "object_type", "source_type",
                  "target_type"):
            if t.get(f):
                after_types[t[f]] += 1

    print("=" * 64)
    print("SCHEMA ENFORCEMENT" + ("  [DRY RUN]" if args.dry_run else ""))
    print("=" * 64)
    print(f"triples in            : {len(triples)}")
    print(f"types before          : {len(before_types)}")
    print(f"types after           : {len(after_types)}  "
          f"{dict(after_types)}")
    print(f"self-loops removed    : {n_selfloop}")
    print(f"duplicates removed    : {n_dup}")
    print(f"over-long entities    : {len(flag_long)} "
          + ("(REMOVED)" if args.reject_long else "(flagged only)"))
    print(f"bad type signatures   : {len(flag_sig)} "
          + ("(REMOVED)" if args.reject_bad_sig else "(flagged only)"))
    if unknown:
        print(f"UNKNOWN types mapped to {args.default_type}: "
              f"{dict(unknown)}")
        print("  -> review these: add them to TYPE_MAP or pass --type-map")
    if retyped:
        uniq = sorted({e for e, _ in retyped})
        print(f"descriptor rule       : {len(uniq)} entities called "
              f"Descriptor by the model but absent from the lexicon → "
              f"retyped {args.descriptor_fallback}")
        for e in uniq[:10]:
            print(f"   {e}")
    print(f"triples out           : {len(kept)}")
    if flag_sig:
        print("\nsignature violations (first 5):")
        for s, r, o, st, ot in flag_sig[:5]:
            print(f"  {s} -[{r}]-> {o}   ({st} -> {ot})")

    if args.dry_run:
        print("\n[dry run] nothing written.")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kg["triples"] = kept
    dump_kg(kg, outdir / "tiered_kg_enforced.json")
    with open(outdir / "enforced_triples.jsonl", "w",
              encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps({k_: v for k_, v in t.items()
                                if not k_.startswith("_")},
                               ensure_ascii=False) + "\n")
    with open(outdir / "rejected_triples.jsonl", "w",
              encoding="utf-8") as f:
        for t in rejected:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    with open(outdir / "type_change_log.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from_type", "to_type", "n_occurrences", "examples"])
        for (a, b), rec in sorted(changes.items(), key=lambda x: -x[1]["n"]):
            w.writerow([a, b, rec["n"], "; ".join(rec["examples"])])
    report = {
        "input_kg": args.kg, "n_in": len(triples), "n_out": len(kept),
        "types_before": dict(before_types), "types_after": dict(after_types),
        "n_self_loops": n_selfloop, "n_duplicates": n_dup,
        "n_flag_long": len(flag_long), "n_flag_bad_signature": len(flag_sig),
        "removed_long": args.reject_long,
        "removed_bad_signature": args.reject_bad_sig,
        "unknown_types": dict(unknown),
        "descriptor_rule_retyped": sorted({e for e, _ in retyped}),
        "descriptor_fallback_type": args.descriptor_fallback,
        "default_type": args.default_type, "max_words": args.max_words,
        "policy": "Types are mapped; self-loops and duplicates are removed. "
                  "Over-long entities and invalid relation signatures are "
                  "flagged in place unless explicitly rejected, so that "
                  "schema enforcement never silently drops content.",
    }
    with open(outdir / "enforcement_report.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwritten: {outdir}/tiered_kg_enforced.json (+ logs)")


if __name__ == "__main__":
    main()