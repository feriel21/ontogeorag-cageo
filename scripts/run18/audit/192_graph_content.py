#!/usr/bin/env python3
"""
192 — Content of a graph: what it states about one target, by default mass transport deposits (read-only).

Counts the relations by type, lists the descriptors attached to the target, its causes and
triggers, and its context (occursIn, overlies, underlies, partOf). The target is found by
substrings of the entity names (--terms). Modifies no existing file; writes only into a new
output folder.

Usage (from the run18 worktree)
    python 192_graph_content.py
    python 192_graph_content.py --graph <other_graph.json> --out output/audit/192_content_cleaned
    python 192_graph_content.py --terms "slope failure" --out output/audit/192_content_slope_failure
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_GRAPH = "output/run18/kg_final/flat_run18_enforced_dedup.json"
DEFAULT_OUT = "output/audit/192_content"

SUBJ_KEYS = ["subject", "head", "source", "s", "subj"]
PRED_KEYS = ["predicate", "relation", "rel", "p", "pred", "relation_type"]
OBJ_KEYS = ["object", "tail", "target", "o", "obj"]
TIER_KEYS = ["tier", "tier_label", "tier_before_panel"]
LIST_KEYS = ["relations", "edges", "triples", "triplets", "kg", "data", "items"]

MTD_TERMS = ["mtd", "mass-transport", "mass transport", "masstransport"]


def load_relations(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in LIST_KEYS:
            if k in data and isinstance(data[k], list):
                print(f"[info] relations read under the key '{k}'")
                return data[k]
    sys.exit(f"[error] structure not recognised in {path}: keys = {list(data)[:20]}")


def pick(rec, keys):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return k
    return None


def norm(x):
    return str(x).strip()


def is_mtd(term):
    t = term.lower()
    return any(m in t for m in MTD_TERMS)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--terms", default=",".join(MTD_TERMS),
                    help="substrings that find the target (default: MTD), e.g. --terms 'slope failure'")
    args = ap.parse_args()
    MTD_TERMS[:] = [t.strip().lower() for t in args.terms.split(",") if t.strip()]

    out = Path(args.out)
    if out.exists():
        sys.exit(f"[stop] {out} already exists: nothing is overwritten. Choose another --out.")

    rels = load_relations(args.graph)
    if not rels or not isinstance(rels[0], dict):
        sys.exit("[error] the relations are not dictionaries.")

    first = rels[0]
    ks, kp, ko, kt = pick(first, SUBJ_KEYS), pick(first, PRED_KEYS), pick(first, OBJ_KEYS), pick(first, TIER_KEYS)
    print(f"[info] {len(rels)} relations; fields found: subject={ks}, relation={kp}, object={ko}, tier={kt}")
    print(f"[info] keys of the first relation: {sorted(first.keys())}")
    if not (ks and kp and ko):
        sys.exit("[stop] subject, relation or object field not found: see the keys printed above.")

    out.mkdir(parents=True)

    by_pred = Counter()
    by_pred_tier = defaultdict(Counter)
    mtd_desc = defaultdict(lambda: {"n": 0, "tiers": Counter()})
    causes = defaultdict(lambda: {"n": 0, "tiers": Counter(), "preds": Counter()})
    context = defaultdict(lambda: {"n": 0, "tiers": Counter()})
    mtd_any = 0

    CAUSAL = {"causes", "triggers", "controls", "affects"}
    CONTEXT = {"occursin", "overlies", "underlies", "partof"}

    for r in rels:
        s, p, o = norm(r.get(ks, "")), norm(r.get(kp, "")), norm(r.get(ko, ""))
        tier = norm(r.get(kt, "?")) if kt else "?"
        pl = p.lower()
        by_pred[p] += 1
        by_pred_tier[p][tier] += 1
        s_mtd, o_mtd = is_mtd(s), is_mtd(o)
        if s_mtd or o_mtd:
            mtd_any += 1
        if pl == "hasdescriptor" and s_mtd:
            mtd_desc[o]["n"] += 1
            mtd_desc[o]["tiers"][tier] += 1
        if pl in CAUSAL and o_mtd:
            causes[s]["n"] += 1
            causes[s]["tiers"][tier] += 1
            causes[s]["preds"][p] += 1
        if pl in CONTEXT and s_mtd:
            key = f"{p} -> {o}"
            context[key]["n"] += 1
            context[key]["tiers"][tier] += 1

    tiers_all = sorted({t for c in by_pred_tier.values() for t in c})

    # 1. relations by type
    rows = [[p, n] + [by_pred_tier[p][t] for t in tiers_all] for p, n in by_pred.most_common()]
    write_csv(out / "1_relations_by_type.csv", ["relation", "total"] + [f"tier_{t}" for t in tiers_all], rows)

    # 2. descriptors of the target (last column left empty, to fill in: visible on seismic data?)
    rows = [[d, v["n"], "; ".join(f"{t}:{c}" for t, c in sorted(v["tiers"].items())), ""]
            for d, v in sorted(mtd_desc.items(), key=lambda x: -x[1]["n"])]
    write_csv(out / "2_target_descriptors.csv", ["descriptor", "n_relations", "tiers", "visible_on_seismic (to fill in)"], rows)

    # 3. causes and triggers of the target
    rows = [[c, v["n"], "; ".join(f"{p}:{n}" for p, n in v["preds"].most_common()),
             "; ".join(f"{t}:{n}" for t, n in sorted(v["tiers"].items()))]
            for c, v in sorted(causes.items(), key=lambda x: -x[1]["n"])]
    write_csv(out / "3_target_causes_and_triggers.csv", ["cause", "n_relations", "relations", "tiers"], rows)

    # 4. context of the target
    rows = [[k, v["n"], "; ".join(f"{t}:{n}" for t, n in sorted(v["tiers"].items()))]
            for k, v in sorted(context.items(), key=lambda x: -x[1]["n"])]
    write_csv(out / "4_target_context.csv", ["relation -> object", "n_relations", "tiers"], rows)

    # summary
    n_desc_rel = sum(v["n"] for v in mtd_desc.values())
    n_cause_rel = sum(v["n"] for v in causes.values())
    n_ctx_rel = sum(v["n"] for v in context.values())
    lines = [
        "# 192 - Graph content",
        f"Graph: {args.graph}",
        f"Relations: {len(rels)}; relations that involve the target (subject or object): {mtd_any}",
        "",
        "## Relations by type",
        *[f"- {p} : {n}" for p, n in by_pred.most_common()],
        "",
        f"## Descriptors of the target: {len(mtd_desc)} distinct, {n_desc_rel} relations",
        *[f"- {d} ({v['n']})" for d, v in sorted(mtd_desc.items(), key=lambda x: -x[1]['n'])],
        "",
        f"## Causes and triggers of the target: {len(causes)} distinct, {n_cause_rel} relations",
        *[f"- {c} ({v['n']})" for c, v in sorted(causes.items(), key=lambda x: -x[1]['n'])],
        "",
        f"## Context of the target: {len(context)} distinct, {n_ctx_rel} relations",
        *[f"- {k} ({v['n']})" for k, v in sorted(context.items(), key=lambda x: -x[1]['n'])],
        "",
        "Target found by the substrings: " + ", ".join(MTD_TERMS),
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:5 + len(by_pred)]))
    print(f"\n[ok] target descriptors: {len(mtd_desc)} | causes: {len(causes)} | context: {len(context)}")
    print(f"[ok] outputs in {out}/ (summary.md and 4 CSV files)")


if __name__ == "__main__":
    main()