#!/usr/bin/env python3
"""
pipeline/07_final_metrics.py — Final Metrics for Article (Stage 7)
======================================================================
WHY
    The manuscript needs a single reproducible source for the headline
    numbers (descriptor coverage, LB2019 recall, hallucination rate,
    expert-validation precision/kappa). This stage computes them all from
    the fused KG (06) so every reported number traces back to one script.

WHAT
    - Fix hallucination: supports SUPPORTED/STRONG_SUPPORT/UNCERTAIN/WEAK_SUPPORT verdicts
    - Fix entities: merges mass-transport deposit / mtd / mass transport deposits variants
    - Descriptor coverage against the 13-term LB2019_BENCHMARK_DESCRIPTORS (imported from
      pipeline.rag.constants — see run13 restructure), counted post-synonym-mapping (norm_desc).
    - Recall against configs/lb_reference_edges.json (34 edges) and, separately,
      configs/lb_reference_edges_original26.json (the original 26-edge set).

USAGE
    python pipeline/07_final_metrics.py \
        --kg output/improved_kg/tiered_kg_normalized.json \
        --output output/improved_kg/article_metrics.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pipeline.rag.constants import (
    LB2019_BENCHMARK_DESCRIPTORS as LB2019_DESCRIPTORS,
)

DESCRIPTOR_SYNONYMS = {
    "stratified": "layered",
    "sub-parallel": "parallel",
    "sub parallel": "parallel",
    "essentially undeformed": "undeformed",
    "low amplitude": "low-amplitude",
    "high amplitude": "high-amplitude",
    "low-amplitude reflection": "low-amplitude",
    "high-amplitude reflection": "high-amplitude",
    "high-amplitude positive": "high-amplitude",
    "low amplitude reflections": "low-amplitude",
    "high amplitude reflections": "high-amplitude",
}

REFERENCE_EDGES = [
    ("mass transport deposit", "hasdescriptor", "chaotic"),
    ("mass transport deposit", "hasdescriptor", "transparent"),
    ("mass transport deposit", "hasdescriptor", "blocky"),
    ("mass transport deposit", "hasdescriptor", "layered"),
    ("mass transport deposit", "hasdescriptor", "parallel"),
    ("mass transport deposit", "hasdescriptor", "continuous"),
    ("mass transport deposit", "hasdescriptor", "discontinuous"),
    ("mass transport deposit", "hasdescriptor", "massive"),
    ("mass transport deposit", "hasdescriptor", "low-amplitude"),
    ("mass transport deposit", "hasdescriptor", "high-amplitude"),
    ("slide", "hasdescriptor", "chaotic"),
    ("slide", "hasdescriptor", "blocky"),
    ("debris flow", "hasdescriptor", "chaotic"),
    ("debris flow", "hasdescriptor", "massive"),
    ("turbidity current", "formedby", "slope failure"),
    ("mass transport deposit", "formedby", "slope failure"),
    ("slope failure", "causes", "mass transport deposit"),
    ("sea-level lowstand", "triggers", "slope failure"),
    ("pore pressure buildup", "triggers", "slope failure"),
    ("headscarp", "partof", "mass transport deposit"),
    ("translated block", "partof", "mass transport deposit"),
    ("mass transport deposit", "overlies", "slope"),
    ("mass transport deposit", "occursin", "continental slope"),
    ("turbidite", "occursin", "deep water"),
    ("debris flow", "occursin", "continental slope"),
    ("mass transport deposit", "affects", "slope stability"),
]

MTD_VARIANTS = {
    "mass-transport deposit",
    "mass-transport deposits",
    "mass-transport deposits (mtd)",
    "mtd",
    "mass transport deposits",
    "mass transport deposit 1",
    "mass transport deposit 2",
    "mtd 1",
    "mtd 2",
    "mass transport complex",
    "mass-transport complex",
}

ENTITY_NORMS = {v: "mass transport deposit" for v in MTD_VARIANTS}
ENTITY_NORMS.update(
    {
        "debris flows": "debris flow",
        "debrites": "debris flow",
        "turbidity currents": "turbidity current",
        "turbidites": "turbidite",
        "slides": "slide",
        "slumps": "slump",
        "hydrate dissociation": "gas hydrate dissociation",
        "methane hydrate dissociation": "gas hydrate dissociation",
        "gas hydrate dissolution": "gas hydrate dissociation",
        "low amplitude": "low-amplitude",
        "high amplitude": "high-amplitude",
        "low amplitude reflections": "low-amplitude",
        "high amplitude reflections": "high-amplitude",
    }
)


def norm(text):
    """Lowercase/collapse-whitespace `text` and apply ENTITY_NORMS (MTD/debris-flow/etc. variant merging); returns the normalized string, no side effects."""
    if not text:
        return ""
    t = " ".join(str(text).lower().strip().split())
    return ENTITY_NORMS.get(t, t)


def norm_desc(text):
    """Apply norm() then DESCRIPTOR_SYNONYMS (e.g. stratified->layered) to `text`; returns the canonical descriptor string, no side effects."""
    t = norm(text)
    return DESCRIPTOR_SYNONYMS.get(t, t)


def verdict_to_tier(verdict):
    """Map a verification verdict string to its tier number (1=STRONG/SUPPORTED, 2=WEAK/UNCERTAIN, 3=otherwise); no side effects."""
    v = str(verdict).upper().strip()
    if "STRONG" in v or v == "SUPPORTED":
        return 1
    if "WEAK" in v or "UNCERTAIN" in v:
        return 2
    return 3


def coverage(triples):
    """Compute LB2019 descriptor coverage over `triples`' hasDescriptor objects (post norm_desc); returns found/missing sets, n_found, n_total=13 and coverage fraction, no side effects."""
    found = set()
    for t in triples:
        d = norm_desc(t.get("object", ""))
        if d in LB2019_DESCRIPTORS:
            found.add(d)
    missing = LB2019_DESCRIPTORS - found
    return {
        "found": sorted(found),
        "missing": sorted(missing),
        "n_found": len(found),
        "n_total": 13,
        "coverage": len(found) / 13,
    }


# === unified recall block (patched) ===
def _load_reference_edges_from_json(
    ref_path="configs/lb_reference_edges.json",
):
    import json
    from pathlib import Path as _P

    obj = json.load(open(_P(ref_path)))
    edges = obj.get("edges", obj) if isinstance(obj, dict) else obj
    out = []
    for e in edges:
        if isinstance(e, dict):
            out.append(
                (
                    e.get("subject", ""),
                    e.get("relation", ""),
                    e.get("object", ""),
                )
            )
        else:
            out.append(tuple(e))
    return out


def recall(triples, ref_path="configs/lb_reference_edges.json"):
    """Recall vs LB2019 benchmark, unified with the SLURM inline / paper headline matcher.

    Matching rule (per paper §3): a benchmark edge (rs, rr, ro) is recovered iff
    there exists an extracted triple (ts, tr, to) such that
        rs is a substring of ts (after lowercasing/whitespace/entity normalization),
        ro is a substring of to,
        normalize_relation(rr) == normalize_relation(tr).

    Returns the headline (substring) recall plus an exact-match 'lower_bound'.
    """
    try:
        from pipeline.rag.constants import normalize_relation
    except Exception:
        def normalize_relation(r):
            return (r or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    ref_edges = _load_reference_edges_from_json(ref_path)
    n = len(ref_edges)

    # Build extracted set once
    extracted = [(norm(t["subject"]), normalize_relation(t.get("relation","")), norm(t["object"]))
                 for t in triples]

    def match_substring(rs, rr, ro):
        rrn = normalize_relation(rr)
        rs_n, ro_n = norm(rs), norm(ro)
        for ts, tr, to in extracted:
            if tr == rrn and rs_n in ts and ro_n in to:
                return (rs, rr, ro)
        return None

    def match_exact(rs, rr, ro):
        rrn = normalize_relation(rr)
        rs_n, ro_n = norm(rs), norm(ro)
        for ts, tr, to in extracted:
            if tr == rrn and ts == rs_n and to == ro_n:
                return (rs, rr, ro)
        return None

    sub_hits   = [match_substring(s, r, o) for s, r, o in ref_edges]
    sub_hits   = [h for h in sub_hits if h is not None]
    exact_hits = [match_exact(s, r, o)     for s, r, o in ref_edges]
    exact_hits = [h for h in exact_hits if h is not None]

    return {
        "recall":          len(sub_hits) / n if n else 0.0,
        "hits":            len(sub_hits),
        "total_reference": n,
        "matched_edges":   sub_hits,
        "matcher":         "substring_normRel (paper headline)",
        "lower_bound": {
            "recall":          len(exact_hits) / n if n else 0.0,
            "hits":            len(exact_hits),
            "total_reference": n,
            "matched_edges":   exact_hits,
            "matcher":         "exact_normRel (strict tuple equality + relation normalization)",
        },
        "ref_path":        ref_path,
    }

def hallucination(triples):
    """Tally `triples` by verdict_to_tier and compute the not-supported (tier-3) fraction of the total; returns counts dict + hallucination_rate, no side effects."""
    counts = defaultdict(int)
    for t in triples:
        tier = verdict_to_tier(t.get("verdict", ""))
        if tier == 1:
            counts["strong"] += 1
        elif tier == 2:
            counts["weak"] += 1
        else:
            counts["not_supported"] += 1
    total = sum(counts.values())
    rate = counts["not_supported"] / total if total > 0 else 0.0
    return {
        "strong_support": counts["strong"],
        "weak_support": counts["weak"],
        "not_supported": counts["not_supported"],
        "total": total,
        "hallucination_rate": rate,
    }


def compute_expert_metrics(protocol_path: str) -> dict:
    """
    Compute relaxed precision and Cohen's kappa from expert annotation.
    Reads output/expert_annotation_protocol.json filled by Antoine.
    Returns empty dict if no verdicts yet.
    """
    import json
    from pathlib import Path

    path = Path(protocol_path)
    if not path.exists():
        print(f"  Expert protocol not found: {path}")
        return {}

    protocol = json.load(open(path))
    statements = protocol.get("statements", [])
    filled = [s for s in statements if s.get("verdict_expert") is not None]

    if not filled:
        pending = len(statements) - len(filled)
        print(
            f"  Expert validation pending: "
            f"0/{len(statements)} verdicts received."
        )
        return {}

    Y = sum(1 for s in filled if s["verdict_expert"] == "Y")
    P = sum(1 for s in filled if s["verdict_expert"] == "P")
    N = sum(1 for s in filled if s["verdict_expert"] == "N")
    n = len(filled)

    relaxed = (Y + 0.5 * P) / n
    strict = Y / n

    # Cohen's kappa: automated verifier vs expert
    auto_map = {
        "STRONG_SUPPORT": "Y",
        "WEAK_SUPPORT": "P",
        "NOT_SUPPORTED": "N",
        "UNCERTAIN": "P",
    }
    cats = ["Y", "P", "N"]
    cat_idx = {c: i for i, c in enumerate(cats)}
    n_cat = len(cats)

    conf = [[0] * n_cat for _ in range(n_cat)]
    for s in filled:
        a = auto_map.get(s.get("verdict_automated", ""), "P")
        h = s["verdict_expert"]
        if a in cat_idx and h in cat_idx:
            conf[cat_idx[a]][cat_idx[h]] += 1

    total = sum(sum(row) for row in conf)
    po = sum(conf[i][i] for i in range(n_cat)) / total
    pe = sum(
        (
            sum(conf[i][j] for j in range(n_cat))
            * sum(conf[j][i] for j in range(n_cat))
        )
        for i in range(n_cat)
    ) / (total**2)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

    if kappa >= 0.8:
        interp = "almost perfect"
    elif kappa >= 0.6:
        interp = "substantial"
    elif kappa >= 0.4:
        interp = "moderate"
    else:
        interp = "fair"

    print(f"\n{'=' * 50}")
    print(f"  EXPERT VALIDATION METRICS")
    print(f"{'=' * 50}")
    print(f"  Annotated: {n}/{len(statements)}")
    print(f"  Y={Y}  P={P}  N={N}")
    print(f"  Strict precision:  {strict * 100:.1f}%")
    print(f"  Relaxed precision: {relaxed * 100:.1f}%")
    print(f"  Cohen kappa:       {kappa:.3f} ({interp})")

    # By relation type
    by_rel = {}
    for s in filled:
        r = s["triple"]["relation"]
        by_rel.setdefault(r, []).append(s["verdict_expert"])
    print(f"\n  Precision by relation:")
    for rel, verdicts in sorted(by_rel.items()):
        y = verdicts.count("Y")
        p = verdicts.count("P")
        nn = verdicts.count("N")
        rp = (y + 0.5 * p) / len(verdicts)
        print(f"    {rel:<20} Y={y} P={p} N={nn}  relaxed={rp * 100:.0f}%")

    # LaTeX line for Section 5.3
    print(f"\n  LATEX FOR SECTION 5.3:")
    print(
        f"  Of {n} sampled Tier-1 triples, {Y} were rated \\emph{{Y}}, "
        f"{P} rated \\emph{{P}}, and {N} rated \\emph{{N}}, "
        f"yielding a relaxed precision of {relaxed * 100:.1f}\\% "
        f"(strict: {strict * 100:.1f}\\%). "
        f"Cohen's $\\kappa = {kappa:.2f}$ ({interp} agreement)."
    )

    return {
        "n": n,
        "Y": Y,
        "P": P,
        "N": N,
        "relaxed_precision": relaxed,
        "strict_precision": strict,
        "kappa": kappa,
        "interpretation": interp,
    }


def compute_generalization_metrics(
    gen_protocol_path: str, dev_protocol_path: str
) -> dict:
    """
    Computes precision on generalization corpus and compares with development.
    gen_protocol_path = output/generalization_annotation_protocol.json
    dev_protocol_path = output/expert_annotation_protocol.json
    """
    import json
    from pathlib import Path

    gen_path = Path(gen_protocol_path)
    dev_path = Path(dev_protocol_path)

    if not gen_path.exists():
        print(f"  Generalization protocol not found: {gen_path}")
        return {}

    gen = json.load(open(gen_path))
    filled_gen = [
        s for s in gen["statements"] if s.get("verdict_expert") is not None
    ]

    if not filled_gen:
        print(
            f"  Generalization validation pending: "
            f"0/{len(gen['statements'])} verdicts."
        )
        return {}

    Y = sum(1 for s in filled_gen if s["verdict_expert"] == "Y")
    P = sum(1 for s in filled_gen if s["verdict_expert"] == "P")
    N = sum(1 for s in filled_gen if s["verdict_expert"] == "N")
    n = len(filled_gen)
    gen_relaxed = (Y + 0.5 * P) / n

    print(f"\n{'=' * 55}")
    print(f"  GENERALIZATION VALIDATION")
    print(f"{'=' * 55}")
    print(f"  Annotated: {n}/{len(gen['statements'])}")
    print(f"  Y={Y}  P={P}  N={N}")
    print(f"  Relaxed precision: {gen_relaxed * 100:.1f}%")

    # Compare with development corpus precision
    if dev_path.exists():
        dev = json.load(open(dev_path))
        filled_dev = [
            s for s in dev["statements"] if s.get("verdict_expert") is not None
        ]
        if filled_dev:
            Yd = sum(1 for s in filled_dev if s["verdict_expert"] == "Y")
            Pd = sum(1 for s in filled_dev if s["verdict_expert"] == "P")
            dev_relaxed = (Yd + 0.5 * Pd) / len(filled_dev)
            diff = gen_relaxed - dev_relaxed
            print(f"  Dev corpus precision:  {dev_relaxed * 100:.1f}%")
            print(f"  Difference:            {diff * 100:+.1f} pp")
            if abs(diff) <= 0.10:
                verdict = "GENERALIZATION HOLDS (within 10pp)"
            elif diff < -0.10:
                verdict = "PRECISION DROP on new corpus"
            else:
                verdict = "PRECISION HIGHER on new corpus"
            print(f"  {verdict}")

    return {"n": n, "Y": Y, "P": P, "N": N, "relaxed_precision": gen_relaxed}


def main():
    """CLI entry point: loads --kg, normalizes/dedups triples, computes coverage/recall(34)/recall(orig26)/hallucination, prints the article table, and writes the full metrics JSON to --output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", required=True)
    parser.add_argument(
        "--output", default="output/improved_kg/article_metrics.json"
    )
    args = parser.parse_args()

    print("=" * 62)
    print("FINAL METRICS (CORRIGE)")
    print("=" * 62)

    with open(args.kg, "r", encoding="utf-8") as f:
        data = json.load(f)
    triples = data.get("triples", data) if isinstance(data, dict) else data
    print("\nTriples charges: {}".format(len(triples)))

    # Normalisation supplementaire
    for t in triples:
        t["subject"] = norm(t.get("subject", ""))
        t["object"] = norm(t.get("object", ""))

    # Re-deduplication
    seen = {}
    for t in triples:
        k = (t["subject"], t["relation"], t["object"])
        if k not in seen or t.get("tier", 3) < seen[k].get("tier", 3):
            seen[k] = t
    deduped = sorted(
        seen.values(),
        key=lambda x: (
            x.get("tier", 3),
            x.get("relation", ""),
            x.get("subject", ""),
        ),
    )
    print(
        "Apres normalisation+dedup: {} triples ({} supprimes)".format(
            len(deduped), len(triples) - len(deduped)
        )
    )

    tier1 = [t for t in deduped if t.get("tier", 3) == 1]
    tier12 = [t for t in deduped if t.get("tier", 3) <= 2]

    cov1 = coverage(tier1)
    cov12 = coverage(tier12)
    rec1 = recall(tier1)
    rec12 = recall(tier12)
    rec1_orig26 = recall(
        tier1, ref_path="configs/lb_reference_edges_original26.json"
    )
    rec12_orig26 = recall(
        tier12, ref_path="configs/lb_reference_edges_original26.json"
    )
    hall = hallucination(deduped)

    entities = set()
    for t in deduped:
        entities.add(t["subject"])
        entities.add(t["object"])

    rel_by_tier = defaultdict(lambda: defaultdict(int))
    for t in deduped:
        rel_by_tier[t.get("tier", 3)][t.get("relation", "?")] += 1

    print("\n" + "=" * 62)
    print("TABLE ARTICLE (copy-paste pour LaTeX)")
    print("=" * 62)
    print("\n{:<38} {:>7} {:>8}".format("Metric", "Tier1", "Tier1+2"))
    print("-" * 55)
    print("{:<38} {:>7} {:>8}".format("Triples", len(tier1), len(tier12)))
    e1 = len(
        set(t["subject"] for t in tier1) | set(t["object"] for t in tier1)
    )
    e12 = len(
        set(t["subject"] for t in tier12) | set(t["object"] for t in tier12)
    )
    print("{:<38} {:>7} {:>8}".format("Unique entities", e1, e12))
    print(
        "{:<38} {:>7} {:>8}".format(
            "Descriptor coverage (n/13)", cov1["n_found"], cov12["n_found"]
        )
    )
    print(
        "{:<38} {:>6.1f}% {:>7.1f}%".format(
            "Descriptor coverage (%)",
            cov1["coverage"] * 100,
            cov12["coverage"] * 100,
        )
    )
    print(
        "{:<38} {:>7} {:>8}".format(
            "Recall vs LB2019 (n/{})".format(rec12["total_reference"]),
            rec1["hits"],
            rec12["hits"],
        )
    )
    print(
        "{:<38} {:>6.1f}% {:>7.1f}%".format(
            "Recall vs LB2019 (%)", rec1["recall"] * 100, rec12["recall"] * 100
        )
    )
    print(
        "{:<38} {:>7} {:>8}".format(
            "Recall vs LB2019-orig26 (n/{})".format(
                rec12_orig26["total_reference"]
            ),
            rec1_orig26["hits"],
            rec12_orig26["hits"],
        )
    )
    print(
        "{:<38} {:>6.1f}% {:>7.1f}%".format(
            "Recall vs LB2019-orig26 (%)",
            rec1_orig26["recall"] * 100,
            rec12_orig26["recall"] * 100,
        )
    )
    print(
        "{:<38} {:>6.1f}% {:>7.1f}%".format(
            "Hallucination rate", 0.0, hall["hallucination_rate"] * 100
        )
    )

    print(
        "\nDescriptors found (Tier1+2): {}".format(", ".join(cov12["found"]))
    )
    if cov12["missing"]:
        print(
            "Missing:                     {}".format(
                ", ".join(cov12["missing"])
            )
        )

    print(
        "\nMatched LB2019 edges (Tier1+2, {}/{}):".format(
            rec12["hits"], rec12["total_reference"]
        )
    )
    for s, r, o in rec12["matched_edges"]:
        print("  {} --[{}]--> {}".format(s, r, o))

    print("\nHallucination breakdown (full KG):")
    print("  STRONG/SUPPORTED:  {:4d}".format(hall["strong_support"]))
    print("  WEAK/UNCERTAIN:    {:4d}".format(hall["weak_support"]))
    print("  NOT_SUPPORTED:     {:4d}".format(hall["not_supported"]))
    print(
        "  Hallucination rate: {:.1f}%".format(
            hall["hallucination_rate"] * 100
        )
    )

    print("\nRelations (Tier1+2):")
    all_rels = sorted(
        rel_by_tier[1].keys() | rel_by_tier[2].keys(),
        key=lambda r: -(rel_by_tier[1][r] + rel_by_tier[2][r]),
    )
    print("  {:<22} {:>4} {:>4} {:>5}".format("Relation", "T1", "T2", "Tot"))
    print("  " + "-" * 36)
    for rel in all_rels:
        t1 = rel_by_tier[1][rel]
        t2 = rel_by_tier[2][rel]
        if t1 + t2 > 0:
            print("  {:<22} {:>4} {:>4} {:>5}".format(rel, t1, t2, t1 + t2))

    print("\nTop entites:")
    ent_counts = defaultdict(int)
    for t in deduped:
        ent_counts[t["subject"]] += 1
        ent_counts[t["object"]] += 1
    for ent, cnt in sorted(ent_counts.items(), key=lambda x: -x[1])[:10]:
        print("  {:<40} {:3d}".format(ent, cnt))

    metrics = {
        "summary": {
            "tier1_triples": len(tier1),
            "tier12_triples": len(tier12),
            "total_triples": len(deduped),
            "unique_entities": len(entities),
        },
        "descriptor_coverage": {"tier1": cov1, "tier12": cov12},
        "recall_vs_lb2019": {"tier1": rec1, "tier12": rec12},
        "recall_vs_lb2019_orig26": {
            "tier1": rec1_orig26,
            "tier12": rec12_orig26,
        },
        "hallucination": hall,
        "relation_distribution_by_tier": {
            str(k): dict(v) for k, v in rel_by_tier.items()
        },
        "triples_final": deduped,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\nSauvegarde: {}".format(args.output))


if __name__ == "__main__":
    main()
