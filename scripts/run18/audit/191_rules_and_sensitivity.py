#!/usr/bin/env python3
"""191_rules_and_sensitivity.py: read-only, no model, no run.

1. Joins controls.jsonl and the Llama and Mistral verdicts by m4_index, checking
   on each line that subject, relation and object are identical (stops otherwise).
2. Recomputes explicit rules and states which ones reproduce the counts of
   negatives_report.json and decision_rules.json (checked, not assumed).
3. Rates per perturbation type with 95 % Wilson intervals.
4. Sensitivity: same rates without the ambiguous records (introduced entity
   present in the passage, relation of the same family, subject = object).
5. Originals: verdicts of the self-audit compared with those of the delivered
   graph (join by content key), and overlap of the originals set aside with the
   UNCERTAIN relations.

Usage (from the run18 worktree):
  python ~/ontogeorag/191_rules_and_sensitivity.py --repo . \
      --outdir output/audit/191_rules_and_sensitivity
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

FAMILIES = {"causal": {"causes", "triggers", "controls", "affects", "formedby"},
            "stratigraphic": {"overlies", "underlies"},
            "spatial": {"occursin", "partof"}}
CLASSES = ["entity_sub", "passage_perm", "relation_sub", "inversion"]


def norm(x):
    s = unicodedata.normalize("NFKC", str(x or "")).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def nrel(x):
    return norm(x).replace(" ", "")


def fam(r):
    r = nrel(r)
    for f, m in FAMILIES.items():
        if r in m:
            return f
    return r


def rate(k, n):
    return k / n if n else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def g(d, path):
    for k in path.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stress", default="output/run18/stress_test")
    ap.add_argument("--graph", default="output/run18/kg_final/flat_run18_enforced_dedup.json")
    a = ap.parse_args()
    R = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(a.repo, x))
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} already exists: no output is overwritten.")
    os.makedirs(outdir)
    out = ["191_rules_and_sensitivity: read-only."]

    st = R(a.stress)
    ctrl = jsonl(os.path.join(st, "controls.jsonl"))
    verifiers = {}
    for j in ("llama", "mistral"):
        v = jsonl(os.path.join(st, j, "m4_verdicts.jsonl"))
        by_index = {}
        for r in v:
            i = r.get("m4_index")
            if i in by_index:
                sys.exit(f"m4_index {i} duplicated in {j}: stop.")
            by_index[i] = r
        verifiers[j] = by_index
    # --- 1. checked join
    lines = []
    for i, c in enumerate(ctrl):
        rec = {"i": i, "c": c}
        for j, by_index in verifiers.items():
            r = by_index.get(i)
            if r is None:
                sys.exit(f"{j}: m4_index {i} missing: stop.")
            if (norm(r["subject"]), nrel(r["relation"]), norm(r["object"])) != \
               (norm(c["subject"]), nrel(c["relation"]), norm(c["object"])):
                sys.exit(f"{j}: m4_index {i} does not match record {i}: stop.")
            rec[j] = r
        lines.append(rec)
    out.append(f"join by m4_index checked on the content: {len(lines)}/{len(ctrl)} lines, "
               f"0 mismatch, for llama and mistral")

    # --- 2. explicit rules
    def ev(rec, j):
        return str(g(rec[j], "evidence.verdict"))

    def bl(rec, j):
        return str(g(rec[j], "blind.verdict"))

    rules = {
        "sentence NOT_SUPPORTED llama": lambda r: ev(r, "llama") == "NOT_SUPPORTED",
        "sentence NOT_SUPPORTED mistral": lambda r: ev(r, "mistral") == "NOT_SUPPORTED",
        "sentence NOT_SUPPORTED, at least one verifier": lambda r: "NOT_SUPPORTED" in (ev(r, "llama"), ev(r, "mistral")),
        "sentence NOT_SUPPORTED, both verifiers": lambda r: ev(r, "llama") == ev(r, "mistral") == "NOT_SUPPORTED",
        "sentence NOT or blind IMPLAUSIBLE, llama": lambda r: ev(r, "llama") == "NOT_SUPPORTED" or bl(r, "llama") == "IMPLAUSIBLE",
        "sentence NOT or blind IMPLAUSIBLE, mistral": lambda r: ev(r, "mistral") == "NOT_SUPPORTED" or bl(r, "mistral") == "IMPLAUSIBLE",
        "sentence NOT or blind IMPLAUSIBLE, at least one verifier":
            lambda r: any(ev(r, j) == "NOT_SUPPORTED" or bl(r, j) == "IMPLAUSIBLE" for j in ("llama", "mistral")),
        "sentence not SUPPORTED (NOT or PARTIALLY), both verifiers":
            lambda r: ev(r, "llama") != "SUPPORTED" and ev(r, "mistral") != "SUPPORTED",
        "blind IMPLAUSIBLE, at least one verifier": lambda r: "IMPLAUSIBLE" in (bl(r, "llama"), bl(r, "mistral")),
    }
    # published counts to reproduce
    published = {}
    for j in ("llama", "mistral"):
        try:
            with open(os.path.join(st, j, "negatives_report.json"), encoding="utf-8") as f:
                nr = json.load(f)
            published[f"negatives_report {j}"] = (nr["overall"]["tp"], nr["overall"]["fp"])
        except Exception as e:
            out.append(f"negatives_report {j} not read: {e}")
    try:
        with open(os.path.join(st, "decision_rules.json"), encoding="utf-8") as f:
            rd = json.load(f)
        for name, block in rd.items():
            if isinstance(block, dict) and "corrupted_flagged" in block:
                published[f"decision_rules {name}"] = (block["corrupted_flagged"], block["originals_flagged"])
    except Exception as e:
        out.append(f"decision_rules.json not read: {e}")
    txt = os.path.join(st, "decision_rules.txt")
    if os.path.exists(txt):
        out.append(f"\n--- content of {txt}")
        with open(txt, encoding="utf-8", errors="replace") as f:
            out += [l.rstrip("\n") for l in f]

    pert = [r for r in lines if r["c"]["label"] != "original"]
    orig = [r for r in lines if r["c"]["label"] == "original"]
    out.append(f"\n=== 2. Recomputed rules (perturbed {len(pert)}, originals {len(orig)}) ===")
    results = {}
    for name, f in rules.items():
        k = sum(1 for r in pert if f(r))
        m = sum(1 for r in orig if f(r))
        results[name] = (k, m)
        equal = [p for p, v in published.items() if v == (k, m)]
        out.append(f"   {name:<55} perturbed set aside {k:>3}/{len(pert)}, originals set aside {m:>3}/{len(orig)}"
                   + (f"   = {', '.join(equal)}" if equal else ""))
    out.append("   published counts not reproduced by these rules: " +
               (", ".join(f"{p} {v}" for p, v in published.items() if v not in results.values()) or "none"))

    # --- 3 and 4. rates per type, with and without the ambiguous records
    originals_by_index = {str(r["c"]["orig_index"]): r["c"] for r in orig}

    def ambiguous(r):
        c = r["c"]
        o = originals_by_index.get(str(c["orig_index"]))
        s, ob, p = norm(c["subject"]), norm(c["object"]), nrel(c["relation"])
        if s == ob:
            return "loop subject = object"
        if o is None:
            return ""
        passage = norm(c.get("supporting_passage"))
        cl = c["corruption_class"]
        if cl == "entity_sub":
            new = [e for e, e0 in ((s, norm(o["subject"])), (ob, norm(o["object"]))) if e != e0]
            if new and all(e in passage for e in new):
                return "introduced entity present in the passage"
        if cl == "relation_sub" and fam(p) == fam(o["relation"]) and p != nrel(o["relation"]):
            return "relation of the same family"
        if cl == "passage_perm" and s in passage and ob in passage:
            return "subject and object in the new passage"
        return ""

    amb = Counter()
    csv_rows = []
    for r in pert:
        r["amb"] = ambiguous(r)
        if r["amb"]:
            amb[(r["c"]["corruption_class"], r["amb"])] += 1
    out.append("\n=== 3. Ambiguous records kept for the sensitivity ===")
    for (cl, m), n in sorted(amb.items()):
        out.append(f"   {cl:<14} {m} : {n}")

    for title, keep in (("all perturbed", lambda r: True),
                          ("without the ambiguous records", lambda r: not r["amb"])):
        out.append(f"\n=== 4. Rates per type, {title} (Wilson 95 %) ===")
        for name, f in rules.items():
            out.append(f"   -- {name}")
            for cl in CLASSES:
                gr = [r for r in pert if r["c"]["corruption_class"] == cl and keep(r)]
                k = sum(1 for r in gr if f(r))
                lo, hi = wilson(k, len(gr))
                out.append(f"      {cl:<14} rejected {k:>3}/{len(gr):<3} = {rate(k, len(gr)):.3f} [{lo:.3f}, {hi:.3f}]")
                csv_rows.append([title, name, cl, len(gr), k, round(rate(k, len(gr)), 4), round(lo, 4), round(hi, 4)])
            gr = [r for r in pert if keep(r)]
            k = sum(1 for r in gr if f(r))
            lo, hi = wilson(k, len(gr))
            out.append(f"      {'all':<14} rejected {k:>3}/{len(gr):<3} = {rate(k, len(gr)):.3f} [{lo:.3f}, {hi:.3f}]")
            m = sum(1 for r in orig if not f(r))
            lo2, hi2 = wilson(m, len(orig))
            out.append(f"      {'originals':<14} retained {m}/{len(orig)} = {rate(m, len(orig)):.3f} [{lo2:.3f}, {hi2:.3f}]")
            csv_rows.append([title, name, "all", len(gr), k, round(rate(k, len(gr)), 4), round(lo, 4), round(hi, 4)])
            csv_rows.append([title, name, "originals_retained", len(orig), m, round(rate(m, len(orig)), 4),
                               round(lo2, 4), round(hi2, 4)])

    # --- 5. originals against the delivered graph
    out.append("\n=== 5. Originals of the self-audit and delivered graph ===")
    with open(R(a.graph), encoding="utf-8") as f:
        graph = json.load(f)
    if isinstance(graph, dict):
        graph = next(v for v in graph.values() if isinstance(v, list))
    idx = defaultdict(list)
    for r in graph:
        idx[(norm(r["subject"]), nrel(r["relation"]), norm(r["object"]))].append(r)
    agreement = Counter()
    crossed = defaultdict(Counter)
    without = 0
    for r in orig:
        c = r["c"]
        m = idx.get((norm(c["subject"]), nrel(c["relation"]), norm(c["object"])), [])
        if len(m) != 1:
            without += 1
            continue
        gv = str(g(m[0], "m4.evidence_verdict") or "")
        sv = f"{ev(r, 'llama')}|{ev(r, 'mistral')}"
        agreement["identical" if gv == sv else "different"] += 1
        dec = str(g(m[0], "m4.decision"))
        for name, f in rules.items():
            crossed[name][(dec, "set aside" if f(r) else "kept")] += 1
    out.append(f"   originals joined to the graph by content key: {len(orig) - without}/{len(orig)} "
               f"({without} without a unique match)")
    out.append(f"   verdicts on the sentence (llama|mistral), self-audit against delivered graph: {dict(agreement)}")
    out.append("   panel decision in the graph x rule on the original:")
    for name, t in crossed.items():
        out.append(f"      {name:<55} " + ", ".join(f"{d} {e} {n}" for (d, e), n in sorted(t.items())))

    with open(os.path.join(outdir, "rates_per_type.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "rule", "class", "n", "k", "rate", "wilson_low", "wilson_high"])
        w.writerows(csv_rows)
    with open(os.path.join(outdir, "ambiguous_records.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["m4_index", "class", "reason", "subject", "relation", "object", "llama_sentence",
                    "mistral_sentence", "passage"])
        for r in pert:
            if r["amb"]:
                c = r["c"]
                w.writerow([r["i"], c["corruption_class"], r["amb"], c["subject"], c["relation"], c["object"],
                            ev(r, "llama"), ev(r, "mistral"), str(c.get("supporting_passage"))[:300]])
    with open(os.path.join(outdir, "_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()