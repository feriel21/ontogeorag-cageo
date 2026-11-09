#!/usr/bin/env python3
"""193_descriptor_coverage.py: read-only, no model, no run.

Explains the difference in the coverage of the 13 LB2019 descriptors:
  189 finds 13/13 (hasDescriptor objects, NFKC, lowercase, hyphen = space);
  article_metrics.json (07) finds 12/13 (Tier 1 and 2) and 11/13 (Tier 1).

For each descriptor and each graph (delivered 364, cleaned 281), counts the
hasDescriptor relations whose object matches:
  exact      : identical string
  lowercase  : identical after strip().lower()
  normalised : NFKC, lowercase, hyphen and underscore read as a space (rule of 189)
and gives the breakdown by tier. Also prints the lines of 07 that compute the
coverage, so that the rule of 07 is read rather than assumed.

Usage (from the run18 worktree):
  python ~/ontogeorag/193_descriptor_coverage.py --repo . \
      --outdir output/audit/193_descriptor_coverage
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter


def normalise(x):
    s = unicodedata.normalize("NFKC", str(x or "")).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def load(p):
    with open(p, encoding="utf-8") as f:
        g = json.load(f)
    if isinstance(g, dict):
        g = next(v for v in g.values() if isinstance(v, list))
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--delivered-graph", default="output/run18/kg_final/flat_run18_enforced_dedup.json")
    ap.add_argument("--cleaned-graph", default="output/audit/188_nettoyage_v2/graphe_nettoye.json")
    ap.add_argument("--script-07", default="pipeline/07_final_metrics.py")
    a = ap.parse_args()
    R = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(a.repo, x))
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} already exists: no output is overwritten.")
    os.makedirs(outdir)
    sys.path.insert(0, os.path.abspath(a.repo))
    from pipeline.rag.constants import LB2019_BENCHMARK_DESCRIPTORS as LB  # read-only

    out = ["193_descriptor_coverage: read-only.",
           f"LB2019 descriptors ({len(LB)}): " + ", ".join(repr(d) for d in LB)]
    for name, path in (("delivered graph", R(a.delivered_graph)), ("cleaned graph 188", R(a.cleaned_graph))):
        g = load(path)
        desc = [r for r in g if str(r.get("relation")).strip().lower() == "hasdescriptor"]
        out.append(f"\n=== {name}: {path}, {len(g)} relations, {len(desc)} hasDescriptor ===")
        out.append(f"   {'descriptor':<16} {'exact':>6} {'lower':>8} {'normalised':>10}   tiers (normalised)  forms found")
        coverage = Counter()
        for d in LB:
            ex = [r for r in desc if str(r.get("object")) == d]
            mi = [r for r in desc if str(r.get("object")).strip().lower() == str(d).strip().lower()]
            no = [r for r in desc if normalise(r.get("object")) == normalise(d)]
            for key, l in (("exact", ex), ("lowercase", mi), ("normalised", no)):
                coverage[key] += bool(l)
                coverage[key + " tier1"] += any(str(r.get("tier")) == "1" for r in l)
            tiers = Counter(str(r.get("tier")) for r in no)
            forms = Counter(str(r.get("object")) for r in no)
            out.append(f"   {d!s:<16} {len(ex):>6} {len(mi):>8} {len(no):>10}   "
                       f"{dict(sorted(tiers.items()))!s:<18}  " + ", ".join(f"{f!r} {n}" for f, n in forms.items()))
        out.append("   coverage: " + ", ".join(f"{k} {v}/{len(LB)}" for k, v in coverage.items()))

    s07 = R(a.script_07)
    out.append(f"\n=== lines of {s07} about descriptors ===")
    if os.path.exists(s07):
        with open(s07, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        keep = set()
        for i, l in enumerate(lines):
            if re.search(r"descriptor|DESCRIPTOR|coverage", l):
                keep.update(range(max(0, i - 2), min(len(lines), i + 3)))
        previous = -2
        for i in sorted(keep):
            if i != previous + 1:
                out.append("   ...")
            out.append(f"   {i + 1:>5}  {lines[i].rstrip()}")
            previous = i
    else:
        out.append("   not found")

    with open(os.path.join(outdir, "_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()