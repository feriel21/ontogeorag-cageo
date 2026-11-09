#!/usr/bin/env python3
"""192_missing_information.py: read-only, no model, no run.

Gathers the last items missing for the manuscript:

  1. Evidence check applied to the graphs (364 and 281 relations): a relation is
     flagged if at least one verifying-LLM gives NOT_SUPPORTED in
     m4.evidence_verdict (Llama|Mistral).
     Check: on the 364-relation graph, the total must be 79 (self-audit, 191).
     Breakdown by tier, by panel decision and by relation type.
  2. Contributing papers (distinct references in provenance.sources), in the
     graph with provenance (187) and in the cleaned graph (188).
  3. Metadata of the verifying-LLMs of the run (model, number of relations,
     passage limit, duration).
  4. Statistics of the steps (calls, extractions, rejections): every numerical
     field of the files *stats*.json, *report*.json and article_metrics.json.
  5. SLURM jobs of 22 Oct. 2026 (sacct): name, duration, GPU, node, state.

Writes only into --outdir and refuses to overwrite an existing folder.

Usage (from the run18 worktree):
  python ~/ontogeorag/192_missing_information.py --repo . --outdir output/audit/192_missing_information
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import unicodedata
from collections import Counter


def norm(x):
    return " ".join(unicodedata.normalize("NFKC", str(x or "")).lower().split())


def load_graph(p):
    with open(p, encoding="utf-8") as f:
        g = json.load(f)
    if isinstance(g, dict):
        g = next(v for v in g.values() if isinstance(v, list))
    return g


def m4(r, key):
    v = r.get("m4")
    if isinstance(v, dict):
        return v.get(key)
    return r.get(f"m4.{key}")


def flagged(r):
    ev = str(m4(r, "evidence_verdict") or "")
    return "NOT_SUPPORTED" in ev.split("|")


def check(name, path, out, expected=None):
    if not os.path.exists(path):
        out.append(f"[{name}] not found: {path}")
        return
    g = load_graph(path)
    sig = [r for r in g if flagged(r)]
    empty = sum(1 for r in g if not m4(r, "evidence_verdict"))
    line = f"[{name}] {path}: {len(g)} relations, flagged {len(sig)}"
    if expected is not None:
        line += f" (expected {expected}: {'OK' if len(sig) == expected else 'DIFFERENT'})"
    out.append(line)
    out.append(f"   relations without m4.evidence_verdict: {empty}")
    for title, f in (("tier", lambda r: str(r.get("tier"))),
                     ("panel decision", lambda r: str(m4(r, "decision"))),
                     ("relation type", lambda r: str(r.get("relation")))):
        tot = Counter(f(r) for r in g)
        s = Counter(f(r) for r in sig)
        out.append(f"   by {title}: " + ", ".join(f"{k} {s.get(k, 0)}/{n}" for k, n in tot.most_common()))
    verifiers = Counter()
    for r in sig:
        a, _, b = str(m4(r, "evidence_verdict")).partition("|")
        verifiers[("Llama" if a == "NOT_SUPPORTED" else "") + ("+" if a == b == "NOT_SUPPORTED" else "") +
              ("Mistral" if b == "NOT_SUPPORTED" else "")] += 1
    out.append("   flagging verifier: " + ", ".join(f"{k} {n}" for k, n in verifiers.most_common()))


def papers(name, path, out):
    if not os.path.exists(path):
        out.append(f"[{name}] not found: {path}")
        return
    g = load_graph(path)
    refs, without = Counter(), 0
    for r in g:
        src = r.get("provenance", {}).get("sources") if isinstance(r.get("provenance"), dict) \
            else r.get("provenance.sources")
        if isinstance(src, str):
            try:
                src = json.loads(src)
            except Exception:
                src = None
        if not src:
            without += 1
            continue
        for s in {norm(x.get("reference")) for x in src if isinstance(x, dict) and x.get("reference")}:
            refs[s] += 1
    out.append(f"[{name}] {len(g)} relations; distinct contributing papers: {len(refs)}; "
               f"relations without a source: {without}")


def scalars(obj, path="", out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            scalars(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, (int, float, bool)) or (isinstance(obj, str) and len(obj) < 120):
        out.append(f"{path} = {obj}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run", default="output/run18")
    ap.add_argument("--delivered-graph", default="output/run18/kg_final/flat_run18_enforced_dedup.json")
    ap.add_argument("--provenance-graph", default="output/audit/187_provenance/graphe_avec_provenance.json")
    ap.add_argument("--cleaned-graph", default="output/audit/188_nettoyage_v2/graphe_nettoye.json")
    ap.add_argument("--start", default="2026-10-22T00:00:00")
    ap.add_argument("--end", default="2026-10-23T12:00:00")
    ap.add_argument("--jobs", nargs="*", default=[], help="SLURM job numbers (otherwise: every job of the period)")
    a = ap.parse_args()
    R = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(a.repo, x))
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} already exists: no output is overwritten.")
    os.makedirs(outdir)
    out = ["192_missing_information: read-only."]

    out.append("\n=== 1. Evidence check applied to the graphs ===")
    out.append("rule: flagged if at least one verifier returns NOT_SUPPORTED (m4.evidence_verdict, Llama|Mistral)")
    check("delivered graph", R(a.delivered_graph), out, expected=79)
    check("cleaned graph 188", R(a.cleaned_graph), out)

    out.append("\n=== 2. Contributing papers ===")
    papers("graph with provenance 187", R(a.provenance_graph), out)
    papers("cleaned graph 188", R(a.cleaned_graph), out)

    out.append("\n=== 3. Metadata of the verifiers of the run ===")
    for f in sorted(glob.glob(os.path.join(R(a.run), "m4", "*", "m4_run_meta.json"))):
        with open(f, encoding="utf-8") as h:
            m = json.load(h)
        out.append(f"[{os.path.relpath(f, a.repo)}] " + ", ".join(
            f"{k}={os.path.basename(str(v)) if k in ('model', 'input') else v}" for k, v in m.items()))

    out.append("\n=== 4. Statistics of the steps ===")
    patterns = ["*stats*.json", "*report*.json", "*summary*.json", "article_metrics.json"]
    seen = set()
    for pattern in patterns:
        for f in sorted(glob.glob(os.path.join(R(a.run), "**", pattern), recursive=True)):
            if f in seen or "stress_test" in f or "checkpoint" in f:
                continue
            seen.add(f)
            try:
                with open(f, encoding="utf-8") as h:
                    obj = json.load(h)
            except Exception as e:
                out.append(f"[{os.path.relpath(f, a.repo)}] not read: {e}")
                continue
            s = scalars(obj)
            out.append(f"[{os.path.relpath(f, a.repo)}] {len(s)} values")
            out += ["   " + x for x in s[:80]]
            if len(s) > 80:
                out.append(f"   ... {len(s) - 80} more values, see the file")

    out.append("\n=== 5. Jobs SLURM ===")
    fmt = "--format=JobID,JobName%35,Start,End,Elapsed,AllocTRES%70,NodeList%20,State"
    cmd = ["sacct", "-X", fmt]
    cmd += ["-j", ",".join(a.jobs)] if a.jobs else ["-S", a.start, "-E", a.end]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out.append("command: " + " ".join(cmd))
        out += ["   " + l for l in r.stdout.splitlines()]
        if r.stderr.strip():
            out.append("   error: " + r.stderr.strip()[:300])
    except Exception as e:
        out.append(f"sacct failed: {e}")
    out.append("The manifest steps run from 2026-10-22 11:19 to 18:15: keep the job or jobs that cover this period.")

    with open(os.path.join(outdir, "_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()