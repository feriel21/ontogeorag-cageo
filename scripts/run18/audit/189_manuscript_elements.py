#!/usr/bin/env python3
"""189_manuscript_elements.py: read-only, no model, no run.

Gathers, from the existing run18 outputs, the items still missing to write the
manuscript:

  A. Schema of the files read (to check the detected fields).
  B. The graphs before cleaning (delivered graph) and after cleaning (output of 188):
     relations, entities, relation types, tiers, decisions, flags, coverage of the
     13 LB2019 descriptors, provenance joined by content key.
  C. Perturbed records: rejection rate per perturbation type and per verifier or
     rule, retention rate of the originals, 95 % Wilson intervals.
  D. Ambiguous records per perturbation type (a perturbation that could remain
     supported by the sentence).
  E. Qualification markers (output of 186) and existing summaries of 184, 186,
     187 and 188.
  F. Hardware, duration and calls: manifests and SLURM logs.

The script writes only into --outdir and refuses to overwrite an existing folder.
Joins use normalised content keys, never positions.

Usage (from the run18 worktree):
  python ~/ontogeorag/189_manuscript_elements.py --repo . --run output/run18 \
      --outdir output/audit/189_manuscript_elements
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Reading constants (not pipeline parameters)
# --------------------------------------------------------------------------
CAND_SUBJECT = ["source", "subject", "head", "subj", "s"]
CAND_REL = ["relation", "predicate", "rel", "r"]
CAND_OBJECT = ["target", "object", "tail", "obj", "o"]
CAND_SENTENCE = ["evidence_sentence", "evidence", "sentence", "sentence_text", "quote",
               "evidence_text", "support_sentence", "text"]
CAND_TIER = ["tier"]

# Directed predicates: an inversion generally makes the relation false.
DIRECTED = {"causes", "triggers", "controls", "affects", "overlies", "underlies",
            "partof", "formedby", "occursin", "hasdescriptor", "indicates",
            "evidences"}
# Families of neighbouring predicates: a change inside a family
# can remain supported by the sentence.
FAMILIES = {
    "causal": {"causes", "triggers", "controls", "affects", "formedby"},
    "stratigraphic": {"overlies", "underlies"},
    "spatial": {"occursin", "partof"},
    "descriptive": {"hasdescriptor"},
    "association": {"relatedto", "indicates", "evidences"},
}
VERDICT_PATTERN = re.compile(r"SUPPORT|ACCEPT|REJECT|UNCERTAIN|PLAUSIBLE|NOT_|IMPLAUSIBLE",
                           re.I)
REJECTION_PATTERN = re.compile(r"NOT[_ ]SUPPORTED|REJECT", re.I)
RECORDS_PATTERN = re.compile(r"stress|control|corrupt|adversar|perturb", re.I)
MAX_LEVELS = 12          # above this length, a field is treated as free text


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def norm(x):
    """Text key: NFKC (ligatures), lowercase, hyphen and underscore read as a space."""
    if x is None:
        return ""
    s = unicodedata.normalize("NFKC", str(x)).lower()
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def norm_rel(x):
    return norm(x).replace(" ", "")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def flatten(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = json.dumps(v2, ensure_ascii=False) if isinstance(v2, (dict, list)) else v2
        else:
            out[k] = v
    return out


def load(path):
    """Return a list of flattened dicts, or [] if the file is not tabular."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".jsonl":
            with open(path, encoding="utf-8") as f:
                return [flatten(json.loads(l)) for l in f if l.strip().startswith("{")]
        if ext == ".json":
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [flatten(r) for r in obj if isinstance(r, dict)]
            if isinstance(obj, dict):
                for key in ["relations", "triples", "edges", "records", "controls",
                            "items", "data", "rows"]:
                    if isinstance(obj.get(key), list):
                        return [flatten(r) for r in obj[key] if isinstance(r, dict)]
                for v in obj.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return [flatten(r) for r in v if isinstance(r, dict)]
            return []
        if ext in (".csv", ".tsv"):
            with open(path, encoding="utf-8", newline="") as f:
                return [dict(r) for r in csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ",")]
    except Exception as e:  # unreadable file: reported, then skipped
        print(f"  [unreadable] {path}: {e}", file=sys.stderr)
    return []


def key(keys, candidates, excluded_prefix=None):
    """Find a field: exact name, then last segment, then substring."""
    keys = [k for k in keys if not (excluded_prefix and k.lower().startswith(excluded_prefix))]
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c in lower:
            return lower[c]
    for c in candidates:
        for k in keys:
            if k.lower().split(".")[-1] == c:
                return k
    for c in candidates:
        if len(c) > 2:
            for k in keys:
                if c in k.lower():
                    return k
    return None


def all_keys(recs):
    seen = []
    for r in recs[:500]:
        for k in r:
            if k not in seen:
                seen.append(k)
    return seen


def categorical_fields(recs, keys):
    out = {}
    for k in keys:
        vals = Counter(str(r.get(k)) for r in recs)
        if 1 < len(vals) <= MAX_LEVELS:
            out[k] = vals
    return out


def triple_keys(recs, excluded_prefix=None):
    ks = all_keys(recs)
    return (key(ks, CAND_SUBJECT, excluded_prefix), key(ks, CAND_REL, excluded_prefix),
            key(ks, CAND_OBJECT, excluded_prefix))


def content_key(r, s, p, o):
    return (norm(r.get(s)), norm_rel(r.get(p)), norm(r.get(o)))


# --------------------------------------------------------------------------
# A. Schemas
# --------------------------------------------------------------------------
def schema(name, recs):
    lines = [f"== {name}: {len(recs)} records"]
    ks = all_keys(recs)
    for k in ks:
        vals = Counter(str(r.get(k)) for r in recs)
        ex = str(recs[0].get(k))[:80].replace("\n", " ") if recs else ""
        lines.append(f"   {k} | {len(vals)} distinct values | e.g. {ex}")
    return lines


# --------------------------------------------------------------------------
# B. Graphs
# --------------------------------------------------------------------------
def lb13(repo):
    sys.path.insert(0, os.path.abspath(repo))
    try:
        from pipeline.rag.constants import LB2019_BENCHMARK_DESCRIPTORS  # read-only
        return [norm(x) for x in LB2019_BENCHMARK_DESCRIPTORS]
    except Exception as e:
        print(f"  [constants] LB2019_BENCHMARK_DESCRIPTORS not found: {e}", file=sys.stderr)
        return None


def graph_stats(name, recs, lb, provenance_sets, out):
    s, p, o = triple_keys(recs)
    if not (s and p and o):
        out.append(f"[{name}] subject/relation/object fields not found: see schemas.txt")
        return None
    ent = {norm(r.get(s)) for r in recs} | {norm(r.get(o)) for r in recs}
    types = Counter(str(r.get(p)) for r in recs)
    unique_keys = {content_key(r, s, p, o) for r in recs}
    out.append(f"[{name}] {len(recs)} relations ({len(unique_keys)} distinct content keys), "
               f"{len(ent)} distinct entities (normalised)")
    out.append(f"   detected fields: subject={s} relation={p} object={o}")
    out.append("   relation types: " + ", ".join(f"{k} {v}" for k, v in types.most_common()))
    for k, vals in categorical_fields(recs, all_keys(recs)).items():
        if k in (p,):
            continue
        out.append(f"   {k}: " + ", ".join(f"{a} {b}" for a, b in vals.most_common()))
    if lb is not None:
        desc = {norm(r.get(o)) for r in recs if norm_rel(r.get(p)) == "hasdescriptor"}
        covered = [d for d in lb if d in desc]
        missing = [d for d in lb if d not in desc]
        out.append(f"   coverage of the {len(lb)} LB2019 descriptors: {len(covered)}/{len(lb)}; "
                   f"absent: {', '.join(missing) or 'none'}")
    for pname, (precs, ps, pp, po) in provenance_sets.items():
        index = defaultdict(list)
        for r in precs:
            index[content_key(r, ps, pp, po)].append(r)
        joined = [index[content_key(r, s, p, o)][0] for r in recs if content_key(r, s, p, o) in index]
        out.append(f"   provenance via {pname}: {len(joined)}/{len(recs)} relations joined by content key")
        for k, vals in categorical_fields(joined, all_keys(joined)).items():
            if k in (ps, pp, po):
                continue
            out.append(f"      {k}: " + ", ".join(f"{a} {b}" for a, b in vals.most_common()))
    return {"name": name, "relations": len(recs), "entities": len(ent),
            **{f"type:{k}": v for k, v in types.items()}}


# --------------------------------------------------------------------------
# C and D. Perturbed records
# --------------------------------------------------------------------------
def category(v):
    t = norm(v)
    if t in ("", "none", "nan", "null"):
        return "original"
    if re.search(r"orig|clean|unmodified|identity|^control$|negative", t):
        return "original"
    if re.search(r"sentence|evidence|quote", t):
        return "swapped_sentence"
    if re.search(r"inver|revers|flip|direction|subject object", t):
        return "inversion"
    if re.search(r"relation|predicate|rel ", t):
        return "type_changed"
    if re.search(r"entity|object|subject|substit|replace|argument", t):
        return "replaced_entity"
    return "other:" + t


def type_field(recs):
    best, best_score = None, 0
    for k, vals in categorical_fields(recs, all_keys(recs)).items():
        cats = {category(v) for v in vals}
        score = len([c for c in cats if not c.startswith("other")])
        if score > best_score:
            best, best_score = k, score
    return best if best_score >= 2 else None


def verdict_fields(recs):
    out = []
    for k, vals in categorical_fields(recs, all_keys(recs)).items():
        if any(VERDICT_PATTERN.search(str(v)) for v in vals):
            out.append(k)
    return out


def ambiguity(cat, r, ctrl, orig, k_sentence):
    sentence = norm(r.get(k_sentence)) if k_sentence else ""
    s, p, o = (norm(r.get(ctrl[0])), norm_rel(r.get(ctrl[1])), norm(r.get(ctrl[2])))
    if cat == "swapped_sentence":
        if not sentence:
            return "undetermined (no sentence)"
        return "ambiguous" if (s and s in sentence and o and o in sentence) else "not ambiguous"
    if cat == "replaced_entity":
        if not sentence:
            return "undetermined (no sentence)"
        if all(orig):
            os_, oo = norm(r.get(orig[0])), norm(r.get(orig[2]))
            new = [e for e, e0 in ((s, os_), (o, oo)) if e and e != e0]
        else:
            new = [s, o]
        return "ambiguous" if new and all(e in sentence for e in new) else "not ambiguous"
    if cat == "inversion":
        return "not ambiguous" if p in DIRECTED else "ambiguous (undirected predicate)"
    if cat == "type_changed":
        if not orig[1]:
            return "undetermined (original relation missing)"
        p0 = norm_rel(r.get(orig[1]))
        same = any(p in f and p0 in f for f in FAMILIES.values())
        return "ambiguous (same family)" if same else "not ambiguous"
    return "not assessed"


def analyse_records(name, recs, out, csv_rows, ambiguity_rows, expected):
    kt = type_field(recs)
    if not kt:
        out.append(f"[{name}] perturbation type field not detected: see schemas.txt")
        return
    verdicts = verdict_fields(recs)
    ctrl = triple_keys(recs, excluded_prefix="orig")
    ks = all_keys(recs)
    orig = tuple(key([k for k in ks if k.lower().startswith("orig")], c) for c in (CAND_SUBJECT, CAND_REL, CAND_OBJECT))
    k_sentence = key(ks, CAND_SENTENCE)
    by_category = defaultdict(list)
    for r in recs:
        by_category[category(r.get(kt))].append(r)
    n_orig = len(by_category.get("original", []))
    n_corr = sum(len(v) for c, v in by_category.items() if c != "original")
    out.append(f"[{name}] type field = {kt}; originals {n_orig}, perturbed {n_corr} "
               f"(expected {expected[0]} and {expected[1]}: {'OK' if (n_orig, n_corr) == expected else 'DIFFERENT'})")
    out.append(f"   values of the type field: " + ", ".join(f"{v} -> {category(v)} ({n})"
               for v, n in Counter(str(r.get(kt)) for r in recs).most_common()))
    out.append(f"   detected verdict fields: {', '.join(verdicts) or 'none'}")
    out.append(f"   sentence = {k_sentence}; presented triple = {ctrl}; original triple = {orig}")
    for kv in verdicts:
        out.append(f"   -- {kv} (rejected = value containing NOT_SUPPORTED or REJECT)")
        for cat in sorted(by_category):
            g = by_category[cat]
            k = sum(1 for r in g if REJECTION_PATTERN.search(str(r.get(kv))))
            lo, hi = wilson(k, len(g))
            if cat == "original":
                kept = len(g) - k
                clo, chi = wilson(kept, len(g))
                out.append(f"      {cat:<18} retained {kept}/{len(g)} = {kept/len(g):.3f} "
                           f"[{clo:.3f}, {chi:.3f}]")
            else:
                out.append(f"      {cat:<18} rejected {k}/{len(g)} = {k/len(g):.3f} [{lo:.3f}, {hi:.3f}]")
            dist = Counter(str(r.get(kv)) for r in g)
            out.append("         values: " + ", ".join(f"{a} {b}" for a, b in dist.most_common()))
            csv_rows.append([name, kv, cat, len(g), k, round(k / len(g), 4) if g else "",
                               round(lo, 4), round(hi, 4)])
    out.append("   ambiguous records per type:")
    for cat in sorted(c for c in by_category if c != "original"):
        summary = Counter()
        for r in by_category[cat]:
            a = ambiguity(cat, r, ctrl, orig, k_sentence)
            summary[a] += 1
            if a.startswith("ambiguous"):
                ambiguity_rows.append([name, cat, a, r.get(ctrl[0]), r.get(ctrl[1]), r.get(ctrl[2]),
                                   str(r.get(k_sentence))[:300] if k_sentence else ""])
        out.append(f"      {cat:<18} " + ", ".join(f"{a} {b}" for a, b in summary.most_common()))


# --------------------------------------------------------------------------
# F. Hardware, duration, calls
# --------------------------------------------------------------------------
MANIFEST_PATTERN = re.compile(r"time|duration|elapsed|gpu|device|host|node|start|end|date|calls|n_", re.I)
SLURM_PATTERN = re.compile(r"NVIDIA|A100|GPU|CUDA_VISIBLE|Elapsed|elapsed|SLURM_JOB_ID|JobId|"
                         r"^real\s|Start|End|duration", re.I)


def walk(obj, path="", out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            walk(v, f"{path}[{i}]", out)
    else:
        if MANIFEST_PATTERN.search(path.split(".")[-1]):
            out.append(f"{path} = {str(obj)[:100]}")
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--run", default="output/run18")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--graph-before", default=None,
                    help="default: <run>/kg_final/flat_run18_enforced_dedup.json")
    ap.add_argument("--folder-188", default="output/audit/188_nettoyage_v2")
    ap.add_argument("--provenance-folders", nargs="*",
                    default=["output/audit/184_trace_nfkc_v2", "output/audit/187_provenance"])
    ap.add_argument("--folder-186", default="output/audit/186_scope")
    ap.add_argument("--records", nargs="*", default=None,
                    help="files of the perturbed records; default: search under output/")
    ap.add_argument("--expected", nargs=2, type=int, default=[364, 364],
                    help="expected number of originals and of perturbed records")
    ap.add_argument("--slurm-glob", nargs="*", default=["slurm*.out", "logs/**/*.out",
                                                          "slurm/**/*.out", "slurm/**/*.err"])
    a = ap.parse_args()

    R = lambda x: x if os.path.isabs(x) else os.path.join(a.repo, x)
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} already exists: choose another --outdir (no output is overwritten).")
    os.makedirs(outdir)

    report, schemas = [], []
    report.append("189_manuscript_elements: read-only. Numbers to be checked against the outputs quoted.")

    # ---- B. graphs
    report.append("\n=== B. Graphs ===")
    before = R(a.graph_before or os.path.join(a.run, "kg_final", "flat_run18_enforced_dedup.json"))
    graphs = []
    if os.path.exists(before):
        graphs.append(("before cleaning: " + os.path.relpath(before, a.repo), load(before)))
    else:
        report.append(f"delivered graph not found: {before}")
    for f in sorted(glob.glob(os.path.join(R(a.folder_188), "**", "*"), recursive=True)):
        if os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".csv", ".tsv"):
            recs = load(f)
            if recs and all(triple_keys(recs)):
                graphs.append(("output of 188: " + os.path.relpath(f, a.repo), recs))
    provenance_sets = {}
    for d in a.provenance_folders:
        for f in sorted(glob.glob(os.path.join(R(d), "**", "*"), recursive=True)):
            if os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".csv", ".tsv"):
                recs = load(f)
                tr = triple_keys(recs) if recs else (None, None, None)
                if recs and all(tr):
                    provenance_sets[os.path.relpath(f, a.repo)] = (recs, *tr)
                    schemas += schema("provenance " + os.path.relpath(f, a.repo), recs)
    lb = lb13(a.repo)
    report.append(f"LB2019 descriptors loaded: {len(lb) if lb else 'no'}")
    graph_rows = []
    for name, recs in graphs:
        schemas += schema(name, recs)
        st = graph_stats(name, recs, lb, provenance_sets, report)
        if st:
            graph_rows.append(st)
    if graph_rows:
        cols = sorted({k for d in graph_rows for k in d})
        with open(os.path.join(outdir, "graphs.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(graph_rows)

    # ---- C and D. perturbed records
    report.append("\n=== C and D. Perturbed records ===")
    if a.records:
        files = [R(x) for x in a.records]
    else:
        files = [f for f in glob.glob(os.path.join(R("output"), "**", "*"), recursive=True)
                    if RECORDS_PATTERN.search(os.path.basename(f))
                    and os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".csv", ".tsv")
                    and not os.path.abspath(f).startswith(os.path.abspath(outdir))]
    csv_rows, ambiguity_rows = [], []
    if not files:
        report.append("no record file found: run again with --records <file>")
    for f in sorted(files):
        recs = load(f)
        if not recs:
            continue
        name = os.path.relpath(f, a.repo)
        schemas += schema("records " + name, recs)
        analyse_records(name, recs, report, csv_rows, ambiguity_rows, tuple(a.expected))
    with open(os.path.join(outdir, "records_per_type.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "verdict_field", "category", "n", "n_rejected", "rejection_rate",
                    "wilson_low", "wilson_high"])
        w.writerows(csv_rows)
    with open(os.path.join(outdir, "ambiguities.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "category", "status", "subject", "relation", "object", "sentence"])
        w.writerows(ambiguity_rows)

    # ---- E. qualification markers and existing summaries
    report.append("\n=== E. Qualification markers (186) and existing summaries ===")
    for f in sorted(glob.glob(os.path.join(R(a.folder_186), "**", "*"), recursive=True)):
        if os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".csv", ".tsv"):
            recs = load(f)
            if not recs:
                continue
            name = os.path.relpath(f, a.repo)
            schemas += schema("186 " + name, recs)
            report.append(f"[{name}] {len(recs)} records")
            for k, vals in categorical_fields(recs, all_keys(recs)).items():
                report.append(f"   {k}: " + ", ".join(f"{x} {n}" for x, n in vals.most_common()))
    for d in [a.folder_186, a.folder_188] + list(a.provenance_folders):
        for f in sorted(glob.glob(os.path.join(R(d), "_*.txt"))):
            report.append(f"--- copy of {os.path.relpath(f, a.repo)} (first 80 lines)")
            with open(f, encoding="utf-8", errors="replace") as h:
                report += [l.rstrip("\n") for l in h.readlines()[:80]]

    # ---- F. hardware, duration, calls
    report.append("\n=== F. Hardware, duration, calls ===")
    for f in sorted(glob.glob(os.path.join(R(a.run), "**", "*manifest*"), recursive=True))[:10]:
        try:
            with open(f, encoding="utf-8") as h:
                obj = json.load(h)
            found = walk(obj)
            report.append(f"[{os.path.relpath(f, a.repo)}] {len(found)} relevant fields")
            report += ["   " + t for t in found[:60]]
        except Exception as e:
            report.append(f"[{os.path.relpath(f, a.repo)}] not read: {e}")
    seen_logs = 0
    for pattern in a.slurm_glob:
        for f in sorted(glob.glob(R(pattern), recursive=True)):
            if seen_logs >= 40:
                break
            seen_logs += 1
            with open(f, encoding="utf-8", errors="replace") as h:
                lines = [l.strip() for l in h if SLURM_PATTERN.search(l)][:8]
            if lines:
                report.append(f"[{os.path.relpath(f, a.repo)}]")
                report += ["   " + l[:160] for l in lines]
    report.append("For the exact duration and the GPU of a job: "
                  "sacct -j <JOBID> --format=JobID,JobName%30,Elapsed,AllocTRES%60,NodeList")

    with open(os.path.join(outdir, "_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(os.path.join(outdir, "schemas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(schemas) + "\n")
    print("\n".join(report))
    print(f"\nOutputs: {outdir}/_summary.txt, schemas.txt, graphs.csv, "
          f"records_per_type.csv, ambiguities.csv")


if __name__ == "__main__":
    main()