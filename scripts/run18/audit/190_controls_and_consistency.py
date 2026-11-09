#!/usr/bin/env python3
"""190_controls_and_consistency.py: read-only, no model, no run.

Completes 189 on the points left empty and on new points:

  1. Consistency of the delivered graph and of the cleaned graph:
     tier x tier_label x tier_original, tier x verdict x m4.decision,
     m4.flags x verdict, relations whose content key is duplicated.
  2. Files of the perturbed records: inventory of the stress_test and 181 folders
     (size, sha, number of records, fields), identity of the controls.jsonl files.
  3. Verdicts of the verifying-LLMs on the 728 records: search for the files that
     hold them, join on common fields (never on line order without saying so),
     full distribution of the values per perturbation type.
  4. Ambiguous records per perturbation type, from supporting_passage and from
     the original relation found through orig_index:
       passage_perm : does the swapped passage already name the subject and the object?
       entity_sub   : does the introduced entity appear in the passage?
       relation_sub : original relation -> introduced relation (matrix)
       inversion    : inverted relations
  5. Manifest steps with their duration, and sacct if job numbers are given.

Writes only into --outdir and refuses to overwrite an existing folder.

Usage (from the run18 worktree):
  python ~/ontogeorag/190_controls_and_consistency.py --repo . \
      --outdir output/audit/190_controls_and_consistency
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

VERDICT = re.compile(r"SUPPORT|ACCEPT|REJECT|UNCERTAIN|PLAUSIBLE|NOT_|IMPLAUSIBLE|KEEP|DROP|DISCARD",
                     re.I)
EXT = (".json", ".jsonl", ".csv", ".tsv")
MAX_MOD = 12

FAMILIES = {
    "causal": {"causes", "triggers", "controls", "affects", "formedby"},
    "stratigraphic": {"overlies", "underlies"},
    "spatial": {"occursin", "partof"},
}


def norm(x):
    if x is None:
        return ""
    s = unicodedata.normalize("NFKC", str(x)).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def nrel(x):
    return norm(x).replace(" ", "")


def family(r):
    for f, members in FAMILIES.items():
        if nrel(r) in members:
            return f
    return nrel(r)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def flatten(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = json.dumps(v2, ensure_ascii=False) if isinstance(v2, (dict, list)) else v2
        elif isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def load(path):
    """Return (list of flattened records, raw object if the JSON is not tabular)."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".jsonl":
            with open(path, encoding="utf-8") as f:
                return [flatten(json.loads(l)) for l in f if l.strip().startswith("{")], None
        if ext == ".json":
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [flatten(r) for r in obj if isinstance(r, dict)], None
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return [flatten(r) for r in v if isinstance(r, dict)], obj
            return [], obj
        if ext in (".csv", ".tsv"):
            with open(path, encoding="utf-8", newline="") as f:
                return [dict(r) for r in csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ",")], None
    except Exception as e:
        print(f"  [unreadable] {path}: {e}", file=sys.stderr)
    return [], None


def keys(recs):
    seen = []
    for r in recs[:500]:
        for k in r:
            if k not in seen:
                seen.append(k)
    return seen


def scalars(obj, path="", out=None, limit=80):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            scalars(v, f"{path}.{k}" if path else k, out, limit)
    elif isinstance(obj, list):
        if obj and not isinstance(obj[0], (dict, list)):
            out.append(f"{path} = {str(obj)[:120]}")
        else:
            for i, v in enumerate(obj[:5]):
                scalars(v, f"{path}[{i}]", out, limit)
    else:
        out.append(f"{path} = {str(obj)[:120]}")
    return out


def table(recs, a, b, out, title):
    t = Counter((str(r.get(a)), str(r.get(b))) for r in recs)
    out.append(f"   {title} ({a} x {b}):")
    for (x, y), n in sorted(t.items()):
        out.append(f"      {x:<22} {y:<35} {n}")


# --------------------------------------------------------------------------
def consistency(name, path, out):
    recs, _ = load(path)
    if not recs:
        out.append(f"[{name}] not found or empty: {path}")
        return
    out.append(f"[{name}] {path}: {len(recs)} relations (sha {sha(path)})")
    table(recs, "tier", "tier_label", out, "tier and label")
    table(recs, "tier", "tier_original", out, "tier and original tier")
    t = Counter((str(r.get("tier")), str(r.get("verdict")), str(r.get("m4.decision"))) for r in recs)
    out.append("   tier x verdict x m4.decision:")
    for k, n in sorted(t.items()):
        out.append(f"      {' | '.join(k):<45} {n}")
    table(recs, "m4.flags", "verdict", out, "flags and verdict")
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        groups[(norm(r.get("subject")), nrel(r.get("relation")), norm(r.get("object")))].append(r)
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    out.append(f"   duplicated content keys after normalisation: {len(duplicates)}")
    for k, v in duplicates.items():
        for r in v:
            out.append(f"      {r.get('subject')!r} --{r.get('relation')}--> {r.get('object')!r} "
                       f"| tier {r.get('tier')} | origin {r.get('origin')} | decision {r.get('m4.decision')}")


# --------------------------------------------------------------------------
def inventory(folders, out):
    files = []
    for d in folders:
        if not os.path.isdir(d):
            out.append(f"[absent] {d}")
            continue
        for f in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if not os.path.isfile(f):
                continue
            size = os.path.getsize(f)
            line = f"   {f} | {size} B | sha {sha(f)}"
            if os.path.splitext(f)[1].lower() in EXT and size < 300_000_000:
                recs, raw = load(f)
                line += f" | {len(recs)} records"
                if recs:
                    line += " | fields: " + ", ".join(keys(recs))[:400]
                files.append((f, recs, raw))
            out.append(line)
    return files


def verdict_fields(recs):
    res = []
    for k in keys(recs):
        vals = Counter(str(r.get(k)) for r in recs)
        if 1 < len(vals) <= MAX_MOD * 3 and any(VERDICT.search(v) for v in vals):
            res.append(k)
    return res


def join_fields(records, recs):
    """Common fields usable for a join, from the most to the least reliable."""
    kc, kr = set(keys(records)), set(keys(recs))
    for fieldset in (["orig_index", "label", "corruption_class", "subject", "relation", "object"],
                ["orig_index", "label", "corruption_class"],
                ["orig_index", "label"],
                ["subject", "relation", "object", "label"],
                ["subject", "relation", "object"]):
        if all(c in kc and c in kr for c in fieldset):
            return fieldset
    return None


def key_value(r, fieldset):
    return tuple(nrel(r.get(c)) if c == "relation" else norm(r.get(c)) for c in fieldset)


def verdicts(records, files, records_path, out, rows):
    found = False
    for f, recs, raw in files:
        if os.path.abspath(f) == os.path.abspath(records_path):
            continue
        if not recs:
            if raw is not None:
                out.append(f"   [json summary] {f}")
                out += ["      " + s for s in scalars(raw)]
            continue
        kv = verdict_fields(recs)
        if not kv:
            if raw is not None:
                out.append(f"   [json summary] {f}")
                out += ["      " + s for s in scalars(raw)]
            continue
        fieldset = join_fields(records, recs)
        mode = "fields " + ",".join(fieldset) if fieldset else None
        if fieldset:
            idx = defaultdict(list)
            for r in recs:
                idx[key_value(r, fieldset)].append(r)
            pairs = []
            ambiguous = 0
            for c in records:
                m = idx.get(key_value(c, fieldset), [])
                if len(m) == 1:
                    pairs.append((c, m[0]))
                elif len(m) > 1:
                    ambiguous += 1
            out.append(f"   [verdicts] {f}: {len(recs)} rec.; join on {mode}: "
                       f"{len(pairs)}/{len(records)} unique, {ambiguous} multiple")
        elif len(recs) == len(records):
            pairs = list(zip(records, recs))
            mode = "LINE ORDER (no common field): to be confirmed before quoting"
            out.append(f"   [verdicts] {f}: {len(recs)} rec.; join by {mode}")
        else:
            out.append(f"   [verdicts not joinable] {f}: {len(recs)} rec., fields {keys(recs)[:15]}")
            continue
        if not pairs:
            continue
        found = True
        for k in kv:
            by_category = defaultdict(Counter)
            for c, r in pairs:
                cat = "original" if str(c.get("label")) == "original" else str(c.get("corruption_class"))
                by_category[cat][str(r.get(k))] += 1
            out.append(f"      -- {k}")
            for cat in sorted(by_category):
                n = sum(by_category[cat].values())
                out.append(f"         {cat:<14} n={n:<4} " +
                           ", ".join(f"{v} {x}" for v, x in by_category[cat].most_common()))
                for v, x in by_category[cat].items():
                    rows.append([f, mode, k, cat, n, v, x])
    if not found:
        out.append("   NO joinable verdict file: send the list above, "
                   "or run again with --verdicts <file>")


# --------------------------------------------------------------------------
def ambiguities(records, out, rows):
    originals = {str(c.get("orig_index")): c for c in records if str(c.get("label")) == "original"}
    pert = [c for c in records if str(c.get("label")) != "original"]
    out.append(f"   originals {len(originals)}, perturbed {len(pert)}; "
               f"perturbed without a matching original: "
               f"{sum(1 for c in pert if str(c.get('orig_index')) not in originals)}")
    summary = defaultdict(Counter)
    matrix = Counter()
    inv = Counter()
    for c in pert:
        cat = str(c.get("corruption_class"))
        o = originals.get(str(c.get("orig_index")))
        s, p, ob = norm(c.get("subject")), nrel(c.get("relation")), norm(c.get("object"))
        passage = norm(c.get("supporting_passage"))
        status = "not assessed"
        detail = ""
        if o is None:
            status = "original not found"
        elif cat == "passage_perm":
            p0 = norm(o.get("supporting_passage"))
            if passage == p0:
                status = "passage identical to the original"
            else:
                ns, no = s in passage, ob in passage
                status = {(True, True): "ambiguous (subject and object in the new passage)",
                          (True, False): "subject only in the new passage",
                          (False, True): "object only in the new passage",
                          (False, False): "not ambiguous"}[(ns, no)]
        elif cat == "entity_sub":
            s0, o0 = norm(o.get("subject")), norm(o.get("object"))
            new = [e for e, e0 in ((s, s0), (ob, o0)) if e != e0]
            detail = " / ".join(new)
            if not new:
                status = "no entity changed"
            elif all(e in passage for e in new):
                status = "ambiguous (introduced entity present in the passage)"
            else:
                status = "not ambiguous"
        elif cat == "relation_sub":
            r0 = nrel(o.get("relation"))
            matrix[(o.get("relation"), c.get("relation"))] += 1
            detail = f"{o.get('relation')} -> {c.get('relation')}"
            status = "ambiguous (same family)" if family(r0) == family(p) and r0 != p else "not ambiguous"
        elif cat == "inversion":
            inv[str(c.get("relation"))] += 1
            status = "to judge (directed relation?)"
        summary[cat][status] += 1
        rows.append([cat, status, detail, c.get("orig_index"), c.get("subject"), c.get("relation"),
                       c.get("object"), str(c.get("supporting_passage"))[:300]])
    for cat in sorted(summary):
        out.append(f"   {cat:<14} " + ", ".join(f"{s} {n}" for s, n in summary[cat].most_common()))
    out.append("   relation_sub, original relation -> introduced relation:")
    for (a, b), n in matrix.most_common():
        out.append(f"      {a} -> {b}: {n}")
    out.append("   inversion, inverted relations: " + ", ".join(f"{a} {n}" for a, n in inv.most_common()))
    out.append("   relations present in the records: " +
               ", ".join(f"{a} {n}" for a, n in Counter(str(c.get('relation')) for c in records).most_common()))
    out.append("   Lexical indicator (substring after normalisation): read ambiguities.csv before quoting.")


# --------------------------------------------------------------------------
def manifest(path, out):
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:
        out.append(f"manifest not read: {path}: {e}")
        return
    out.append(f"[{path}] sha {sha(path)}")
    for k, v in m.items():
        if not isinstance(v, (dict, list)):
            out.append(f"   {k} = {str(v)[:150]}")
        elif isinstance(v, dict):
            out += ["   " + s for s in scalars(v, k, limit=40)]
    start, end = None, None
    for i, st in enumerate(m.get("steps", [])):
        if not isinstance(st, dict):
            continue
        desc = ", ".join(f"{k}={str(v)[:70]}" for k, v in st.items()
                         if k not in ("started", "ended") and not isinstance(v, (dict, list)))
        duration = ""
        try:
            a = datetime.fromisoformat(st["started"])
            b = datetime.fromisoformat(st["ended"])
            start = start or a
            end = b
            s = int((b - a).total_seconds())
            duration = f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"
        except Exception:
            pass
        out.append(f"   step {i} | {duration:<10} | {desc}")
    if start and end:
        s = int((end - start).total_seconds())
        out.append(f"   from the first start to the last end: {s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--graph-before", default="output/run18/kg_final/flat_run18_enforced_dedup.json")
    ap.add_argument("--graph-after", default="output/audit/188_nettoyage_v2/graphe_nettoye.json")
    ap.add_argument("--records", default="output/run18/stress_test/controls.jsonl")
    ap.add_argument("--folders", nargs="*", default=["output/run18/stress_test",
                                                      "output/audit/181_rules_llama",
                                                      "output/audit/181_rules_mistral"])
    ap.add_argument("--verdicts", nargs="*", default=None,
                    help="verdict files to join (otherwise: all those of --folders)")
    ap.add_argument("--manifest", default="output/run18/manifest.json")
    ap.add_argument("--jobs", nargs="*", default=[], help="SLURM job numbers for sacct")
    a = ap.parse_args()

    R = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(a.repo, x))
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} already exists: choose another --outdir (no output is overwritten).")
    os.makedirs(outdir)
    out = ["190_controls_and_consistency: read-only."]

    out.append("\n=== 1. Consistency of the graphs ===")
    consistency("before cleaning", R(a.graph_before), out)
    consistency("after cleaning", R(a.graph_after), out)

    out.append("\n=== 2. Inventory of the record folders ===")
    files = inventory([R(d) for d in a.folders], out)
    shas = defaultdict(list)
    for f, _, _ in files:
        if os.path.basename(f) == "controls.jsonl":
            shas[sha(f)].append(f)
    out.append("   identity of the controls.jsonl files: " +
               "; ".join(f"sha {h}: {len(v)} file(s)" for h, v in shas.items()))

    records_path = R(a.records)
    records, _ = load(records_path)
    out.append(f"\n=== 3. Verdicts on the records ({records_path}, {len(records)} rec.) ===")
    verdict_rows = []
    if a.verdicts:
        verdict_files = [(R(f), *load(R(f))) for f in a.verdicts]
    else:
        verdict_files = files
    if records:
        verdicts(records, verdict_files, records_path, out, verdict_rows)
    with open(os.path.join(outdir, "verdicts_per_type.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "join", "field", "category", "n_category", "value", "count"])
        w.writerows(verdict_rows)

    out.append("\n=== 4. Ambiguous records per perturbation type ===")
    ambiguity_rows = []
    if records:
        ambiguities(records, out, ambiguity_rows)
    with open(os.path.join(outdir, "ambiguities.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "status", "detail", "orig_index", "subject", "relation", "object", "passage"])
        w.writerows(ambiguity_rows)

    out.append("\n=== 5. Steps, durations, hardware ===")
    manifest(R(a.manifest), out)
    for j in a.jobs:
        try:
            r = subprocess.run(["sacct", "-j", j, "--format=JobID,JobName%30,Elapsed,AllocTRES%70,NodeList%20,State"],
                               capture_output=True, text=True, timeout=60)
            out.append(f"[sacct {j}]")
            out += ["   " + l for l in r.stdout.splitlines()]
        except Exception as e:
            out.append(f"[sacct {j}] failed: {e}")

    with open(os.path.join(outdir, "_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nOutputs: {outdir}/_summary.txt, verdicts_per_type.csv, ambiguities.csv")


if __name__ == "__main__":
    main()