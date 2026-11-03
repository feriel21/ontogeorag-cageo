#!/usr/bin/env python3
"""scripts/tools/compare_runs.py -- compare two output folders of the run18 pipeline.

Read-only, CPU only. Checks a rerun (--new) against the recorded run18
outputs distributed in results/run18/ (--ref), so that anyone who reruns the
pipeline can compare the results themselves. Nothing is modified in either
folder; the report goes to --out, which must not exist yet.

Both folders have the layout written by workflows/run_pipeline_run18.sh or
slurm/run18_clean.sh:
    pass_a/ pass_b/            raw, repaired, verified, cleaned triples
    kg/tiered_kg_<run>.json    fused graph before the judges
    m4_panel/                  judges' decisions and quarantine
    kg_final/flat_<run>_enforced_dedup.json   delivered graph
    kg_final/article_metrics.json

Usage:
    python scripts/tools/compare_runs.py --ref results/run18 \
        --new output/run18_rerun --out output/compare_run18

Report:
    <out>/compare_report.txt          everything printed on screen
    <out>/relations_only_ref.tsv   delivered relations only in --ref
    <out>/relations_only_new.tsv   delivered relations only in --new
    <out>/metrics_side_by_side.tsv        article_metrics.json side by side

A relation is identified by (subject, relation, object), lower-cased and with
spaces normalised. Two runs are not expected to be identical: pass b samples
at temperature 0.3, and greedy decoding in bf16 can differ on another GPU.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def norm(x):
    return " ".join(str(x or "").lower().split())


def key(t):
    s = t.get("subject", t.get("head", ""))
    r = t.get("relation", t.get("predicate", ""))
    o = t.get("object", t.get("tail", ""))
    return (norm(s), norm(r), norm(o))


def read_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def count_lines(p):
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def sha12(p):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:12]


def find_one(folder, pattern):
    hits = sorted(folder.glob(pattern))
    return hits[0] if hits else None


def load_flat(folder):
    p = find_one(folder, "kg_final/flat_*_enforced_dedup.json")
    if p is None:
        return None, None
    d = json.loads(p.read_text(encoding="utf-8"))
    triples = d["triples"] if isinstance(d, dict) and "triples" in d else d
    return p, triples


def leaves(obj, prefix=""):
    """Flatten numeric and string leaves of a JSON object into {path: value}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(leaves(v, prefix + "/" + str(k) if prefix else str(k)))
    elif isinstance(obj, list):
        if len(obj) <= 30 and all(not isinstance(v, (dict, list)) for v in obj):
            out[prefix] = json.dumps(obj, ensure_ascii=False)
        else:
            out[prefix] = "[list of {}]".format(len(obj))
    else:
        out[prefix] = obj
    return out


def fmt(v):
    if v is None:
        return "absent"
    if isinstance(v, float):
        return "{:.4f}".format(v)
    return str(v)


def row(label, a, b):
    diff = ""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = b - a
        diff = "{:+d}".format(d) if isinstance(d, int) else "{:+.4f}".format(d)
        if a:
            diff += "  ({:+.0%})".format(d / float(a))
    say("  {:<46} {:>12} {:>12}   {}".format(label, fmt(a), fmt(b), diff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference folder, e.g. results/run18")
    ap.add_argument("--new", required=True, help="rerun output folder to check")
    ap.add_argument("--out", required=True, help="report folder (must not exist)")
    ap.add_argument("--examples", type=int, default=15,
                    help="relations listed on screen for each side (default 15)")
    args = ap.parse_args()

    ref = Path(args.ref).expanduser()
    new = Path(args.new).expanduser()
    out = Path(args.out).expanduser()
    for p in (ref, new):
        if not p.is_dir():
            print("STOP: folder not found: {}".format(p), file=sys.stderr)
            return 2
    if out.exists():
        print("STOP: {} exists, nothing is overwritten.".format(out), file=sys.stderr)
        return 2

    say("=" * 78)
    say("compare_runs -- two run18 output folders side by side (read-only)")
    say("=" * 78)
    say("ref : {}".format(ref))
    say("new : {}".format(new))
    say()

    # ---- 1. counts at each stage -----------------------------------
    say("--- 1. NUMBER OF RECORDS AT EACH STAGE " + "-" * 39)
    say("  {:<46} {:>12} {:>12}   {}".format("file", "ref", "new", "new - ref"))
    stages = []
    for P in ("a", "b"):
        stages += [
            "pass_{}/raw_triples.jsonl".format(P),
            "pass_{}/triples_repaired.jsonl".format(P),
            "pass_{}/verified_triples.jsonl".format(P),
            "pass_{}/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl".format(P),
        ]
    stages += ["m4_panel/m4_quarantine.jsonl"]
    for s in stages:
        row(s, count_lines(ref / s), count_lines(new / s))
    say()

    # ---- 2. pass a: same extraction? --------------------------------
    say("--- 2. PASS A (temperature 0) " + "-" * 48)
    pa_ref, pa_new = ref / "pass_a/raw_triples.jsonl", new / "pass_a/raw_triples.jsonl"
    h_ref, h_new = sha12(pa_ref), sha12(pa_new)
    say("  sha256 raw_triples   ref {}   new {}   {}".format(
        fmt(h_ref), fmt(h_new),
        "IDENTICAL" if h_ref and h_ref == h_new else "different"))
    if pa_ref.exists() and pa_new.exists():
        kr = collections.Counter(key(t) for t in read_jsonl(pa_ref))
        kn = collections.Counter(key(t) for t in read_jsonl(pa_new))
        common = set(kr) & set(kn)
        union = set(kr) | set(kn)
        say("  distinct (s, r, o)   ref {}   new {}   common {}   Jaccard {:.3f}".format(
            len(kr), len(kn), len(common), len(common) / float(len(union)) if union else 0.0))
    say()

    # ---- 3. delivered graph ------------------------------------------
    say("--- 3. DELIVERED GRAPH (kg_final/flat_*_enforced_dedup.json) " + "-" * 16)
    p_ref, t_ref = load_flat(ref)
    p_new, t_new = load_flat(new)
    say("  ref file: {}".format(p_ref))
    say("  new file: {}".format(p_new))
    if t_ref is None or t_new is None:
        say("  [!] delivered graph missing on one side; section skipped.")
    else:
        say("  {:<46} {:>12} {:>12}   {}".format("", "ref", "new", "new - ref"))
        row("relations", len(t_ref), len(t_new))
        tr = collections.Counter(str(t.get("tier")) for t in t_ref)
        tn = collections.Counter(str(t.get("tier")) for t in t_new)
        for tier in sorted(set(tr) | set(tn)):
            row("tier {}".format(tier), tr.get(tier, 0), tn.get(tier, 0))
        dr = collections.Counter(str((t.get("m4") or {}).get("decision") if isinstance(t.get("m4"), dict) else None) for t in t_ref)
        dn = collections.Counter(str((t.get("m4") or {}).get("decision") if isinstance(t.get("m4"), dict) else None) for t in t_new)
        for dec in sorted(set(dr) | set(dn)):
            row("judges' decision {}".format(dec), dr.get(dec, 0), dn.get(dec, 0))
        er = {norm(t.get(k)) for t in t_ref for k in ("subject", "object")}
        en = {norm(t.get(k)) for t in t_new for k in ("subject", "object")}
        row("distinct entities", len(er), len(en))
        docs_r = collections.Counter()
        docs_n = collections.Counter()
        for t, c in [(x, docs_r) for x in t_ref] + [(x, docs_n) for x in t_new]:
            for d in (t.get("supporting_papers") or []):
                c[str(d)] += 1
        if docs_r or docs_n:
            row("articles cited as support", len(docs_r), len(docs_n))
        say()

        say("  relation types:")
        rr = collections.Counter(norm(t.get("relation")) for t in t_ref)
        rn = collections.Counter(norm(t.get("relation")) for t in t_new)
        for r in sorted(set(rr) | set(rn), key=lambda x: -(rr.get(x, 0) + rn.get(x, 0))):
            row("  " + r, rr.get(r, 0), rn.get(r, 0))
        say()

        kr = {key(t): t for t in t_ref}
        kn = {key(t): t for t in t_new}
        common = set(kr) & set(kn)
        only_r = sorted(set(kr) - set(kn))
        only_n = sorted(set(kn) - set(kr))
        union = set(kr) | set(kn)
        say("  same (s, r, o) in both : {}   only ref : {}   only new : {}   Jaccard {:.3f}".format(
            len(common), len(only_r), len(only_n), len(common) / float(len(union)) if union else 0.0))
        t1r = {k for k, t in kr.items() if str(t.get("tier")) == "1"}
        t1n = {k for k, t in kn.items() if str(t.get("tier")) == "1"}
        if t1r or t1n:
            say("  tier 1 in both         : {}   (ref tier 1: {}, new tier 1: {})".format(
                len(t1r & t1n), len(t1r), len(t1n)))
        same_tier = sum(1 for k in common if str(kr[k].get("tier")) == str(kn[k].get("tier")))
        say("  common relations with the same tier: {} of {}".format(same_tier, len(common)))
        say()

        out.mkdir(parents=True)
        for name, keys, src in (("relations_only_ref.tsv", only_r, kr),
                                ("relations_only_new.tsv", only_n, kn)):
            with open(out / name, "w", encoding="utf-8") as f:
                f.write("subject\trelation\tobject\ttier\tevidence\n")
                for k in keys:
                    ev = " ".join(str(src[k].get("evidence", "")).split())[:300]
                    f.write("\t".join(list(k) + [str(src[k].get("tier")), ev]) + "\n")
        for title, keys, src in (("only in ref", only_r, kr), ("only in new", only_n, kn)):
            say("  examples {} (first {}):".format(title, min(args.examples, len(keys))))
            for k in keys[:args.examples]:
                say("    [tier {}] {} --{}--> {}".format(src[k].get("tier"), k[0], k[1], k[2]))
            say()

    # ---- 4. article_metrics.json ---------------------------------------
    say("--- 4. kg_final/article_metrics.json " + "-" * 41)
    m_ref, m_new = ref / "kg_final/article_metrics.json", new / "kg_final/article_metrics.json"
    if not (m_ref.exists() and m_new.exists()):
        say("  [!] article_metrics.json missing on one side; section skipped.")
    else:
        lr = leaves(json.loads(m_ref.read_text(encoding="utf-8")))
        ln = leaves(json.loads(m_new.read_text(encoding="utf-8")))
        out.mkdir(parents=True, exist_ok=True)
        n_diff = 0
        with open(out / "metrics_side_by_side.tsv", "w", encoding="utf-8") as f:
            f.write("metric\tref\tnew\tsame\n")
            for k in sorted(set(lr) | set(ln)):
                a, b = lr.get(k), ln.get(k)
                same = a == b
                n_diff += 0 if same else 1
                f.write("{}\t{}\t{}\t{}\n".format(k, fmt(a), fmt(b), "yes" if same else "no"))
        say("  {} values compared, {} differ. Differences:".format(len(set(lr) | set(ln)), n_diff))
        for k in sorted(set(lr) | set(ln)):
            a, b = lr.get(k), ln.get(k)
            if a != b and not (isinstance(a, str) and a.startswith("[list")):
                row(k[-46:], a, b)
    say()

    # ---- 5. manifests ---------------------------------------------------
    say("--- 5. MANIFEST " + "-" * 62)
    for side, folder in (("ref", ref), ("new", new)):
        mp = folder / "manifest.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            say("  {}: {} steps, commit {}, PYTHONHASHSEED {}".format(
                side, len(m.get("steps", [])), str(m.get("git_commit", "?"))[:10],
                m.get("pythonhashseed", "not recorded")))
        else:
            say("  {}: manifest.json absent".format(side))
    say()

    out.mkdir(parents=True, exist_ok=True)
    (out / "compare_report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("Written: {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
