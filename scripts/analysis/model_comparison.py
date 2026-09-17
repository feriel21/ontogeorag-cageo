#!/usr/bin/env python3
"""Controlled comparison of three extraction models (Table 4).

Every argument of the production extractor is held identical to the delivered
run except --model and --output:

    index      run16E index (1,758 chunks)
    queries    configs/descriptor_queries_v2.jsonl (287 queries)
    schema     configs/ontology_schema.json
    retrieval  --bm25-topn 20 --min-bm25 2.0 --top-k 5 --max-chars 2800
               --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
    decoding   --temperature 0.0, one pass

A single pass at temperature 0.0 removes sampling noise, so differences come
from the models. Outputs are compared as raw extraction, before verification,
because the primary verifier is one of the compared models.

Residual confound: the instruction template was developed with Qwen in the
loop, which favours that model. The comparison removes every other difference.

Stages
    --stage plan     resolve model snapshots and write the commands; run nothing
    --stage run      run the three extractions sequentially (GPU)
    --stage report   score the outputs and write the LaTeX table
    --stage all      run, then report

Outputs (under output/audit/): 95p_model_comparison_report.json,
tab_model_comparison.tex, 95p_commands.sh. The commands actually executed for
the published table are kept as model_comparison_commands.sh.

Default paths follow the original layout (~/ontogeorag for code and
configuration, ~/ontogeorag_D for the run16E index); --code-root selects the
code checkout used for extraction.

Historical name: 95p_comparaison_controlee_modeles.py

Usage
    python scripts/analysis/model_comparison.py --stage plan
    python scripts/analysis/model_comparison.py --stage run
    python scripts/analysis/model_comparison.py --stage report
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
SCRIPTS_ROOT = HOME / "ontogeorag"
DATA_ROOT = HOME / "ontogeorag_D"
HF_HUB = HOME / ".cache" / "huggingface" / "hub"

EXTRACTOR = SCRIPTS_ROOT / "pipeline" / "02_extract_triples.py"
INDEX_DIR = DATA_ROOT / "output" / "run16E" / "index"
QUERIES = SCRIPTS_ROOT / "configs" / "descriptor_queries_v2.jsonl"
SCHEMA = SCRIPTS_ROOT / "configs" / "ontology_schema.json"
OUT_ROOT = DATA_ROOT / "output" / "model_comparison"

BENCH_26 = SCRIPTS_ROOT / "configs" / "lb_reference_edges_original26.json"
BENCH_34 = SCRIPTS_ROOT / "configs" / "lb_reference_edges.json"

MODELS = [
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "qwen"),
    ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "llama"),
    ("Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2", "mistral"),
]

FIXED = ["--schema", str(SCHEMA),
         "--queries", str(QUERIES),
         "--index-dir", str(INDEX_DIR),
         "--backend", "hf",
         "--top-k", "5",
         "--bm25-topn", "20",
         "--min-bm25", "2.0",
         "--max-chars", "2800",
         "--temperature", "0.0",
         "--reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"]

OUT_DIR = SCRIPTS_ROOT / "output" / "audit"
OUT_JSON = OUT_DIR / "95p_model_comparison_report.json"
OUT_TEX = OUT_DIR / "tab_model_comparison.tex"
OUT_SH = OUT_DIR / "95p_commands.sh"

MANIFESTS = [SCRIPTS_ROOT / "output" / "run16E" / "manifest.json",
             DATA_ROOT / "output" / "run16E" / "manifest.json"]

# set by main() from --code-root; the extractor and PYTHONPATH follow it
CODE_ROOT = SCRIPTS_ROOT


def extractor_path():
    return CODE_ROOT / "pipeline" / "02_extract_triples.py"


def sha256_of(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def manifest_extractor_sha():
    """The sha256 the delivered run recorded for its extraction step."""
    for p in MANIFESTS:
        if not p.exists():
            continue
        try:
            man = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for s in man.get("steps") or []:
            if isinstance(s, dict) and str(s.get("step", "")).startswith("extract"):
                return s.get("sha256"), str(p)
    return None, None


def check_code_identity():
    """Compare the extractor on the chosen code root with the delivered run."""
    recorded, man_path = manifest_extractor_sha()
    current = sha256_of(extractor_path())
    print(SUB)
    print("extractor: " + str(extractor_path()))
    print("  sha256 now      " + str(current))
    print("  sha256 in run16E " + str(recorded))
    if recorded and current and recorded == current:
        print("  MATCH: this is the code that produced the delivered run.")
        print("  The comparison can be reported as run with the delivered pipeline.")
        return {"match": True, "current": current, "recorded": recorded, "manifest": man_path}
    print("  DIFFERENT: this code is not the code of the delivered run.")
    print("  The three models still share one code base, so the comparison between")
    print("  them remains valid, but it cannot be attached to run16E. To attach it,")
    print("  run from a clean checkout of the recorded commit:")
    print("    git -C " + str(SCRIPTS_ROOT) + " worktree add /tmp/run16E_code <commit>")
    print("    python3 95p_comparaison_controlee_modeles.py --code-root /tmp/run16E_code --stage run")
    return {"match": False, "current": current, "recorded": recorded, "manifest": man_path}


def child_env():
    """The pipeline package is imported by name, so the repository root must be
    on PYTHONPATH. Running a script by path puts only the script's own
    directory on sys.path, which is why a bare invocation fails."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(CODE_ROOT) + (os.pathsep + existing if existing else "")
    return env

SEP = "=" * 78
SUB = "-" * 78


def norm(s):
    return " ".join(str(s).split()).strip().lower()


def load_pipeline_normalisers():
    """Use the pipeline's own normalisers when available, so that the
    comparison is scored the way the pipeline scores everything else."""
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        c = importlib.import_module("pipeline.rag.constants")
    except Exception:
        return None, None, None
    ne = getattr(c, "normalize_entity", None)
    nr = getattr(c, "normalize_relation", None)
    nd = getattr(c, "normalize_descriptor_fuzzy", None) or getattr(c, "normalize_descriptor", None)
    return ne, nr, nd


def resolve_snapshot(repo_id):
    """Find the local snapshot directory for a hub repo id."""
    folder = "models--" + repo_id.replace("/", "--")
    base = HF_HUB / folder / "snapshots"
    if not base.is_dir():
        return None, []
    snaps = sorted([p for p in base.iterdir() if p.is_dir()])
    if not snaps:
        return None, []
    return snaps[-1], [p.name for p in snaps]


def build_commands():
    cmds = []
    for label, repo, slug in MODELS:
        snap, all_snaps = resolve_snapshot(repo)
        out = OUT_ROOT / slug / "raw_triples.jsonl"
        entry = {"label": label, "repo": repo, "slug": slug,
                 "snapshot": str(snap) if snap else None,
                 "revision": snap.name[:7] if snap else None,
                 "all_snapshots": all_snaps,
                 "output": str(out)}
        if snap is None:
            entry["cmd"] = None
        else:
            entry["cmd"] = ([sys.executable, str(extractor_path())] + FIXED
                            + ["--model", str(snap), "--output", str(out)])
        cmds.append(entry)
    return cmds


def stage_plan(cmds, report):
    print(SEP)
    print("PLAN - identical arguments except --model and --output")
    print(SEP)
    missing = [c for c in cmds if c["cmd"] is None]
    for c in cmds:
        print(SUB)
        print(c["label"])
        print("  repo      " + c["repo"])
        if c["snapshot"] is None:
            print("  snapshot  NOT FOUND under " + str(HF_HUB))
            continue
        print("  snapshot  " + c["snapshot"])
        if len(c["all_snapshots"]) > 1:
            print("  WARNING: several snapshots present, the last was chosen: "
                  + ", ".join(s[:7] for s in c["all_snapshots"]))
        print("  output    " + c["output"])
    print(SUB)
    if missing:
        print("cannot run: no local snapshot for " + ", ".join(c["label"] for c in missing))
        print("  download the snapshot first, or remove that model from MODELS.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["#!/bin/bash", "set -euo pipefail", "",
             "# the pipeline package is imported by name, so the repository root",
             "# must be on PYTHONPATH; running the script by path is not enough",
             "export PYTHONPATH=" + str(CODE_ROOT) + "${PYTHONPATH:+:$PYTHONPATH}", ""]
    for c in cmds:
        if c["cmd"]:
            lines.append("mkdir -p " + str(Path(c["output"]).parent))
            lines.append(" ".join(c["cmd"]))
            lines.append("")
    OUT_SH.write_text("\n".join(lines), encoding="utf-8")
    print("commands written to " + str(OUT_SH))
    print("  review them, then either run that file or use --stage run")
    report["plan"] = cmds
    report["code_identity"] = check_code_identity()


def stage_run(cmds, report):
    print(SEP)
    print("RUN")
    print(SEP)
    timings = {}
    for c in cmds:
        if not c["cmd"]:
            print("skipped (no snapshot): " + c["label"])
            continue
        Path(c["output"]).parent.mkdir(parents=True, exist_ok=True)
        print(SUB)
        print("running " + c["label"])
        print("  " + " ".join(c["cmd"][:6]) + " ...")
        t0 = time.time()
        proc = subprocess.run(c["cmd"], cwd=str(CODE_ROOT), env=child_env())
        dt = time.time() - t0
        timings[c["slug"]] = {"seconds": round(dt, 1), "returncode": proc.returncode}
        print("  exit " + str(proc.returncode) + "   " + str(round(dt, 1)) + " s")
        if proc.returncode != 0:
            print("  -> failed: the output of this model is incomplete and must not be scored.")
    report["timings"] = timings


def as_text(v):
    """Models sometimes emit a list or a nested object where the schema asks
    for a string. Coerce, and let the caller count how often it happens."""
    if isinstance(v, str):
        return v, False
    if isinstance(v, (list, tuple)):
        parts = [as_text(x)[0] for x in v]
        parts = [p for p in parts if p]
        return (parts[0] if parts else ""), True
    if isinstance(v, dict):
        for k in ("name", "label", "text", "value"):
            if isinstance(v.get(k), str):
                return v[k], True
        return "", True
    if v is None:
        return "", False
    return str(v), True


def read_triples(path, ne, nr):
    out = []
    malformed = 0
    p = Path(path)
    if not p.exists():
        return out, malformed
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        s, f1 = as_text(rec.get("source", rec.get("subject", "")))
        r, f2 = as_text(rec.get("relation", ""))
        o, f3 = as_text(rec.get("target", rec.get("object", "")))
        if f1 or f2 or f3:
            malformed += 1
        if not s or not o:
            continue
        s2 = norm(ne(s)) if ne else norm(s)
        r2 = norm(nr(r)) if nr else norm(r)
        o2 = norm(ne(o)) if ne else norm(o)
        out.append((s2, r2, o2))
    return out, malformed


def load_bench(path, ne, nr):
    if not Path(path).exists():
        return set()
    obj = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    edges = obj.get("edges", obj if isinstance(obj, list) else [])
    out = set()
    for e in edges:
        if isinstance(e, dict):
            s, r, o = e.get("subject"), e.get("relation"), e.get("object")
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            s, r, o = e[0], e[1], e[2]
        else:
            continue
        s2 = norm(ne(s)) if ne else norm(s)
        r2 = norm(nr(r)) if nr else norm(r)
        o2 = norm(ne(o)) if ne else norm(o)
        out.add((s2, r2, o2))
    return out


def recall(triples, bench):
    strict = {t for t in bench if t in set(triples)}
    loose = set()
    for bs, br, bo in bench:
        for s, r, o in triples:
            if r != br:
                continue
            if (bs in s or s in bs) and (bo in o or o in bo):
                loose.add((bs, br, bo))
                break
    return len(strict), len(loose)


def stage_report(cmds, report):
    print(SEP)
    print("REPORT")
    print(SEP)
    ne, nr, nd = load_pipeline_normalisers()
    print("normalisers: " + ("pipeline.rag.constants" if ne else "fallback (lowercase, whitespace)"))
    b26 = load_bench(BENCH_26, ne, nr)
    b34 = load_bench(BENCH_34, ne, nr)
    print("benchmarks: " + str(len(b26)) + " and " + str(len(b34)) + " edges")

    rows = []
    sets = {}
    vocab = set()
    try:
        c = importlib.import_module("pipeline.rag.constants")
        vocab = {norm(x) for x in getattr(c, "KNOWN_DESCRIPTORS", [])}
    except Exception:
        pass
    print("declared descriptor vocabulary: " + str(len(vocab)) + " terms")

    # determinism check: the fresh Qwen pass against the delivered pass A
    delivered_a = DATA_ROOT / "output" / "run16E" / "pass_a" / "raw_triples.jsonl"
    ref_a, _ = read_triples(delivered_a, ne, nr) if delivered_a.exists() else ([], 0)

    for c_ in cmds:
        tr, malformed = read_triples(c_["output"], ne, nr)
        if not tr:
            print(SUB)
            print(c_["label"] + ": no output at " + str(c_["output"]))
            continue
        uniq = set(tr)
        ents = {x for s, _, o in uniq for x in (s, o)}
        descs = {o for s, r, o in uniq if r == "hasdescriptor"}
        in_vocab = descs & vocab if vocab else set()
        share = (100.0 * len(in_vocab) / len(descs)) if descs else 0.0
        s26, l26 = recall(uniq, b26)
        s34, l34 = recall(uniq, b34)
        rel = Counter(r for _, r, _ in uniq)
        secs = (report.get("timings", {}).get(c_["slug"], {}) or {}).get("seconds")
        rows.append({"label": c_["label"], "revision": c_["revision"],
                     "raw": len(tr), "distinct": len(uniq),
                     "entities": len(ents), "descriptors": len(descs),
                     "descriptors_in_vocab": len(in_vocab),
                     "vocab_share": round(share, 1),
                     "malformed_records": malformed,
                     "recall26_strict": s26, "recall26_loose": l26,
                     "recall34_strict": s34, "recall34_loose": l34,
                     "relations": dict(rel), "seconds": secs})
        sets[c_["label"]] = uniq
        print(SUB)
        print(c_["label"] + "   rev " + str(c_["revision"]))
        print("  raw triples            " + str(len(tr)))
        print("  distinct triples       " + str(len(uniq)))
        print("  distinct entities      " + str(len(ents)))
        print("  distinct descriptors   " + str(len(descs))
              + "   of which in the declared vocabulary: " + str(len(in_vocab))
              + " (" + str(round(share, 1)) + " per cent)")
        print("  malformed records      " + str(malformed)
              + "   (a list or object where the schema asks for a string)")
        print("  recall / 26            strict " + str(s26) + "   substring " + str(l26))
        print("  recall / 34            strict " + str(s34) + "   substring " + str(l34))
        print("  relation types         " + ", ".join(k + "=" + str(v) for k, v in rel.most_common(6)))
        if ref_a and c_["slug"] == "qwen":
            same = set(ref_a) & uniq
            print("  determinism check against the delivered pass A:")
            print("    delivered " + str(len(set(ref_a))) + " distinct, fresh "
                  + str(len(uniq)) + ", shared " + str(len(same)))
            if len(set(ref_a)) == len(uniq) == len(same):
                print("    identical, as expected at temperature 0")
            else:
                print("    NOT identical. Understand this before publishing the table:")
                print("    at temperature 0 the same code and model should reproduce.")
            report["determinism"] = {"delivered": len(set(ref_a)),
                                     "fresh": len(uniq), "shared": len(same)}

    if len(sets) >= 2:
        print(SUB)
        print("pairwise agreement on distinct triples:")
        labels = list(sets)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = sets[labels[i]], sets[labels[j]]
                inter, union = a & b, a | b
                ea = {x for s, _, o in a for x in (s, o)}
                eb = {x for s, _, o in b for x in (s, o)}
                je = len(ea & eb) / len(ea | eb) if (ea | eb) else 0.0
                print("  " + labels[i] + " vs " + labels[j])
                print("    triples shared " + str(len(inter)) + "   Jaccard "
                      + str(round(len(inter) / len(union), 3) if union else 0.0))
                print("    entities Jaccard " + str(round(je, 3)))
                report.setdefault("pairwise", {})[labels[i] + " | " + labels[j]] = {
                    "triples_shared": len(inter),
                    "triples_jaccard": round(len(inter) / len(union), 4) if union else 0.0,
                    "entities_jaccard": round(je, 4)}

    if rows:
        tex = ["% generated by 95p_comparaison_controlee_modeles.py",
               "% identical arguments except --model; one pass at temperature 0",
               "\\begin{table}[htbp]", "\\centering",
               "\\caption{Extraction models compared under identical conditions:",
               "the same index, the same 287 queries, the same two-stage retrieval",
               "and the same instruction template, in a single pass at temperature 0.",
               "Only the extraction model differs. Counts are of distinct items in the",
               "raw extraction output, before verification and canonicalisation.",
               "\\emph{In vocabulary} is the share of distinct descriptor values that",
               "belong to the 40-term declared vocabulary. Recall is over the 26-edge",
               "benchmark under strict tuple equality. The instruction template was",
               "developed with Qwen in the loop, which favours it.}",
               "\\label{tab:model-comparison}",
               "\\begin{tabular}{lrrrrr}", "\\hline",
               "Model & Triples & Entities & Descriptors & In vocabulary & Recall/26 \\\\",
               "\\hline"]
        for r in rows:
            tex.append(" & ".join([
                r["label"], str(r["distinct"]), str(r["entities"]),
                str(r["descriptors"]),
                str(r["descriptors_in_vocab"]) + " (" + str(r["vocab_share"]) + "\\%)",
                str(r["recall26_strict"])]) + " \\\\")
        tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_TEX.write_text("\n".join(tex), encoding="utf-8")
        print(SUB)
        print("LaTeX table written to " + str(OUT_TEX))
    report["rows"] = rows


def main():
    global CODE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["plan", "run", "report", "all"], default="plan")
    ap.add_argument("--code-root", default=str(SCRIPTS_ROOT),
                    help="repository root to run the extractor from; point it at a "
                         "clean checkout of the delivered commit to attach the "
                         "comparison to run16E")
    args = ap.parse_args()
    CODE_ROOT = Path(args.code_root).expanduser()
    if not extractor_path().exists():
        print("FATAL: no extractor at " + str(extractor_path()))
        sys.exit(2)

    report = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "protocol": {"fixed_arguments": FIXED, "varied": ["--model", "--output"],
                           "passes": 1, "temperature": "0.0",
                           "residual_confound": "instruction template developed with Qwen in the loop"}}
    cmds = build_commands()
    if args.stage == "plan":
        stage_plan(cmds, report)
    elif args.stage == "run":
        stage_plan(cmds, report)
        stage_run(cmds, report)
    elif args.stage == "report":
        stage_report(cmds, report)
    else:
        stage_plan(cmds, report)
        stage_run(cmds, report)
        stage_report(cmds, report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(SEP)
    print("report written: " + str(OUT_JSON))
    print(SEP)


if __name__ == "__main__":
    main()