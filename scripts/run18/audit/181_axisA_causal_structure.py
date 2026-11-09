#!/usr/bin/env python3
"""
181_axisA_causal_structure.py

Read-only verification of axis A: does the delivered graph let us rebuild,
from several papers, a causal structure around MTDs, and which links are
corroborated by several papers?

Python 3.10, standard library only. Never writes outside --outdir, never
overwrites. Stops on the first failed control.

Rules applied throughout
  * The unit of corroboration is the PAPER (doc_id from located_provenance),
    never the chunk.
  * A relation absent from the graph is "not extracted", never "absent from
    the literature". Every absence in the outputs is labelled that way.
  * Chains are built ONLY from extracted relations. Nothing is completed
    from outside knowledge. A junction that exists only after family fusion
    is flagged for manual reading (possible case 3).
  * Modality / conditionality is NOT inferred. The script only prepares the
    sentences and lexical cues; cue counts are not results.

Run:
  source ~/kg_test/venv/bin/activate
  python -u 181_axisA_causal_structure.py
Then send back:  <outdir>/report.md  (and the CSVs if asked).
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ------------------------------------------------------------------ controls
EXPECTED_TRIPLES = 232
EXPECTED_ENTITIES = 216
EXPECTED_PREDICATES = {"hasDescriptor": 83, "causes": 40, "occursIn": 38, "triggers": 29,
                       "affects": 25, "controls": 8, "overlies": 4, "partOf": 4, "underlies": 1}
# configs/entity_families.csv, commit 1e52e9b: 128 rows in 21 families, of which
# 2 are singletons (systeme sedimentaire, sea level); the 19 multi-member
# families cover 126 entities (the figures quoted in the hand-off).
EXPECTED_FAMILY_ROWS = 128
EXPECTED_FAMILIES_ALL = 21
EXPECTED_FAMILIES = 19
EXPECTED_MULTI_ENTITIES = 126

CAUSAL = ("causes", "triggers", "controls", "affects")
MAX_CHAIN_EDGES = 4
MAX_PATHS = 20000

ROOTS = ["~/ontogeorag", "~/ontogeorag_D", "~/ontogeorag_release"]
DEFAULT_GRAPH = "~/ontogeorag_release/results/run16E/graph_with_located_provenance.json"
DEFAULT_FAMILIES = "~/ontogeorag_release/configs/entity_families.csv"
DEFAULT_SHEET = "~/ontogeorag/output/audit/knowledge_form_sheet.csv"
DEFAULT_CHUNKS = "~/ontogeorag_D/output/run16E/index/chunks.jsonl"
DEFAULT_OUTDIR = "~/ontogeorag/output/audit/181_axisA"

# Seeds and targets are matched on SURFACE entity names; the matched lists are
# printed in the report so they can be checked by eye.
SEED_RE = re.compile(r"pressure|overpress", re.I)
SEED_EXCLUDE_RE = re.compile(r"ridge", re.I)
TARGET_MTD_RE = re.compile(r"mass[- ]transport|\bmtds?\b", re.I)
TARGET_FAIL_RE = re.compile(r"failure|instabilit|landslid|mass[- ]wasting|mass movement|slump|\bslides?\b", re.I)

KEYWORDS = ["pore pressure", "overpressure", "over-pressure", "excess pore", "effective stress",
            "sedimentation", "fluid", "seismic", "earthquake", "hydrate", "instabilit",
            "slope failure", "failure", "trigger", "precondition", "promot", "favo",
            "lead to", "leads to", "result", "associated", "mechanism", "condition",
            "loading", "storm", "erosion", "oversteepen", "weak layer"]

CUES = {
    "hedge": r"\b(may|might|could|can|possibly|possible|probably|probable|likely|suggest\w*|propos\w*|"
             r"interpret\w*|thought to|believed|hypothes\w*|appears? to|seems?|potential\w*)\b",
    "condition": r"\b(if|when|where|whereas|only|unless|provided|under|during|above|below|deeper|"
                 r"shallower|depth|beyond|except|depend\w*)\b",
    "association": r"\b(associated|related|correlat\w*|linked|coincid\w*|accompan\w*)\b",
    "preparation": r"\b(precondition\w*|predispos\w*|favou?r\w*|promot\w*|contribut\w*|prone|"
                   r"weak\w*|reduc\w*|lower\w*)\b",
    "trigger": r"\b(trigger\w*|initiat\w*|earthquake\w*|seismic\w*|storm\w*)\b",
}
CUES = {k: re.compile(v, re.I) for k, v in CUES.items()}
ANNOTATION_COLUMNS = ["direct_causality", "precondition", "favouring_factor", "trigger",
                      "association", "possibility", "hypothesis", "conditional",
                      "context_dependent", "condition_text", "reader_note"]


# ------------------------------------------------------------------ helpers
def stop(msg):
    print("STOP: " + msg, file=sys.stderr)
    sys.exit(1)


def norm(s):
    return " ".join(str(s or "").lower().split())


def sha8(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()[:12]


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def count_items(p):
    try:
        if p.suffix == ".json":
            with open(p) as f:
                d = json.load(f)
            if isinstance(d, dict):
                for k in ("triples", "edges", "relations"):
                    if isinstance(d.get(k), list):
                        return k + "=" + str(len(d[k]))
                return "keys=" + ",".join(list(d)[:6])
            return "list=" + str(len(d))
        if p.suffix == ".csv":
            with open(p, newline="", encoding="utf-8-sig") as f:
                return "rows=" + str(sum(1 for _ in csv.reader(f)) - 1)
        if p.suffix == ".jsonl":
            with open(p) as f:
                return "lines=" + str(sum(1 for line in f if line.strip()))
        if p.suffix == ".md":
            with open(p) as f:
                return "lines=" + str(sum(1 for _ in f))
    except Exception as e:  # inventory only, never used as data
        return "unreadable (" + type(e).__name__ + ")"
    return ""


# ------------------------------------------------------------------ section 1
INVENTORY_PATTERNS = [
    re.compile(r"graph_with_located_provenance.*\.json$"),
    re.compile(r"flat_run16E.*located.*\.json$"),
    re.compile(r"entity_families.*\.csv$"),
    re.compile(r"knowledge_form.*\.(csv|md)$"),
    re.compile(r"PROTOCOLE\.md$"),
    re.compile(r"corpus_table\.md$"),
    re.compile(r".*_evidence\.md$"),
]
PRUNE = {".git", "venv", "__pycache__", "node_modules", ".ipynb_checkpoints", "hf_cache"}


def inventory():
    found = []
    for root in ROOTS:
        r = Path(root).expanduser()
        if not r.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if d not in PRUNE]
            in_146 = "146" in dirpath
            for fn in filenames:
                p = Path(dirpath) / fn
                hit = any(pat.search(fn) for pat in INVENTORY_PATTERNS)
                hit = hit or (in_146 and fn in ("relations.csv", "entities.csv"))
                hit = hit or (fn == "chunks.jsonl" and "run16E" in dirpath)
                if hit:
                    st = p.stat()
                    found.append([str(p), st.st_size,
                                  datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="minutes"),
                                  count_items(p)])
    return sorted(found)


# ------------------------------------------------------------------ loading
def load_graph(path):
    if not path.is_file():
        stop("graph not found: " + str(path))
    with open(path) as f:
        d = json.load(f)
    triples = d["triples"] if isinstance(d, dict) else d
    if len(triples) != EXPECTED_TRIPLES:
        stop("graph has " + str(len(triples)) + " triples, expected 232 (old state?)")
    ents = set()
    for t in triples:
        for k in ("subject", "relation", "object"):
            if k not in t:
                stop("triple without '" + k + "': " + json.dumps(t)[:200])
        if "located_provenance" not in t:
            stop("triple without located_provenance: " + t["subject"] + " | " + t["object"])
        ents.add(norm(t["subject"]))
        ents.add(norm(t["object"]))
    if len(ents) != EXPECTED_ENTITIES:
        stop("graph has " + str(len(ents)) + " entities, expected 216")
    pc = Counter(t["relation"] for t in triples)
    if dict(pc) != EXPECTED_PREDICATES:
        stop("predicate distribution differs: " + json.dumps(dict(pc)))
    return triples, ents


def load_families(path, ents):
    if not path.is_file():
        stop("families not found: " + str(path))
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "entity" not in rows[0] or "family" not in rows[0]:
        stop("families file needs columns entity,family")
    fam = {}
    for r in rows:
        e = norm(r["entity"])
        if e in fam:
            stop("entity listed twice in families: " + e)
        fam[e] = norm(r["family"])
    size = Counter(fam.values())
    multi = {f for f, n in size.items() if n >= 2}
    n_multi_ent = sum(size[f] for f in multi)
    got = (len(fam), len(size), len(multi), n_multi_ent)
    want = (EXPECTED_FAMILY_ROWS, EXPECTED_FAMILIES_ALL, EXPECTED_FAMILIES, EXPECTED_MULTI_ENTITIES)
    if got != want:
        stop("families: rows/families/multi-member families/entities in them = " + str(got)
             + ", expected " + str(want))
    missing = sorted(e for e in fam if e not in ents)
    if missing:
        stop("family entities absent from the graph (join would be wrong): " + "; ".join(missing))
    return fam


def load_corpus_labels():
    labels = {}
    for root in ROOTS:
        p = Path(root).expanduser() / "data" / "corpus_table.md"
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                labels["__lines__"] = [line.strip() for line in f if line.strip().startswith("|")]
            labels["__path__"] = str(p)
            return labels
    return labels


def doc_label(doc, labels):
    for line in labels.get("__lines__", []):
        if doc in line:
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            return " / ".join(cells[:4])[:120]
    return ""


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--families", default=DEFAULT_FAMILIES)
    ap.add_argument("--sheet", default=DEFAULT_SHEET)
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if outdir.exists():
        stop("output directory exists, refusing to overwrite: " + str(outdir))

    rep = []  # report.md lines
    def H(title):
        rep.append("")
        rep.append("## " + title)
        rep.append("")

    rep.append("# 181 axis A: causal structure, provenance, corroboration")
    rep.append("generated " + datetime.datetime.now().isoformat(timespec="seconds"))
    rep.append("RULE: a relation absent from the graph = not extracted, never absent from the literature.")

    # ---- 1. files
    H("1. Files")
    inv = inventory()
    rep.append("| path | bytes | modified | items |")
    rep.append("|---|---|---|---|")
    for p, size, mt, cnt in inv:
        rep.append("| " + p + " | " + str(size) + " | " + mt + " | " + cnt + " |")
    graph_p = Path(args.graph).expanduser()
    fam_p = Path(args.families).expanduser()
    triples, ents = load_graph(graph_p)
    fam = load_families(fam_p, ents)
    labels = load_corpus_labels()
    rep.append("")
    rep.append("USED graph: " + str(graph_p) + " sha " + sha8(graph_p)
               + " | 232 triples, 216 entities, predicates as verified: OK")
    rep.append("USED families: " + str(fam_p) + " sha " + sha8(fam_p) + " | 128 rows, 21 families (19 multi-member covering 126 entities): OK")
    rep.append("corpus labels: " + labels.get("__path__", "corpus_table.md NOT FOUND (doc_ids shown raw)"))

    # cross-check other located-provenance copies (content keys, never positions)
    def key3(t):
        return (norm(t["subject"]), t["relation"], norm(t["object"]))
    ref = {key3(t): sorted(set((t.get("located_provenance") or {}).get("doc_ids") or [])) for t in triples}
    for p, _, _, _ in inv:
        if p.endswith(".json") and "located" in p and Path(p) != graph_p:
            try:
                with open(p) as f:
                    d = json.load(f)
                other = d["triples"] if isinstance(d, dict) else d
                oth = {key3(t): sorted(set((t.get("located_provenance") or {}).get("doc_ids") or []))
                       for t in other}
                same = oth == ref
                rep.append("cross-check " + p + ": " + ("IDENTICAL keys and doc_ids" if same else
                           "DIFFERS (" + str(len(set(oth) ^ set(ref))) + " keys differ, "
                           + str(sum(1 for k in ref if k in oth and oth[k] != ref[k])) + " doc lists differ)"))
            except Exception as e:
                rep.append("cross-check " + p + ": unreadable " + type(e).__name__)

    # ---- per-triple records
    F = lambda e: fam.get(norm(e), norm(e))
    recs = []
    for i, t in enumerate(triples):
        lp = t.get("located_provenance") or {}
        docs = sorted(set(lp.get("doc_ids") or []))
        m4 = t.get("m4") or {}
        recs.append({
            "row": i, "s": norm(t["subject"]), "r": t["relation"], "o": norm(t["object"]),
            "fs": F(t["subject"]), "fo": F(t["object"]), "docs": docs,
            "match": lp.get("match", ""), "tier": t.get("tier", ""),
            "panel": m4.get("decision", ""), "ev": " ".join(str(t.get("evidence", "")).split()),
        })
    mc = Counter(r["match"] for r in recs)
    nd = Counter(len(r["docs"]) for r in recs)
    rep.append("located_provenance match: " + json.dumps(dict(mc))
               + " | docs per relation: " + json.dumps({str(k): v for k, v in sorted(nd.items())}))
    multi = [r for r in recs if len(r["docs"]) > 1]
    if multi:
        rep.append("WARNING: " + str(len(multi)) + " relations carry several doc_ids for ONE evidence sentence."
                   " Same sentence in several files = likely duplicate documents, NOT independent corroboration:")
        for r in multi:
            rep.append("  - " + r["s"] + " | " + r["r"] + " | " + r["o"] + " -> " + ", ".join(r["docs"]))

    all_docs = sorted({d for r in recs for d in r["docs"]})
    raw_rel_fields = sorted({k for t in triples for k in t if re.search(r"raw|orig|surface", k, re.I)})
    rep.append("fields possibly holding the pre-mapping predicate: " + (", ".join(raw_rel_fields) or "none"))

    # ---- 2. causal + keyword relations
    H("2. Causal and keyword relations (all predicates scanned)")
    kw_re = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.I)
    rows2 = []
    for r in recs:
        is_c = r["r"] in CAUSAL
        hit_ent = bool(kw_re.search(r["s"] + " " + r["o"]))
        hit_ev = bool(kw_re.search(r["ev"]))
        if is_c or hit_ent or hit_ev:
            rows2.append(r)
    rep.append("causal predicates (" + ", ".join(CAUSAL) + "): "
               + str(sum(1 for r in recs if r["r"] in CAUSAL)) + " relations")
    rep.append("non-causal predicates matched by keyword (entity or sentence): "
               + str(sum(1 for r in rows2 if r["r"] not in CAUSAL))
               + " (listed in 02_causal_and_keyword.csv, NOT treated as causal)")
    rep.append("note: the delivered graph has only the 9 canonical predicates; surface forms such as"
               " promotes / leads to / associated with were mapped before delivery. Their wording"
               " survives only in the evidence sentence.")

    # ---- 5. families (computed early, used everywhere)
    def group(key_fn, subset):
        g = defaultdict(list)
        for r in subset:
            g[key_fn(r)].append(r)
        return g

    k_surf = lambda r: (r["s"], r["r"], r["o"])
    k_fused = lambda r: (r["fs"], r["r"], r["fo"])
    k_fclass = lambda r: (r["fs"], "CAUSAL" if r["r"] in CAUSAL else r["r"], r["fo"])

    def ndocs(members):
        return len({d for m in members for d in m["docs"]})

    def dist(g):
        c = Counter()
        for members in g.values():
            n = ndocs(members)
            c["0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else ">=4"] += 1
        return {b: c.get(b, 0) for b in ("0", "1", "2", "3", ">=4")}

    # ---- 8. corroboration distributions
    H("8. Corroboration (unit = paper)")
    causal_recs = [r for r in recs if r["r"] in CAUSAL]
    rep.append("| subset | level | distinct relations | 0 papers | 1 | 2 | 3 | >=4 |")
    rep.append("|---|---|---|---|---|---|---|---|")
    for name, subset in (("all 232", recs), ("causal 102", causal_recs)):
        for lvl, fn in (("surface", k_surf), ("families, predicate kept", k_fused),
                        ("families, causal predicates pooled", k_fclass)):
            g = group(fn, subset)
            d = dist(g)
            rep.append("| " + name + " | " + lvl + " | " + str(len(g)) + " | "
                       + " | ".join(str(d[b]) for b in ("0", "1", "2", "3", ">=4")) + " |")
    rep.append("")
    rep.append("Most corroborated relations after fusion (causal predicates pooled, all predicates):")
    g_all = group(k_fclass, recs)
    ranked = sorted(g_all.items(), key=lambda kv: (-ndocs(kv[1]), kv[0]))
    rows8 = []
    for key, members in ranked:
        docs = sorted({d for m in members for d in m["docs"]})
        variants = sorted({m["s"] + " | " + m["r"] + " | " + m["o"] for m in members})
        before = max((len(m["docs"]) for m in members), default=0)
        rows8.append([key[0], key[1], key[2], len(docs), before, len(variants),
                      " || ".join(variants), " | ".join(docs)])
    for row in rows8[:25]:
        rep.append("- [" + str(row[3]) + " papers after fusion, max " + str(row[4]) + " before; "
                   + str(row[5]) + " surface variants] " + row[0] + " --" + row[1] + "--> " + row[2])

    # ---- 5. what fusion adds
    H("5. Effect of family fusion")
    rep.append("Relations whose paper count grows with fusion (causal predicates pooled):")
    gains = [x for x in rows8 if x[3] > x[4]]
    rep.append("count: " + str(len(gains)) + " fused relations gain at least one paper")
    for x in gains[:30]:
        rep.append("- " + x[0] + " --" + x[1] + "--> " + x[2] + ": " + str(x[4]) + " -> " + str(x[3])
                   + " papers | variants: " + x[6])
    rep.append("")
    rep.append("Family membership of every surface form matching 'pressure' (pressure ridge excluded):")
    for e in sorted(ents):
        if SEED_RE.search(e) and not SEED_EXCLUDE_RE.search(e):
            rep.append("- " + e + " -> family: " + (fam[e] if e in fam else "(none, stays alone)"))

    # ---- graph for chains (families, causal only)
    G = defaultdict(lambda: defaultdict(list))
    Gs = defaultdict(set)  # surface adjacency for 'exists before fusion'
    for r in causal_recs:
        G[r["fs"]][r["fo"]].append(r)
        Gs[r["s"]].add(r["o"])
    fam_members = defaultdict(set)
    for e in ents:
        fam_members[F(e)].add(e)
    def is_target(fnode, which):
        rx = TARGET_MTD_RE if which == "mtd" else TARGET_FAIL_RE
        return any(rx.search(e) for e in fam_members[fnode])
    t_mtd = {n for n in fam_members if is_target(n, "mtd")}
    t_fail = {n for n in fam_members if is_target(n, "fail") and n not in t_mtd}
    seeds = {F(e) for e in ents if SEED_RE.search(e) and not SEED_EXCLUDE_RE.search(e)}

    # ---- 6. overpressure sub-network
    H("6. Overpressure sub-network (1 hop, causal predicates, seeds = 'pressure' forms)")
    rep.append("seed families: " + "; ".join(sorted(seeds)))
    sub = [r for r in causal_recs if r["fs"] in seeds or r["fo"] in seeds]
    nodes6 = sorted({r["s"] for r in sub} | {r["o"] for r in sub})
    rep.append("relations: " + str(len(sub)) + " | surface nodes: " + str(len(nodes6))
               + " | distinct papers: " + str(len({d for r in sub for d in r["docs"]})))
    rep.append("")
    rep.append("| subject | relation | object | role of seed | papers (this relation) | papers (fused) | evidence |")
    rep.append("|---|---|---|---|---|---|---|")
    g_f = group(k_fclass, causal_recs)
    rows6 = []
    for r in sorted(sub, key=lambda x: (x["fs"] not in seeds, x["s"], x["o"])):
        role = "seed->" if r["fs"] in seeds else "->seed"
        if r["fs"] in seeds and r["fo"] in seeds:
            role = "seed->seed"
        fused_n = ndocs(g_f[k_fclass(r)])
        rows6.append([r["s"], r["r"], r["o"], role, " | ".join(r["docs"]), len(r["docs"]), fused_n,
                      r["match"], r["tier"], r["ev"]])
        rep.append("| " + r["s"] + " | " + r["r"] + " | " + r["o"] + " | " + role + " | "
                   + (", ".join(r["docs"]) or "NONE") + " | " + str(fused_n) + " | "
                   + r["ev"][:220].replace("|", "/") + " |")

    # ---- 3/7. chains
    H("3 and 7. Chains built only from extracted causal relations (families)")
    rep.append("MTD target families: " + "; ".join(sorted(t_mtd)))
    rep.append("failure/instability target families: " + "; ".join(sorted(t_fail)))
    rep.append("CHECK these target lists by eye: a family can match by accident.")
    targets = t_mtd | t_fail
    paths = []
    def dfs(path):
        if len(paths) >= MAX_PATHS:
            return
        last = path[-1]
        if len(path) >= 3 and last in targets:
            paths.append(list(path))
        if len(path) - 1 >= MAX_CHAIN_EDGES or last in t_mtd:
            return
        for nxt in G.get(last, {}):
            if nxt not in path:
                path.append(nxt)
                dfs(path)
                path.pop()
    for start in list(G):
        if start not in t_mtd:
            dfs([start])
    if len(paths) >= MAX_PATHS:
        rep.append("WARNING: path cap reached (" + str(MAX_PATHS) + ")")

    def classify(path):
        link_docs, junction_flags, surface_ok = [], [], True
        for a, b in zip(path, path[1:]):
            link_docs.append({d for m in G[a][b] for d in m["docs"]})
        for i in range(1, len(path) - 1):
            ins = {m["o"] for m in G[path[i - 1]][path[i]]}
            outs = {m["s"] for m in G[path[i]][path[i + 1]]}
            if not ins & outs:
                surface_ok = False
                junction_flags.append(path[i] + ": in=" + "/".join(sorted(ins)) + " out=" + "/".join(sorted(outs)))
        if any(not d for d in link_docs):
            case = "INCOMPLETE_PROVENANCE"
        elif set.intersection(*link_docs):
            case = "CASE2_ONE_PAPER_COVERS_ALL_LINKS"
        else:
            case = "CASE1_DISTRIBUTED"
        if junction_flags:
            case += "+CASE3_CANDIDATE_JOIN_VIA_FUSION"
        return case, link_docs, junction_flags, surface_ok

    rows7 = []
    cc = Counter()
    for p in paths:
        case, ld, jf, sok = classify(p)
        cc[case] += 1
        rows7.append([len(p) - 1, " -> ".join(p), case, "yes" if sok else "no",
                      any(n in seeds for n in p),
                      " || ".join(",".join(sorted(d)) for d in ld), " || ".join(jf)])
    rep.append("chains (>=2 links, ending at a target, max " + str(MAX_CHAIN_EDGES) + " links): "
               + str(len(paths)))
    rep.append("by case: " + json.dumps(dict(cc)))
    rep.append("chains through an overpressure seed: " + str(sum(1 for r in rows7 if r[4])))
    rep.append("")
    rep.append("| links | chain | case | exists before fusion | papers per link |")
    rep.append("|---|---|---|---|---|")
    order = sorted(rows7, key=lambda r: (not r[4], "CASE1" not in r[2], "CASE3" in r[2], -r[0], r[1]))
    for r in order[:40]:
        rep.append("| " + str(r[0]) + " | " + r[1] + " | " + r[2] + " | " + r[3] + " | " + r[5] + " |")

    rep.append("")
    rep.append("Direct factors (one extracted link into a target):")
    rows_direct = []
    for a in sorted(G):
        for b in sorted(G[a]):
            if b in targets and a not in targets:
                docs = sorted({d for m in G[a][b] for d in m["docs"]})
                rels = sorted({m["r"] for m in G[a][b]})
                rows_direct.append([a, "/".join(rels), b, len(docs), " | ".join(docs)])
                rep.append("- " + a + " --" + "/".join(rels) + "--> " + b + " : " + str(len(docs)) + " paper(s)")

    rep.append("")
    rep.append("Seeds with NO extracted path to any target (= extraction/representation gap, NOT literature absence):")
    reach = {}
    for s in seeds:
        seen, stack = {s}, [s]
        while stack:
            n = stack.pop()
            for m in G.get(n, {}):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        reach[s] = sorted(seen & targets)
        if not reach[s]:
            rep.append("- " + s)
    rep.append("(end of list)")

    # ---- 9. modality sheet
    H("9. Conditionality and modality: what can and cannot be said now")
    rel_mtd = [r for r in causal_recs
               if r["fs"] in targets or r["fo"] in targets or r["fs"] in seeds or r["fo"] in seeds
               or any(r["fs"] in p.split(" -> ") for p in [x[1] for x in rows7])]
    cue_c = Counter()
    rows9 = []
    for r in rel_mtd:
        flags = [k for k, rx in CUES.items() if rx.search(r["ev"])]
        for k in flags:
            cue_c[k] += 1
        rows9.append([r["s"], r["r"], r["o"], " | ".join(r["docs"]), ",".join(flags), r["ev"]]
                     + [""] * len(ANNOTATION_COLUMNS))
    rep.append("causal relations in the MTD / failure / overpressure neighbourhood: " + str(len(rel_mtd)))
    rep.append("lexical cue counts (NOT results, a cue is not a meaning): " + json.dumps(dict(cue_c)))
    sheet_p = Path(args.sheet).expanduser()
    if sheet_p.is_file():
        with open(sheet_p, newline="", encoding="utf-8-sig") as f:
            srows = list(csv.DictReader(f))
        hdr = list(srows[0]) if srows else []
        cols = {c.lower(): c for c in hdr}
        sc = next((cols[c] for c in ("subject", "head") if c in cols), None)
        pc_ = next((cols[c] for c in ("predicate", "relation") if c in cols), None)
        oc = next((cols[c] for c in ("object", "tail") if c in cols), None)
        if sc and pc_ and oc and "h1_final" in hdr and "h2_final" in hdr:
            skey = {(norm(x[sc]), x[pc_].strip(), norm(x[oc])): x for x in srows}
            joined = [skey.get((r["s"], r["r"], r["o"])) for r in rel_mtd]
            nj = sum(1 for x in joined if x is not None)
            done = [x for x in joined if x is not None and x["h1_final"].strip() and x["h2_final"].strip()]
            rep.append("knowledge_form_sheet: joined " + str(nj) + "/" + str(len(rel_mtd))
                       + " by content key; annotated (h1_final and h2_final): " + str(len(done)))
            if done and len(done) == len(rel_mtd):
                rep.append("H1 on this subset: " + json.dumps(dict(Counter(x["h1_final"].strip() for x in done))))
                rep.append("H2 on this subset: " + json.dumps(dict(Counter(x["h2_final"].strip() for x in done))))
            else:
                rep.append("proportions NOT reported: annotation incomplete on this subset."
                           " Annotate exactly the rows of 09_modality_to_annotate.csv.")
        else:
            rep.append("knowledge_form_sheet found but columns not recognised: " + ", ".join(hdr))
    else:
        rep.append("knowledge_form_sheet not found: proportions not available.")

    # ---- 11. Storegga
    H("11. Storegga / lost condition")
    st_re = re.compile(r"storegga|\b30\s?m\b|\b80\s?m\b", re.I)
    hits = [r for r in recs if st_re.search(r["ev"])]
    for r in hits:
        rep.append("- TRIPLE: " + r["s"] + " --" + r["r"] + "--> " + r["o"] + " | papers: "
                   + (", ".join(r["docs"]) or "NONE") + " | tier " + str(r["tier"]))
        rep.append("  SENTENCE: " + r["ev"])
    if not hits:
        rep.append("no evidence sentence mentions Storegga, 30 m or 80 m")
    ch_p = Path(args.chunks).expanduser()
    if ch_p.is_file():
        ctx = []
        with open(ch_p) as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                txt = " ".join(str(c.get("text", "")).split())
                if re.search(r"storegga", txt, re.I) and re.search(r"\b30\s?m\b", txt):
                    for sent in re.split(r"(?<=[.!?])\s+", txt):
                        if re.search(r"\b30\s?m\b", sent):
                            ctx.append((c.get("doc_id", ""), sent[:400]))
        rep.append("")
        rep.append("corpus context (chunks mentioning Storegga, sentences with '30 m'): " + str(len(ctx)))
        for d, s in ctx[:8]:
            rep.append("  - [" + str(d) + "] " + s)
    else:
        rep.append("chunks file not found at " + str(ch_p) + ": corpus context skipped")

    # ---- 10. candidate strong examples (ranking only, judgement is manual)
    H("10. Candidate strong examples (to READ, not to accept)")
    rep.append("A. Targets fed by several papers through different factors (convergence):")
    conv = []
    for b in sorted(targets):
        into = [(a, {d for m in G[a][b] for d in m["docs"]}) for a in G if b in G[a]]
        docs = set().union(*[d for _, d in into]) if into else set()
        if len(into) >= 2:
            conv.append((len(docs), b, into))
    for n, b, into in sorted(conv, key=lambda x: -x[0])[:6]:
        rep.append("- " + b + ": " + str(len(into)) + " incoming factors, " + str(n) + " papers: "
                   + "; ".join(a + " [" + ",".join(sorted(d)) + "]" for a, d in sorted(into)))
    rep.append("B. Distributed chains without fusion-only junctions: "
               + str(sum(1 for r in rows7 if r[2] == "CASE1_DISTRIBUTED"))
               + " (see 07_chains.csv, filter case == CASE1_DISTRIBUTED)")

    # ---- 12. rule audit
    H("12. Absence rule")
    rep.append("Outputs that describe an absence: section 3/7 'Seeds with NO extracted path' and the"
               " 0-paper buckets of section 8. Both are labelled as extraction not observed."
               " No statement in this report infers absence from the literature.")

    # ---- write
    outdir.mkdir(parents=True)
    write_csv(outdir / "01_inventory.csv", ["path", "bytes", "modified", "items"], inv)
    write_csv(outdir / "02_causal_and_keyword.csv",
              ["subject", "relation", "object", "family_s", "family_o", "is_causal_predicate",
               "papers", "n_papers", "match", "tier", "panel", "evidence"],
              [[r["s"], r["r"], r["o"], r["fs"], r["fo"], r["r"] in CAUSAL, " | ".join(r["docs"]),
                len(r["docs"]), r["match"], r["tier"], r["panel"], r["ev"]] for r in rows2])
    write_csv(outdir / "05_08_corroboration_fused.csv",
              ["family_s", "relation_class", "family_o", "papers_after_fusion", "max_papers_before",
               "n_surface_variants", "surface_variants", "papers"], rows8)
    write_csv(outdir / "06_overpressure_subnetwork.csv",
              ["subject", "relation", "object", "seed_role", "papers", "n_papers", "papers_fused",
               "match", "tier", "evidence"], rows6)
    write_csv(outdir / "07_chains.csv",
              ["n_links", "chain", "case", "exists_before_fusion", "through_overpressure",
               "papers_per_link", "fusion_only_junctions"], order)
    write_csv(outdir / "07_direct_factors.csv", ["factor_family", "relations", "target_family",
                                                "n_papers", "papers"], rows_direct)
    write_csv(outdir / "09_modality_to_annotate.csv",
              ["subject", "relation", "object", "papers", "lexical_cues", "evidence"] + ANNOTATION_COLUMNS,
              rows9)
    doc_rows = [[d, doc_label(d, labels)] for d in all_docs]
    write_csv(outdir / "00_papers.csv", ["doc_id", "corpus_table_label"], doc_rows)
    with open(outdir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("\nwrote " + str(outdir))
    print("OK")


if __name__ == "__main__":
    main()