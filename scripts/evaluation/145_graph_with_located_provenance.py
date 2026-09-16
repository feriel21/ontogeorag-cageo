#!/usr/bin/env python3
"""
145_graph_with_located_provenance.py

Writes a copy of the delivered graph in which each relation gains a
`located_provenance` field: the chunk(s) and document(s) of the index in
which its evidence sentence is found. The delivered file is never modified.

Rules: F1 and F2 of script 144, unchanged.
  F1  compare lowercase [a-z0-9] only, on evidence and chunk text.
  F2  exact if the whole normalised evidence is in a chunk; partial if one of
      three 40-character windows (start, middle, end) is; not_found otherwise.
Consistency: must reproduce 144 exactly (exact 164, partial 64, not_found 4;
single document 226, two 1, three 1, none 4) or the script stops.
No GPU, no model. Refuses to overwrite its output directory.
"""

import glob
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HOME = Path.home()
ROOTS = [HOME / "ontogeorag_D", HOME / "ontogeorag"]
OUT_DIR = HOME / "ontogeorag" / "output" / "audit" / "145_provenance"
WINDOW = 40
LIST_KEYS = ("triples", "relations", "edges", "records", "items")
EXPECT_MATCH = {"exact": 164, "partial": 64, "not_found": 4}
EXPECT_NDOCS = {0: 4, 1: 226, 2: 1, 3: 1}


def die(msg):
    print("STOP: " + msg, file=sys.stderr, flush=True)
    if OUT_DIR.is_dir() and not any(OUT_DIR.iterdir()):
        OUT_DIR.rmdir()
    sys.exit(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_one(pattern):
    hits = sorted({Path(p).resolve() for r in ROOTS for p in glob.glob(str(r / pattern), recursive=True)})
    if not hits:
        die("no file for " + pattern)
    shas = {sha256(h) for h in hits}
    if len(shas) > 1:
        die("distinct files match " + pattern + ": " + str(hits))
    return hits[0], shas.pop()


def alnum(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def locate(ev, chunk_alnum):
    e = alnum(ev)
    exact = [i for i, c in enumerate(chunk_alnum) if e and e in c]
    if exact:
        return "exact", exact
    if len(e) <= WINDOW:
        return "not_found", []
    mid = (len(e) - WINDOW) // 2
    wins = [e[:WINDOW], e[mid:mid + WINDOW], e[-WINDOW:]]
    partial = [i for i, c in enumerate(chunk_alnum) if any(w in c for w in wins)]
    return ("partial", partial) if partial else ("not_found", [])


def main():
    if OUT_DIR.exists():
        die("output directory exists, refusing to overwrite: " + str(OUT_DIR))
    OUT_DIR.mkdir(parents=True)

    cp, cs = find_one("output/**/run16E/index/chunks.jsonl")
    with open(cp) as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    if len(chunks) != 1758:
        die("chunks = " + str(len(chunks)))
    chunk_alnum = [alnum(c["text"]) for c in chunks]

    gp, gs = find_one("output/run16E/kg_final/flat_run16E_enforced_dedup.json")
    with open(gp) as f:
        raw = json.load(f)
    list_key = None
    if isinstance(raw, list):
        graph = raw
    else:
        named = [k for k in LIST_KEYS if isinstance(raw.get(k), list)]
        lists = [k for k, v in raw.items() if isinstance(v, list)]
        if len(named) == 1:
            list_key = named[0]
        elif len(lists) == 1:
            list_key = lists[0]
        else:
            die("cannot pick the record list among " + str(sorted(raw)))
        graph = raw[list_key]
    print("graph container: " + ("list" if list_key is None else "dict, key " + repr(list_key)
          + ", other keys " + str(sorted(k for k in raw if k != list_key))), flush=True)
    if len(graph) != 232:
        die("graph has " + str(len(graph)) + " records, expected 232")

    out, match_c, ndocs_c = [], Counter(), Counter()
    for rec in graph:
        status, ids = locate(rec.get("evidence") or "", chunk_alnum)
        docs = sorted({str(chunks[i]["doc_id"]) for i in ids})
        match_c[status] += 1
        ndocs_c[len(docs)] += 1
        new = dict(rec)
        new["located_provenance"] = {
            "match": status,
            "doc_ids": docs,
            "chunk_ids": [chunks[i]["chunk_id"] for i in ids],
            "method": "evidence sentence located in the index (lowercase alphanumeric "
                      "comparison; partial = one of three 40-character windows)",
        }
        out.append(new)

    if dict(match_c) != EXPECT_MATCH:
        die("match counts " + str(dict(match_c)) + " differ from 144 " + str(EXPECT_MATCH))
    if dict(ndocs_c) != EXPECT_NDOCS:
        die("document counts " + str(dict(ndocs_c)) + " differ from 144 " + str(EXPECT_NDOCS))

    if list_key is None:
        container = out
    else:
        container = dict(raw)
        container[list_key] = out
    with open(OUT_DIR / "flat_run16E_with_located_provenance.json", "w") as f:
        json.dump(container, f, indent=1)
    summary = {"inputs": {"chunks": [str(cp), cs], "graph": [str(gp), gs]},
               "match": dict(match_c), "documents_per_relation": {str(k): v for k, v in sorted(ndocs_c.items())},
               "not_found": [[r["subject"], r["relation"], r["object"]] for r in out
                             if r["located_provenance"]["match"] == "not_found"],
               "multiple_documents": [[r["subject"], r["relation"], r["object"], r["located_provenance"]["doc_ids"]]
                                      for r in out if len(r["located_provenance"]["doc_ids"]) > 1]}
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=1), flush=True)
    print("[done] " + str(OUT_DIR), flush=True)


if __name__ == "__main__":
    main()