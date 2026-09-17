#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialise the 26-edge benchmark from the constant embedded in the validation stage.

The 26-edge benchmark is defined in the code as the constant LB_REFERENCE_EDGES of
pipeline/04_clean_validate.py. This script reads that constant through the syntax tree,
without importing or executing the pipeline module, and writes it as a reference file
(configs/lb_26_original.json by default), in the record format of an existing
benchmark file (--like).

It then compares the 26 edges with the 34-edge benchmark and reports any of the 26
missing from the 34, so that the 34 can be checked to be the 26 plus added edges.

The procedure by which the constant itself was derived from the reference ontology
is not recorded in the code.

Refuses to overwrite its output unless --force is given. Modifies no pipeline file.
The default --like path is the historical location; in this repository the 34-edge
file is configs/lb_reference_edges.json.

Historical name: 50_extract_lb26_reference.py

Usage
    python scripts/tools/benchmark26_from_pipeline_constant.py --source pipeline/04_clean_validate.py --like configs/lb_reference_edges.json --out output/lb_26_original.json
"""

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path


def extract_constant(py_path, name):
    """Read a literal constant through the syntax tree, without importing the module."""
    src = Path(py_path).read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name \
                    and node.value is not None:
                return ast.literal_eval(node.value)
    raise KeyError("{} not found in {}".format(name, py_path))


def norm(s):
    return " ".join(str(s).lower().split())


def load_ref_edges(path):
    """Read a reference file; return (normalised edges, record structure)."""
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, list):
        items, container = data, None
    elif isinstance(data, dict):
        container = None
        for k in ("edges", "reference_edges", "triples", "items", "data"):
            if isinstance(data.get(k), list):
                items, container = data[k], k
                break
        else:
            raise ValueError("unrecognised reference format: {}".format(path))
    else:
        raise ValueError("unrecognised reference format: {}".format(path))

    edges, keys = [], None
    for it in items:
        if isinstance(it, dict):
            if keys is None:
                keys = list(it.keys())
            s = it.get("subject") or it.get("s") or it.get("source")
            r = it.get("relation") or it.get("r") or it.get("predicate")
            o = it.get("object") or it.get("o") or it.get("target")
            edges.append((norm(s), norm(r), norm(o)))
        elif isinstance(it, (list, tuple)) and len(it) >= 3:
            edges.append((norm(it[0]), norm(it[1]), norm(it[2])))
    return edges, {"container": container, "item_keys": keys,
                   "is_list": isinstance(data, list)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pipeline/04_clean_validate.py")
    ap.add_argument("--constant", default="LB_REFERENCE_EDGES")
    ap.add_argument("--like", default="configs/lb_extended_benchmark.json",
                    help="existing benchmark file whose record format is copied")
    ap.add_argument("--out", default="configs/lb_26_original.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(os.path.expanduser(args.out))
    if out.exists() and not args.force:
        print("ERROR: {} already exists; refusing to overwrite (use --force).".format(out),
              file=sys.stderr)
        sys.exit(2)

    print("=" * 74)
    print("26-edge benchmark from the pipeline constant")
    print("=" * 74)

    raw = extract_constant(args.source, args.constant)
    print("source   : {} :: {}".format(args.source, args.constant))
    print("items    : {}".format(len(raw)))
    if raw:
        print("example  : {}".format(raw[0]))
    print()

    edges26 = []
    for it in raw:
        if isinstance(it, (list, tuple)) and len(it) >= 3:
            edges26.append((norm(it[0]), norm(it[1]), norm(it[2])))
        elif isinstance(it, dict):
            edges26.append((norm(it.get("subject")), norm(it.get("relation")),
                            norm(it.get("object"))))
    if len(edges26) != len(raw):
        print("[!] {} of {} items could not be read".format(
            len(raw) - len(edges26), len(raw)))

    dup = len(edges26) - len(set(edges26))
    if dup:
        print("[!] {} duplicate edge(s) in the constant".format(dup))

    # target record format
    like = Path(os.path.expanduser(args.like))
    fmt = {"container": None, "item_keys": ["subject", "relation", "object"],
           "is_list": True}
    edges34 = []
    if like.is_file():
        edges34, fmt = load_ref_edges(like)
        print("format copied from: {}".format(like))
        print("  sha256        : {}".format(sha256(like)))
        print("  edges         : {}".format(len(edges34)))
        print("  record keys   : {}".format(fmt["item_keys"]))
        print()
    else:
        print("[!] {} not found: default record format used".format(like))
        print()

    # compare the 26 edges with the 34
    if edges34:
        s26, s34 = set(edges26), set(edges34)
        inter = s26 & s34
        print("--- 26 vs 34 -------------------------------------------")
        print("  of the 26, present in the 34 : {}".format(len(inter)))
        print("  of the 26, ABSENT from the 34: {}".format(len(s26 - s34)))
        for e in sorted(s26 - s34):
            print("    - {} | {} | {}".format(*e))
        print("  of the 34, outside the 26    : {}".format(len(s34 - s26)))
        for e in sorted(s34 - s26):
            print("    + {} | {} | {}".format(*e))
        if len(s26 - s34) == 0 and len(s34 - s26) == len(s34) - len(s26):
            print("  -> consistent: the 34 are the 26 plus the added edges")
        else:
            print("  -> INCONSISTENT: the 34 do not contain exactly the 26")
        print()

    # write the 26 edges in the target format
    items = []
    for s, r, o in edges26:
        keys = fmt["item_keys"] or ["subject", "relation", "object"]
        if set(["subject", "relation", "object"]).issubset(set(keys)):
            d = {"subject": s, "relation": r, "object": o}
        elif set(["s", "r", "o"]).issubset(set(keys)):
            d = {"s": s, "r": r, "o": o}
        elif set(["source", "predicate", "target"]).issubset(set(keys)):
            d = {"source": s, "predicate": r, "target": o}
        else:
            d = {"subject": s, "relation": r, "object": o}
        items.append(d)

    payload = items if fmt["is_list"] else {fmt["container"] or "edges": items}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print("Written: {}  ({} edges)".format(out, len(items)))
    print("sha256: {}".format(sha256(out)))
    print()
    print("benchmark file for recall evaluation: {}".format(out))


if __name__ == "__main__":
    main()