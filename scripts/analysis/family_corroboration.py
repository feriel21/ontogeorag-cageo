#!/usr/bin/env python3
"""Corroboration of each delivered relation across the corpus, at concept-family level.

Supports Section 2.3.5 (protocol), Section 3.10 and Figure 7:
    - number of papers in which the two arguments of a relation co-occur in
      the same indexed passage, counted at surface-string level and at
      concept-family level;
    - relations resting on at most one paper under each count;
    - the family grouping itself (26 families with more than one form in the
      published count: 19 from the manual table, 7 from the automatic rule).

Why families. The graph keeps several surface forms per concept ("pore
pressure", "excess pore pressure", "overpressure"). Counted per string,
co-occurrences are split across those forms and each form looks weakly
corroborated. Counting per family measures the concept instead of the phrasing.

Grouping
    Automatic rule: union-find joining two entities when the tokens of one are
    contained in those of the other, they share the same nominal head, and they
    are not polar opposites (high/low, stable/unstable, un-/dis-/non-/in-).
    Manual table (--families, columns entity,family): overrides the automatic
    rule for the entities it covers; the others keep the automatic grouping.

Reading the counts
    * Family-level counting can only raise a count, never lower it, so it cannot
      create the finding that a relation rests on a single paper.
    * Co-occurrence in a passage is not assertion of the relation: the count is
      a corroboration signal, not a verification.
    * A count of zero means the family names do not co-occur under the indexed
      segmentation, not that the relation lacks an evidence sentence.

Outputs
    --out            one row per relation: tier, families, forms per family,
                     papers_surface, papers_family, delta
    <out>_groups.csv every family with more than one form
The zero/one split, the surface-level count and the means and medians by tier
reported in the article are computed from the --out file.

Read-only with respect to the pipeline. No language model is involved.
Default paths follow the original layout (~/ontogeorag_D).

Historical name: 100_family_corroboration.py

Usage
    python scripts/analysis/family_corroboration.py \
        --graph /path/to/run16E/kg_final/flat_run16E_enforced_dedup.json \
        --chunks /path/to/run16E/index/chunks.jsonl \
        --families configs/entity_families.csv \
        --out output/family_corroboration.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_GRAPH = str(Path.home() / "ontogeorag_D" / "output" / "run16E" /
                    "kg_final" / "flat_run16E_enforced_dedup.json")
DEFAULT_CHUNKS = str(Path.home() / "ontogeorag_D" / "output" / "run16E" /
                     "index" / "chunks.jsonl")

STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "is"}
PROCESS_WRAPPERS = {"development", "formation", "generation", "geometry",
                    "emplacement", "initiation", "growth", "occurrence"}
NEGATION_PREFIXES = ("un", "dis", "non", "in")
POLAR_PAIRS = [("high", "low"), ("thick", "thin"), ("increased", "decreased"),
               ("elevated", "reduced"), ("upper", "lower"),
               ("stability", "instability"), ("stable", "unstable")]


def singular(t: str) -> str:
    if len(t) <= 4:
        return t
    if t.endswith("ies"):
        return t[:-3] + "y"
    if t.endswith("sses"):
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def toks(text: str) -> list:
    text = str(text).lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [singular(t) for t in text.split() if t and t not in STOP]


def norm(text: str) -> str:
    return " ".join(toks(text))


def antonyms(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    for p in NEGATION_PREFIXES:
        if na == p + nb or nb == p + na:
            return True
    ta, tb = set(toks(a)), set(toks(b))
    for l, r in POLAR_PAIRS:
        if (l in ta and r in tb) or (r in ta and l in tb):
            return True
    return False


def group_concepts(entities: list):
    """Union-find over: token subset + same nominal head + not polar."""
    parent = {e: e for e in entities}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if len(ra) <= len(rb):
            parent[rb] = ra
        else:
            parent[ra] = rb

    decisions = []
    ents = sorted(entities)
    for i, a in enumerate(ents):
        ta = set(toks(a))
        if not ta:
            continue
        ha = toks(a)[-1]
        for b in ents[i + 1:]:
            tb = set(toks(b))
            if not tb:
                continue
            if toks(b)[-1] != ha or antonyms(a, b):
                continue
            if not (ta <= tb or tb <= ta):
                continue
            extra = (tb - ta) or (ta - tb)
            decisions.append({
                "a": a, "b": b,
                "crosses_process_wrapper":
                    "yes" if extra & PROCESS_WRAPPERS else "no",
            })
            union(a, b)

    fam = {e: find(e) for e in entities}
    members = defaultdict(list)
    for e, f in fam.items():
        members[f].append(e)
    return fam, {k: sorted(v) for k, v in members.items()}, decisions


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS)
    ap.add_argument("--out", default="family_corroboration.csv")
    ap.add_argument("--families", default=None,
                     help="Optional CSV with columns entity,family: manually "
                          "specified families overriding the automatic rule for the "
                          "entities they cover. The automatic rule requires a shared "
                          "nominal head, so it does not group 'deep water' with "
                          "'deepwater basin'.")
    args = ap.parse_args()

    for f in (args.graph, args.chunks):
        if not Path(f).exists():
            print("ERROR: not found: " + f, file=sys.stderr)
            return 1

    raw = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    records = raw.get("triples", raw) if isinstance(raw, dict) else raw
    triples = []
    for r in records:
        s = str(r.get("subject") or "").strip().lower()
        o = str(r.get("object") or "").strip().lower()
        rel = str(r.get("relation") or "").strip()
        if s and o and rel:
            try:
                tier = int(r.get("tier") or 1)
            except (TypeError, ValueError):
                tier = 1
            triples.append({"subject": s, "relation": rel, "object": o,
                            "tier": tier})

    entities = sorted({t["subject"] for t in triples} | {t["object"] for t in triples})
    fam, members, decisions = group_concepts(entities)
    source = "automatic rule (subset + shared nominal head + antonym guard)"

    if args.families:
        if not Path(args.families).exists():
            print("ERROR: families file not found: " + args.families, file=sys.stderr)
            return 1
        manual, n_read = {}, 0
        with open(args.families, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ent = str(row.get("entity") or row.get("form") or "").strip().lower()
                grp = str(row.get("family") or row.get("family_expert") or "").strip().lower()
                if ent and grp:
                    manual[ent] = grp
                    n_read += 1
        covered = sum(1 for e in entities if e in manual)
        print("  families file: " + str(n_read) + " rows, covering " +
              str(covered) + " of " + str(len(entities)) + " entities")
        if covered:
            # expert families win; entities absent from the file keep the
            # automatic grouping, so coverage gaps degrade gracefully
            for e in entities:
                if e in manual:
                    fam[e] = "expert::" + manual[e]
            members = defaultdict(list)
            for e, f in fam.items():
                members[f].append(e)
            members = {k: sorted(v) for k, v in members.items()}
            source = ("expert families for " + str(covered) + " entities, "
                      "automatic rule for the rest")
    n_multi = sum(1 for m in members.values() if len(m) > 1)

    print("Graph : " + args.graph)
    print("  " + str(len(triples)) + " relations, " + str(len(entities)) + " entities")
    print("  grouping: " + source)
    print("  " + str(len(members)) + " concept groups (" + str(n_multi) +
          " with more than one surface form)")

    # ---- index the corpus ------------------------------------------------
    chunks = []
    for line in open(args.chunks, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        doc = str(c.get("doc_id") or c.get("source_file") or "?")
        doc = re.sub(r"-checkpoint$", "", doc)
        chunks.append((doc, " " + norm(c.get("text", "")) + " "))
    papers = sorted({d for d, _ in chunks})
    print("  " + str(len(chunks)) + " chunks, " + str(len(papers)) + " papers\n")

    # Papers in which each entity surface form occurs, computed once.
    ent_papers = {}
    for e in entities:
        needle = " " + norm(e) + " "
        if not needle.strip():
            ent_papers[e] = set()
            continue
        ent_papers[e] = {d for d, txt in chunks if needle in txt}

    def papers_for(forms):
        out = set()
        for f in forms:
            out |= ent_papers.get(f, set())
        return out

    # Co-occurrence must be within the SAME chunk, as in 08.
    ent_chunks = {}
    for e in entities:
        needle = " " + norm(e) + " "
        ent_chunks[e] = {i for i, (_, txt) in enumerate(chunks)
                         if needle and needle in txt}
    chunk_doc = [d for d, _ in chunks]

    def cooc_papers(subj_forms, obj_forms):
        cs = set()
        for f in subj_forms:
            cs |= ent_chunks.get(f, set())
        co = set()
        for f in obj_forms:
            co |= ent_chunks.get(f, set())
        return {chunk_doc[i] for i in (cs & co)}

    rows = []
    for t in triples:
        surf = cooc_papers([t["subject"]], [t["object"]])
        fs, fo = fam[t["subject"]], fam[t["object"]]
        famp = cooc_papers(members[fs], members[fo])
        rows.append({
            "subject": t["subject"], "relation": t["relation"],
            "object": t["object"], "tier": t["tier"],
            "family_subject": fs, "family_object": fo,
            "n_forms_subject": len(members[fs]),
            "n_forms_object": len(members[fo]),
            "papers_surface": len(surf),
            "papers_family": len(famp),
            "delta": len(famp) - len(surf),
        })

    n = len(rows)
    one_surf = sum(1 for r in rows if r["papers_surface"] <= 1)
    one_fam = sum(1 for r in rows if r["papers_family"] <= 1)
    crossed = sum(1 for r in rows
                  if r["papers_surface"] <= 1 < r["papers_family"])
    moved = sum(1 for r in rows if r["delta"] > 0)

    print("=" * 70)
    print("CORROBORATION: SURFACE FORM VS CONCEPT FAMILY")
    print("=" * 70)
    print("  relations resting on a single paper")
    print("    surface level : " + str(one_surf) + " / " + str(n) +
          "  (" + format(one_surf / n, ".1%") + ")")
    print("    family level  : " + str(one_fam) + " / " + str(n) +
          "  (" + format(one_fam / n, ".1%") + ")")
    print("    crossing 1 -> 2+ papers : " + str(crossed))
    print("  relations whose count rises at all : " + str(moved))

    mean_s = sum(r["papers_surface"] for r in rows) / n
    mean_f = sum(r["papers_family"] for r in rows) / n
    print("  mean supporting papers : " + format(mean_s, ".2f") +
          " -> " + format(mean_f, ".2f"))

    print()
    if one_surf:
        share = crossed / one_surf
        print("  VERDICT")
        if share >= 0.30:
            print("    " + format(share, ".0%") + " of the single-paper relations are")
            print("    single-paper only because the concept is split across surface")
            print("    forms: the surface-level count mostly reflects redundancy.")
        elif share >= 0.10:
            print("    " + format(share, ".0%") + " of the single-paper relations move.")
            print("    Redundancy explains part of the figure, not most of it.")
        else:
            print("    Only " + format(share, ".0%") + " move: thin corroboration does not")
            print("    come from the graph's surface forms.")

    print()
    print("  Largest gains:")
    for r in sorted(rows, key=lambda x: -x["delta"])[:10]:
        if r["delta"] <= 0:
            break
        print("    +" + str(r["delta"]) + "  " + r["subject"] + " --" +
              r["relation"] + "--> " + r["object"] +
              "   (" + str(r["papers_surface"]) + " -> " +
              str(r["papers_family"]) + ")")

    wrap = [d for d in decisions if d["crosses_process_wrapper"] == "yes"]
    if wrap:
        print()
        print("  " + str(len(wrap)) + " grouping decision(s) cross a process/object")
        print("  boundary (e.g. 'development of X' with 'X'); listed in the CSV.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: -x["delta"]))

    with open(args.out.replace(".csv", "_groups.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["family", "n_forms", "forms"])
        w.writeheader()
        for k, v in sorted(members.items()):
            if len(v) > 1:
                w.writerow({"family": k, "n_forms": len(v),
                            "forms": " | ".join(v)})

    print()
    print("Written: " + args.out + " and " +
          args.out.replace(".csv", "_groups.csv"))
    print("Co-occurrence bounds consensus; it does not prove assertion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())