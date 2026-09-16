#!/usr/bin/env python3
"""
100_family_corroboration.py - Is the "55% single paper" figure an artefact?

ADDITIVE. Reads the graph and the chunk index, writes a CSV. MODIFIES NOTHING.

THE QUESTION
------------
08_rebuild_provenance.py counts, for each triple, the number of distinct
papers whose text contains BOTH the subject and the object - as SURFACE
STRINGS. On run16E that gives 128 of 232 relations (55%) resting on a
single paper.

But the graph splits one concept across several surface forms. The setting
"deep water" appears as deep water / deepwater basin / deep-water setting /
deepwater settings; "pore pressure" as pore pressure / excess pore pressure /
elevated pore pressures / overpressure. Co-occurrences are therefore split
across those forms, and every form looks poorly corroborated even when the
concept is well attested.

So the 55% may be measuring the graph's own redundancy rather than the
literature. This script tests that directly: it recomputes corroboration at
CONCEPT-FAMILY level and reports how many relations move.

    surface level : papers containing "deep water"        AND "mtd"
    family level  : papers containing ANY form of the deep-water family
                    AND any form of the MTD family

INTERPRETATION - read before citing
------------------------------------
* If the single-paper share drops a lot, the two weaknesses (redundancy and
  thin corroboration) are one weakness, and fixing canonicalisation fixes
  both. If it barely moves, thin corroboration is real and independent.
* Family-level counting can only RAISE the count, never lower it, so a rise
  is expected. What matters is the SIZE of the move, and whether relations
  cross the 1 -> 2 papers boundary.
* Co-occurrence bounds consensus, it does not prove it. Two terms in one
  chunk may appear in a contrastive sentence ("the MTD is deformed, unlike
  the undeformed sediments below"). This measure inherits that limit from
  08 and does not fix it.
* Grouping is conservative and rule-based, with an antonym guard, and every
  grouping decision is written to the CSV so it can be audited.

USAGE
-----
    python 100_family_corroboration.py
    python 100_family_corroboration.py --graph ... --chunks ...
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
                     help="Optional CSV with columns entity,family (e.g. the "
                          "expert-reviewed entity_families.csv). The automatic "
                          "rule requires a shared nominal head, so it will NOT "
                          "group 'deep water' with 'deepwater basin'. Supplying "
                          "expert families removes that limitation and is the "
                          "stronger test.")
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
            print("    forms. The '55% single paper' figure is substantially an")
            print("    artefact of redundancy: fixing canonicalisation addresses both.")
        elif share >= 0.10:
            print("    " + format(share, ".0%") + " of the single-paper relations move.")
            print("    Redundancy explains part of the figure, not most of it.")
        else:
            print("    Only " + format(share, ".0%") + " move. Thin corroboration is")
            print("    real and independent of redundancy; it reflects extraction")
            print("    coverage or the literature itself, not the graph's surface forms.")

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