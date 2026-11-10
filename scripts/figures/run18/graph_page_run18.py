#!/usr/bin/env python3
"""Interactive page of the run18 knowledge graph (one self-contained HTML file).

The page shows the cleaned graph (281 relations) or the delivered graph (364 relations, with the
83 relations removed by the written rules drawn in grey), by entity name or by name family.
Clicking an arrow shows the evidence sentence, the source papers with the deduced section, the
tier, the panel decision and the evidence-check verdict of each verifying-LLM.

Inputs (all in this repository): the graph with located provenance (187), the cleaned graph, the
removed and flagged relations (188), and the table of name families. The script checks the counts
of the manuscript and writes nothing if one of them differs.

The graph library vis-network 9.1.9 is loaded from a CDN, or embedded with --vis-js, which makes
the page work offline.

Usage
    python scripts/figures/run18/graph_page_run18.py --out output/figures/ontogeorag_graph_run18.html
    python scripts/figures/run18/graph_page_run18.py --out page.html --vis-js vis-network.min.js
"""
import argparse
import csv
import json
import re
from pathlib import Path

import run18_style as S

EXPECTED = {"delivered": 364, "cleaned": 281, "removed": 83, "entities_delivered": 272,
            "entities_cleaned": 230, "families_cleaned": 139, "flagged": 79,
            "tier1": 197, "accepted": 238}

# rule and flag labels of 188 are written in French in its outputs; the page shows them in English
RULES = [("R1 bibliographie", "R1 reference list"),
         ("R2 etiquette", "R2 local label"),
         ("R3 sujet absent des phrases sources", "R3 subject absent from the source sentences")]
FLAGS = [("S1 lieu nomme", "S1 named place"),
         ("S2 commentaire du modele", "S2 comment of the model"),
         ("S3 legende de figure probable", "S3 probable figure caption"),
         ("S4 sujet repris par un pronom", "S4 subject given by a pronoun")]
WORDS = [("(sujet code ", "(subject code "), ("(objet code ", "(object code "),
         ("(sujet etiquette ", "(subject label "), ("(objet etiquette ", "(object label ")]
CDN = "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def english(text, table):
    for fr, en in table + WORDS:
        text = text.replace(fr, en)
    return text


def short_reference(ref, doc_id):
    """'Bull, S., Cartwright, J. & Huuse, M. (2009) A review...' -> 'Bull et al. (2009)'."""
    m = re.match(r"^(.*?)\((\d{4}[a-z]?)\)", ref or "")
    if not m:
        return doc_id
    authors = [a.strip() for a in re.split(r",\s*(?:&\s*)?|\s&\s", m.group(1)) if a.strip()]
    surnames = [a for a in authors if not re.fullmatch(r"(?:[A-Z][a-z]?\.\s*-?)+", a)]
    if not surnames:
        return doc_id
    if len(surnames) == 1:
        name = surnames[0]
    elif len(surnames) == 2:
        name = surnames[0] + " & " + surnames[1]
    else:
        name = surnames[0] + " et al."
    return "%s (%s)" % (name, m.group(2))


def key(subject, relation, obj):
    return (subject, relation, obj)


def build():
    delivered = S.triples(S.WITH_PROVENANCE)
    cleaned = {key(t["subject"], t["relation"], t["object"]) for t in S.triples(S.CLEANED)}
    families = {}
    with open(S.FAMILIES, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            families[row["entity"]] = row["family"] or row["entity"]
    removed = {}
    with open(S.REMOVED, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            removed[key(row["sujet"], row["relation"], row["objet"])] = english(row["regles"], RULES)
    flags = {}
    with open(S.REMOVED.parent / "relations_signalees.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            flags[key(row["sujet"], row["relation"], row["objet"])] = english(row["signalements"], FLAGS)

    relations, types = [], {}
    for t in delivered:
        k = key(t["subject"], t["relation"], t["object"])
        llama, mistral = (t["m4"]["evidence_verdict"].split("|") + ["", ""])[:2]
        sources, seen = [], set()
        for s in t["provenance"]["sources"]:
            item = (s.get("reference") or s.get("doc_id"), s.get("section_deduite"), s.get("passage"))
            if item in seen:
                continue
            seen.add(item)
            sources.append({"short": short_reference(s.get("reference"), s.get("doc_id")),
                            "reference": s.get("reference") or "", "doi": s.get("doi") or "",
                            "section": s.get("section_deduite") or "", "passage": s.get("passage") or "",
                            "match": s.get("statut") or "", "caption": bool(s.get("legende_figure_probable"))})
        relations.append({
            "s": t["subject"], "r": t["relation"], "o": t["object"],
            "fs": S.label(families.get(t["subject"], t["subject"])),
            "fo": S.label(families.get(t["object"], t["object"])),
            "tier": int(t["tier"]), "panel": t["m4"]["decision"],
            "llama": llama, "mistral": mistral, "flagged": S.flagged(t),
            "kept": k in cleaned, "rules": removed.get(k, ""), "flags": flags.get(k, ""),
            "evidence": str(t.get("evidence", "")).strip().strip('"').strip(),
            "sources": sources,
        })
        types.setdefault(t["subject"], t["subject_type"])
        types.setdefault(t["object"], t["object_type"])

    kept = [r for r in relations if r["kept"]]
    m = {"delivered": len(relations), "cleaned": len(kept), "removed": sum(1 for r in relations if r["rules"]),
         "entities_delivered": len({x for r in relations for x in (r["s"], r["o"])}),
         "entities_cleaned": len({x for r in kept for x in (r["s"], r["o"])}),
         "families_cleaned": len({x for r in kept for x in (r["fs"], r["fo"])}),
         "flagged": sum(r["flagged"] for r in relations),
         "tier1": sum(r["tier"] == 1 for r in relations),
         "accepted": sum(r["panel"] == "ACCEPT" for r in relations)}
    print(m)
    S.check(m, EXPECTED)
    if any(r["kept"] == bool(r["rules"]) for r in relations):
        raise SystemExit("ERROR: a relation is both kept and removed, or neither; nothing written")
    entity_types = {n: {"label": lab, "colour": col} for n, (lab, col) in S.ENTITY_TYPES.items()}
    return {"relations": relations, "types": types, "entity_types": entity_types,
            "relation_colours": S.RELATION_COLOURS, "counts": m}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="output/figures/ontogeorag_graph_run18.html")
    ap.add_argument("--vis-js", default=None, help="embed this copy of vis-network.min.js instead of the CDN")
    args = ap.parse_args()
    data = build()
    template = (Path(__file__).with_name("graph_page_run18_template.html")).read_text(encoding="utf-8")
    if args.vis_js:
        lib = "<script>" + Path(args.vis_js).read_text(encoding="utf-8").replace("</script", "<\\/script") + "</script>"
    else:
        lib = '<script src="%s"></script>' % CDN
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("<!--VIS-->", lib).replace("/*DATA*/null", payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print("written: %s (%d kB)" % (out, out.stat().st_size // 1024))


if __name__ == "__main__":
    main()
