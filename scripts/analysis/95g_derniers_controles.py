#!/usr/bin/env python3
"""
95g_derniers_controles.py

The last disk questions before writing. Four independent checks, one pass.

  A  Reconcile the two compound-descriptor counts. The single-word counts
     already agree exactly between the two implementations, so the
     disagreement lives in the compound pattern, not in the corpus.
     Four counting variants are run side by side and each is tested
     against the previously published values.

  B  Does the hyphen/whitespace problem also affect KNOWN_SETTINGS?
     Same machinery as the descriptors, plus a check for members of the
     set that collapse onto each other.

  C  216 vs 217 entities. Every plausible counting convention is applied
     to the delivered graph so the 217 can be attributed to one of them,
     or shown to belong to none.

  D  Surface variants of `gas hydrate dissociation` in the delivered graph.

DECISION RULES, FIXED HERE
--------------------------
  R2 (unchanged)  attested iff >=1 match, word-boundary, case-insensitive,
                  internal hyphen may be hyphen OR whitespace.
  R6  A counting variant EXPLAINS a published value iff it reproduces it
      exactly. Approximate agreement is not agreement.
  R7  `X- Y` (hyphen then whitespace) is a PDF line-break form of the
      hyphenated spelling. `XY` (no separator) is a PDF concatenation
      artefact. Neither is evidence that the literature writes the term
      in two words. Only `X Y` is.

Read-only. Writes one JSON report.

Usage:
    python3 95g_derniers_controles.py
"""

import importlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
SCRIPTS_ROOT = HOME / "ontogeorag"
DATA_ROOT = HOME / "ontogeorag_D"

INDEX_CANDIDATES = [
    DATA_ROOT / "output" / "run16E" / "index" / "chunks.jsonl",
    SCRIPTS_ROOT / "output" / "run16E" / "index" / "chunks.jsonl",
]
GRAPH_CANDIDATES = [
    SCRIPTS_ROOT / "output" / "run16E" / "kg_final" / "flat_run16E_enforced_dedup.json",
    DATA_ROOT / "output" / "run16E" / "kg_final" / "flat_run16E_enforced_dedup.json",
]
PRIOR_SCRIPT = SCRIPTS_ROOT / "96_descriptor_frequencies_run16E.py"

TEXT_KEY_PREFERENCE = ["text", "chunk", "chunk_text", "content", "passage", "body"]
DESCRIPTOR_RELATION = "hasdescriptor"

# values reported by the earlier implementation, to be explained or refuted
PUBLISHED = {
    "high-amplitude": 103,
    "low-amplitude": 50,
    "moderate-amplitude": 7,
    "variable-amplitude": 4,
    "wedge-shaped": 19,
    "sheet-like": 4,
}

OUT_DIR = SCRIPTS_ROOT / "output" / "audit"
OUT_JSON = OUT_DIR / "95g_derniers_controles_report.json"

SEP = "=" * 78
SUB = "-" * 78


def first_existing(candidates):
    for c in candidates:
        if c.exists():
            return c
    return None


def load_constants():
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        return importlib.import_module("pipeline.rag.constants")
    except Exception:
        path = SCRIPTS_ROOT / "pipeline" / "rag" / "constants.py"
        spec = importlib.util.spec_from_file_location("ontogeorag_constants", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def load_chunks(path):
    texts = []
    key_used = None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if key_used is None:
                for cand in TEXT_KEY_PREFERENCE:
                    if isinstance(rec.get(cand), str):
                        key_used = cand
                        break
            val = rec.get(key_used)
            texts.append(val if isinstance(val, str) else "")
    return texts, key_used


# separator alternatives, in increasing permissiveness
VARIANTS = [
    ("literal", None),                 # exact term, hyphen as written
    ("opt_single", r"[- ]?"),          # hyphen, single space, or nothing
    ("ws_plus", r"[-\s]+"),            # one or more hyphen/whitespace  (rule R2)
    ("ws_star", r"[-\s]*"),            # same, plus concatenation
]


def build(term, sep):
    parts = [re.escape(p) for p in term.split("-") if p]
    if sep is None:
        return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    return re.compile(r"\b" + sep.join(parts) + r"\b", re.IGNORECASE)


def classify_form(term, matched):
    """R7: which surface class does this match belong to?"""
    m = matched.lower()
    hyphen = term.lower()
    spaced = term.replace("-", " ").lower()
    joined = term.replace("-", "").lower()
    collapsed = " ".join(m.split())
    if m == hyphen:
        return "hyphenated"
    if collapsed == spaced:
        return "spaced"
    if m == joined:
        return "concatenated"
    if "- " in m or "-\n" in m or re.search(r"-\s", m):
        return "line_break"
    return "other"


def count_variants(terms, texts):
    out = {}
    for term in terms:
        per_variant = {}
        for name, sep in VARIANTS:
            pat = build(term, sep)
            total = 0
            chunks = 0
            forms = Counter()
            classes = Counter()
            for t in texts:
                if not t:
                    continue
                hits = pat.findall(t)
                if hits:
                    total += len(hits)
                    chunks += 1
                    for h in hits:
                        h = str(h)
                        forms[" ".join(h.split()).lower() if "\n" in h else h.lower()] += 1
                        classes[classify_form(term, h)] += 1
            per_variant[name] = {"occurrences": total, "chunks": chunks,
                                 "forms": dict(forms.most_common(8)),
                                 "classes": dict(classes)}
        out[term] = per_variant
    return out


# --------------------------------------------------------------------------

def part_a(texts, report):
    print(SEP)
    print("PART A - reconciling the compound-descriptor counts  (rules R6, R7)")
    print(SEP)

    if PRIOR_SCRIPT.exists():
        print("prior implementation: " + str(PRIOR_SCRIPT))
        keep = re.compile(r"re\.compile|re\.escape|\bopen\(|chunks|jsonl|replace\(|\[\-|findall|IGNORECASE",
                          re.IGNORECASE)
        lines = PRIOR_SCRIPT.read_text(encoding="utf-8", errors="replace").splitlines()
        shown = 0
        for i, line in enumerate(lines, start=1):
            if keep.search(line):
                print("  " + str(i).rjust(4) + "  " + line.strip()[:120])
                shown += 1
            if shown >= 40:
                print("  ... (truncated)")
                break
        report["prior_script"] = {"path": str(PRIOR_SCRIPT), "n_lines": len(lines)}
    else:
        print("prior implementation NOT FOUND at " + str(PRIOR_SCRIPT))
        report["prior_script"] = None
    print(SUB)

    terms = sorted(PUBLISHED)
    counts = count_variants(terms, texts)
    names = [n for n, _ in VARIANTS]
    print("descriptor".ljust(22) + "published" + "".join(n.rjust(12) for n in names))
    print(SUB)
    explained = {}
    for term in terms:
        row = counts[term]
        line = term.ljust(22) + str(PUBLISHED[term]).rjust(9)
        matches = []
        for n in names:
            v = row[n]["occurrences"]
            mark = "*" if v == PUBLISHED[term] else " "
            line += (str(v) + mark).rjust(12)
            if v == PUBLISHED[term]:
                matches.append(n)
        print(line)
        explained[term] = matches
    print(SUB)
    print("* marks a variant that reproduces the published value exactly (rule R6)")
    print()

    all_names = set(names)
    consistent = [n for n in names if all(n in explained[t] for t in terms)]
    if consistent:
        print("VERDICT: the published values are reproduced across all six descriptors by: "
              + ", ".join(consistent))
        print("  -> the earlier implementation used that separator rule. The disagreement")
        print("     with the shipped count is a rule difference, not a corpus difference.")
    else:
        print("VERDICT: no single variant reproduces all six published values.")
        for t in terms:
            print("  " + t.ljust(22) + ("explained by " + ", ".join(explained[t])
                                        if explained[t] else "explained by NO variant"))
        print("  -> the earlier numbers do not come from a separator rule alone.")
        print("     Do not carry them into the manuscript; use the measured breakdown below.")
    print()

    print("surface breakdown under the most permissive variant (rule R7):")
    print("term".ljust(22) + "hyphenated".rjust(12) + "spaced".rjust(9)
          + "line_break".rjust(12) + "concatenated".rjust(14) + "  total")
    print(SUB)
    breakdown = {}
    for term in terms:
        cls = counts[term]["ws_star"]["classes"]
        total = counts[term]["ws_star"]["occurrences"]
        print(term.ljust(22)
              + str(cls.get("hyphenated", 0)).rjust(12)
              + str(cls.get("spaced", 0)).rjust(9)
              + str(cls.get("line_break", 0)).rjust(12)
              + str(cls.get("concatenated", 0)).rjust(14)
              + "  " + str(total))
        breakdown[term] = {"hyphenated": cls.get("hyphenated", 0),
                           "spaced": cls.get("spaced", 0),
                           "line_break": cls.get("line_break", 0),
                           "concatenated": cls.get("concatenated", 0),
                           "other": cls.get("other", 0),
                           "total": total,
                           "forms": counts[term]["ws_star"]["forms"]}
    print(SUB)
    print("The claim 'the literature writes it in two words' rests on the SPACED column only.")
    print("line_break and concatenated are PDF extraction artefacts of the hyphenated form.")
    print()
    report["part_a"] = {"published": PUBLISHED, "variants": counts,
                        "explained_by": explained, "consistent_variant": consistent,
                        "surface_breakdown": breakdown}


def part_b(mod, texts, report):
    print(SEP)
    print("PART B - does the same problem affect KNOWN_SETTINGS?")
    print(SEP)
    settings = sorted({str(s).strip().lower() for s in getattr(mod, "KNOWN_SETTINGS", [])})
    print("KNOWN_SETTINGS: " + str(len(settings)))

    collapsed = defaultdict(list)
    for s in settings:
        collapsed[re.sub(r"[-\s]+", " ", s)].append(s)
    dupes = {k: v for k, v in collapsed.items() if len(v) > 1}
    if dupes:
        print(SUB)
        print("members that collapse onto each other under hyphen/space normalisation:")
        for k, v in sorted(dupes.items()):
            print("  " + k.ljust(30) + " <- " + ", ".join(v))
        print("  -> the set counts them separately; any coverage score over 21 is inflated.")
    else:
        print("  no member collapses onto another.")

    compound = [s for s in settings if "-" in s]
    counts = count_variants(compound, texts) if compound else {}
    if compound:
        print(SUB)
        print("hyphenated settings, literal vs normalised:")
        print("setting".ljust(30) + "literal".rjust(9) + "ws_plus".rjust(9) + "  delta")
        print(SUB)
        affected = []
        for s in compound:
            lit = counts[s]["literal"]["occurrences"]
            nrm = counts[s]["ws_plus"]["occurrences"]
            print(s.ljust(30) + str(lit).rjust(9) + str(nrm).rjust(9)
                  + "  " + ("+" + str(nrm - lit) if nrm > lit else str(nrm - lit)))
            if nrm > lit:
                affected.append(s)
        print(SUB)
        if affected:
            print("AFFECTED: " + ", ".join(affected))
            print("  -> the same undercount as the descriptors. Any settings figure computed")
            print("     by literal matching is a lower bound.")
        else:
            print("no hyphenated setting is undercounted by literal matching.")
    report["part_b"] = {"n_settings": len(settings), "settings": settings,
                        "collapsing_members": {k: v for k, v in dupes.items()},
                        "compound_counts": counts}
    print()


def part_c(graph_path, report):
    print(SEP)
    print("PART C - 216 vs 217 entities")
    print(SEP)
    raw = json.loads(graph_path.read_text(encoding="utf-8", errors="replace"))
    triples = raw.get("triples", []) if isinstance(raw, dict) else raw

    subj_raw, obj_raw = [], []
    desc_obj = set()
    for t in triples:
        if not isinstance(t, dict):
            continue
        s = str(t.get("subject", ""))
        o = str(t.get("object", ""))
        subj_raw.append(s)
        obj_raw.append(o)
        if str(t.get("relation", "")).strip().lower() == DESCRIPTOR_RELATION:
            desc_obj.add(o.strip().lower())

    def uniq(vals, fn):
        return {fn(v) for v in vals if fn(v)}

    ident = lambda x: x
    strip = lambda x: x.strip()
    lower = lambda x: x.strip().lower()

    conventions = {
        "subjects+objects, raw": uniq(subj_raw + obj_raw, ident),
        "subjects+objects, stripped": uniq(subj_raw + obj_raw, strip),
        "subjects+objects, stripped+lowercased": uniq(subj_raw + obj_raw, lower),
        "subjects only, lowercased": uniq(subj_raw, lower),
        "objects only, lowercased": uniq(obj_raw, lower),
    }
    all_lower = uniq(subj_raw + obj_raw, lower)
    conventions["subjects+objects minus descriptor objects"] = all_lower - desc_obj

    print("triples: " + str(len(triples)))
    for name, s in conventions.items():
        print("  " + str(len(s)).rjust(5) + "   " + name)
    print(SUB)
    hits = [n for n, s in conventions.items() if len(s) == 217]
    if hits:
        print("217 is produced by: " + "; ".join(hits))
        print("  -> the 217 is a counting convention, not a second graph. Name the convention")
        print("     in the manuscript and use one throughout.")
    else:
        print("NO convention on this graph yields 217.")
        print("  -> the 217 comes from another artefact or another run. Treat 216 as the")
        print("     figure and drop the 217 rather than reconciling it.")

    case_pairs = defaultdict(set)
    for v in uniq(subj_raw + obj_raw, strip):
        case_pairs[v.lower()].add(v)
    multi = {k: sorted(v) for k, v in case_pairs.items() if len(v) > 1}
    if multi:
        print(SUB)
        print("entities differing only by case (" + str(len(multi)) + "):")
        for k, v in sorted(multi.items())[:15]:
            print("  " + " | ".join(v))
    report["part_c"] = {"n_triples": len(triples),
                        "counts": {k: len(v) for k, v in conventions.items()},
                        "yields_217": hits,
                        "case_only_variants": multi}
    print()


def part_d(graph_path, report):
    print(SEP)
    print("PART D - surface variants of `gas hydrate dissociation`")
    print(SEP)
    raw = json.loads(graph_path.read_text(encoding="utf-8", errors="replace"))
    triples = raw.get("triples", []) if isinstance(raw, dict) else raw
    roles = defaultdict(Counter)
    for t in triples:
        if not isinstance(t, dict):
            continue
        for role in ("subject", "object"):
            v = str(t.get(role, "")).strip()
            if "hydrate" in v.lower():
                roles[v.lower()][role] += 1
    if not roles:
        print("no entity containing 'hydrate' in the delivered graph.")
    else:
        print("form".ljust(46) + "subject".rjust(9) + "object".rjust(8) + "  total")
        print(SUB)
        for form in sorted(roles, key=lambda k: -sum(roles[k].values())):
            c = roles[form]
            print(form.ljust(46) + str(c.get("subject", 0)).rjust(9)
                  + str(c.get("object", 0)).rjust(8) + "  " + str(sum(c.values())))
        print(SUB)
        print("distinct forms: " + str(len(roles)))
        print("  -> these are the merge candidates. Merging changes the entity count,")
        print("     so decide before quoting 216.")
    report["part_d"] = {form: dict(c) for form, c in roles.items()}
    print()


def main():
    report = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "rules": ["R6 a variant explains a published value iff it reproduces it exactly",
                        "R7 'X- Y' is a line-break form, 'XY' a concatenation artefact; only 'X Y' is a spaced spelling"]}

    index_path = first_existing(INDEX_CANDIDATES)
    graph_path = first_existing(GRAPH_CANDIDATES)
    if index_path is None or graph_path is None:
        print("FATAL: index or graph not found.")
        for c in INDEX_CANDIDATES + GRAPH_CANDIDATES:
            print("  " + str(c) + ("  [present]" if c.exists() else "  [absent]"))
        sys.exit(2)

    texts, key_used = load_chunks(index_path)
    print(SEP)
    print("INPUTS")
    print(SEP)
    print("index : " + str(index_path) + "   chunks: " + str(len(texts)) + "   field: " + str(key_used))
    print("graph : " + str(graph_path))
    print()
    report["inputs"] = {"index": str(index_path), "n_chunks": len(texts),
                        "text_field": key_used, "graph": str(graph_path)}

    mod = load_constants()
    part_a(texts, report)
    part_b(mod, texts, report)
    part_c(graph_path, report)
    part_d(graph_path, report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(SEP)
    print("report written: " + str(OUT_JSON))
    print(SEP)


if __name__ == "__main__":
    main()