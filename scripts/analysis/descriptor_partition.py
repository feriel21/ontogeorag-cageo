#!/usr/bin/env python3
"""Partition of the extraction vocabulary against the corpus and the delivered graph.

Supports Section 3.5 and Figure 4 (1 absent, 8 attested but not extracted,
31 extracted, out of 40) and the coverage of the 13 reference descriptors
(12 recovered).

Three checks in one pass:
  (1) status of run16F, an exploratory branch, recorded so that it is not
      mistaken for the delivered configuration;
  (2) frequency of each descriptor over the run16E index, with hyphen and
      whitespace treated as equivalent;
  (3) coverage of the reference descriptors measured on the delivered graph.

Decision rules (fixed before the counts were read)
  R1  run16F is a deliverable candidate only if it has BOTH an enforced,
      deduplicated graph under kg_final/ AND a manifest.json with
      git_status_clean true. Otherwise it is an exploratory branch.
  R2  A descriptor is ATTESTED iff it matches at least once in the indexed
      corpus: case-insensitive, whole-word, each internal hyphen matching a
      hyphen OR whitespace. No stemming, no plural. Zero matches = ABSENT.
  R3  A descriptor is EXTRACTED iff it is, lower-cased and stripped, the
      object of at least one hasDescriptor triple of the delivered graph.
      Matching by content only, never by position.
  R4  Input paths come from explicit candidate lists and are reported.
  R5  A descriptor is STRUCTURALLY UNREACHABLE iff it is a key of
      DESCRIPTOR_SYNONYMS or DESCRIPTOR_FUZZY_MAP mapped to a different
      descriptor: normalisation replaces it before the graph is written.

Read-only with respect to the pipeline. Writes output/audit/95d_partition_report.json.
Default paths follow the original layout (~/ontogeorag, ~/ontogeorag_D).

Historical name: 95d_fermer_partition_descripteurs.py

Usage
    python scripts/analysis/descriptor_partition.py
    python scripts/analysis/descriptor_partition.py --vocab-attr NAME --lb-attr NAME
    python scripts/analysis/descriptor_partition.py --list-constants
"""

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
SCRIPTS_ROOT = HOME / "ontogeorag"
DATA_ROOT = HOME / "ontogeorag_D"

# R4: explicit candidates only.
INDEX_CANDIDATES = [
    DATA_ROOT / "output" / "run16E" / "index" / "chunks.jsonl",
    SCRIPTS_ROOT / "output" / "run16E" / "index" / "chunks.jsonl",
]
GRAPH_CANDIDATES = [
    SCRIPTS_ROOT / "output" / "run16E" / "kg_final" / "flat_run16E_enforced_dedup.json",
    DATA_ROOT / "output" / "run16E" / "kg_final" / "flat_run16E_enforced_dedup.json",
]
RUN16F_ROOTS = [
    SCRIPTS_ROOT / "output" / "run16F",
    DATA_ROOT / "output" / "run16F",
]
RUN16E_ROOTS = [
    SCRIPTS_ROOT / "output" / "run16E",
    DATA_ROOT / "output" / "run16E",
]

EXPECTED_CHUNKS = 1758  # verified on disk; tripwire only, not an assumption
TEXT_KEY_PREFERENCE = ["text", "chunk", "chunk_text", "content", "passage", "body"]
DESCRIPTOR_RELATION = "hasdescriptor"
NORMALISATION_TABLES = ["DESCRIPTOR_SYNONYMS", "DESCRIPTOR_FUZZY_MAP"]

OUT_DIR = SCRIPTS_ROOT / "output" / "audit"
OUT_JSON = OUT_DIR / "95d_partition_report.json"

SEP = "=" * 78
SUB = "-" * 78


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def mtime(path):
    try:
        ts = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def first_existing(candidates):
    for c in candidates:
        if c.exists():
            return c
    return None


def norm_pattern(descriptor):
    """R2: internal hyphens may be hyphen or whitespace."""
    parts = [re.escape(p) for p in descriptor.split("-") if p]
    joined = r"[-\s]+".join(parts)
    return re.compile(r"\b" + joined + r"\b", re.IGNORECASE)


def literal_pattern(descriptor):
    return re.compile(r"\b" + re.escape(descriptor) + r"\b", re.IGNORECASE)


def stem_pattern(descriptor):
    """Diagnostic only, applied to descriptors that come out at zero."""
    head = descriptor.split("-")[0]
    stem = head[:8] if len(head) >= 4 else head
    return stem, re.compile(r"\b" + re.escape(stem) + r"\w*", re.IGNORECASE)


# --------------------------------------------------------------------------
# (1) run16F status
# --------------------------------------------------------------------------

def audit_run16F():
    print(SEP)
    print("PART 1 - run16F status  (rule R1)")
    print(SEP)

    report = {"roots": [], "verdict": None, "reason": None}

    found_any = False
    for root in RUN16F_ROOTS:
        entry = {"path": str(root), "exists": root.exists()}
        print("root: " + str(root) + ("   [present]" if root.exists() else "   [absent]"))
        if not root.exists():
            report["roots"].append(entry)
            continue
        found_any = True

        subdirs = sorted([p.name for p in root.iterdir() if p.is_dir()])
        files = sorted([p.name for p in root.iterdir() if p.is_file()])
        entry["subdirs"] = subdirs
        entry["files"] = files
        print("  subdirs: " + (", ".join(subdirs) if subdirs else "(none)"))
        print("  files  : " + (", ".join(files) if files else "(none)"))

        kg_final = root / "kg_final"
        enforced = []
        if kg_final.is_dir():
            for p in sorted(kg_final.iterdir()):
                if p.is_file() and p.suffix == ".json":
                    tag = "  <-- enforced/dedup" if ("enforced" in p.name or "dedup" in p.name) else ""
                    print("  kg_final/" + p.name + "   " + str(p.stat().st_size) + " B   " + str(mtime(p)) + tag)
                    if "enforced" in p.name or "dedup" in p.name:
                        enforced.append(p.name)
        else:
            print("  kg_final/ : ABSENT")
        entry["kg_final_exists"] = kg_final.is_dir()
        entry["enforced_graphs"] = enforced

        # every json under the root, so nothing hides one level down
        all_json = sorted(str(p.relative_to(root)) for p in root.rglob("*.json"))
        entry["all_json"] = all_json
        print("  all .json under root (" + str(len(all_json)) + "):")
        for rel in all_json:
            print("    " + rel + "   " + str(mtime(root / rel)))

        man = root / "manifest.json"
        entry["manifest"] = None
        if man.exists():
            try:
                m = json.loads(man.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                m = {"_parse_error": repr(exc)}
            entry["manifest"] = m
            print("  manifest.json keys: " + ", ".join(sorted(str(k) for k in m)))
            for k in ("run", "git_commit", "git_describe", "git_status_clean", "benchmark"):
                if k in m:
                    print("    " + k + " = " + repr(m[k]))
        else:
            print("  manifest.json : ABSENT")

        report["roots"].append(entry)

    # comparison anchor: newest artefact under run16E
    e_root = first_existing(RUN16E_ROOTS)
    if e_root is not None:
        newest = None
        for p in e_root.rglob("*"):
            if p.is_file():
                try:
                    ts = p.stat().st_mtime
                except OSError:
                    continue
                if newest is None or ts > newest[0]:
                    newest = (ts, p)
        if newest is not None:
            report["run16E_newest_artifact"] = {"path": str(newest[1]), "mtime": mtime(newest[1])}
            print(SUB)
            print("newest artefact under run16E: " + str(newest[1]))
            print("  mtime: " + str(mtime(newest[1])))

    # R1
    if not found_any:
        report["verdict"] = "ABSENT"
        report["reason"] = "no run16F directory under either root"
    else:
        candidate = False
        for entry in report["roots"]:
            if not entry.get("exists"):
                continue
            man = entry.get("manifest") or {}
            if entry.get("enforced_graphs") and man.get("git_status_clean") is True:
                candidate = True
        report["verdict"] = "DELIVERABLE_CANDIDATE" if candidate else "EXPLORATORY_BRANCH"
        report["reason"] = (
            "enforced/dedup graph AND manifest.git_status_clean true"
            if candidate
            else "missing enforced/dedup graph and/or clean manifest (rule R1)"
        )

    print(SUB)
    print("VERDICT run16F: " + report["verdict"])
    print("  reason: " + report["reason"])
    print("  -> an exploratory branch is not a delivered configuration; run16E is.")
    print()
    return report


# --------------------------------------------------------------------------
# vocabulary from constants.py
# --------------------------------------------------------------------------

def load_constants():
    sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        return importlib.import_module("pipeline.rag.constants")
    except Exception as exc_pkg:
        path = SCRIPTS_ROOT / "pipeline" / "rag" / "constants.py"
        if not path.exists():
            print("FATAL: cannot import pipeline.rag.constants and " + str(path) + " does not exist")
            print("  package import error: " + repr(exc_pkg))
            sys.exit(2)
        spec = importlib.util.spec_from_file_location("ontogeorag_constants", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc_file:
            print("FATAL: constants.py failed to execute standalone: " + repr(exc_file))
            print("  package import error was: " + repr(exc_pkg))
            sys.exit(2)
        return mod


def string_containers(mod):
    """name -> list of strings, for every str container attribute."""
    out = {}
    for name, val in vars(mod).items():
        if name.startswith("_"):
            continue
        items = None
        if isinstance(val, dict):
            keys = list(val.keys())
            if keys and all(isinstance(k, str) for k in keys):
                items = keys
        elif isinstance(val, (list, tuple, set, frozenset)):
            vals = list(val)
            if vals and all(isinstance(v, str) for v in vals):
                items = vals
        if items is not None:
            out[name] = sorted({s.strip().lower() for s in items if s.strip()})
    return out


def pick_attr(containers, size, hint, explicit, label):
    if explicit:
        if explicit not in containers:
            print("FATAL: --" + label + "-attr " + explicit + " is not a string container in constants.py")
            sys.exit(2)
        return explicit
    exact = [n for n, v in containers.items() if len(v) == size]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        hinted = [n for n in exact if hint in n.lower()]
        if len(hinted) == 1:
            return hinted[0]
    print("FATAL: cannot identify the " + label + " set (expected size " + str(size) + ").")
    print("  candidates of that size: " + (", ".join(exact) if exact else "(none)"))
    print("  string containers in constants.py, name -> size:")
    for n in sorted(containers, key=lambda k: (-len(containers[k]), k)):
        print("    " + str(len(containers[n])).rjust(4) + "  " + n)
    print("  rerun with --" + label + "-attr NAME")
    sys.exit(2)


def unreachable_descriptors(mod, vocab):
    """R5: vocabulary members that a normalisation table maps onto another
    canonical, and are therefore removed before the graph is written."""
    vocab_set = set(vocab)
    out = {}
    tables_seen = {}
    for table_name in NORMALISATION_TABLES:
        table = getattr(mod, table_name, None)
        if not isinstance(table, dict):
            tables_seen[table_name] = None
            continue
        tables_seen[table_name] = len(table)
        for k, v in table.items():
            key = str(k).strip().lower()
            val = str(v).strip().lower() if isinstance(v, str) else None
            if key in vocab_set and val and val != key:
                out.setdefault(key, []).append({"table": table_name, "target": val,
                                                "target_in_vocab": val in vocab_set})
    return out, tables_seen


# --------------------------------------------------------------------------
# (2) corpus counting
# --------------------------------------------------------------------------

def load_chunks(path):
    texts = []
    key_used = None
    bad = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                continue
            if not isinstance(rec, dict):
                bad += 1
                continue
            if key_used is None:
                for cand in TEXT_KEY_PREFERENCE:
                    if isinstance(rec.get(cand), str):
                        key_used = cand
                        break
                if key_used is None:
                    best, blen = None, -1
                    for k, v in rec.items():
                        if isinstance(v, str) and len(v) > blen:
                            best, blen = k, len(v)
                    key_used = best
            val = rec.get(key_used)
            texts.append(val if isinstance(val, str) else "")
    return texts, key_used, bad


def count_corpus(vocab, texts):
    rows = {}
    for d in vocab:
        lit_re = literal_pattern(d)
        nrm_re = norm_pattern(d)
        lit_occ = nrm_occ = nrm_chunks = 0
        surfaces = Counter()
        for t in texts:
            if not t:
                continue
            lit_occ += len(lit_re.findall(t))
            hits = nrm_re.findall(t)
            if hits:
                nrm_occ += len(hits)
                nrm_chunks += 1
                for h in hits:
                    surfaces[" ".join(str(h).split()).lower()] += 1
        row = {
            "literal_occurrences": lit_occ,
            "normalised_occurrences": nrm_occ,
            "normalised_chunks": nrm_chunks,
            "delta": nrm_occ - lit_occ,
            "surface_forms": dict(surfaces.most_common(6)),
        }
        if nrm_occ == 0:
            stem, st_re = stem_pattern(d)
            forms = Counter()
            n = 0
            for t in texts:
                if not t:
                    continue
                for h in st_re.findall(t):
                    forms[str(h).lower()] += 1
                    n += 1
            row["zero_probe"] = {"stem": stem, "occurrences": n, "forms": dict(forms.most_common(6))}
        rows[d] = row
    return rows


# --------------------------------------------------------------------------
# (3) delivered graph
# --------------------------------------------------------------------------

def load_graph_descriptors(path):
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(raw, dict):
        triples = raw.get("triples", [])
        meta = raw.get("meta", {})
    elif isinstance(raw, list):
        triples = raw
        meta = {}
    else:
        print("FATAL: unexpected graph top-level type: " + type(raw).__name__)
        sys.exit(2)

    counts = Counter()
    tiers = defaultdict(set)
    subjects = defaultdict(set)
    n_desc_triples = 0
    for t in triples:
        if not isinstance(t, dict):
            continue
        rel = str(t.get("relation", "")).strip().lower()
        if rel != DESCRIPTOR_RELATION:
            continue
        n_desc_triples += 1
        obj = str(t.get("object", "")).strip().lower()
        if not obj:
            continue
        counts[obj] += 1
        tiers[obj].add(t.get("tier"))
        subjects[obj].add(str(t.get("subject", "")).strip().lower())
    return {
        "n_triples_total": len(triples),
        "n_hasDescriptor_triples": n_desc_triples,
        "counts": counts,
        "tiers": {k: sorted(str(x) for x in v) for k, v in tiers.items()},
        "subjects": {k: sorted(v) for k, v in subjects.items()},
        "meta_keys": sorted(str(k) for k in meta) if isinstance(meta, dict) else [],
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-attr", default=None, help="name of the full descriptor set in constants.py")
    ap.add_argument("--lb-attr", default=None, help="name of the LB2019 descriptor set in constants.py")
    ap.add_argument("--vocab-size", type=int, default=40)
    ap.add_argument("--lb-size", type=int, default=13)
    ap.add_argument("--list-constants", action="store_true", help="print string containers and exit")
    args = ap.parse_args()

    report = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "rules": ["R1 run16F candidate iff enforced/dedup graph AND git_status_clean true",
                        "R2 attested iff >=1 match, word-boundary, case-insensitive, hyphen or whitespace",
                        "R3 extracted iff object of a hasDescriptor triple in the delivered graph",
                        "R4 explicit paths only, never built by analogy"]}

    report["run16F"] = audit_run16F()

    mod = load_constants()
    containers = string_containers(mod)
    print(SEP)
    print("VOCABULARY  (source: " + str(getattr(mod, "__file__", "constants.py")) + ")")
    print(SEP)
    if args.list_constants:
        for n in sorted(containers, key=lambda k: (-len(containers[k]), k)):
            print(str(len(containers[n])).rjust(4) + "  " + n)
        return

    vocab_attr = pick_attr(containers, args.vocab_size, "desc", args.vocab_attr, "vocab")
    lb_attr = pick_attr(containers, args.lb_size, "lb", args.lb_attr, "lb")
    vocab = containers[vocab_attr]
    lb = containers[lb_attr]
    print("full set  : " + vocab_attr + "  (n=" + str(len(vocab)) + ")")
    print("LB2019 set: " + lb_attr + "  (n=" + str(len(lb)) + ")")
    missing_from_vocab = [d for d in lb if d not in set(vocab)]
    if missing_from_vocab:
        print("WARNING: LB2019 members absent from the full set: " + ", ".join(missing_from_vocab))
    unreachable, tables_seen = unreachable_descriptors(mod, vocab)
    print("normalisation tables: " + ", ".join(
        t + "=" + (str(n) if n is not None else "absent/not-a-dict") for t, n in tables_seen.items()))
    if unreachable:
        print("STRUCTURALLY UNREACHABLE (rule R5): " + str(len(unreachable)))
        for d in sorted(unreachable):
            for route in unreachable[d]:
                print("  " + d.ljust(24) + " -> " + route["target"]
                      + "   via " + route["table"]
                      + ("   [target also in the vocabulary]" if route["target_in_vocab"] else ""))
        lb_unreachable = sorted(set(unreachable) & set(lb))
        if lb_unreachable:
            print("  of which LB2019 members: " + ", ".join(lb_unreachable))
            print("  -> PREDICTION from the tables: attainable LB2019 coverage would be "
                  + str(len(lb) - len(lb_unreachable)) + "/" + str(len(lb)) + ".")
            print("     PART 3 measures it against the graph. The measurement wins.")
    else:
        print("STRUCTURALLY UNREACHABLE (rule R5): none")
    report["vocabulary"] = {"vocab_attr": vocab_attr, "lb_attr": lb_attr,
                            "vocab": vocab, "lb2019": lb,
                            "lb_not_in_vocab": missing_from_vocab,
                            "normalisation_tables": tables_seen,
                            "structurally_unreachable": unreachable}
    print()

    index_path = first_existing(INDEX_CANDIDATES)
    if index_path is None:
        print("FATAL: no chunks.jsonl found. Candidates checked:")
        for c in INDEX_CANDIDATES:
            print("  " + str(c))
        sys.exit(2)
    texts, key_used, bad = load_chunks(index_path)
    print(SEP)
    print("CORPUS")
    print(SEP)
    print("index : " + str(index_path))
    print("chunks: " + str(len(texts)) + "   text field: " + str(key_used) + "   unparsed lines: " + str(bad))
    if len(texts) != EXPECTED_CHUNKS:
        print("WARNING: expected " + str(EXPECTED_CHUNKS) + " chunks, got " + str(len(texts)) + " - resolve before using any count")
    report["corpus"] = {"index_path": str(index_path), "n_chunks": len(texts),
                        "text_field": key_used, "unparsed_lines": bad,
                        "expected_chunks": EXPECTED_CHUNKS}
    print()

    rows = count_corpus(vocab, texts)

    graph_path = first_existing(GRAPH_CANDIDATES)
    if graph_path is None:
        print("FATAL: delivered graph not found. Candidates checked:")
        for c in GRAPH_CANDIDATES:
            print("  " + str(c))
        sys.exit(2)
    g = load_graph_descriptors(graph_path)
    print(SEP)
    print("DELIVERED GRAPH")
    print(SEP)
    print("graph : " + str(graph_path))
    print("triples: " + str(g["n_triples_total"]) + "   hasDescriptor: " + str(g["n_hasDescriptor_triples"]))
    print("distinct descriptor objects: " + str(len(g["counts"])))
    report["graph"] = {"path": str(graph_path),
                       "n_triples_total": g["n_triples_total"],
                       "n_hasDescriptor_triples": g["n_hasDescriptor_triples"],
                       "n_distinct_objects": len(g["counts"])}
    print()

    vocab_set = set(vocab)
    extracted = set(g["counts"])
    off_vocab = sorted(extracted - vocab_set)
    if off_vocab:
        print("descriptor objects outside the declared vocabulary (" + str(len(off_vocab)) + "):")
        for d in off_vocab:
            print("  " + d + "   x" + str(g["counts"][d]))
        print()
    report["off_vocabulary_objects"] = {d: g["counts"][d] for d in off_vocab}

    # ---- partition -------------------------------------------------------
    absent, attested_not_extracted, extracted_in_vocab = [], [], []
    unreachable_class = []
    anomaly_unattested, anomaly_declared_unreachable = [], []
    for d in vocab:
        occ = rows[d]["normalised_occurrences"]
        is_ext = d in extracted
        if is_ext and occ == 0:
            anomaly_unattested.append(d)
        if is_ext:
            extracted_in_vocab.append(d)
            if d in unreachable:
                anomaly_declared_unreachable.append(d)
        elif d in unreachable:
            unreachable_class.append(d)
        elif occ == 0:
            absent.append(d)
        else:
            attested_not_extracted.append(d)

    print(SEP)
    print("PART 2 - descriptor partition  (rules R2, R3)")
    print(SEP)
    header = ("descriptor".ljust(26) + "literal".rjust(8) + "normal".rjust(8)
              + "delta".rjust(7) + "chunks".rjust(8) + "  extracted  tier(s)")
    print(header)
    print(SUB)
    for d in sorted(vocab, key=lambda x: (-rows[x]["normalised_occurrences"], x)):
        r = rows[d]
        tier_s = ",".join(g["tiers"].get(d, [])) if d in extracted else ""
        print(d.ljust(26)
              + str(r["literal_occurrences"]).rjust(8)
              + str(r["normalised_occurrences"]).rjust(8)
              + ("+" + str(r["delta"]) if r["delta"] > 0 else str(r["delta"])).rjust(7)
              + str(r["normalised_chunks"]).rjust(8)
              + ("     yes" if d in extracted else "      no").rjust(11)
              + "  " + tier_s)
    print(SUB)
    print("ABSENT (" + str(len(absent)) + "): " + (", ".join(sorted(absent)) if absent else "(none)"))
    print("STRUCTURALLY UNREACHABLE (" + str(len(unreachable_class)) + "): "
          + (", ".join(sorted(unreachable_class)) if unreachable_class else "(none)"))
    print("ATTESTED NOT EXTRACTED (" + str(len(attested_not_extracted)) + "): "
          + (", ".join(sorted(attested_not_extracted)) if attested_not_extracted else "(none)"))
    print("EXTRACTED (" + str(len(extracted_in_vocab)) + ")")
    print("PARTITION: " + str(len(absent)) + " + " + str(len(unreachable_class))
          + " + " + str(len(attested_not_extracted))
          + " + " + str(len(extracted_in_vocab)) + " = " + str(len(vocab)))
    print("  (absent + unreachable + attested-not-extracted + extracted)")
    if anomaly_unattested:
        print("ANOMALY A: extracted but zero corpus match under R2: "
              + ", ".join(sorted(anomaly_unattested)))
        print("  -> a surface-form mismatch that the counter does not model.")
    if anomaly_declared_unreachable:
        print("ANOMALY B: declared unreachable by a normalisation table, yet present in the graph: "
              + ", ".join(sorted(anomaly_declared_unreachable)))
        print("  -> the declared synonymy is NOT enforced on the delivered graph.")
        print("     Both the term and its declared target coexist as distinct objects.")
        print("     This is a defect to declare, not a count to correct.")

    for d in sorted(absent):
        probe = rows[d].get("zero_probe", {})
        print("  zero probe on " + d + ": stem " + str(probe.get("stem"))
              + " -> " + str(probe.get("occurrences")) + " occurrences  "
              + str(probe.get("forms")))
        print("  (diagnostic only; R2 is the rule that classifies)")

    report["partition"] = {"absent": sorted(absent),
                           "structurally_unreachable": sorted(unreachable_class),
                           "attested_not_extracted": sorted(attested_not_extracted),
                           "extracted": sorted(extracted_in_vocab),
                           "anomalies_extracted_but_unattested": sorted(set(anomaly_unattested)),
                           "anomalies_declared_unreachable_yet_present": sorted(set(anomaly_declared_unreachable)),
                           "counts": [len(absent), len(unreachable_class),
                                      len(attested_not_extracted), len(extracted_in_vocab)]}
    report["descriptor_rows"] = rows
    print()

    # ---- LB2019 coverage -------------------------------------------------
    print(SEP)
    print("PART 3 - LB2019 coverage in the delivered graph  (rule R3)")
    print(SEP)
    print("descriptor".ljust(26) + "normal".rjust(8) + "  in graph  tier(s)   n_triples  subjects")
    print(SUB)
    covered, uncovered = [], []
    lb_detail = {}
    for d in sorted(lb):
        occ = rows.get(d, {}).get("normalised_occurrences")
        in_graph = d in extracted
        tier_s = ",".join(g["tiers"].get(d, [])) if in_graph else ""
        n_tr = g["counts"].get(d, 0)
        subs = g["subjects"].get(d, [])
        print(d.ljust(26)
              + (str(occ) if occ is not None else "n/a").rjust(8)
              + ("       yes" if in_graph else "        no")
              + "  " + tier_s.ljust(8)
              + str(n_tr).rjust(9) + "  " + ", ".join(subs[:3]) + (" ..." if len(subs) > 3 else ""))
        (covered if in_graph else uncovered).append(d)
        lb_detail[d] = {"normalised_occurrences": occ, "in_graph": in_graph,
                        "tiers": g["tiers"].get(d, []), "n_triples": n_tr, "subjects": subs}
    print(SUB)
    print("coverage: " + str(len(covered)) + "/" + str(len(lb)) + "   (measured on the delivered graph)")
    predicted_unreachable = sorted(set(lb) & set(unreachable))
    refuted = sorted(set(predicted_unreachable) & set(covered))
    if predicted_unreachable:
        print("tables predicted unreachable: " + ", ".join(predicted_unreachable))
        if refuted:
            print("  REFUTED by the graph: " + ", ".join(refuted) + " is present.")
            print("  -> the normalisation rule exists in constants.py but is not applied.")
            print("  -> there is no coverage ceiling below " + str(len(lb)) + "/" + str(len(lb))
                  + "; report the defect instead of a ceiling.")
    print("cause of each miss:")
    for d in uncovered:
        if d in unreachable:
            tgt = ", ".join(r["target"] for r in unreachable[d])
            cause = "UNREACHABLE, normalised onto " + tgt + " (design decision, not a failure)"
        elif d in set(absent):
            cause = "ABSENT from the corpus"
        else:
            cause = "attested in the corpus but not extracted (extraction failure)"
        print("  " + d.ljust(24) + cause)
    print("  -> these are different phenomena and must not be reported as one score.")
    gap = sorted(set(uncovered) - set(attested_not_extracted) - set(absent) - set(unreachable))
    if gap:
        print("GAP: uncovered yet classified as extracted: " + ", ".join(gap))
    report["lb2019_coverage"] = {"covered": covered, "uncovered": uncovered,
                                 "n_covered": len(covered), "detail": lb_detail, "gap": gap}
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    OUT_JSON.write_text(payload, encoding="utf-8")
    print(SEP)
    print("report written: " + str(OUT_JSON))
    print(SEP)


if __name__ == "__main__":
    main()