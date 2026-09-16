"""
kg_io.py — Shared, format-tolerant I/O for OntoGeoRAG KG analysis suite.

Handles BOTH serialization formats observed in the project:
  * run11 style : list of [key, value] pairs -> dict(...) with key 'triples',
                  each triple carrying 'tier' (int).
  * M4 style    : dict with keys {meta, tier1, tier2, quarantine}, each a list
                  of triples.

Also loads the chunk index (chunks.jsonl from step 01) and resolves
chunk_id -> paper_id with tolerant field detection, so provenance can be
rebuilt without modifying any existing pipeline script.

No domain lexicon is hardcoded here: descriptors, objects, settings are always
derived from the KG content itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ── tolerant key sets ─────────────────────────────────────────────────
SUBJECT_KEYS = ("subject", "source", "head", "subj", "s")
OBJECT_KEYS = ("object", "target", "tail", "obj", "o")
RELATION_KEYS = ("relation", "predicate", "rel", "p")
CHUNKID_KEYS = (
    "selected_chunk_ids",
    "selected_chunks",
    "chunk_ids",
    "chunks",
    "evidence_chunk_ids",
    "supporting_chunks",
    "chunk_id",
    "retrieved_chunk_ids",
)
M4_VERDICT_KEYS = (
    "m4_verdict",
    "m4",
    "verdict",
    "m4_final",
    "panel_verdict",
    "verification",
)
PAPER_KEYS = (
    "paper_id",
    "paper",
    "doc_id",
    "doc",
    "source_paper",
    "pdf",
    "source_pdf",
    "source",
    "file",
    "filename",
    "document",
)
CHUNK_INDEX_ID_KEYS = ("chunk_id", "id", "cid", "uid")


def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def get_subject(t: dict) -> str:
    return str(_first(t, SUBJECT_KEYS, "")).strip()


def get_object(t: dict) -> str:
    return str(_first(t, OBJECT_KEYS, "")).strip()


def get_relation(t: dict) -> str:
    return str(_first(t, RELATION_KEYS, "")).strip()


def get_m4_verdict(t: dict) -> str:
    v = _first(t, M4_VERDICT_KEYS, "")
    if isinstance(v, dict):
        v = _first(v, ("final", "verdict", "label"), "")
    return str(v).upper()


def get_chunk_ids(t: dict) -> list:
    """Collect chunk ids from any known provenance field, incl. per-pass
    sub-records (pass_a / pass_b) if present. Returns de-duplicated list."""
    out = []

    def _collect(val):
        if val is None:
            return
        if isinstance(val, (list, tuple, set)):
            for v in val:
                _collect(v)
        elif isinstance(val, dict):
            for kk in CHUNKID_KEYS:
                if kk in val:
                    _collect(val[kk])
        else:
            out.append(str(val))

    for k in CHUNKID_KEYS:
        if k in t:
            _collect(t[k])
    for passes_key in (
        "pass_a",
        "pass_b",
        "passes",
        "provenance",
        "_provenance",
    ):
        if passes_key in t and isinstance(t[passes_key], (dict, list)):
            _collect(t[passes_key])
    # de-dup, keep order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


# ── KG loading ────────────────────────────────────────────────────────


def load_kg(path: str | Path) -> dict:
    """Return {'format': 'run11'|'m4', 'meta': dict, 'triples': [dict, ...]}.

    Every triple is annotated in-place with:
        _tier   : 1, 2, or 0 (quarantine / unknown)
        _status : 'active' or 'quarantine'
    Original triple dicts are preserved (annotation keys start with '_').
    """
    raw = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(raw, list):  # run11 list-of-pairs serialization
        raw = dict(raw)

    meta = raw.get("meta", raw.get("metadata", {})) or {}

    if "triples" in raw:  # run11 style
        triples = list(raw["triples"])
        for t in triples:
            t["_tier"] = int(t.get("tier", 0) or 0)
            t["_status"] = "active"
        return {"format": "run11", "meta": meta, "triples": triples}

    if "tier1" in raw or "tier2" in raw:  # M4 style
        triples = []
        for key, tier, status in (
            ("tier1", 1, "active"),
            ("tier2", 2, "active"),
            ("quarantine", 0, "quarantine"),
        ):
            for t in raw.get(key, []) or []:
                t["_tier"] = tier
                t["_status"] = status
                triples.append(t)
        return {"format": "m4", "meta": meta, "triples": triples}

    raise ValueError(
        f"Unrecognized KG format in {path}: top-level keys = {list(raw)[:8]}"
    )


def dump_kg(kg: dict, path: str | Path) -> None:
    """Write back in the same structural format it was loaded from,
    stripping only the private '_' annotations that duplicate structure."""
    triples = kg["triples"]

    def _clean(t):
        return {k: v for k, v in t.items() if not k.startswith("_")}

    if kg["format"] == "m4":
        out = {
            "meta": kg.get("meta", {}),
            "tier1": [
                _clean(t)
                for t in triples
                if t["_tier"] == 1 and t["_status"] == "active"
            ],
            "tier2": [
                _clean(t)
                for t in triples
                if t["_tier"] == 2 and t["_status"] == "active"
            ],
            "quarantine": [
                _clean(t) for t in triples if t["_status"] == "quarantine"
            ],
        }
    else:
        out = {
            "meta": kg.get("meta", {}),
            "triples": [_clean(t) for t in triples],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ── chunk index (step 01 output) ──────────────────────────────────────


def load_chunk_index(chunks_path: str | Path) -> dict:
    """Return {chunk_id(str) -> paper_id(str)} from chunks.jsonl.

    Field names are auto-detected on the first record; a summary of what was
    detected is printed so mis-detection is immediately visible.
    """
    chunks_path = Path(chunks_path)
    index, id_key, paper_key = {}, None, None
    with open(chunks_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if id_key is None:
                id_key = next(
                    (k for k in CHUNK_INDEX_ID_KEYS if k in rec), None
                )
                paper_key = next((k for k in PAPER_KEYS if k in rec), None)
                print(
                    f"[kg_io] chunk index field detection: "
                    f"id_key={id_key!r}, paper_key={paper_key!r} "
                    f"(record keys: {sorted(rec.keys())})"
                )
                if id_key is None or paper_key is None:
                    raise ValueError(
                        "Could not detect chunk_id / paper fields in "
                        f"{chunks_path}. First record keys: "
                        f"{sorted(rec.keys())}. Extend CHUNK_INDEX_ID_KEYS "
                        "or PAPER_KEYS in kg_io.py accordingly."
                    )
            index[str(rec[id_key])] = normalize_paper_id(str(rec[paper_key]))
    print(
        f"[kg_io] chunk index loaded: {len(index)} chunks, "
        f"{len(set(index.values()))} papers"
    )
    return index


def normalize_paper_id(p: str) -> str:
    """Strip path and extension so 'data/corpus/Foo et al 2019.pdf' and
    'Foo et al 2019' resolve to the same paper id. Also strips the
    '-checkpoint' suffix left by Jupyter .ipynb_checkpoints duplicates
    (observed corpus contamination: each paper indexed twice)."""
    p = p.replace("\\", "/").split("/")[-1]
    p = re.sub(r"\.(pdf|txt|json)$", "", p, flags=re.I)
    p = re.sub(r"-checkpoint$", "", p, flags=re.I)
    return p.strip()


# ── figure style (project conventions) ────────────────────────────────
BLUE = "#2E5E8C"
TERRACOTTA = "#C4622D"
PALETTE = [
    BLUE,
    TERRACOTTA,
    "#6B8F71",
    "#8C6BB1",
    "#B1A16B",
    "#5E8CA0",
    "#A05E5E",
    "#708090",
]


def apply_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlecolor": "black",
            "axes.titlesize": 18,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    return plt


def load_chunk_records(chunks_path):
    """Return list of {chunk_id, paper, text} with duplicate texts removed
    (checkpoint copies collapse to one record). Field names auto-detected."""
    import hashlib

    chunks_path = Path(chunks_path)
    records, id_key, paper_key, text_key = [], None, None, None
    seen_hash = set()
    n_dup = 0
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if id_key is None:
                id_key = next(
                    (k for k in CHUNK_INDEX_ID_KEYS if k in rec), None
                )
                paper_key = next((k for k in PAPER_KEYS if k in rec), None)
                text_key = next(
                    (
                        k
                        for k in ("text", "content", "chunk_text", "passage")
                        if k in rec
                    ),
                    None,
                )
                print(
                    f"[kg_io] chunk records: id={id_key!r}, "
                    f"paper={paper_key!r}, text={text_key!r}"
                )
                if not all((id_key, paper_key, text_key)):
                    raise ValueError(
                        f"field detection failed: {sorted(rec.keys())}"
                    )
            paper = normalize_paper_id(str(rec[paper_key]))
            text = str(rec[text_key])
            h = hashlib.md5((paper + "|" + text).encode()).hexdigest()
            if h in seen_hash:  # checkpoint duplicate
                n_dup += 1
                continue
            seen_hash.add(h)
            records.append(
                {"chunk_id": str(rec[id_key]), "paper": paper, "text": text}
            )
    papers = {r["paper"] for r in records}
    print(
        f"[kg_io] chunk records: {len(records)} unique chunks "
        f"({n_dup} checkpoint/duplicate chunks removed), "
        f"{len(papers)} papers"
    )
    return records
