#!/usr/bin/env python3
"""
180_blind50_knowledge_form.py

Blind second-annotator subsample for the knowledge-form study (H1, H2, H4).
Standard library only, Python 3.10. Never overwrites a file. Stops on the
first failed check.

Two sub-commands:

  make     Draws 25 causal rows (causes, triggers, affects, controls) and
           25 hasDescriptor rows from the annotation sheet with a recorded
           seed, shuffles them, assigns blind ids B01..B50 and writes a sheet
           that shows ONLY subject, predicate, object and evidence sentence.
           No row id, no paper, no tier, no pipeline score, no suggestion,
           no first-annotator answer. Also writes a private key and a meta file.

  unblind  Reads the sheet filled by the second annotator and the private key,
           checks integrity and completeness, checks that the first annotator
           has finished the same 50 rows, and writes a second-annotator sheet
           keyed by the original row id, with the column names h1_final,
           h2_final, h4_final of the main sheet.

Usage on the cluster:
  python -u 180_blind50_knowledge_form.py make
  python -u 180_blind50_knowledge_form.py unblind \
      --filled ~/ontogeorag/output/audit/knowledge_form_blind50/blind50_filled_john.csv
"""
import argparse
import csv
import datetime
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- constants
EXPECTED_ROWS = 232
CAUSAL = {"causes", "triggers", "affects", "controls"}  # 40 + 29 + 25 + 8
DESCRIPTOR = "hasdescriptor"
EXPECTED_CAUSAL = 102
EXPECTED_DESCRIPTOR = 83
N_CAUSAL = 25
N_DESCRIPTOR = 25
DEFAULT_SEED = 20260927

DEFAULT_SHEET = "~/ontogeorag/output/audit/knowledge_form_sheet.csv"
DEFAULT_OUTDIR = "~/ontogeorag/output/audit/knowledge_form_blind50"
BLIND_NAME = "blind50_for_second_annotator.csv"
KEY_NAME = "blind50_key_PRIVATE.csv"
META_NAME = "blind50_meta.json"
SECOND_NAME = "second_annotator.csv"

CANDIDATES = {
    "id": ["row_id", "id", "triple_id", "idx", "index", "rid"],
    "subject": ["subject", "head", "subj", "s"],
    "predicate": ["predicate", "relation", "rel", "p"],
    "object": ["object", "tail", "obj", "o"],
    "evidence": ["evidence", "sentence", "evidence_sentence", "evidence_text", "text"],
}
FINAL_COLS = {"h1": "h1_final", "h2": "h2_final", "h4": "h4_final"}
APPLIES = {"causal": ("h1", "h2"), "descriptor": ("h4",)}
NA = "n/a"
# A source column copied into the blind sheet must not carry any of these.
FORBIDDEN_TOKENS = ("suggest", "final", "tier", "metric", "verdict", "panel",
                    "decision", "support", "paper", "doc", "score", "note")


# ---------------------------------------------------------------- helpers
def stop(msg):
    print("STOP: " + msg, file=sys.stderr)
    sys.exit(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    if not path.is_file():
        stop("file not found: " + str(path))
    # Lines starting with '#' are comments (the annotation sheet opens with the
    # coding rules, e.g. "# h1_final: conditional | unconditional").
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    rows = list(reader)
    header = reader.fieldnames or []
    return header, rows


def resolve(header, role, override):
    if override:
        if override not in header:
            stop("column '" + override + "' (for " + role + ") not in header: " + ", ".join(header))
        return override
    lower = {c.lower(): c for c in header}
    for cand in CANDIDATES[role]:
        if cand in lower:
            return lower[cand]
    stop("no column found for " + role + ". Header is: " + ", ".join(header)
         + ". Pass --" + role + "-col explicitly.")


def id_sort_key(value):
    try:
        return (0, int(value), "")
    except ValueError:
        return (1, 0, value)


def stratum_of(predicate):
    p = predicate.strip().lower()
    if p in CAUSAL:
        return "causal"
    if p == DESCRIPTOR:
        return "descriptor"
    return None


def refuse_existing(paths):
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        stop("refusing to overwrite: " + ", ".join(existing))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------- make
def cmd_make(args):
    sheet = Path(args.sheet).expanduser()
    outdir = Path(args.outdir).expanduser()
    blind_p, key_p, meta_p = outdir / BLIND_NAME, outdir / KEY_NAME, outdir / META_NAME
    refuse_existing([blind_p, key_p, meta_p])

    header, rows = read_csv(sheet)
    cols = {r: resolve(header, r, getattr(args, r + "_col")) for r in CANDIDATES}
    print("columns used:", json.dumps(cols))

    for role in ("subject", "predicate", "object", "evidence"):
        name = cols[role].lower()
        bad = [t for t in FORBIDDEN_TOKENS if t in name]
        if bad:
            stop("column '" + cols[role] + "' resolved for " + role
                 + " looks like a pipeline or annotation column (" + ", ".join(bad) + ")")
    for fc in FINAL_COLS.values():
        if fc not in header:
            stop("expected column '" + fc + "' not in sheet: wrong file?")

    if len(rows) != EXPECTED_ROWS:
        stop("sheet has " + str(len(rows)) + " rows, expected " + str(EXPECTED_ROWS))

    ids = [r[cols["id"]].strip() for r in rows]
    if any(not i for i in ids):
        stop("empty id in column " + cols["id"])
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        stop("duplicate ids: " + ", ".join(dup[:10]))

    strata = {"causal": [], "descriptor": []}
    for r in rows:
        s = stratum_of(r[cols["predicate"]])
        if s:
            strata[s].append(r)
    if len(strata["causal"]) != EXPECTED_CAUSAL or len(strata["descriptor"]) != EXPECTED_DESCRIPTOR:
        counts = Counter(r[cols["predicate"]].strip() for r in rows)
        stop("strata sizes causal=" + str(len(strata["causal"])) + " descriptor="
             + str(len(strata["descriptor"])) + ", expected 102 and 83. Predicates: "
             + json.dumps(dict(counts)))

    for s, members in strata.items():
        empty = [m[cols["id"]] for m in members if not m[cols["evidence"]].strip()]
        if empty:
            stop("empty evidence sentence in stratum " + s + " for ids: " + ", ".join(empty))

    rng = random.Random(args.seed)
    picked = []
    for s, n in (("causal", N_CAUSAL), ("descriptor", N_DESCRIPTOR)):
        pool = sorted(strata[s], key=lambda r: id_sort_key(r[cols["id"]].strip()))
        for r in rng.sample(pool, n):
            picked.append((s, r))
    rng.shuffle(picked)

    blind_header = ["blind_id", "subject", "predicate", "object", "evidence_sentence",
                    "h1", "h2", "h4", "comment"]
    blind_rows, key_rows = [], []
    for k, (s, r) in enumerate(picked, start=1):
        bid = "B{:02d}".format(k)
        h = {hk: (NA if hk not in APPLIES[s] else "") for hk in ("h1", "h2", "h4")}
        blind_rows.append([bid, r[cols["subject"]], r[cols["predicate"]], r[cols["object"]],
                           r[cols["evidence"]], h["h1"], h["h2"], h["h4"], ""])
        key_rows.append([bid, r[cols["id"]].strip(), s, r[cols["subject"]],
                         r[cols["predicate"]], r[cols["object"]]])

    # Final guard: nothing but the whitelisted fields leaves the sheet.
    if set(blind_header) != {"blind_id", "subject", "predicate", "object",
                             "evidence_sentence", "h1", "h2", "h4", "comment"}:
        stop("blind header changed unexpectedly")
    for r in blind_rows:
        for cell in r[5:8]:
            if cell not in ("", NA):
                stop("pre-filled answer cell in blind sheet: " + r[0])

    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(blind_p, blind_header, blind_rows)
    write_csv(key_p, ["blind_id", cols["id"], "stratum", "subject", "predicate", "object"], key_rows)
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__)),
        "sheet": str(sheet),
        "sheet_sha256_at_sampling": sha256(sheet),
        "seed": args.seed,
        "strata_sizes": {"causal": EXPECTED_CAUSAL, "descriptor": EXPECTED_DESCRIPTOR},
        "sample_sizes": {"causal": N_CAUSAL, "descriptor": N_DESCRIPTOR},
        "causal_predicates": sorted(CAUSAL),
        "columns": cols,
        "sampled_ids": sorted([k[1] for k in key_rows], key=id_sort_key),
        "blind_sheet_columns": blind_header,
    }
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    pc = Counter(k[4].strip() for k in key_rows)
    print("sampled predicates:", json.dumps(dict(pc)))
    print("wrote", blind_p, "(send this one)")
    print("wrote", key_p, "(PRIVATE: do not send)")
    print("wrote", meta_p)
    print("OK")


# ---------------------------------------------------------------- unblind
def cmd_unblind(args):
    outdir = Path(args.outdir).expanduser()
    key_p = Path(args.key).expanduser() if args.key else outdir / KEY_NAME
    filled_p = Path(args.filled).expanduser()
    sheet = Path(args.sheet).expanduser()
    out_p = Path(args.out).expanduser() if args.out else outdir / SECOND_NAME
    refuse_existing([out_p])

    key_header, key_rows = read_csv(key_p)
    id_col = key_header[1]
    key = {r["blind_id"]: r for r in key_rows}
    if len(key) != N_CAUSAL + N_DESCRIPTOR:
        stop("key has " + str(len(key)) + " rows")

    _, filled = read_csv(filled_p)
    fb = [r.get("blind_id", "").strip() for r in filled]
    if len(fb) != len(set(fb)):
        stop("duplicate blind_id in filled sheet")
    if set(fb) != set(key):
        stop("blind ids differ from key. missing: " + ", ".join(sorted(set(key) - set(fb)))
             + " extra: " + ", ".join(sorted(set(fb) - set(key))))

    for r in filled:
        k = key[r["blind_id"].strip()]
        for f in ("subject", "predicate", "object"):
            if r.get(f, "").strip() != k[f].strip():
                stop(r["blind_id"] + ": " + f + " differs from key (row edited or shifted?)")

    missing = []
    for r in filled:
        s = key[r["blind_id"].strip()]["stratum"]
        for hk in APPLIES[s]:
            v = r.get(hk, "").strip()
            if not v or v.lower() == NA:
                missing.append(r["blind_id"] + ":" + hk)
    if missing:
        stop("second annotator left " + str(len(missing)) + " cells empty: " + ", ".join(missing))

    s_header, s_rows = read_csv(sheet)
    if id_col not in s_header:
        stop("id column '" + id_col + "' of the key not in sheet")
    by_id = {r[id_col].strip(): r for r in s_rows}

    # The first annotator must be done on these rows before unblinding.
    canon = {hk: {} for hk in FINAL_COLS}
    not_done = []
    for k in key.values():
        row = by_id.get(k[id_col].strip())
        if row is None:
            stop("id " + k[id_col] + " of key not in sheet")
        for hk in APPLIES[k["stratum"]]:
            v = row.get(FINAL_COLS[hk], "").strip()
            if not v:
                not_done.append(k[id_col] + ":" + FINAL_COLS[hk])
    if not_done:
        stop("first annotator has not finished " + str(len(not_done))
             + " cells of the 50 sampled rows: " + ", ".join(not_done[:20]))
    for row in s_rows:
        for hk, fc in FINAL_COLS.items():
            v = row.get(fc, "").strip()
            if v:
                canon[hk][v.lower()] = v

    out_rows, unseen = [], []
    for r in filled:
        k = key[r["blind_id"].strip()]
        vals = {hk: "" for hk in FINAL_COLS}
        for hk in APPLIES[k["stratum"]]:
            v = r[hk].strip()
            if v.lower() in canon[hk]:
                vals[hk] = canon[hk][v.lower()]
            elif args.accept_new_codes:
                vals[hk] = v
            else:
                unseen.append(r["blind_id"] + ":" + hk + "=" + v)
        out_rows.append([k[id_col], vals["h1"], vals["h2"], vals["h4"]])
    if unseen:
        stop("values absent from the first annotator's codes (typo or new code?): "
             + ", ".join(unseen) + ". Codes in use: " + json.dumps(
                 {hk: sorted(set(c.values())) for hk, c in canon.items()})
             + ". If the code is a valid PROTOCOLE.md code, rerun with --accept-new-codes.")

    out_rows.sort(key=lambda x: id_sort_key(x[0]))
    write_csv(out_p, [id_col, FINAL_COLS["h1"], FINAL_COLS["h2"], FINAL_COLS["h4"]], out_rows)
    for hk in FINAL_COLS:
        dist = Counter(x[1 + list(FINAL_COLS).index(hk)] for x in out_rows)
        dist.pop("", None)
        print(hk, "second annotator:", json.dumps(dict(dist)))
    print("wrote", out_p)
    print("OK")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make")
    m.add_argument("--sheet", default=DEFAULT_SHEET)
    m.add_argument("--outdir", default=DEFAULT_OUTDIR)
    m.add_argument("--seed", type=int, default=DEFAULT_SEED)
    for role in CANDIDATES:
        m.add_argument("--" + role + "-col", dest=role + "_col", default=None)

    u = sub.add_parser("unblind")
    u.add_argument("--filled", required=True)
    u.add_argument("--sheet", default=DEFAULT_SHEET)
    u.add_argument("--outdir", default=DEFAULT_OUTDIR)
    u.add_argument("--key", default=None)
    u.add_argument("--out", default=None)
    u.add_argument("--accept-new-codes", action="store_true",
                   help="keep a second-annotator code never used by the first annotator")

    args = ap.parse_args()
    if args.cmd == "make":
        cmd_make(args)
    else:
        cmd_unblind(args)


if __name__ == "__main__":
    main()