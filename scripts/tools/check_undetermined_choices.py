#!/usr/bin/env python3
"""scripts/tools/check_undetermined_choices.py -- decide whether the failed checks
of 184 come only from undetermined descriptor choices that have no effect.

Read-only, CPU only. Called by workflows/postprocess_run18.sh when
184_trace_requetes_et_pertes.py reports failed checks (exit code 3).

Why this check exists
    Script 04 replaces a descriptor object that contains several descriptors of
    the vocabulary (for example "chaotic with layered") by the first of them in
    the iteration order of a Python set, which depends on PYTHONHASHSEED. To
    replay 04, script 184 reads the choice made by the run in the output of 04
    (canonical_triples_v5.jsonl). When the relation is removed or merged as a
    duplicate inside 04, the choice is written nowhere: 184 marks it "non
    determinable" and counts a failed check (ECHEC). 184 is kept byte-identical
    to the code of the recorded run, so the case is handled here.

The failed checks are accepted, and the cleaning goes on, only if all of the
following hold:
    1. every failed check of 184 is a check of ambiguous objects whose only
       defect is "non determinable" (no incomplete matching, no order conflict);
    2. every other check of 184 passed, in particular "sorties de 04 =
       canonical_triples_v5.jsonl" for each pass: the replayed output of 04 is
       the output of the run, relation by relation;
    3. every undetermined object leaves the pipeline inside 04 (verdict filter,
       validation, duplicate or self-loop), according to suivi_extractions.tsv,
       so that no choice of 04 can reach the fused graph.
Otherwise the script exits with code 3 and the cleaning must stop.

Usage:
    python scripts/tools/check_undetermined_choices.py \
        --trace <postprocess>/184_trace --outdir <postprocess>/184_undetermined_check

Output: <outdir>/summary.txt (also printed). --outdir must not exist yet.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

# exit stages of suivi_extractions.tsv that are inside 04 (see 184, section B)
EXITS_IN_04 = {"04_verdict", "04_validation", "doublon_04", "boucle_04"}
AMBIGUOUS = re.compile(r"^ECHEC passe ([ab]) : \d+ objet\(s\) ambigu\(s\), \d+ choix lus dans la sortie du run"
                       r"  \[(\d+) non determinable\(s\)\]$")


def read_tsv(p):
    csv.field_size_limit(sys.maxsize)
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="output folder of 184 (184_trace)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    trace, out = Path(args.trace), Path(args.outdir)
    if out.exists():
        sys.exit("STOP: %s exists, nothing is overwritten." % out)
    for f in ("_resume.txt", "normalisations_ambigues.tsv", "suivi_extractions.tsv"):
        if not (trace / f).is_file():
            sys.exit("STOP: %s not found in %s" % (f, trace))

    lines = (trace / "_resume.txt").read_text(encoding="utf-8").splitlines()
    checks = [ln.rstrip() for ln in lines if ln.startswith("OK    ") or ln.startswith("ECHEC ")]
    failed = [c for c in checks if c.startswith("ECHEC ")]
    problems = []
    undetermined_per_pass = {}
    if not failed:
        problems.append("184 reports no failed check: nothing to accept here")
    for c in failed:
        m = AMBIGUOUS.match(c)
        if m:
            undetermined_per_pass[m.group(1)] = int(m.group(2))
        else:
            problems.append("failed check that is not an undetermined choice: " + c)
    for p in sorted(undetermined_per_pass):
        ok04 = [c for c in checks if c.startswith("OK    passe %s : sorties de 04 = canonical_triples_v5.jsonl" % p)]
        if not ok04:
            problems.append("pass %s: the check of the output of 04 did not pass" % p)

    rows = [r for r in read_tsv(trace / "normalisations_ambigues.tsv")
            if r.get("origine_du_choix") == "non determinable"]
    found = {}
    for r in rows:
        found[r["passe"]] = found.get(r["passe"], 0) + 1
    if found != undetermined_per_pass:
        problems.append("undetermined objects in normalisations_ambigues.tsv %s differ from the failed checks %s"
                        % (found, undetermined_per_pass))

    trace_rows = {(r["passe"], r["ligne"]): r for r in read_tsv(trace / "suivi_extractions.tsv")}
    report = []
    for r in rows:
        t = trace_rows.get((r["passe"], r["ligne"]))
        stage = t["etape_sortie"] if t else "(line not found)"
        inside = t is not None and stage in EXITS_IN_04
        report.append("  pass %s, line %s: %r, candidates %s, leaves at %s%s"
                      % (r["passe"], r["ligne"], r["objet_brut"], r["candidats"], stage or "(enters the fusion)",
                         "" if inside else "  <- reaches the output of 04"))
        if not inside:
            problems.append("pass %s line %s: the undetermined object does not leave the pipeline inside 04"
                            % (r["passe"], r["ligne"]))

    accepted = not problems
    txt = ["=== check of the failed checks of 184 ===",
           "184 checks: %d, failed: %d" % (len(checks), len(failed)),
           "undetermined descriptor choices:"] + (report or ["  none"]) + [""]
    if accepted:
        txt.append("ACCEPTED: the failed checks are only undetermined choices of objects that leave the pipeline "
                   "inside 04, and the replayed output of 04 equals the run's. They have no effect on the graph; "
                   "the cleaning goes on.")
    else:
        txt += ["NOT ACCEPTED:"] + ["  " + p for p in problems] + ["The cleaning stops; read 184_trace/_resume.txt."]
    out.mkdir(parents=True)
    (out / "summary.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(txt), flush=True)
    sys.exit(0 if accepted else 3)


if __name__ == "__main__":
    main()
