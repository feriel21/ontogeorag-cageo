#!/usr/bin/env python3
"""Check that the repository files are the ones recorded in PROVENANCE.tsv and in the run manifest.

Checks, all read-only:

1. PROVENANCE.tsv lists every file copied into this repository, with its origin
   (a path at the recorded commit, or a later script) and its SHA-256.
   When a file is listed several times, the last line is the current one.
   Each listed file still present in the repository must have that checksum;
   a listed file that is no longer present is counted, not reported as an error.
2. The manifest of a run records the SHA-256 of the script that executed each
   pipeline stage. Each of those scripts must exist in the repository with
   that checksum, so the code here is the code that ran. Steps without a
   script (the index copy of run18) are counted and skipped.
3. Optional (--postprocess): the postprocess manifest of run18 records the
   first 12 hexadecimal characters of the SHA-256 of each cleaning script and
   input table. Each of those files must have that checksum prefix.

Exit code 0 only if every requested check passes.

Usage
    python tools/verify_provenance.py --manifest results/run16E/manifest.json
    python tools/verify_provenance.py --manifest results/run18/manifest.json \\
        --postprocess results/run18/postprocess/postprocess_manifest.txt
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=".", help="repository root (default: .)")
    ap.add_argument("--provenance", default="PROVENANCE.tsv")
    ap.add_argument("--manifest", default="results/run16E/manifest.json")
    ap.add_argument("--postprocess", default=None,
                    help="postprocess manifest of run18 (lines 'sha256 <prefix> <file>')")
    args = ap.parse_args()
    repo = Path(args.repo)
    errors = 0

    with open(repo / args.provenance, encoding="utf-8", newline="") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(f, delimiter="\t")]
    latest = {}  # a file may be listed more than once; the last line is the current one
    for row in rows:
        latest[row["destination"]] = row
    for dest, row in sorted(latest.items()):
        path = repo / dest
        if not path.is_file():
            continue  # removed from the repository after being recorded
        if sha256(path) != row["sha256"]:
            print("MODIFIED  " + dest)
            errors += 1
    print("provenance: %d files listed, %d present, %d modified"
          % (len(latest), sum((repo / d).is_file() for d in latest), errors))

    manifest = json.load(open(repo / args.manifest, encoding="utf-8"))
    steps = manifest["steps"]
    steps = steps.values() if isinstance(steps, dict) else steps
    steps = list(steps)
    scripts = {s["script"]: s["sha256"] for s in steps if s.get("script")}
    no_script = sum(1 for s in steps if not s.get("script"))
    bad = 0
    for script, checksum in sorted(scripts.items()):
        path = repo / script
        if not path.is_file() or sha256(path) != checksum:
            print("NOT AS RUN  " + script)
            bad += 1
    print("manifest %s (commit %s): %d stage scripts, %d identical to the recorded run; %d steps without a script"
          % (args.manifest, str(manifest.get("git_commit"))[:7], len(scripts), len(scripts) - bad, no_script))

    post_bad = 0
    if args.postprocess:
        listed = []
        for line in open(repo / args.postprocess, encoding="utf-8"):
            parts = line.split()
            if len(parts) == 3 and parts[0] == "sha256":
                listed.append((parts[2], parts[1]))
        for name, prefix in listed:
            path = repo / name
            if not path.is_file() or not sha256(path).startswith(prefix):
                print("NOT AS RUN  " + name)
                post_bad += 1
        print("postprocess %s: %d files, %d identical to the recorded cleaning"
              % (args.postprocess, len(listed), len(listed) - post_bad))

    sys.exit(1 if errors or bad or post_bad else 0)


if __name__ == "__main__":
    main()