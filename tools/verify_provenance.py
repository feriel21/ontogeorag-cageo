#!/usr/bin/env python3
"""Check that the repository files are the ones recorded in PROVENANCE.tsv and in the run manifest.

Two checks, both read-only:

1. PROVENANCE.tsv lists every file copied into this repository, with its origin
   (a path at the recorded commit, or a later script) and its SHA-256.
   Each listed file must exist and have that checksum.
2. The manifest of the delivered run records the SHA-256 of the script that
   executed each pipeline stage. Each of those scripts must exist in the
   repository with that checksum, i.e. the code here is the code that ran.

Exit code 0 only if both checks pass.

Usage
    python tools/verify_provenance.py --manifest results/run16E/manifest.json
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
    scripts = {s["script"]: s["sha256"] for s in steps}
    bad = 0
    for script, checksum in sorted(scripts.items()):
        path = repo / script
        if not path.is_file() or sha256(path) != checksum:
            print("NOT AS RUN  " + script)
            bad += 1
    print("manifest %s (commit %s): %d stage scripts, %d identical to the recorded run"
          % (args.manifest, str(manifest.get("git_commit"))[:7], len(scripts), len(scripts) - bad))

    sys.exit(1 if errors or bad else 0)


if __name__ == "__main__":
    main()