#!/usr/bin/env python3
"""Replay the CPU-only final stages of a recorded run and compare with its manifest.

Stages replayed, with the exact command lines stored in the manifest:
    schema_enforce  analysis_suite/04b_schema_enforce.py
    flatten         analysis_suite/24_kg_to_flat.py
    metrics         pipeline/07_final_metrics.py

Input: the panel output of the run (m4_panel/tiered_kg_m4.json), whose checksum
must match the manifest. The replay writes into --out; the recorded run
directory in each command line is replaced by --out.

Comparison. Some outputs store the absolute path of their input, so a replay
in another directory changes the file bytes without changing the content.
Each replayed file is therefore compared after writing the recorded run
directory back in place of --out. The result must match the checksum
(first 16 hex characters of SHA-256) recorded in the manifest.

Exit code 0 only if every stage succeeds and every compared artefact matches.

Usage
    PYTHONPATH=. python tools/replay_cpu_steps.py \\
        --manifest results/run16E/manifest.json \\
        --panel-output /path/to/run16E/m4_panel/tiered_kg_m4.json \\
        --out output/replay_run16E
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

STAGES = ["schema_enforce", "flatten", "metrics"]
PANEL_ARTEFACT = "m4_panel/tiered_kg_m4.json"
COMPARED_PREFIXES = ("kg_enforced/", "kg_final/")


def fail(msg):
    print("ERROR: " + msg, flush=True)
    sys.exit(1)


def sha16(data):
    return hashlib.sha256(data).hexdigest()[:16]


def argv_list(step):
    return step["argv"].split() if isinstance(step["argv"], str) else list(step["argv"])


def recorded_run_dir(steps):
    """Run directory used at record time: the directory holding m4_panel/ in the --kg argument."""
    argv = argv_list(steps["schema_enforce"])
    kg = Path(argv[argv.index("--kg") + 1])
    return str(kg.parent.parent)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=".", help="repository root holding the scripts (default: .)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--panel-output", required=True, help="m4_panel/tiered_kg_m4.json of the run")
    ap.add_argument("--out", required=True, help="output directory (must not exist)")
    args = ap.parse_args()

    repo, out = Path(args.repo).resolve(), Path(args.out).resolve()
    if out.exists():
        fail("output directory already exists: " + str(out))
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    steps = manifest["steps"]
    steps = {s["step"]: s for s in (steps.values() if isinstance(steps, dict) else steps)}
    expected = manifest["artifact_sha256"]
    missing = [s for s in STAGES if s not in steps]
    if missing:
        fail("stages missing from the manifest: " + str(missing))

    panel = Path(args.panel_output)
    if sha16(panel.read_bytes()) != expected[PANEL_ARTEFACT]:
        fail("panel output checksum differs from the manifest")
    for name in STAGES:
        script = repo / steps[name]["script"]
        if not script.is_file() or hashlib.sha256(script.read_bytes()).hexdigest() != steps[name]["sha256"]:
            fail("script missing or different from the manifest: " + str(script))

    recorded = recorded_run_dir(steps)
    (out / "m4_panel").mkdir(parents=True)
    shutil.copy2(panel, out / PANEL_ARTEFACT)
    env = dict(os.environ, PYTHONPATH=str(repo))

    for name in STAGES:
        argv = [a.replace(recorded, str(out)) for a in argv_list(steps[name])]
        result = subprocess.run([sys.executable, str(repo / steps[name]["script"])] + argv,
                                cwd=repo, env=env, capture_output=True, text=True)
        (out / (name + ".log")).write_text(result.stdout + "\n--- stderr ---\n" + result.stderr,
                                           encoding="utf-8")
        print("%-15s exit %d" % (name, result.returncode), flush=True)
        if result.returncode != 0:
            fail("stage %s failed, see %s" % (name, out / (name + ".log")))

    comparison, all_match = [], True
    for artefact, checksum in sorted(expected.items()):
        if not artefact.startswith(COMPARED_PREFIXES):
            continue
        path = out / artefact
        if not path.is_file():
            status, got = "MISSING", None
        else:
            raw = path.read_bytes()
            normalised = raw.replace(str(out).encode(), recorded.encode())
            got = sha16(normalised)
            if sha16(raw) == checksum:
                status = "identical"
            elif got == checksum:
                status = "identical except paths"
            else:
                status = "DIFFERENT"
        all_match &= status.startswith("identical")
        comparison.append({"artefact": artefact, "expected": checksum, "replayed": got, "status": status})
        print("%-24s %s" % (status, artefact), flush=True)

    (out / "replay_report.json").write_text(
        json.dumps({"manifest": args.manifest, "recorded_run_dir": recorded,
                    "comparison": comparison}, indent=2), encoding="utf-8")
    if not all_match:
        fail("replayed artefacts differ from the manifest")
    print("all compared artefacts match the manifest")


if __name__ == "__main__":
    main()