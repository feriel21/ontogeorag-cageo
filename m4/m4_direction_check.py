#!/usr/bin/env python3
"""
m4_direction_check.py — Targeted directional verification pass
================================================================

Motivation (measured, not assumed): the negative-control characterization
showed the evidence judge detects passage mismatches at 92% and entity
substitutions at 73%, but direction inversions at only 36% — inverted
causal relations are mostly absorbed into PARTIALLY_SUPPORTED. This pass
closes that measured gap with a question the judge answers well: instead
of "is the triple supported?", it asks "WHICH direction does the passage
state?" — a forced-choice discrimination task.

Applies to ACCEPTED triples with a directional relation only.

Verdicts:
  FORWARD     passage states subject -> object (as extracted)
  REVERSE     passage states object -> subject (direction error!)
  UNDIRECTED  passage states an association without direction
  ABSENT      passage does not relate the two entities

Outputs:
  m4_direction_verdicts.jsonl
  m4_direction_summary.json
Triples judged REVERSE get flag `direction_error` (candidate demotion);
UNDIRECTED gets `direction_unstated` (residual risk, documented).

Usage (GPU):
    python m4_direction_check.py \
        --kg        ~/ontogeorag/output/run11_kg/tiered_kg_run11.json \
        --decisions ~/ontogeorag/output/m4/m4_decisions.jsonl \
        --output    ~/ontogeorag/output/m4 \
        --model     meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

from m4_config import DEFAULT_MODEL, MAX_EVIDENCE_CHARS, get_glosses
from m4_verify import (
    evidence_from_provenance,
    generate,
    load_model,
    load_triples,
    parse_pass,
    triple_fields,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m4dir")

DIRECTIONAL = {
    "causes",
    "triggers",
    "controls",
    "affects",
    "overlies",
    "underlies",
    "evolvesTo",
    "indicates",
    "formedBy",
}

DIR_SYSTEM = (
    "You are a strict scientific fact-checker. You will determine the "
    "DIRECTION of a relation as stated by a source passage. Use ONLY the "
    "provided passage. Follow the output format exactly."
)

DIR_PROMPT = """\
The following relation was extracted from the passage below:

  Entity A: {subject}
  Relation: {relation} (meaning: A {gloss} B)
  Entity B: {object}

=== SOURCE PASSAGE ===
{evidence}
=== END SOURCE PASSAGE ===

Question: according to THIS PASSAGE ONLY, in which direction does the
relation hold between Entity A and Entity B?

STEP 1 — QUOTE: Copy the sentence(s) relating A and B. If none, write
"NO EVIDENCE FOUND".

STEP 2 — REASONING: 1-2 sentences on the direction the passage states.

STEP 3 — VERDICT: choose exactly one:
  FORWARD     the passage states A {gloss} B (as extracted)
  REVERSE     the passage states B {gloss} A (opposite direction)
  UNDIRECTED  the passage associates A and B without a clear direction
  ABSENT      the passage does not relate A and B

Format EXACTLY:
QUOTE: <...>
REASONING: <...>
VERDICT: <FORWARD or REVERSE or UNDIRECTED or ABSENT>
"""

DIR_VERDICTS = ("FORWARD", "REVERSE", "UNDIRECTED", "ABSENT")


def load_jsonl(path):
    """Read `path` as one JSON object per line; no side effects."""
    out = []
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    """CLI entry point: for every ACCEPTed directional-relation triple in --decisions, asks the --model verifier which direction its passage states, and writes m4_direction_verdicts.jsonl + m4_direction_summary.json to --output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = load_triples(Path(args.kg).expanduser())
    decisions = load_jsonl(args.decisions)
    glosses = get_glosses()

    targets = []
    for d in decisions:
        if d["m4_decision"] == "ACCEPT" and d["relation"] in DIRECTIONAL:
            idx = d["m4_index"]
            passage = (
                evidence_from_provenance(triples[idx])
                if idx < len(triples)
                else ""
            )
            if passage:
                targets.append((d, passage))
    if args.limit:
        targets = targets[: args.limit]
    log.info(f"Directional ACCEPTed triples to check: {len(targets)}")

    tok, mdl = load_model(args.model)

    out_path = out_dir / "m4_direction_verdicts.jsonl"
    counts = Counter()
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, (d, passage) in enumerate(targets):
            gloss = glosses.get(d["relation"], d["relation"])
            raw = generate(
                tok,
                mdl,
                DIR_SYSTEM,
                DIR_PROMPT.format(
                    subject=d["subject"],
                    relation=d["relation"],
                    object=d["object"],
                    gloss=gloss,
                    evidence=passage[:MAX_EVIDENCE_CHARS],
                ),
            )
            parsed = parse_pass(raw, DIR_VERDICTS, ("QUOTE", "REASONING"))
            v = parsed["verdict"]
            counts[v] += 1

            flags = []
            if v == "REVERSE":
                flags.append("direction_error")
            elif v == "UNDIRECTED":
                flags.append("direction_unstated")
            elif v == "ABSENT":
                flags.append("direction_check_absent")

            fout.write(
                json.dumps(
                    {
                        "m4_index": d["m4_index"],
                        "subject": d["subject"],
                        "relation": d["relation"],
                        "object": d["object"],
                        "tier": d.get("tier"),
                        "evidence_verdict": d["evidence_verdict"],
                        "direction_verdict": v,
                        "quote": parsed.get("quote", ""),
                        "reasoning": parsed.get("reasoning", ""),
                        "flags": flags,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if (i + 1) % 10 == 0 or (i + 1) == len(targets):
                log.info(
                    f"[{i + 1}/{len(targets)}] {v:<11s} "
                    f"({(time.time() - t0) / (i + 1):.1f}s/triple)"
                )

    n = len(targets)
    summary = {
        "model": args.model,
        "n_checked": n,
        "verdicts": dict(counts),
        "rates": {k: round(v / n, 4) for k, v in counts.items()} if n else {},
        "interpretation": {
            "FORWARD": "direction confirmed as extracted",
            "REVERSE": "direction error — candidate demotion",
            "UNDIRECTED": "association without stated direction — "
            "residual risk, documented",
            "ABSENT": "inconsistent with prior ACCEPT — inspect",
        },
    }
    (out_dir / "m4_direction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Verdicts: {out_path}")


if __name__ == "__main__":
    main()
