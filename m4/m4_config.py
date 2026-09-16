#!/usr/bin/env python3
"""
m4_config.py — Shared constants and prompts for the M4 Independent
Cross-Family Verifier.

Design notes
------------
* M4 is deliberately SELF-CONTAINED: it does not import from pipeline/rag/
  so that the verifier shares no code path with the extractor. If the
  ontogeorag repo is importable, relation glosses are loaded from
  pipeline.rag.constants for consistency; otherwise the local fallback
  dictionary below is used (kept in sync manually — see M4_GLOSSES).

* The verdict vocabulary is INTENTIONALLY DIFFERENT from the Qwen verifier
  (STRONG_SUPPORT / WEAK_SUPPORT / NOT_SUPPORTED). Using different labels
  (SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED for the evidence pass;
  PLAUSIBLE / UNCERTAIN / IMPLAUSIBLE for the blind pass) avoids label
  anchoring across instruments and makes explicit that M4 measures two
  different properties: parametric plausibility vs textual support.
"""

# ── Model ──────────────────────────────────────────────────────────────
# Cross-family requirement: the extractor is Qwen 2.5-7B-Instruct, so the
# verifier MUST come from a different model family / training lineage.
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Generation parameters. Greedy decoding (do_sample=False) for full
# determinism and reproducibility of verdicts.
GEN_KWARGS = dict(
    max_new_tokens=300,
    do_sample=False,
    temperature=None,  # ignored when do_sample=False; set explicitly to
    top_p=None,  # silence transformers warnings on some versions
)

MAX_EVIDENCE_CHARS = 1500  # same truncation as 03_verify / expD


# ── Relation glosses ───────────────────────────────────────────────────
# Local fallback, aligned with pipeline/rag/constants.py RELATION_GLOSSES.
# If the ontogeorag package is importable, the pipeline version wins.
M4_GLOSSES = {
    "hasDescriptor": "is characterised in seismic data by the descriptor",
    "formedBy": "is formed by the process",
    "triggers": "triggers / initiates",
    "causes": "causes / produces",
    "controls": "controls / influences",
    "occursIn": "occurs in the depositional setting",
    "overlies": "lies stratigraphically above",
    "underlies": "lies stratigraphically below",
    "partOf": "is a component / part of",
    "evolvesTo": "evolves or transforms into",
    "indicates": "is diagnostic evidence of",
}


def get_glosses() -> dict:
    """Prefer the pipeline's own glosses when available (run from repo root)."""
    try:
        from pipeline.rag.constants import RELATION_GLOSSES  # type: ignore

        merged = dict(M4_GLOSSES)
        merged.update(RELATION_GLOSSES)
        return merged
    except Exception:
        return dict(M4_GLOSSES)


# ── Pass 1 — BLIND (parametric plausibility, NO source text) ──────────
BLIND_SYSTEM = (
    "You are an expert marine geologist specialised in mass-transport "
    "deposits (MTDs) and seismic interpretation. You will judge whether a "
    "knowledge-graph triple is geologically plausible, using ONLY your own "
    "geological knowledge. No source text is provided and none should be "
    "assumed. Follow the output format exactly."
)

BLIND_PROMPT = """\
Judge the geological plausibility of the following knowledge-graph triple.

=== TRIPLE ===
  Subject:  {subject}
  Relation: {relation} (meaning: subject {gloss} object)
  Object:   {object}
=== END TRIPLE ===

STEP 1 — REASONING: In 1-3 sentences, explain whether this relation is
consistent with established geological knowledge about mass-transport
deposits, submarine slope processes, and seismic facies.

STEP 2 — VERDICT: Choose exactly one:
  PLAUSIBLE    — consistent with established geological knowledge
  UNCERTAIN    — possible but context-dependent, or you lack the knowledge
  IMPLAUSIBLE  — contradicts established geological knowledge

Format EXACTLY:
REASONING: <...>
VERDICT: <PLAUSIBLE or UNCERTAIN or IMPLAUSIBLE>
"""

BLIND_VERDICTS = ("PLAUSIBLE", "UNCERTAIN", "IMPLAUSIBLE")


# ── Pass 2 — EVIDENCE (textual support, source passage provided) ──────
EVIDENCE_SYSTEM = (
    "You are a strict scientific fact-checker. You will judge whether a "
    "knowledge-graph triple is supported by a source passage. Use ONLY the "
    "provided passage. Do NOT use any geological knowledge beyond it. "
    "Follow the output format exactly."
)

EVIDENCE_PROMPT = """\
Determine whether the source passage supports the claimed triple.

=== SOURCE PASSAGE ===
{evidence}
=== END SOURCE PASSAGE ===

=== CLAIMED TRIPLE ===
  Subject:  {subject}
  Relation: {relation} (meaning: subject {gloss} object)
  Object:   {object}
=== END TRIPLE ===

STEP 1 — QUOTE: Copy the most relevant sentence(s) from the passage.
If none, write "NO EVIDENCE FOUND".

STEP 2 — REASONING: In 1-2 sentences, explain whether the quoted text
supports the triple. Use ONLY the passage.

STEP 3 — VERDICT: Choose exactly one:
  SUPPORTED            — the passage explicitly states the relation
  PARTIALLY_SUPPORTED  — the passage implies it, or supports it with caveats
  NOT_SUPPORTED        — the passage does not state or imply the relation

Format EXACTLY:
QUOTE: <...>
REASONING: <...>
VERDICT: <SUPPORTED or PARTIALLY_SUPPORTED or NOT_SUPPORTED>
"""

EVIDENCE_VERDICTS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED")


# ── Aggregation (decision matrix + continuous confidence) ─────────────
# Numeric mapping for the continuous confidence score.
EVIDENCE_SCORE = {
    "SUPPORTED": 1.0,
    "PARTIALLY_SUPPORTED": 0.5,
    "NOT_SUPPORTED": 0.0,
}
BLIND_SCORE = {"PLAUSIBLE": 1.0, "UNCERTAIN": 0.5, "IMPLAUSIBLE": 0.0}

# Textual support dominates: the pipeline's central claim is literature
# grounding, so the evidence verdict carries most of the weight. The blind
# verdict is mainly DIAGNOSTIC (over-interpretation / parametric-risk
# detection), which is why its weight is low.
W_EVIDENCE = 0.7
W_BLIND = 0.3

ACCEPT_THRESHOLD = 0.70  # confidence >= 0.70 -> ACCEPT
REJECT_THRESHOLD = 0.30  # confidence <= 0.30 -> REJECT
# in between -> UNCERTAIN

# Special diagnostic flag: blind says PLAUSIBLE but evidence says
# NOT_SUPPORTED  ->  the triple "sounds right" but is not in the text.
# This is exactly the parametric-contamination signature (cf. Exp B).
PARAMETRIC_RISK = ("PLAUSIBLE", "NOT_SUPPORTED")
