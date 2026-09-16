"""
pipeline/rag/constants.py — Shared ontology constants
========================================================
WHY
    Relation names, descriptor vocabularies and LB2019 ground truth were
    previously copy-pasted (and drifted) across pipeline stages — see the
    run13 restructure that unified them here. Importing from this single
    module is what keeps 02/03/04/06/07 in agreement about what a valid
    relation or descriptor is.

WHAT
    Relation ontology (ALLOWED_RELATIONS, RELATION_MAP, RELATION_GLOSSES)
    and normalization helpers; LB2019 ground truth (LB2019_BENCHMARK_DESCRIPTORS,
    LB2019_REFERENCE_EDGES) and the extraction closed-world vocabulary
    (KNOWN_DESCRIPTORS, KNOWN_SETTINGS); entity/descriptor normalization
    helpers used throughout the pipeline.

Single source of truth — import from here in every pipeline script.
"""

# ── Relation ontology ─────────────────────────────────────────────────

ALLOWED_RELATIONS: set[str] = {
    "hasDescriptor",
    "occursIn",
    "formedBy",
    "partOf",
    "triggers",
    "causes",
    "controls",
    "affects",
    "overlies",
    "underlies",
    "indicates",
    "evidences",
    "relatedTo",
}

RELATION_MAP: dict[str, str] = {
    # Canonical forms
    "hasdescriptor": "hasDescriptor",
    "occursin": "occursIn",
    "formedby": "formedBy",
    "partof": "partOf",
    "triggeredby": "triggers",
    "triggered_by": "triggers",
    "overlays": "overlies",
    "relatedto": "relatedTo",
    "related_to": "relatedTo",
    "related to": "relatedTo",
    "hasfeature": "hasDescriptor",
    "ischaracterizedby": "hasDescriptor",
    "characterizedby": "hasDescriptor",
    "locatedin": "occursIn",
    "foundin": "occursIn",
    # Extended — recovers filtered Qwen outputs
    "contains": "partOf",
    "results in": "causes",
    "hasthickness": "hasDescriptor",
    "hasrelation": "relatedTo",
    "isunconformablyoverlainby": "underlies",
    "develops into": "causes",
    "induces": "triggers",
    "differentially_loads": "affects",
    "can_combine_with": "relatedTo",
    "can_lead_to": "causes",
    "formedat": "occursIn",
    "initiatedby": "formedBy",
    "locatedat": "occursIn",
    "hascomponent": "partOf",
    "rootedon": "overlies",
    "leads to": "causes",
    "serves as": "hasDescriptor",
    "ispartof": "partOf",
    "catastrophically fails": "affects",
    # Common LLM paraphrases
    "is characterized by": "hasDescriptor",
    "is described as": "hasDescriptor",
    "exhibits": "hasDescriptor",
    "displays": "hasDescriptor",
    "shows": "hasDescriptor",
    "is located in": "occursIn",
    "is found in": "occursIn",
    "deposited in": "occursIn",
    "is formed by": "formedBy",
    "formed by": "formedBy",
    "is triggered by": "triggers",
    "triggered by": "triggers",
    "is caused by": "causes",
    "caused by": "causes",
    "is part of": "partOf",
    "lies above": "overlies",
    "lies below": "underlies",
    "is above": "overlies",
    "is below": "underlies",
    # Extended — recovers filtered Qwen outputs
    # Keys must be lowercased + spaces/underscores/hyphens removed
    # (matches normalize_relation: key = rel.lower().replace(" ","").replace("_","").replace("-",""))
    "resultsin": "causes",  # "results in"
    "leadsto": "causes",  # "leads to"
    "canleadto": "causes",  # "can_lead_to"
    "developsinto": "causes",  # "develops into"
    "servesas": "hasDescriptor",  # "serves as"
    "differentiallyloads": "affects",  # "differentially_loads"
    "cancombinewith": "relatedTo",  # "can_combine_with"
    "catastrophicallyfails": "affects",  # "catastrophically fails"
    "depositedin": "occursIn",  # "deposited in"
    "islocatedin": "occursIn",  # "is located in"
    "isfoundin": "occursIn",  # "is found in"
    "isformedby": "formedBy",  # "is formed by"
    "istriggeredby": "triggers",  # "is triggered by"
    "triggeredby": "triggers",  # "triggered by"
    "iscausedby": "causes",  # "is caused by"
    "causedby": "causes",  # "caused by"
    "ispartof": "partOf",  # "is part of"
    "liesabove": "overlies",  # "lies above"
    "liesbelow": "underlies",  # "lies below"
    "isabove": "overlies",  # "is above"
    "isbelow": "underlies",  # "is below"
    "isdescribedas": "hasDescriptor",  # "is described as"
    "exhibits": "hasDescriptor",
    "displays": "hasDescriptor",
    "transitionsinto": "relatedTo",  # "transitions into"
    "gradatesinto": "relatedTo",  # "grades into"
    "evolvesfrom": "formedBy",  # "evolves from"
    "evolvesfrominto": "relatedTo",
    "transitionsto": "relatedTo",
    "ischaracterizedby": "hasDescriptor",  # "is characterized by"
    "isunconformablyoverlainby": "underlies",  # "isUnconformablyOverlainBy"
    "hasthickness": "hasDescriptor",  # "hasThickness"
    "hasrelation": "relatedTo",  # "hasRelation"
    "formedat": "occursIn",  # "formedAt"
    "initiatedby": "formedBy",  # "initiatedBy"
    "locatedat": "occursIn",  # "locatedAt"
    "hascomponent": "partOf",  # "hasComponent"
    "rootedon": "overlies",  # "rootedOn"
    "contains": "partOf",  # "contains"
}

RELATION_GLOSSES: dict[str, str] = {
    "hasDescriptor": "is characterised by / exhibits",
    "occursIn": "is found in / located in",
    "formedBy": "is formed by / produced by",
    "partOf": "is a part / component of",
    "triggers": "initiates / triggers",
    "causes": "directly produces",
    "controls": "governs / regulates",
    "affects": "influences / modifies",
    "overlies": "is stratigraphically above",
    "underlies": "is stratigraphically below",
    "indicates": "indicates",
    "evidences": "provides evidence for",
    "relatedTo": "is related to",
}


# ── LB2019 ground truth ───────────────────────────────────────────────

# The true 13-term LB2019 descriptor benchmark (single source of truth —
# do not redefine locally; import this everywhere, including 04 and 07).
LB2019_BENCHMARK_DESCRIPTORS: set[str] = {
    "blocky",
    "chaotic",
    "continuous",
    "discontinuous",
    "high-amplitude",
    "hummocky",
    "layered",
    "low-amplitude",
    "massive",
    "parallel",
    "stratified",
    "transparent",
    "undeformed",
}

# Back-compat alias for callers outside pipeline/ (e.g. experiments/) that
# still import the old name.
LB2019_DESCRIPTORS = LB2019_BENCHMARK_DESCRIPTORS

DESCRIPTOR_SYNONYMS: dict[str, str] = {
    "stratified": "layered",
    "sub-parallel": "parallel",
    "sub parallel": "parallel",
    "essentially undeformed": "undeformed",
    "low amplitude": "low-amplitude",
    "high amplitude": "high-amplitude",
    "low-amplitude reflection": "low-amplitude",
    "high-amplitude reflection": "high-amplitude",
}

# All 26 LB2019 reference edges (ground truth)
LB2019_REFERENCE_EDGES: list[tuple[str, str, str]] = [
    ("mass transport deposit", "hasDescriptor", "chaotic"),
    ("mass transport deposit", "hasDescriptor", "transparent"),
    ("mass transport deposit", "hasDescriptor", "hummocky"),
    ("mass transport deposit", "hasDescriptor", "blocky"),
    ("mass transport deposit", "hasDescriptor", "discontinuous"),
    ("mass transport deposit", "hasDescriptor", "massive"),
    ("turbidite", "hasDescriptor", "parallel"),
    ("turbidite", "hasDescriptor", "continuous"),
    ("turbidite", "hasDescriptor", "layered"),
    ("turbidite", "hasDescriptor", "high-amplitude"),
    ("debris flow", "hasDescriptor", "chaotic"),
    ("debris flow", "hasDescriptor", "hummocky"),
    ("slide", "hasDescriptor", "blocky"),
    ("slide", "hasDescriptor", "undeformed"),
    ("hemipelagite", "hasDescriptor", "parallel"),
    ("hemipelagite", "hasDescriptor", "continuous"),
    ("hemipelagite", "hasDescriptor", "low-amplitude"),
    ("slope failure", "causes", "mass transport deposit"),
    ("earthquake", "triggers", "slope failure"),
    ("pore pressure", "controls", "slope failure"),
    ("turbidity current", "formedBy", "debris flow"),
    ("mass transport deposit", "occursIn", "continental slope"),
    ("mass transport deposit", "occursIn", "abyssal plain"),
    ("debris flow", "occursIn", "continental slope"),
    ("turbidite", "occursIn", "basin floor"),
    ("slide", "overlies", "hemipelagite"),
]


# ── Known geological terms ─────────────────────────────────────────────

KNOWN_DESCRIPTORS: set[str] = LB2019_DESCRIPTORS | {
    "mounded",
    "divergent",
    "convergent",
    "wavy",
    "contorted",
    "folded",
    "faulted",
    "deformed",
    "disrupted",
    "draping",
    "onlapping",
    "erosional",
    "aggradational",
    "progradational",
    "retrogradational",
    "tabular",
    "lenticular",
    "wedge-shaped",
    "sheet-like",
    "channelised",
    "irregular",
    "smooth",
    "rough",
    "thick",
    "thin",
    "variable-amplitude",
    "moderate-amplitude",
}

KNOWN_SETTINGS: set[str] = {
    "continental slope",
    "continental shelf",
    "continental margin",
    "abyssal plain",
    "basin floor",
    "submarine canyon",
    "channel",
    "deep-water environment",
    "deep-water environments",
    "deep water",
    "passive margin",
    "active margin",
    "accretionary prism",
    "trench",
    "mid-ocean ridge",
    "seamount",
    "delta",
    "fan",
    "submarine fan",
    "levee",
    "overbank",
}

# Entity normalization map (MTD variants → canonical)
MTD_VARIANTS: set[str] = {
    "mass-transport deposit",
    "mass-transport deposits",
    "mass-transport deposits (mtd)",
    "mtd",
    "mass transport deposits",
    "mass transport deposit 1",
    "mass transport deposit 2",
    "mtd 1",
    "mtd 2",
    "mass transport complex",
    "mass-transport complex",
}

ENTITY_NORMS: dict[str, str] = {
    v: "mass transport deposit" for v in MTD_VARIANTS
}
ENTITY_NORMS.update(
    {
        "debris flows": "debris flow",
        "debrites": "debris flow",
        "turbidity currents": "turbidity current",
        "turbidites": "turbidite",
        "slides": "slide",
        "slumps": "slump",
        "low amplitude": "low-amplitude",
        "high amplitude": "high-amplitude",
        "low amplitude reflections": "low-amplitude",
        "high amplitude reflections": "high-amplitude",
    }
)


def normalize_relation(rel: str) -> str:
    """Normalize a relation string to canonical form."""
    if not rel:
        return ""
    key = (
        rel.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    )
    return RELATION_MAP.get(key, rel.strip())


def normalize_entity(text: str) -> str:
    """Lowercase, strip, collapse whitespace, apply entity norms."""
    import re

    t = re.sub(r"\s+", " ", (text or "").lower().strip()).rstrip(".,;:")
    return ENTITY_NORMS.get(t, t)


def normalize_descriptor(text: str) -> str:
    """Apply descriptor synonyms after entity normalization."""
    t = normalize_entity(text)
    return DESCRIPTOR_SYNONYMS.get(t, t)


# Extended descriptor fuzzy matching — handles verbose LLM outputs
DESCRIPTOR_FUZZY_MAP = {
    # chaotic variants
    "chaotic seismic facies": "chaotic",
    "chaotic facies": "chaotic",
    "chaotic reflection": "chaotic",
    "slightly chaotic facies": "chaotic",
    "chaotic and discontinuous": "chaotic",
    # transparent variants
    "transparent seismic facies": "transparent",
    "transparent facies": "transparent",
    "acoustically transparent": "transparent",
    # hummocky variants
    "hummocky and irregular": "hummocky",
    "hummocky seismic facies": "hummocky",
    "hummocky surface": "hummocky",
    # high-amplitude variants
    "high-amplitude seismic facies": "high-amplitude",
    "high amplitude seismic facies": "high-amplitude",
    "high amplitude reflections": "high-amplitude",
    "high-amplitude reflections": "high-amplitude",
    "high amplitude reflection": "high-amplitude",
    # low-amplitude variants
    "low-amplitude seismic facies": "low-amplitude",
    "low amplitude reflections": "low-amplitude",
    "low to medium amplitude, discontinuous reflectors": "low-amplitude",
    "high-frequency, low-amplitude reflectors": "low-amplitude",
    "low amplitude seismic facies": "low-amplitude",
    # discontinuous variants
    "discontinuous seismic facies": "discontinuous",
    "discontinuous reflections": "discontinuous",
    "discontinuous and low-amplitude reflections": "discontinuous",
    "discontinuous wavy seismic events": "discontinuous",
    # layered variants
    "layered seismic facies": "layered",
    "layered reflections": "layered",
    # parallel variants
    "parallel seismic facies": "parallel",
    "parallel to subparallel and continuous seismic events": "parallel",
    "parallel reflections": "parallel",
    # continuous variants
    "continuous seismic facies": "continuous",
    "continuous reflections": "continuous",
    # stratified variants
    "stratified seismic facies": "stratified",
    # blocky variants
    "blocky seismic facies": "blocky",
    "blocky material": "blocky",
    "high amplitude, blocky material": "blocky",
    # massive variants
    "massive seismic facies": "massive",
    # undeformed variants
    "undeformed seismic facies": "undeformed",
    "undeformed reflections": "undeformed",
}


def normalize_descriptor_fuzzy(text: str) -> str:
    """
    Two-pass descriptor normalization:
    1. Exact match in DESCRIPTOR_SYNONYMS (existing)
    2. Fuzzy match in DESCRIPTOR_FUZZY_MAP (verbose LLM outputs)
    3. Substring scan — if any canonical descriptor appears in text, return it
    """
    t = normalize_entity(text).lower().strip()
    # Pass 1: existing synonym map
    if t in DESCRIPTOR_SYNONYMS:
        return DESCRIPTOR_SYNONYMS[t]
    # Pass 2: fuzzy map for verbose outputs
    if t in DESCRIPTOR_FUZZY_MAP:
        return DESCRIPTOR_FUZZY_MAP[t]
    # Pass 3: substring scan over canonical descriptors
    for canon in sorted(LB2019_DESCRIPTORS, key=len, reverse=True):
        if canon in t:
            return canon
    return t


import re as _re


def normalize_descriptor_multi(text: str) -> set:
    """
    Return ALL canonical LB2019 descriptors found in text.
    Uses word-boundary matching to avoid false positives
    (e.g. 'continuous' inside 'discontinuous').
    Fuzzy map is used as additional seed, not a shortcut.
    """
    t = normalize_entity(text).lower().strip()
    found = set()
    # Word-boundary scan over all canonical descriptors
    for canon in LB2019_DESCRIPTORS:
        if _re.search(r"(?<!\w)" + _re.escape(canon) + r"(?!\w)", t):
            found.add(canon)
    # Also check fuzzy map as fallback for non-canonical phrases
    if not found and t in DESCRIPTOR_FUZZY_MAP:
        found.add(DESCRIPTOR_FUZZY_MAP[t])
    return found if found else {t}
