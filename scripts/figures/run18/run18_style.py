"""Shared inputs, colours and helpers of the run18 figure scripts.

Every figure script of this folder reads only files of the repository:
    results/run18/kg_final/flat_run18_enforced_dedup.json        delivered graph (364 relations)
    results/run18/postprocess/187_provenance/...                 graph with located provenance
    results/run18/postprocess/188_nettoyage/...                  cleaned graph (281 relations)
    configs/entity_families_run18.csv                            table of name families
    configs/lb_reference_edges_original26.json                   26 expert relations
Each script checks its counts against the values printed in the manuscript and writes
nothing if one of them differs.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DELIVERED = REPO / "results/run18/kg_final/flat_run18_enforced_dedup.json"
WITH_PROVENANCE = REPO / "results/run18/postprocess/187_provenance/graphe_avec_provenance.json"
PROVENANCE_TABLE = REPO / "results/run18/postprocess/187_provenance/provenance_par_relation.tsv"
CLEANED = REPO / "results/run18/postprocess/188_nettoyage/graphe_nettoye.json"
REMOVED = REPO / "results/run18/postprocess/188_nettoyage/relations_retirees.tsv"
FAMILIES = REPO / "configs/entity_families_run18.csv"
EXPERT_EDGES = REPO / "configs/lb_reference_edges_original26.json"

INK, MUTED, GRID = "#1b1b1b", "#555555", "#d9d9d9"
DARK, MID, LIGHT = "#24456d", "#8497b0", "#d3d9e2"

# entity types recorded in the graph (schema_enforce), with the label shown in the figures
ENTITY_TYPES = {
    "Geological_Object": ("geological object", "#4e79a7"),
    "Process": ("process", "#e15759"),
    "Environmental_Control": ("setting or environmental control", "#59a14f"),
    "Descriptor": ("seismic descriptor", "#b07aa1"),
}
RELATION_COLOURS = {
    "hasDescriptor": "#9c6b98",
    "triggers": "#d62728",
    "causes": "#ff7f0e",
    "controls": "#8c564b",
    "affects": "#e377c2",
    "occursIn": "#2ca02c",
    "partOf": "#1f77b4",
    "overlies": "#17becf",
    "underlies": "#7f7f7f",
    "formedBy": "#bcbd22",
}
CAUSAL = ("causes", "triggers", "controls", "affects")

# three family labels of configs/entity_families_run18.csv are in French; they are shown in English
FAMILY_LABELS = {
    "contexte de marge": "continental margin settings",
    "position bathymetrique": "slope and deep-water settings",
    "systeme sedimentaire": "sedimentary systems",
}


def label(name):
    return FAMILY_LABELS.get(name, name)


def norm(text):
    return " ".join(str(text or "").lower().replace("_", " ").split())


def triples(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["triples"] if isinstance(obj, dict) else obj


def family_of_subject(t):
    return t.get("famille_sujet") or t["subject"]


def family_of_object(t):
    return t.get("famille_objet") or t["object"]


def flagged(t):
    """Evidence check: flagged if at least one verifying-LLM returns NOT_SUPPORTED on the evidence."""
    return "NOT_SUPPORTED" in str((t.get("m4") or {}).get("evidence_verdict", ""))


def check(measured, expected):
    """Stop without writing anything if a measured value differs from the manuscript."""
    diffs = ["%s: %r, expected %r" % (k, measured.get(k), v) for k, v in expected.items()
             if measured.get(k) != v]
    if diffs:
        print("ERROR: the inputs do not reproduce the manuscript; nothing written")
        for d in diffs:
            print("  - " + d)
        sys.exit(1)


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out) + "." + ext, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written: %s.pdf and %s.png" % (out, out))
