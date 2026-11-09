#!/usr/bin/env python3
"""176 — Generates the two run18 production scripts from 02 and 04, which stay unchanged.

PRINCIPLE
    pipeline/02_extract_triples.py and pipeline/04_clean_validate.py are left
    intact.
    The script reads their text and applies exact replacements; each pattern
    must occur exactly once, otherwise the script stops.
    The script writes:
        pipeline/02_extract_triples_run18.py
        pipeline/04_clean_validate_run18.py
        pipeline/RUN18_PATCHES.md   (rules, sha256 before and after, full diff)
    The script refuses to overwrite an existing file.
    The text written into the generated files (header lines, inline comments,
    RUN18_PATCHES.md) is in French and must stay as is: the run18 manifest
    records the sha256 of the generated scripts.

CHANGES (and nothing else)
    02  E1  the examples of the descriptor, causal and context prompts are read
            from --examples (JSON) instead of being written in the code;
        E2  the two counter-examples that named an object/descriptor pair are
            replaced by a generic wording;
    04  N1  the protection against merging covers the whole ontology
            vocabulary (KNOWN_DESCRIPTORS, 40 terms);
        N2  the protection of settings covers KNOWN_SETTINGS (list of 04);
        N3  the list of evaluation edges is removed from the code (empty list),
            so recall is computed only at step 07, not during cleaning.
    The --examples file is refused if an example names both arguments of an
    edge of configs/lb_reference_edges_original26.json.

FORMAT OF --examples
    {"source": "free text: where the examples come from",
     "descriptor": [{"source":..,"source_type":..,"relation":"hasDescriptor","target":..,
                     "target_type":"Descriptor","salience":"typical"}, ...],
     "causal":     [{"source":..,"source_type":..,"relation":..,"target":..,"target_type":..}, ...],
     "context":    [ ... ]}

USAGE (root of the run18 worktree)
    python 176_generer_scripts_run18.py --repo . --examples configs/prompt_examples_run18.json
"""

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

REF26 = "configs/lb_reference_edges_original26.json"
SHA02 = "f2555decd4de68a7cc8c5630c944990a256d8f14cf5ad4809ef0dd03dd7a5eb6"


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(s):
    return " ".join(str(s or "").replace("_", " ").split()).strip().lower()


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        die("%s : le motif apparait %d fois au lieu d'une" % (label, n))
    return text.replace(old, new)


def block(text, start_marker, label):
    """Example block: from the marker to the ']' that closes the list."""
    i = text.find(start_marker)
    if i < 0 or text.count(start_marker) != 1:
        die(label + " : marqueur introuvable ou multiple")
    j = text.find("\n]\n", i)
    if j < 0:
        die(label + " : fin de liste introuvable")
    return text[i:j + 3]


def render(examples, start_marker):
    lines = [start_marker, "["]
    for k, ex in enumerate(examples):
        s = json.dumps(ex, ensure_ascii=False).replace("{", "{{").replace("}", "}}")
        lines.append("  " + s + ("," if k < len(examples) - 1 else ""))
    lines.append("]\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--examples", required=True)
    args = ap.parse_args()
    repo = Path(args.repo)

    p02, p04 = repo / "pipeline/02_extract_triples.py", repo / "pipeline/04_clean_validate.py"
    o02, o04 = repo / "pipeline/02_extract_triples_run18.py", repo / "pipeline/04_clean_validate_run18.py"
    omd = repo / "pipeline/RUN18_PATCHES.md"
    for p in (o02, o04, omd):
        if p.exists():
            die("existe deja, rien n'est ecrase : " + str(p))

    t02, t04 = p02.read_text(encoding="utf-8"), p04.read_text(encoding="utf-8")
    if sha(t02) != SHA02:
        die("02 n'est pas la version de production (sha %s)" % sha(t02)[:12])

    ex = json.load(open(args.examples, encoding="utf-8"))
    edges = [(norm(e["subject"]), norm(e["object"])) for e in
             json.load(open(repo / REF26, encoding="utf-8"))["edges"]]
    for kind in ("descriptor", "causal", "context"):
        if not ex.get(kind):
            die("--examples : liste '%s' vide ou absente" % kind)
        for e in ex[kind]:
            if not {"source", "relation", "target"} <= set(e):
                die("--examples : exemple incomplet dans %s : %s" % (kind, e))
            s, o = norm(e["source"]), norm(e["target"])
            if any(s == a and o == b for a, b in edges):
                die("--examples : l'exemple %s --%s--> %s est une arete d'evaluation" % (e["source"], e["relation"], e["target"]))

    # ---- 02 ---------------------------------------------------------
    n02 = t02
    marker = "GOLD EXAMPLES (from peer-reviewed MTD literature):"
    # the three example blocks, in order: descriptor, causal, context
    parts = n02.split(marker)
    if len(parts) != 4:
        die("02 : %d blocs d'exemples au lieu de 3" % (len(parts) - 1))
    new_marker = "EXAMPLES (format only):"
    rebuilt = parts[0]
    for kind, tail in zip(("descriptor", "causal", "context"), parts[1:]):
        j = tail.find("\n]\n")
        if j < 0:
            die("02 : fin du bloc %s introuvable" % kind)
        rebuilt += render(ex[kind], new_marker) + tail[j + 3:]
    n02 = rebuilt
    n02 = replace_once(
        n02,
        '  {{"source": "mass transport deposit", "target": "chaotic"}}  <- when text says debris flow is chaotic, use "debris flow" not MTD\n'
        '  {{"source": "mass transport deposit", "target": "blocky"}}   <- when text says slide is blocky, use "slide" not MTD\n',
        '  {{"source": "<parent class>", "target": "<descriptor>"}}  <- when the text attributes the descriptor to a sub-type (e.g. a debris flow), use that sub-type as source\n',
        "02 E2")
    HDR = "#!/usr/bin/env python3\n"
    if not n02.startswith(HDR):
        die("02 : entete inattendue")
    n02 = HDR + ("# GENERE par 176_generer_scripts_run18.py a partir de pipeline/02_extract_triples.py (sha %s).\n"
                 "# Voir pipeline/RUN18_PATCHES.md. Ne pas editer a la main.\n" % SHA02[:12]) + n02[len(HDR):]

    # ---- 04 ---------------------------------------------------------
    n04 = t04
    n04 = replace_once(n04,
                       "from pipeline.rag.constants import (\n    LB2019_BENCHMARK_DESCRIPTORS as LB_DESCRIPTORS,\n)",
                       "from pipeline.rag.constants import (\n    KNOWN_DESCRIPTORS as LB_DESCRIPTORS,  # run18 N1 : vocabulaire complet de l'ontologie\n)",
                       "04 N1")
    m = re.search(r"        lb_settings = \{\n(?:            \"[^\"]+\",\n)+        \}\n", n04)
    if not m or n04.count(m.group(0)) != 1:
        die("04 N2 : bloc lb_settings introuvable")
    n04 = n04.replace(m.group(0), "        lb_settings = set(KNOWN_SETTINGS)  # run18 N2 : cadres de l'ontologie\n")
    m = re.search(r"LB_REFERENCE_EDGES = \[\n(?:    \([^\n]*\),\n)+\]\n", n04)
    if not m or n04.count(m.group(0)) != 1:
        die("04 N3 : liste LB_REFERENCE_EDGES introuvable")
    n04 = n04.replace(m.group(0), "LB_REFERENCE_EDGES = []  # run18 N3 : aucune arete d'evaluation dans le pipeline\n")
    if not n04.startswith(HDR):
        die("04 : entete inattendue")
    n04 = HDR + ("# GENERE par 176_generer_scripts_run18.py a partir de pipeline/04_clean_validate.py (sha %s).\n"
                 "# Voir pipeline/RUN18_PATCHES.md. Ne pas editer a la main.\n" % sha(t04)[:12]) + n04[len(HDR):]

    # checks: both generated scripts compile, and 02 names no evaluation pair
    compile(n02, str(o02), "exec")
    compile(n04, str(o04), "exec")
    low02 = n02.lower()
    for a, b in edges:
        for ln in low02.splitlines():
            if '"source"' in ln and '"target"' in ln and ('"%s"' % a) in ln and ('"%s"' % b) in ln:
                die("02 genere contient encore un couple d'evaluation : " + ln.strip())

    o02.write_text(n02, encoding="utf-8")
    o04.write_text(n04, encoding="utf-8")
    diff = []
    for old, new, a, b in ((t02, n02, p02, o02), (t04, n04, p04, o04)):
        diff += difflib.unified_diff(old.splitlines(True), new.splitlines(True), str(a), str(b))
    md = ["# Scripts de production run18", "",
          "Generes par `176_generer_scripts_run18.py`. Les originaux ne sont pas modifies.", "",
          "| fichier | sha256 |", "|---|---|",
          "| %s | `%s` |" % (p02, sha(t02)), "| %s | `%s` |" % (o02, sha(n02)),
          "| %s | `%s` |" % (p04, sha(t04)), "| %s | `%s` |" % (o04, sha(n04)),
          "| %s | `%s` |" % (args.examples, hashlib.sha256(Path(args.examples).read_bytes()).hexdigest()), "",
          "Source des exemples : " + str(ex.get("source", "non renseignee")), "",
          "Changements : 02 E1 exemples lus dans le fichier ci-dessus ; 02 E2 contre-exemples generiques ; "
          "04 N1 protection = KNOWN_DESCRIPTORS ; 04 N2 cadres = KNOWN_SETTINGS ; 04 N3 aucune arete d'evaluation.", "",
          "```diff", "".join(diff), "```", ""]
    omd.write_text("\n".join(md), encoding="utf-8")
    print("ecrit : %s (%s)\necrit : %s (%s)\necrit : %s" % (o02, sha(n02)[:12], o04, sha(n04)[:12], omd))
    print("lignes de diff : %d" % len(diff))


if __name__ == "__main__":
    main()