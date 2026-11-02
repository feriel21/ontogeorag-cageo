#!/usr/bin/env python3
"""176 — Genere les deux scripts de production de run18 a partir de 02 et 04, sans les modifier.

PRINCIPE
    pipeline/02_extract_triples.py et pipeline/04_clean_validate.py restent
    intacts. Ce script lit leur texte, applique des remplacements exacts
    (chacun doit apparaitre exactement une fois, sinon arret), et ecrit :
        pipeline/02_extract_triples_run18.py
        pipeline/04_clean_validate_run18.py
        pipeline/RUN18_PATCHES.md   (regles, sha256 avant/apres, diff complet)
    Il refuse d'ecraser un fichier existant.

CHANGEMENTS (et rien d'autre)
    02  E1  les exemples des prompts descripteur, causal et contexte sont lus
            dans --examples (JSON) au lieu d'etre ecrits dans le code ;
        E2  les deux contre-exemples qui citaient un couple objet/descripteur
            sont remplaces par une formulation generique ;
    04  N1  la protection contre la fusion porte sur tout le vocabulaire de
            l'ontologie (KNOWN_DESCRIPTORS, 40 termes) ;
        N2  la protection des cadres porte sur KNOWN_SETTINGS (liste de 04) ;
        N3  la liste d'aretes d'evaluation est retiree du code (liste vide) :
            le rappel n'est plus calcule pendant le nettoyage, seulement a l'etape 07.
    Le fichier --examples est refuse si un exemple nomme les deux arguments
    d'une arete de configs/lb_reference_edges_original26.json.

FORMAT DE --examples
    {"source": "texte libre : d'ou viennent les exemples",
     "descriptor": [{"source":..,"source_type":..,"relation":"hasDescriptor","target":..,
                     "target_type":"Descriptor","salience":"typical"}, ...],
     "causal":     [{"source":..,"source_type":..,"relation":..,"target":..,"target_type":..}, ...],
     "context":    [ ... ]}

USAGE (racine du worktree run18)
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
    """Bloc d'exemples : du marqueur jusqu'au ']' qui ferme la liste."""
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
    # les trois blocs dans l'ordre : descripteur, causal, contexte
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

    # controles
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