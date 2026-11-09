#!/usr/bin/env python3
"""177 — Candidate sentences for the run18 prompt examples, taken OUTSIDE the evaluated corpus.

WHY
    Prompt examples must come from texts that the pipeline does not evaluate,
    as examples are usually excluded from the evaluation set.
    The archived index step1_contamine_3386 holds papers that are not among
    the 41 papers of the corpus.
    The script extracts candidate sentences from those papers.
    The final examples (6 to 8 per type) are chosen by hand, then checked by
    176 and 174.

STEPS (read-only, CPU)
    1. Out-of-corpus documents are the documents of --index-hors that are
       absent from --index16E, after removing the "-checkpoint" suffix
       (copies of the same paper). The number of distinct out-of-corpus
       documents is printed.
    2. Text is split into sentences, with three families of candidates:
         descriptor : a descriptor of the vocabulary (KNOWN_DESCRIPTORS) and
                      a geological object
         causal     : a causal verb (trigger, cause, control, induce, lead to,
                      result in, promote)
         context    : a setting (KNOWN_SETTINGS) or overlie/underlie
    3. Any sentence that names both arguments of an evaluation edge
       (configs/lb_reference_edges_original26.json) is excluded.
    4. At most --max sentences per family are drawn with a fixed seed.

OUTPUTS (--outdir, never overwritten; file names and messages in French)
    candidats_descriptor.tsv, candidats_causal.tsv, candidats_context.tsv,
    documents_hors_corpus.txt, _resume.txt

USAGE (root of the run18 worktree)
    python 177_candidats_exemples_hors_corpus.py --repo . \
        --index-hors $HOME/ontogeorag/output/_archive/step1_contamine_3386/chunks.jsonl \
        --index16E $HOME/ontogeorag_D/output/run16E/index/chunks.jsonl \
        --outdir output/audit/177_candidats
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

OBJECTS = r"(mass[- ]transport (deposit|complex)e?s?|mtds?|mtcs?|debris[- ]flows?|debrites?|slides?|slumps?|" \
          r"turbidites?|turbidity currents?|hemipelagites?|megaslides?|blocks?|landslides?)"
CAUSAL = r"(trigger(s|ed|ing)?|caus(e|es|ed|ing)|control(s|led|ling)?|induc(e|es|ed|ing)|lead(s)? to|led to|" \
         r"result(s|ed)? in|promot(e|es|ed|ing))"
STRAT = r"(overl(ie|ies|ain|ying)|underl(ie|ies|ain|ying))"


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def norm(s):
    return " ".join(str(s or "").replace("_", " ").split()).strip().lower()


def mentions(term, text):
    return bool(term) and re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text) is not None


def base(doc):
    return re.sub(r"-checkpoint$", "", str(doc))


def sentences(text):
    t = " ".join(str(text).split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t) if 60 <= len(s.strip()) <= 400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--index-hors", required=True)
    ap.add_argument("--index16E", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    out = Path(args.outdir)
    if out.exists():
        die("le dossier existe deja, rien n'est ecrase : " + str(out))
    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    from pipeline.rag.constants import KNOWN_DESCRIPTORS, KNOWN_SETTINGS
    edges = [(norm(e["subject"]), norm(e["object"])) for e in
             json.load(open(repo / "configs/lb_reference_edges_original26.json", encoding="utf-8"))["edges"]]

    ref_docs = {base(json.loads(l)["doc_id"]) for l in open(args.index16E, encoding="utf-8") if l.strip()}
    rows = [json.loads(l) for l in open(args.index_hors, encoding="utf-8") if l.strip()]
    outside = sorted({base(r["doc_id"]) for r in rows} - ref_docs)
    if not outside:
        die("aucun document hors corpus")

    descs = sorted({norm(d) for d in KNOWN_DESCRIPTORS}, key=len, reverse=True)
    sets = sorted({norm(s) for s in KNOWN_SETTINGS}, key=len, reverse=True)
    seen, cands = set(), {"descriptor": [], "causal": [], "context": []}
    n_excl = 0
    for r in rows:
        d = base(r["doc_id"])
        if d not in outside:
            continue
        for s in sentences(r.get("text", "")):
            low = s.lower()
            key = re.sub(r"[^a-z0-9]", "", low)
            if key in seen:
                continue
            seen.add(key)
            if any(mentions(a, low) and mentions(b, low) for a, b in edges):
                n_excl += 1
                continue
            hit_desc = [x for x in descs if mentions(x, low.replace(" amplitude", "-amplitude"))]
            obj = re.search(r"(?<![a-z])" + OBJECTS + r"(?![a-z])", low)
            if hit_desc and obj:
                cands["descriptor"].append((d, s, obj.group(0) + " | " + ", ".join(hit_desc[:4])))
            c = re.search(r"(?<![a-z])" + CAUSAL + r"(?![a-z])", low)
            if c:
                cands["causal"].append((d, s, c.group(0)))
            hit_set = [x for x in sets if mentions(x, low)]
            st = re.search(r"(?<![a-z])" + STRAT + r"(?![a-z])", low)
            if hit_set or st:
                cands["context"].append((d, s, ", ".join(hit_set[:3]) + ((" | " + st.group(0)) if st else "")))

    out.mkdir(parents=True)
    rng = random.Random(args.seed)
    L = ["=== 177 phrases candidates hors corpus pour les exemples du prompt ===",
         "documents de l'index archive : %d ; vrais documents hors corpus (sans copies -checkpoint) : %d"
         % (len({r["doc_id"] for r in rows}), len(outside)),
         "phrases exclues car elles nomment une arete d'evaluation : %d" % n_excl]
    for kind, lst in cands.items():
        pick = lst if len(lst) <= args.max else rng.sample(lst, args.max)
        with open(out / ("candidats_%s.tsv" % kind), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["document", "termes_reperes", "phrase"])
            for d, s, t in sorted(pick):
                w.writerow([d, t, s])
        L.append("%-10s : %d candidates, %d ecrites" % (kind, len(lst), len(pick)))
    (out / "documents_hors_corpus.txt").write_text("\n".join(outside) + "\n", encoding="utf-8")
    L.append("graine : %d" % args.seed)
    txt = "\n".join(L)
    (out / "_resume.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt + "\n[fait] " + str(out), flush=True)


if __name__ == "__main__":
    main()