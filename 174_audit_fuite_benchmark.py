#!/usr/bin/env python3
"""174 — Fuite du benchmark (26 aretes) dans le pipeline : audit et porte d'entree de run18.

POURQUOI
    Un rappel contre le benchmark n'est une mesure que si le pipeline n'a pas
    vu le benchmark. Ce script cherche les 26 aretes de
    configs/lb_reference_edges_original26.json dans tout ce que le pipeline
    lit ou execute avant l'evaluation (etape 07), et s'arrete en code 1 a la
    moindre fuite. Lance sur run16E, il mesure la fuite ; lance avant run18,
    il prouve qu'il n'y en a plus.

QUOI (lecture seule, CPU)
    F1 prompt   : exemples {source, relation, target} du script d'extraction
                  (--extract) qui nomment les deux arguments d'une arete.
    F2 requetes : requetes du fichier --queries dont l'ancre (focus) ou le
                  texte nomme les deux arguments d'une arete, en termes entiers
                  (regle R12 de manifest_parameters_and_benchmark_targeting.py),
                  plus les familles d'ancres LB: et gapfill: (ecrites a partir
                  du graphe expert ou pour combler une arete manquee).
    F3 nettoyage: termes du benchmark proteges nommement dans le script de
                  nettoyage (--clean) : import de LB2019_BENCHMARK_DESCRIPTORS,
                  liste lb_settings, liste LB_REFERENCE_EDGES non vide (rappel
                  calcule pendant le nettoyage).
    Pour chaque arete : vue dans le prompt, nombre de requetes qui la visent,
    protegee au nettoyage. Avec --flat NOM=chemin (repetable) : retrouvee ou non
    (apparieur exact de 07), et rappel separe aretes vues / jamais vues.

USAGE
    python 174_audit_fuite_benchmark.py --repo . \
        --extract pipeline/02_extract_triples.py \
        --queries configs/descriptor_queries_v2.jsonl \
        --clean pipeline/04_clean_validate.py \
        --flat run16E=results/run16E/flat_run16E_enforced_dedup.json \
        --outdir output/audit/174_fuite
    Code de sortie : 0 aucune fuite, 1 fuite trouvee, 2 erreur d'entree.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REF26 = "configs/lb_reference_edges_original26.json"


def die(msg, code=2):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


def norm(s):
    return " ".join(str(s or "").replace("_", " ").split()).strip().lower()


def nrel(r):
    return re.sub(r"[\s_\-]", "", str(r or "").lower())


def mentions(term, text):
    """Terme entier : un terme contenu dans un terme plus long ne compte pas."""
    return bool(term) and re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text) is not None


def anchor_text(focus):
    f = str(focus or "")
    for p in ("combo:", "gapfill:", "lb:", "LB:"):
        if f.lower().startswith(p.lower()):
            f = f[len(p):]
    return " ".join(norm(p) for p in re.split(r"->|\+|:", f))


def family(focus):
    f = str(focus or "")
    for p in ("gapfill:", "LB:", "combo:"):
        if f.startswith(p):
            return p
    return "autre"


def load_07(repo):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("m07", repo / "pipeline" / "07_final_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--extract", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--flat", action="append", default=[], help="NOM=chemin du graphe plat")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    if out.exists():
        die("le dossier existe deja, rien n'est ecrase : " + str(out))
    repo = Path(args.repo).resolve()
    ref = json.load(open(repo / REF26, encoding="utf-8"))
    edges = [(norm(e["subject"]), e["relation"], norm(e["object"])) for e in ref["edges"]]
    if len(edges) != 26:
        die("benchmark : %d aretes, attendu 26" % len(edges))

    # F1 — exemples du prompt
    src = Path(args.extract).read_text(encoding="utf-8")
    examples = []
    for m in re.finditer(r"\{\{(.*?)\}\}", src, flags=re.S):
        g = dict(re.findall(r'"(source|relation|target)":\s*"([^"]+)"', m.group(1)))
        if {"source", "relation", "target"} <= set(g):
            examples.append((norm(g["source"]), g["relation"], norm(g["target"])))
    bad = src.find("BAD EXAMPLES")
    # les exemples "BAD" sont ceux qui suivent immediatement le marqueur ; on les signale a part
    in_prompt = {}
    for e in edges:
        hits = [x for x in examples if x[0] == e[0] and x[2] == e[2]]
        in_prompt[e] = len(hits)

    # F2 — requetes
    Q = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    targeted = {e: [] for e in edges}
    fam_count = {"gapfill:": 0, "LB:": 0}
    flagged = set()
    for i, q in enumerate(Q):
        fam = family(q.get("focus"))
        if fam in fam_count:
            fam_count[fam] += 1
            flagged.add(i)
        a, t = anchor_text(q.get("focus")), norm(q.get("query"))
        for e in edges:
            if (mentions(e[0], a) and mentions(e[2], a)) or (mentions(e[0], t) and mentions(e[2], t)):
                targeted[e].append(i)
                flagged.add(i)

    # F3 — nettoyage
    csrc = Path(args.clean).read_text(encoding="utf-8")
    f3 = {"imports_LB2019_BENCHMARK_DESCRIPTORS": "LB2019_BENCHMARK_DESCRIPTORS" in csrc,
          "computes_lb_recall": re.search(r"LB_REFERENCE_EDGES\s*=\s*\[\s*\(", csrc) is not None,
          "lb_settings": []}
    m = re.search(r"lb_settings\s*=\s*\{(.*?)\}", csrc, flags=re.S)
    if m:
        f3["lb_settings"] = sorted(norm(x) for x in re.findall(r'"([^"]+)"', m.group(1)))
    lb_desc = set()
    if f3["imports_LB2019_BENCHMARK_DESCRIPTORS"]:
        sys.path.insert(0, str(repo))
        from pipeline.rag.constants import LB2019_BENCHMARK_DESCRIPTORS as LBD
        lb_desc = {norm(x) for x in LBD}
    protected = {e: (e[2] in lb_desc and nrel(e[1]) == "hasdescriptor")
                 or e[0] in f3["lb_settings"] or e[2] in f3["lb_settings"] for e in edges}

    # rappel separe (optionnel)
    found = {}
    if args.flat:
        m07 = load_07(repo)
        import os
        cwd = os.getcwd()
        os.chdir(repo)
        for spec in args.flat:
            name, path = spec.split("=", 1)
            path = str(Path(cwd, path)) if not Path(path).is_absolute() else path
            raw = json.load(open(path, encoding="utf-8"))
            g = raw.get("triples", raw) if isinstance(raw, dict) else raw
            r = m07.recall([t for t in g if t.get("tier", 3) <= 2], ref_path=REF26)
            found[name] = {(norm(s), rr, norm(o)) for s, rr, o in r["lower_bound"]["matched_edges"]}
        os.chdir(cwd)

    rows, L = [], []
    for e in edges:
        seen = in_prompt[e] > 0 or len(targeted[e]) > 0 or protected[e]
        row = {"edge": list(e), "in_prompt_examples": in_prompt[e], "n_queries_targeting": len(targeted[e]),
               "protected_in_cleaning": protected[e], "seen_by_pipeline": seen}
        for name in found:
            row["recovered_" + name] = e in found[name]
        rows.append(row)

    n_prompt = sum(1 for e in edges if in_prompt[e])
    n_q = sum(1 for e in edges if targeted[e])
    n_prot = sum(1 for e in edges if protected[e])
    n_seen = sum(1 for r in rows if r["seen_by_pipeline"])
    leak = n_prompt + n_q + n_prot + len(flagged) > 0 or f3["computes_lb_recall"]

    L.append("=== 174 fuite du benchmark 26 dans le pipeline ===")
    L.append("F1 prompt   : %d exemples, dont %d nomment une arete ; %d aretes sur 26 vues dans le prompt"
             % (len(examples), sum(1 for x in examples if any(x[0] == e[0] and x[2] == e[2] for e in edges)), n_prompt))
    L.append("F2 requetes : %d requetes ; familles LB: %d, gapfill: %d ; %d requetes a retirer ; %d aretes visees"
             % (len(Q), fam_count["LB:"], fam_count["gapfill:"], len(flagged), n_q))
    L.append("F3 nettoyage: descripteurs du benchmark importes %s ; lb_settings %s ; calcul de rappel %s ; %d aretes protegees"
             % (f3["imports_LB2019_BENCHMARK_DESCRIPTORS"], f3["lb_settings"] or "aucun", f3["computes_lb_recall"], n_prot))
    L.append("aretes vues par le pipeline (au moins une fuite) : %d / 26 ; jamais vues : %d" % (n_seen, 26 - n_seen))
    for name in found:
        s = sum(1 for r in rows if r["seen_by_pipeline"] and r["recovered_" + name])
        u = sum(1 for r in rows if not r["seen_by_pipeline"] and r["recovered_" + name])
        p = sum(1 for r in rows if r["in_prompt_examples"] and r["recovered_" + name])
        np_ = sum(1 for r in rows if not r["in_prompt_examples"] and r["recovered_" + name])
        L.append("%s : rappel exact %d/26 = vues %d/%d + jamais vues %d/%d ; dans le prompt %d/%d, hors prompt %d/%d"
                 % (name, s + u, s, n_seen, u, 26 - n_seen, p, n_prompt, np_, 26 - n_prompt))
    L.append("")
    L.append("%-52s %6s %4s %5s" % ("arete", "prompt", "req", "prot") + "".join(" %8s" % n for n in found))
    for r in rows:
        e = r["edge"]
        L.append("%-52s %6s %4d %5s" % ("%s --%s--> %s" % (e[0], e[1], e[2]), "oui" if r["in_prompt_examples"] else "-",
                                        r["n_queries_targeting"], "oui" if r["protected_in_cleaning"] else "-")
                 + "".join(" %8s" % ("oui" if r["recovered_" + n] else "-") for n in found))
    L.append("")
    L.append("VERDICT : " + ("FUITE — le rappel sur 26 n'est pas une mesure independante" if leak
                             else "AUCUNE FUITE — le rappel sur 26 est mesurable"))

    out.mkdir(parents=True)
    json.dump({"inputs": {"extract": args.extract, "queries": args.queries, "clean": args.clean, "flat": args.flat},
               "f1_prompt_examples": [list(x) for x in examples], "f2_flagged_query_indexes": sorted(flagged),
               "f2_families": fam_count, "f3": f3, "edges": rows, "leak": leak},
              open(out / "report.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    with open(out / "queries_flagged.jsonl", "w", encoding="utf-8") as f:
        for i in sorted(flagged):
            f.write(json.dumps({"index": i, **Q[i]}, ensure_ascii=False) + "\n")
    txt = "\n".join(L)
    (out / "_resume.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt + "\n[fait] " + str(out), flush=True)
    sys.exit(1 if leak else 0)


if __name__ == "__main__":
    main()