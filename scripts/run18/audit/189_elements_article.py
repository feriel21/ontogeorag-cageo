#!/usr/bin/env python3
"""189_elements_article.py: read-only, no model, no run.

Gathers, from the existing run18 outputs, the items still missing to write the
manuscript:

  A. Schema of the files read (to check the detected fields).
  B. The graphs before cleaning (delivered graph) and after cleaning (output of 188):
     relations, entities, relation types, tiers, decisions, flags, coverage of the
     13 LB2019 descriptors, provenance joined by content key.
  C. Perturbed records: rejection rate per perturbation type and per verifier or
     rule, retention rate of the originals, 95 % Wilson intervals.
  D. Ambiguous records per perturbation type (a perturbation that could remain
     supported by the sentence).
  E. Qualification markers (output of 186) and existing summaries of 184, 186,
     187 and 188.
  F. Hardware, duration and calls: manifests and SLURM logs.

The script writes only into --outdir and refuses to overwrite an existing folder.
Joins use normalised content keys, never positions.

Usage (from the run18 worktree):
  python ~/ontogeorag/189_elements_article.py --repo . --run output/run18 \
      --outdir output/audit/189_elements_article
  Messages and output files are in French.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Reading constants (not pipeline parameters)
# --------------------------------------------------------------------------
CAND_SUJET = ["source", "subject", "head", "subj", "s"]
CAND_REL = ["relation", "predicate", "rel", "r"]
CAND_OBJET = ["target", "object", "tail", "obj", "o"]
CAND_PHRASE = ["evidence_sentence", "evidence", "sentence", "phrase", "quote",
               "evidence_text", "support_sentence", "text"]
CAND_TIER = ["tier"]

# Directed predicates: an inversion generally makes the relation false.
ORIENTES = {"causes", "triggers", "controls", "affects", "overlies", "underlies",
            "partof", "formedby", "occursin", "hasdescriptor", "indicates",
            "evidences"}
# Families of neighbouring predicates: a change inside a family
# can remain supported by the sentence.
FAMILLES = {
    "causal": {"causes", "triggers", "controls", "affects", "formedby"},
    "stratigraphique": {"overlies", "underlies"},
    "spatial": {"occursin", "partof"},
    "descriptif": {"hasdescriptor"},
    "association": {"relatedto", "indicates", "evidences"},
}
MOTIF_VERDICT = re.compile(r"SUPPORT|ACCEPT|REJECT|UNCERTAIN|PLAUSIBLE|NOT_|IMPLAUSIBLE",
                           re.I)
MOTIF_REJET = re.compile(r"NOT[_ ]SUPPORTED|REJECT", re.I)
MOTIF_CONTROLES = re.compile(r"stress|control|corrupt|adversar|perturb", re.I)
MAX_MODALITES = 12          # above this length, a field is treated as free text


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def norm(x):
    """Text key: NFKC (ligatures), lowercase, hyphen and underscore read as a space."""
    if x is None:
        return ""
    s = unicodedata.normalize("NFKC", str(x)).lower()
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def norm_rel(x):
    return norm(x).replace(" ", "")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    demi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - demi), min(1.0, centre + demi))


def aplatir(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = json.dumps(v2, ensure_ascii=False) if isinstance(v2, (dict, list)) else v2
        else:
            out[k] = v
    return out


def charger(chemin):
    """Return a list of flattened dicts, or [] if the file is not tabular."""
    ext = os.path.splitext(chemin)[1].lower()
    try:
        if ext == ".jsonl":
            with open(chemin, encoding="utf-8") as f:
                return [aplatir(json.loads(l)) for l in f if l.strip().startswith("{")]
        if ext == ".json":
            with open(chemin, encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [aplatir(r) for r in obj if isinstance(r, dict)]
            if isinstance(obj, dict):
                for cle in ["relations", "triples", "edges", "records", "controls",
                            "items", "data", "rows"]:
                    if isinstance(obj.get(cle), list):
                        return [aplatir(r) for r in obj[cle] if isinstance(r, dict)]
                for v in obj.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return [aplatir(r) for r in v if isinstance(r, dict)]
            return []
        if ext in (".csv", ".tsv"):
            with open(chemin, encoding="utf-8", newline="") as f:
                return [dict(r) for r in csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ",")]
    except Exception as e:  # unreadable file: reported, then skipped
        print(f"  [lecture impossible] {chemin} : {e}", file=sys.stderr)
    return []


def cle(cles, candidats, prefixe_exclu=None):
    """Find a field: exact name, then last segment, then substring."""
    cles = [k for k in cles if not (prefixe_exclu and k.lower().startswith(prefixe_exclu))]
    bas = {k.lower(): k for k in cles}
    for c in candidats:
        if c in bas:
            return bas[c]
    for c in candidats:
        for k in cles:
            if k.lower().split(".")[-1] == c:
                return k
    for c in candidats:
        if len(c) > 2:
            for k in cles:
                if c in k.lower():
                    return k
    return None


def toutes_cles(recs):
    vues = []
    for r in recs[:500]:
        for k in r:
            if k not in vues:
                vues.append(k)
    return vues


def champs_categoriels(recs, cles):
    out = {}
    for k in cles:
        vals = Counter(str(r.get(k)) for r in recs)
        if 1 < len(vals) <= MAX_MODALITES:
            out[k] = vals
    return out


def cles_triplet(recs, prefixe_exclu=None):
    ks = toutes_cles(recs)
    return (cle(ks, CAND_SUJET, prefixe_exclu), cle(ks, CAND_REL, prefixe_exclu),
            cle(ks, CAND_OBJET, prefixe_exclu))


def cle_contenu(r, s, p, o):
    return (norm(r.get(s)), norm_rel(r.get(p)), norm(r.get(o)))


# --------------------------------------------------------------------------
# A. Schemas
# --------------------------------------------------------------------------
def schema(nom, recs):
    lignes = [f"== {nom} : {len(recs)} enregistrements"]
    ks = toutes_cles(recs)
    for k in ks:
        vals = Counter(str(r.get(k)) for r in recs)
        ex = str(recs[0].get(k))[:80].replace("\n", " ") if recs else ""
        lignes.append(f"   {k} | {len(vals)} valeurs distinctes | ex. {ex}")
    return lignes


# --------------------------------------------------------------------------
# B. Graphs
# --------------------------------------------------------------------------
def lb13(repo):
    sys.path.insert(0, os.path.abspath(repo))
    try:
        from pipeline.rag.constants import LB2019_BENCHMARK_DESCRIPTORS  # read-only
        return [norm(x) for x in LB2019_BENCHMARK_DESCRIPTORS]
    except Exception as e:
        print(f"  [constantes] LB2019_BENCHMARK_DESCRIPTORS introuvable : {e}", file=sys.stderr)
        return None


def stats_graphe(nom, recs, lb, provenances, out):
    s, p, o = cles_triplet(recs)
    if not (s and p and o):
        out.append(f"[{nom}] champs sujet/relation/objet introuvables : voir schemas.txt")
        return None
    ent = {norm(r.get(s)) for r in recs} | {norm(r.get(o)) for r in recs}
    types = Counter(str(r.get(p)) for r in recs)
    cles_uniques = {cle_contenu(r, s, p, o) for r in recs}
    out.append(f"[{nom}] {len(recs)} relations ({len(cles_uniques)} cles de contenu distinctes), "
               f"{len(ent)} entites distinctes (normalisees)")
    out.append(f"   champs detectes : sujet={s} relation={p} objet={o}")
    out.append("   types de relations : " + ", ".join(f"{k} {v}" for k, v in types.most_common()))
    for k, vals in champs_categoriels(recs, toutes_cles(recs)).items():
        if k in (p,):
            continue
        out.append(f"   {k} : " + ", ".join(f"{a} {b}" for a, b in vals.most_common()))
    if lb is not None:
        desc = {norm(r.get(o)) for r in recs if norm_rel(r.get(p)) == "hasdescriptor"}
        couverts = [d for d in lb if d in desc]
        manquants = [d for d in lb if d not in desc]
        out.append(f"   couverture des {len(lb)} descripteurs LB2019 : {len(couverts)}/{len(lb)} ; "
                   f"absents : {', '.join(manquants) or 'aucun'}")
    for pnom, (precs, ps, pp, po) in provenances.items():
        index = defaultdict(list)
        for r in precs:
            index[cle_contenu(r, ps, pp, po)].append(r)
        jointes = [index[cle_contenu(r, s, p, o)][0] for r in recs if cle_contenu(r, s, p, o) in index]
        out.append(f"   provenance via {pnom} : {len(jointes)}/{len(recs)} relations jointes par cle de contenu")
        for k, vals in champs_categoriels(jointes, toutes_cles(jointes)).items():
            if k in (ps, pp, po):
                continue
            out.append(f"      {k} : " + ", ".join(f"{a} {b}" for a, b in vals.most_common()))
    return {"nom": nom, "relations": len(recs), "entites": len(ent),
            **{f"type:{k}": v for k, v in types.items()}}


# --------------------------------------------------------------------------
# C and D. Perturbed records
# --------------------------------------------------------------------------
def categorie(v):
    t = norm(v)
    if t in ("", "none", "nan", "null"):
        return "original"
    if re.search(r"orig|clean|unmodified|identity|^control$|negative", t):
        return "original"
    if re.search(r"sentence|evidence|phrase|quote", t):
        return "phrase_echangee"
    if re.search(r"inver|revers|flip|direction|subject object", t):
        return "inversion"
    if re.search(r"relation|predicate|rel ", t):
        return "type_change"
    if re.search(r"entity|object|subject|substit|replace|argument", t):
        return "entite_remplacee"
    return "autre:" + t


def champ_type(recs):
    meilleur, score_max = None, 0
    for k, vals in champs_categoriels(recs, toutes_cles(recs)).items():
        cats = {categorie(v) for v in vals}
        score = len([c for c in cats if not c.startswith("autre")])
        if score > score_max:
            meilleur, score_max = k, score
    return meilleur if score_max >= 2 else None


def champs_verdict(recs):
    out = []
    for k, vals in champs_categoriels(recs, toutes_cles(recs)).items():
        if any(MOTIF_VERDICT.search(str(v)) for v in vals):
            out.append(k)
    return out


def ambigu(cat, r, ctrl, orig, k_phrase):
    phrase = norm(r.get(k_phrase)) if k_phrase else ""
    s, p, o = (norm(r.get(ctrl[0])), norm_rel(r.get(ctrl[1])), norm(r.get(ctrl[2])))
    if cat == "phrase_echangee":
        if not phrase:
            return "indetermine (pas de phrase)"
        return "ambigu" if (s and s in phrase and o and o in phrase) else "non ambigu"
    if cat == "entite_remplacee":
        if not phrase:
            return "indetermine (pas de phrase)"
        if all(orig):
            os_, oo = norm(r.get(orig[0])), norm(r.get(orig[2]))
            nouvelles = [e for e, e0 in ((s, os_), (o, oo)) if e and e != e0]
        else:
            nouvelles = [s, o]
        return "ambigu" if nouvelles and all(e in phrase for e in nouvelles) else "non ambigu"
    if cat == "inversion":
        return "non ambigu" if p in ORIENTES else "ambigu (predicat non oriente)"
    if cat == "type_change":
        if not orig[1]:
            return "indetermine (relation originale absente)"
        p0 = norm_rel(r.get(orig[1]))
        meme = any(p in f and p0 in f for f in FAMILLES.values())
        return "ambigu (meme famille)" if meme else "non ambigu"
    return "non evalue"


def analyser_controles(nom, recs, out, lignes_csv, lignes_amb, attendu):
    kt = champ_type(recs)
    if not kt:
        out.append(f"[{nom}] champ du type de perturbation non detecte : voir schemas.txt")
        return
    verdicts = champs_verdict(recs)
    ctrl = cles_triplet(recs, prefixe_exclu="orig")
    ks = toutes_cles(recs)
    orig = tuple(cle([k for k in ks if k.lower().startswith("orig")], c) for c in (CAND_SUJET, CAND_REL, CAND_OBJET))
    k_phrase = cle(ks, CAND_PHRASE)
    par_cat = defaultdict(list)
    for r in recs:
        par_cat[categorie(r.get(kt))].append(r)
    n_orig = len(par_cat.get("original", []))
    n_corr = sum(len(v) for c, v in par_cat.items() if c != "original")
    out.append(f"[{nom}] champ type = {kt} ; originaux {n_orig}, perturbes {n_corr} "
               f"(attendu {attendu[0]} et {attendu[1]} : {'OK' if (n_orig, n_corr) == attendu else 'ECART'})")
    out.append(f"   valeurs du champ type : " + ", ".join(f"{v} -> {categorie(v)} ({n})"
               for v, n in Counter(str(r.get(kt)) for r in recs).most_common()))
    out.append(f"   champs de verdict detectes : {', '.join(verdicts) or 'aucun'}")
    out.append(f"   phrase = {k_phrase} ; triplet presente = {ctrl} ; triplet original = {orig}")
    for kv in verdicts:
        out.append(f"   -- {kv} (rejet = valeur contenant NOT_SUPPORTED ou REJECT)")
        for cat in sorted(par_cat):
            g = par_cat[cat]
            k = sum(1 for r in g if MOTIF_REJET.search(str(r.get(kv))))
            lo, hi = wilson(k, len(g))
            if cat == "original":
                cons = len(g) - k
                clo, chi = wilson(cons, len(g))
                out.append(f"      {cat:<18} conservation {cons}/{len(g)} = {cons/len(g):.3f} "
                           f"[{clo:.3f} ; {chi:.3f}]")
            else:
                out.append(f"      {cat:<18} rejet {k}/{len(g)} = {k/len(g):.3f} [{lo:.3f} ; {hi:.3f}]")
            dist = Counter(str(r.get(kv)) for r in g)
            out.append("         valeurs : " + ", ".join(f"{a} {b}" for a, b in dist.most_common()))
            lignes_csv.append([nom, kv, cat, len(g), k, round(k / len(g), 4) if g else "",
                               round(lo, 4), round(hi, 4)])
    out.append("   cas ambigus par type :")
    for cat in sorted(c for c in par_cat if c != "original"):
        bilan = Counter()
        for r in par_cat[cat]:
            a = ambigu(cat, r, ctrl, orig, k_phrase)
            bilan[a] += 1
            if a.startswith("ambigu"):
                lignes_amb.append([nom, cat, a, r.get(ctrl[0]), r.get(ctrl[1]), r.get(ctrl[2]),
                                   str(r.get(k_phrase))[:300] if k_phrase else ""])
        out.append(f"      {cat:<18} " + ", ".join(f"{a} {b}" for a, b in bilan.most_common()))


# --------------------------------------------------------------------------
# F. Hardware, duration, calls
# --------------------------------------------------------------------------
MOTIF_MANIFESTE = re.compile(r"time|duration|elapsed|gpu|device|host|node|start|end|date|calls|n_", re.I)
MOTIF_SLURM = re.compile(r"NVIDIA|A100|GPU|CUDA_VISIBLE|Elapsed|elapsed|SLURM_JOB_ID|JobId|"
                         r"^real\s|Start|End|duree|duration", re.I)


def parcourir(obj, chemin="", out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            parcourir(v, f"{chemin}.{k}" if chemin else k, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            parcourir(v, f"{chemin}[{i}]", out)
    else:
        if MOTIF_MANIFESTE.search(chemin.split(".")[-1]):
            out.append(f"{chemin} = {str(obj)[:100]}")
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--run", default="output/run18")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--graphe-avant", default=None,
                    help="defaut : <run>/kg_final/flat_run18_enforced_dedup.json")
    ap.add_argument("--dossier-188", default="output/audit/188_nettoyage_v2")
    ap.add_argument("--dossiers-provenance", nargs="*",
                    default=["output/audit/184_trace_nfkc_v2", "output/audit/187_provenance"])
    ap.add_argument("--dossier-186", default="output/audit/186_portee")
    ap.add_argument("--controles", nargs="*", default=None,
                    help="fichiers des controles perturbes ; defaut : recherche sous output/")
    ap.add_argument("--attendu", nargs=2, type=int, default=[364, 364],
                    help="nombre attendu d'originaux et de perturbes")
    ap.add_argument("--slurm-glob", nargs="*", default=["slurm*.out", "logs/**/*.out",
                                                          "slurm/**/*.out", "slurm/**/*.err"])
    a = ap.parse_args()

    R = lambda x: x if os.path.isabs(x) else os.path.join(a.repo, x)
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} existe deja : choisis un autre --outdir (aucune sortie n'est ecrasee).")
    os.makedirs(outdir)

    resume, schemas = [], []
    resume.append("189_elements_article : lecture seule. Chiffres a verifier contre les sorties citees.")

    # ---- B. graphs
    resume.append("\n=== B. Graphes ===")
    avant = R(a.graphe_avant or os.path.join(a.run, "kg_final", "flat_run18_enforced_dedup.json"))
    graphes = []
    if os.path.exists(avant):
        graphes.append(("avant nettoyage : " + os.path.relpath(avant, a.repo), charger(avant)))
    else:
        resume.append(f"graphe livre introuvable : {avant}")
    for f in sorted(glob.glob(os.path.join(R(a.dossier_188), "**", "*"), recursive=True)):
        if os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".csv", ".tsv"):
            recs = charger(f)
            if recs and all(cles_triplet(recs)):
                graphes.append(("sortie 188 : " + os.path.relpath(f, a.repo), recs))
    provenances = {}
    for d in a.dossiers_provenance:
        for f in sorted(glob.glob(os.path.join(R(d), "**", "*"), recursive=True)):
            if os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".csv", ".tsv"):
                recs = charger(f)
                tr = cles_triplet(recs) if recs else (None, None, None)
                if recs and all(tr):
                    provenances[os.path.relpath(f, a.repo)] = (recs, *tr)
                    schemas += schema("provenance " + os.path.relpath(f, a.repo), recs)
    lb = lb13(a.repo)
    resume.append(f"descripteurs LB2019 charges : {len(lb) if lb else 'non'}")
    lignes_g = []
    for nom, recs in graphes:
        schemas += schema(nom, recs)
        st = stats_graphe(nom, recs, lb, provenances, resume)
        if st:
            lignes_g.append(st)
    if lignes_g:
        cols = sorted({k for d in lignes_g for k in d})
        with open(os.path.join(outdir, "graphes.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(lignes_g)

    # ---- C and D. perturbed records
    resume.append("\n=== C et D. Controles perturbes ===")
    if a.controles:
        fichiers = [R(x) for x in a.controles]
    else:
        fichiers = [f for f in glob.glob(os.path.join(R("output"), "**", "*"), recursive=True)
                    if MOTIF_CONTROLES.search(os.path.basename(f))
                    and os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".csv", ".tsv")
                    and not os.path.abspath(f).startswith(os.path.abspath(outdir))]
    lignes_csv, lignes_amb = [], []
    if not fichiers:
        resume.append("aucun fichier de controles trouve : relancer avec --controles <fichier>")
    for f in sorted(fichiers):
        recs = charger(f)
        if not recs:
            continue
        nom = os.path.relpath(f, a.repo)
        schemas += schema("controles " + nom, recs)
        analyser_controles(nom, recs, resume, lignes_csv, lignes_amb, tuple(a.attendu))
    with open(os.path.join(outdir, "controles_par_type.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fichier", "champ_verdict", "categorie", "n", "n_rejet", "taux_rejet",
                    "wilson_bas", "wilson_haut"])
        w.writerows(lignes_csv)
    with open(os.path.join(outdir, "ambiguites.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fichier", "categorie", "statut", "sujet", "relation", "objet", "phrase"])
        w.writerows(lignes_amb)

    # ---- E. qualification markers and existing summaries
    resume.append("\n=== E. Nuance (186) et resumes existants ===")
    for f in sorted(glob.glob(os.path.join(R(a.dossier_186), "**", "*"), recursive=True)):
        if os.path.splitext(f)[1].lower() in (".json", ".jsonl", ".csv", ".tsv"):
            recs = charger(f)
            if not recs:
                continue
            nom = os.path.relpath(f, a.repo)
            schemas += schema("186 " + nom, recs)
            resume.append(f"[{nom}] {len(recs)} enregistrements")
            for k, vals in champs_categoriels(recs, toutes_cles(recs)).items():
                resume.append(f"   {k} : " + ", ".join(f"{x} {n}" for x, n in vals.most_common()))
    for d in [a.dossier_186, a.dossier_188] + list(a.dossiers_provenance):
        for f in sorted(glob.glob(os.path.join(R(d), "*resume*.txt"))):
            resume.append(f"--- copie de {os.path.relpath(f, a.repo)} (80 premieres lignes)")
            with open(f, encoding="utf-8", errors="replace") as h:
                resume += [l.rstrip("\n") for l in h.readlines()[:80]]

    # ---- F. hardware, duration, calls
    resume.append("\n=== F. Materiel, duree, appels ===")
    for f in sorted(glob.glob(os.path.join(R(a.run), "**", "*manifest*"), recursive=True))[:10]:
        try:
            with open(f, encoding="utf-8") as h:
                obj = json.load(h)
            trouv = parcourir(obj)
            resume.append(f"[{os.path.relpath(f, a.repo)}] {len(trouv)} champs pertinents")
            resume += ["   " + t for t in trouv[:60]]
        except Exception as e:
            resume.append(f"[{os.path.relpath(f, a.repo)}] non lu : {e}")
    vus = 0
    for motif in a.slurm_glob:
        for f in sorted(glob.glob(R(motif), recursive=True)):
            if vus >= 40:
                break
            vus += 1
            with open(f, encoding="utf-8", errors="replace") as h:
                lignes = [l.strip() for l in h if MOTIF_SLURM.search(l)][:8]
            if lignes:
                resume.append(f"[{os.path.relpath(f, a.repo)}]")
                resume += ["   " + l[:160] for l in lignes]
    resume.append("Pour la duree exacte et le GPU d'un job : "
                  "sacct -j <JOBID> --format=JobID,JobName%30,Elapsed,AllocTRES%60,NodeList")

    with open(os.path.join(outdir, "_resume.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(resume) + "\n")
    with open(os.path.join(outdir, "schemas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(schemas) + "\n")
    print("\n".join(resume))
    print(f"\nSorties : {outdir}/_resume.txt, schemas.txt, graphes.csv, "
          f"controles_par_type.csv, ambiguites.csv")


if __name__ == "__main__":
    main()