#!/usr/bin/env python3
"""190_controles_et_coherence.py : lecture seule, aucun modele, aucun run.

Complete 189 sur les points restes vides ou nouveaux :

  1. Coherence du graphe livre et du graphe nettoye :
     tier x tier_label x tier_original, tier x verdict x m4.decision,
     m4.flags x verdict, relations dont la cle de contenu est en double.
  2. Fichiers des controles : inventaire des dossiers stress_test et 181
     (taille, sha, nombre d'enregistrements, champs), identite des controls.jsonl.
  3. Verdicts des juges sur les 728 controles : recherche des fichiers qui les
     contiennent, jointure sur des champs communs (jamais sur l'ordre sans le dire),
     distribution complete des valeurs par type de perturbation.
  4. Cas ambigus par type de perturbation, a partir de supporting_passage et de
     la relation originale retrouvee par orig_index :
       passage_perm : le passage echange nomme-t-il deja le sujet et l'objet ?
       entity_sub   : l'entite introduite figure-t-elle dans le passage ?
       relation_sub : relation originale -> relation introduite (matrice)
       inversion    : relations inversees
  5. Etapes du manifeste avec leur duree, et sacct si des numeros de job sont donnes.

N'ecrit que dans --outdir, refuse d'ecraser un dossier existant.

Usage (depuis le worktree run18) :
  python ~/ontogeorag/190_controles_et_coherence.py --repo . \
      --outdir output/audit/190_controles_et_coherence
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

VERDICT = re.compile(r"SUPPORT|ACCEPT|REJECT|UNCERTAIN|PLAUSIBLE|NOT_|IMPLAUSIBLE|KEEP|DROP|DISCARD",
                     re.I)
EXT = (".json", ".jsonl", ".csv", ".tsv")
MAX_MOD = 12

FAMILLES = {
    "causal": {"causes", "triggers", "controls", "affects", "formedby"},
    "stratigraphique": {"overlies", "underlies"},
    "spatial": {"occursin", "partof"},
}


def norm(x):
    if x is None:
        return ""
    s = unicodedata.normalize("NFKC", str(x)).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def nrel(x):
    return norm(x).replace(" ", "")


def famille(r):
    for f, membres in FAMILLES.items():
        if nrel(r) in membres:
            return f
    return nrel(r)


def sha(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def aplatir(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = json.dumps(v2, ensure_ascii=False) if isinstance(v2, (dict, list)) else v2
        elif isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def charger(chemin):
    """(liste d'enregistrements aplatis, objet brut si json non tabulaire)."""
    ext = os.path.splitext(chemin)[1].lower()
    try:
        if ext == ".jsonl":
            with open(chemin, encoding="utf-8") as f:
                return [aplatir(json.loads(l)) for l in f if l.strip().startswith("{")], None
        if ext == ".json":
            with open(chemin, encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [aplatir(r) for r in obj if isinstance(r, dict)], None
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return [aplatir(r) for r in v if isinstance(r, dict)], obj
            return [], obj
        if ext in (".csv", ".tsv"):
            with open(chemin, encoding="utf-8", newline="") as f:
                return [dict(r) for r in csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ",")], None
    except Exception as e:
        print(f"  [illisible] {chemin} : {e}", file=sys.stderr)
    return [], None


def cles(recs):
    vues = []
    for r in recs[:500]:
        for k in r:
            if k not in vues:
                vues.append(k)
    return vues


def scalaires(obj, chemin="", out=None, limite=80):
    out = [] if out is None else out
    if len(out) >= limite:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            scalaires(v, f"{chemin}.{k}" if chemin else k, out, limite)
    elif isinstance(obj, list):
        if obj and not isinstance(obj[0], (dict, list)):
            out.append(f"{chemin} = {str(obj)[:120]}")
        else:
            for i, v in enumerate(obj[:5]):
                scalaires(v, f"{chemin}[{i}]", out, limite)
    else:
        out.append(f"{chemin} = {str(obj)[:120]}")
    return out


def tableau(recs, a, b, out, titre):
    t = Counter((str(r.get(a)), str(r.get(b))) for r in recs)
    out.append(f"   {titre} ({a} x {b}) :")
    for (x, y), n in sorted(t.items()):
        out.append(f"      {x:<22} {y:<35} {n}")


# --------------------------------------------------------------------------
def coherence(nom, chemin, out):
    recs, _ = charger(chemin)
    if not recs:
        out.append(f"[{nom}] introuvable ou vide : {chemin}")
        return
    out.append(f"[{nom}] {chemin} : {len(recs)} relations (sha {sha(chemin)})")
    tableau(recs, "tier", "tier_label", out, "tier et etiquette")
    tableau(recs, "tier", "tier_original", out, "tier et tier d'origine")
    t = Counter((str(r.get("tier")), str(r.get("verdict")), str(r.get("m4.decision"))) for r in recs)
    out.append("   tier x verdict x m4.decision :")
    for k, n in sorted(t.items()):
        out.append(f"      {' | '.join(k):<45} {n}")
    tableau(recs, "m4.flags", "verdict", out, "drapeaux et verdict")
    groupes = defaultdict(list)
    for i, r in enumerate(recs):
        groupes[(norm(r.get("subject")), nrel(r.get("relation")), norm(r.get("object")))].append(r)
    doubles = {k: v for k, v in groupes.items() if len(v) > 1}
    out.append(f"   cles de contenu en double apres normalisation : {len(doubles)}")
    for k, v in doubles.items():
        for r in v:
            out.append(f"      {r.get('subject')!r} --{r.get('relation')}--> {r.get('object')!r} "
                       f"| tier {r.get('tier')} | origin {r.get('origin')} | decision {r.get('m4.decision')}")


# --------------------------------------------------------------------------
def inventaire(dossiers, out):
    fichiers = []
    for d in dossiers:
        if not os.path.isdir(d):
            out.append(f"[absent] {d}")
            continue
        for f in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if not os.path.isfile(f):
                continue
            taille = os.path.getsize(f)
            ligne = f"   {f} | {taille} o | sha {sha(f)}"
            if os.path.splitext(f)[1].lower() in EXT and taille < 300_000_000:
                recs, brut = charger(f)
                ligne += f" | {len(recs)} enregistrements"
                if recs:
                    ligne += " | champs : " + ", ".join(cles(recs))[:400]
                fichiers.append((f, recs, brut))
            out.append(ligne)
    return fichiers


def champs_verdict(recs):
    res = []
    for k in cles(recs):
        vals = Counter(str(r.get(k)) for r in recs)
        if 1 < len(vals) <= MAX_MOD * 3 and any(VERDICT.search(v) for v in vals):
            res.append(k)
    return res


def cle_jointure(controles, recs):
    """Champs communs utilisables pour joindre, du plus sur au moins sur."""
    kc, kr = set(cles(controles)), set(cles(recs))
    for jeu in (["orig_index", "label", "corruption_class", "subject", "relation", "object"],
                ["orig_index", "label", "corruption_class"],
                ["orig_index", "label"],
                ["subject", "relation", "object", "label"],
                ["subject", "relation", "object"]):
        if all(c in kc and c in kr for c in jeu):
            return jeu
    return None


def valeur_cle(r, jeu):
    return tuple(nrel(r.get(c)) if c == "relation" else norm(r.get(c)) for c in jeu)


def verdicts(controles, fichiers, chemin_ctrl, out, lignes):
    trouve = False
    for f, recs, brut in fichiers:
        if os.path.abspath(f) == os.path.abspath(chemin_ctrl):
            continue
        if not recs:
            if brut is not None:
                out.append(f"   [resume json] {f}")
                out += ["      " + s for s in scalaires(brut)]
            continue
        kv = champs_verdict(recs)
        if not kv:
            if brut is not None:
                out.append(f"   [resume json] {f}")
                out += ["      " + s for s in scalaires(brut)]
            continue
        jeu = cle_jointure(controles, recs)
        mode = "champs " + ",".join(jeu) if jeu else None
        if jeu:
            idx = defaultdict(list)
            for r in recs:
                idx[valeur_cle(r, jeu)].append(r)
            paires = []
            ambigus = 0
            for c in controles:
                m = idx.get(valeur_cle(c, jeu), [])
                if len(m) == 1:
                    paires.append((c, m[0]))
                elif len(m) > 1:
                    ambigus += 1
            out.append(f"   [verdicts] {f} : {len(recs)} enr. ; jointure sur {mode} : "
                       f"{len(paires)}/{len(controles)} uniques, {ambigus} multiples")
        elif len(recs) == len(controles):
            paires = list(zip(controles, recs))
            mode = "ORDRE DES LIGNES (aucun champ commun) : a confirmer avant de citer"
            out.append(f"   [verdicts] {f} : {len(recs)} enr. ; jointure par {mode}")
        else:
            out.append(f"   [verdicts non joignables] {f} : {len(recs)} enr., champs {cles(recs)[:15]}")
            continue
        if not paires:
            continue
        trouve = True
        for k in kv:
            par = defaultdict(Counter)
            for c, r in paires:
                cat = "original" if str(c.get("label")) == "original" else str(c.get("corruption_class"))
                par[cat][str(r.get(k))] += 1
            out.append(f"      -- {k}")
            for cat in sorted(par):
                n = sum(par[cat].values())
                out.append(f"         {cat:<14} n={n:<4} " +
                           ", ".join(f"{v} {x}" for v, x in par[cat].most_common()))
                for v, x in par[cat].items():
                    lignes.append([f, mode, k, cat, n, v, x])
    if not trouve:
        out.append("   AUCUN fichier de verdicts joignable : envoyer la liste ci-dessus, "
                   "ou relancer avec --verdicts <fichier>")


# --------------------------------------------------------------------------
def ambiguites(controles, out, lignes):
    originaux = {str(c.get("orig_index")): c for c in controles if str(c.get("label")) == "original"}
    pert = [c for c in controles if str(c.get("label")) != "original"]
    out.append(f"   originaux {len(originaux)}, perturbes {len(pert)} ; "
               f"perturbes sans original retrouve : "
               f"{sum(1 for c in pert if str(c.get('orig_index')) not in originaux)}")
    bilan = defaultdict(Counter)
    matrice = Counter()
    inv = Counter()
    for c in pert:
        cat = str(c.get("corruption_class"))
        o = originaux.get(str(c.get("orig_index")))
        s, p, ob = norm(c.get("subject")), nrel(c.get("relation")), norm(c.get("object"))
        passage = norm(c.get("supporting_passage"))
        statut = "non evalue"
        detail = ""
        if o is None:
            statut = "original introuvable"
        elif cat == "passage_perm":
            p0 = norm(o.get("supporting_passage"))
            if passage == p0:
                statut = "passage identique a l'original"
            else:
                ns, no = s in passage, ob in passage
                statut = {(True, True): "ambigu (sujet et objet dans le nouveau passage)",
                          (True, False): "sujet seul dans le nouveau passage",
                          (False, True): "objet seul dans le nouveau passage",
                          (False, False): "non ambigu"}[(ns, no)]
        elif cat == "entity_sub":
            s0, o0 = norm(o.get("subject")), norm(o.get("object"))
            nouvelles = [e for e, e0 in ((s, s0), (ob, o0)) if e != e0]
            detail = " / ".join(nouvelles)
            if not nouvelles:
                statut = "aucune entite changee"
            elif all(e in passage for e in nouvelles):
                statut = "ambigu (entite introduite presente dans le passage)"
            else:
                statut = "non ambigu"
        elif cat == "relation_sub":
            r0 = nrel(o.get("relation"))
            matrice[(o.get("relation"), c.get("relation"))] += 1
            detail = f"{o.get('relation')} -> {c.get('relation')}"
            statut = "ambigu (meme famille)" if famille(r0) == famille(p) and r0 != p else "non ambigu"
        elif cat == "inversion":
            inv[str(c.get("relation"))] += 1
            statut = "a juger (relation orientee ?)"
        bilan[cat][statut] += 1
        lignes.append([cat, statut, detail, c.get("orig_index"), c.get("subject"), c.get("relation"),
                       c.get("object"), str(c.get("supporting_passage"))[:300]])
    for cat in sorted(bilan):
        out.append(f"   {cat:<14} " + ", ".join(f"{s} {n}" for s, n in bilan[cat].most_common()))
    out.append("   relation_sub, relation originale -> relation introduite :")
    for (a, b), n in matrice.most_common():
        out.append(f"      {a} -> {b} : {n}")
    out.append("   inversion, relations inversees : " + ", ".join(f"{a} {n}" for a, n in inv.most_common()))
    out.append("   relations presentes dans les controles : " +
               ", ".join(f"{a} {n}" for a, n in Counter(str(c.get('relation')) for c in controles).most_common()))
    out.append("   Indicateur lexical (sous-chaine apres normalisation) : relire ambiguites.csv avant de citer.")


# --------------------------------------------------------------------------
def manifeste(chemin, out):
    try:
        with open(chemin, encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:
        out.append(f"manifeste non lu : {chemin} : {e}")
        return
    out.append(f"[{chemin}] sha {sha(chemin)}")
    for k, v in m.items():
        if not isinstance(v, (dict, list)):
            out.append(f"   {k} = {str(v)[:150]}")
        elif isinstance(v, dict):
            out += ["   " + s for s in scalaires(v, k, limite=40)]
    debut, fin = None, None
    for i, st in enumerate(m.get("steps", [])):
        if not isinstance(st, dict):
            continue
        desc = ", ".join(f"{k}={str(v)[:70]}" for k, v in st.items()
                         if k not in ("started", "ended") and not isinstance(v, (dict, list)))
        duree = ""
        try:
            a = datetime.fromisoformat(st["started"])
            b = datetime.fromisoformat(st["ended"])
            debut = debut or a
            fin = b
            s = int((b - a).total_seconds())
            duree = f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"
        except Exception:
            pass
        out.append(f"   etape {i} | {duree:<10} | {desc}")
    if debut and fin:
        s = int((fin - debut).total_seconds())
        out.append(f"   du premier debut a la derniere fin : {s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--graphe-avant", default="output/run18/kg_final/flat_run18_enforced_dedup.json")
    ap.add_argument("--graphe-apres", default="output/audit/188_nettoyage_v2/graphe_nettoye.json")
    ap.add_argument("--controles", default="output/run18/stress_test/controls.jsonl")
    ap.add_argument("--dossiers", nargs="*", default=["output/run18/stress_test",
                                                      "output/audit/181_regles_llama",
                                                      "output/audit/181_regles_mistral"])
    ap.add_argument("--verdicts", nargs="*", default=None,
                    help="fichiers de verdicts a joindre (sinon : tous ceux des --dossiers)")
    ap.add_argument("--manifeste", default="output/run18/manifest.json")
    ap.add_argument("--jobs", nargs="*", default=[], help="numeros de jobs SLURM pour sacct")
    a = ap.parse_args()

    R = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(a.repo, x))
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} existe deja : choisis un autre --outdir (aucune sortie n'est ecrasee).")
    os.makedirs(outdir)
    out = ["190_controles_et_coherence : lecture seule."]

    out.append("\n=== 1. Coherence des graphes ===")
    coherence("avant nettoyage", R(a.graphe_avant), out)
    coherence("apres nettoyage", R(a.graphe_apres), out)

    out.append("\n=== 2. Inventaire des dossiers de controles ===")
    fichiers = inventaire([R(d) for d in a.dossiers], out)
    shas = defaultdict(list)
    for f, _, _ in fichiers:
        if os.path.basename(f) == "controls.jsonl":
            shas[sha(f)].append(f)
    out.append("   identite des controls.jsonl : " +
               " ; ".join(f"sha {h} : {len(v)} fichier(s)" for h, v in shas.items()))

    chemin_ctrl = R(a.controles)
    controles, _ = charger(chemin_ctrl)
    out.append(f"\n=== 3. Verdicts sur les controles ({chemin_ctrl}, {len(controles)} enr.) ===")
    lignes_v = []
    if a.verdicts:
        fichiers_v = [(R(f), *charger(R(f))) for f in a.verdicts]
    else:
        fichiers_v = fichiers
    if controles:
        verdicts(controles, fichiers_v, chemin_ctrl, out, lignes_v)
    with open(os.path.join(outdir, "verdicts_par_type.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fichier", "jointure", "champ", "categorie", "n_categorie", "valeur", "effectif"])
        w.writerows(lignes_v)

    out.append("\n=== 4. Cas ambigus par type de perturbation ===")
    lignes_a = []
    if controles:
        ambiguites(controles, out, lignes_a)
    with open(os.path.join(outdir, "ambiguites.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["categorie", "statut", "detail", "orig_index", "sujet", "relation", "objet", "passage"])
        w.writerows(lignes_a)

    out.append("\n=== 5. Etapes, durees, materiel ===")
    manifeste(R(a.manifeste), out)
    for j in a.jobs:
        try:
            r = subprocess.run(["sacct", "-j", j, "--format=JobID,JobName%30,Elapsed,AllocTRES%70,NodeList%20,State"],
                               capture_output=True, text=True, timeout=60)
            out.append(f"[sacct {j}]")
            out += ["   " + l for l in r.stdout.splitlines()]
        except Exception as e:
            out.append(f"[sacct {j}] impossible : {e}")

    with open(os.path.join(outdir, "_resume.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nSorties : {outdir}/_resume.txt, verdicts_par_type.csv, ambiguites.csv")


if __name__ == "__main__":
    main()