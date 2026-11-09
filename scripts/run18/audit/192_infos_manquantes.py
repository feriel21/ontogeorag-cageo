#!/usr/bin/env python3
"""192_infos_manquantes.py : lecture seule, aucun modele, aucun run.

Rassemble les dernieres informations manquantes pour l'article :

  1. Controle sur le passage applique aux graphes (364 et 281) : relation signalee
     si au moins un juge rend NOT_SUPPORTED dans m4.evidence_verdict (Llama|Mistral).
     Verification : sur le graphe de 364, le total doit valoir 79 (stress test, 191).
     Repartition par tier, par decision du panel et par type de relation.
  2. Articles qui contribuent (references distinctes dans provenance.sources),
     dans le graphe avec provenance (187) et dans le graphe nettoye (188).
  3. Meta des juges du run (modele, nombre de relations, plafond du passage, duree).
  4. Statistiques des etapes (appels, extractions, rejets) : tous les champs
     numeriques des fichiers *stats*.json, *report*.json et article_metrics.json.
  5. Jobs SLURM du 22 oct. 2026 (sacct) : nom, duree, GPU, noeud, etat.

N'ecrit que dans --outdir, refuse d'ecraser un dossier existant.

Usage (depuis le worktree run18) :
  python ~/ontogeorag/192_infos_manquantes.py --repo . --outdir output/audit/192_infos_manquantes
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import unicodedata
from collections import Counter


def norm(x):
    return " ".join(unicodedata.normalize("NFKC", str(x or "")).lower().split())


def charger_graphe(p):
    with open(p, encoding="utf-8") as f:
        g = json.load(f)
    if isinstance(g, dict):
        g = next(v for v in g.values() if isinstance(v, list))
    return g


def m4(r, cle):
    v = r.get("m4")
    if isinstance(v, dict):
        return v.get(cle)
    return r.get(f"m4.{cle}")


def signale(r):
    ev = str(m4(r, "evidence_verdict") or "")
    return "NOT_SUPPORTED" in ev.split("|")


def controle(nom, chemin, out, attendu=None):
    if not os.path.exists(chemin):
        out.append(f"[{nom}] introuvable : {chemin}")
        return
    g = charger_graphe(chemin)
    sig = [r for r in g if signale(r)]
    vides = sum(1 for r in g if not m4(r, "evidence_verdict"))
    ligne = f"[{nom}] {chemin} : {len(g)} relations, signalees {len(sig)}"
    if attendu is not None:
        ligne += f" (attendu {attendu} : {'OK' if len(sig) == attendu else 'ECART'})"
    out.append(ligne)
    out.append(f"   relations sans m4.evidence_verdict : {vides}")
    for titre, f in (("tier", lambda r: str(r.get("tier"))),
                     ("decision du panel", lambda r: str(m4(r, "decision"))),
                     ("type de relation", lambda r: str(r.get("relation")))):
        tot = Counter(f(r) for r in g)
        s = Counter(f(r) for r in sig)
        out.append(f"   par {titre} : " + ", ".join(f"{k} {s.get(k, 0)}/{n}" for k, n in tot.most_common()))
    juges = Counter()
    for r in sig:
        a, _, b = str(m4(r, "evidence_verdict")).partition("|")
        juges[("Llama" if a == "NOT_SUPPORTED" else "") + ("+" if a == b == "NOT_SUPPORTED" else "") +
              ("Mistral" if b == "NOT_SUPPORTED" else "")] += 1
    out.append("   juge qui signale : " + ", ".join(f"{k} {n}" for k, n in juges.most_common()))


def articles(nom, chemin, out):
    if not os.path.exists(chemin):
        out.append(f"[{nom}] introuvable : {chemin}")
        return
    g = charger_graphe(chemin)
    refs, sans = Counter(), 0
    for r in g:
        src = r.get("provenance", {}).get("sources") if isinstance(r.get("provenance"), dict) \
            else r.get("provenance.sources")
        if isinstance(src, str):
            try:
                src = json.loads(src)
            except Exception:
                src = None
        if not src:
            sans += 1
            continue
        for s in {norm(x.get("reference")) for x in src if isinstance(x, dict) and x.get("reference")}:
            refs[s] += 1
    out.append(f"[{nom}] {len(g)} relations ; articles distincts qui contribuent : {len(refs)} ; "
               f"relations sans source : {sans}")


def scalaires(obj, chemin="", out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            scalaires(v, f"{chemin}.{k}" if chemin else k, out)
    elif isinstance(obj, (int, float, bool)) or (isinstance(obj, str) and len(obj) < 120):
        out.append(f"{chemin} = {obj}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run", default="output/run18")
    ap.add_argument("--graphe-livre", default="output/run18/kg_final/flat_run18_enforced_dedup.json")
    ap.add_argument("--graphe-provenance", default="output/audit/187_provenance/graphe_avec_provenance.json")
    ap.add_argument("--graphe-nettoye", default="output/audit/188_nettoyage_v2/graphe_nettoye.json")
    ap.add_argument("--debut", default="2026-10-22T00:00:00")
    ap.add_argument("--fin", default="2026-10-23T12:00:00")
    ap.add_argument("--jobs", nargs="*", default=[], help="numeros de jobs SLURM (sinon : tous les jobs de la periode)")
    a = ap.parse_args()
    R = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(a.repo, x))
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} existe deja : aucune sortie n'est ecrasee.")
    os.makedirs(outdir)
    out = ["192_infos_manquantes : lecture seule."]

    out.append("\n=== 1. Controle sur le passage applique aux graphes ===")
    out.append("regle : signalee si au moins un juge rend NOT_SUPPORTED (m4.evidence_verdict, Llama|Mistral)")
    controle("graphe livre", R(a.graphe_livre), out, attendu=79)
    controle("graphe nettoye 188", R(a.graphe_nettoye), out)

    out.append("\n=== 2. Articles qui contribuent ===")
    articles("graphe avec provenance 187", R(a.graphe_provenance), out)
    articles("graphe nettoye 188", R(a.graphe_nettoye), out)

    out.append("\n=== 3. Meta des juges du run ===")
    for f in sorted(glob.glob(os.path.join(R(a.run), "m4", "*", "m4_run_meta.json"))):
        with open(f, encoding="utf-8") as h:
            m = json.load(h)
        out.append(f"[{os.path.relpath(f, a.repo)}] " + ", ".join(
            f"{k}={os.path.basename(str(v)) if k in ('model', 'input') else v}" for k, v in m.items()))

    out.append("\n=== 4. Statistiques des etapes ===")
    motifs = ["*stats*.json", "*report*.json", "*summary*.json", "article_metrics.json"]
    vus = set()
    for mo in motifs:
        for f in sorted(glob.glob(os.path.join(R(a.run), "**", mo), recursive=True)):
            if f in vus or "stress_test" in f or "checkpoint" in f:
                continue
            vus.add(f)
            try:
                with open(f, encoding="utf-8") as h:
                    obj = json.load(h)
            except Exception as e:
                out.append(f"[{os.path.relpath(f, a.repo)}] non lu : {e}")
                continue
            s = scalaires(obj)
            out.append(f"[{os.path.relpath(f, a.repo)}] {len(s)} valeurs")
            out += ["   " + x for x in s[:80]]
            if len(s) > 80:
                out.append(f"   ... {len(s) - 80} valeurs de plus, voir le fichier")

    out.append("\n=== 5. Jobs SLURM ===")
    fmt = "--format=JobID,JobName%35,Start,End,Elapsed,AllocTRES%70,NodeList%20,State"
    cmd = ["sacct", "-X", fmt]
    cmd += ["-j", ",".join(a.jobs)] if a.jobs else ["-S", a.debut, "-E", a.fin]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out.append("commande : " + " ".join(cmd))
        out += ["   " + l for l in r.stdout.splitlines()]
        if r.stderr.strip():
            out.append("   erreur : " + r.stderr.strip()[:300])
    except Exception as e:
        out.append(f"sacct impossible : {e}")
    out.append("Les etapes du manifeste vont de 2026-10-22 11:19 a 18:15 : retenir le ou les jobs qui couvrent cette periode.")

    with open(os.path.join(outdir, "_resume.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()