#!/usr/bin/env python3
"""191_regles_et_sensibilite.py : lecture seule, aucun modele, aucun run.

1. Joint controls.jsonl et les verdicts Llama et Mistral par m4_index, en verifiant
   pour chaque ligne que sujet, relation et objet sont identiques (sinon : arret).
2. Recalcule des regles explicites et indique lesquelles reproduisent les chiffres
   de negatives_report.json et regles_decision.json (verification, pas hypothese).
3. Taux par type de perturbation avec intervalles de Wilson a 95 %.
4. Sensibilite : memes taux sans les cas ambigus (entite introduite presente dans le
   passage, relation de la meme famille, boucle sujet = objet).
5. Originaux : verdicts du stress test compares a ceux du graphe livre (jointure par
   cle de contenu), et recouvrement des originaux ecartes avec les UNCERTAIN.

Usage (depuis le worktree run18) :
  python ~/ontogeorag/191_regles_et_sensibilite.py --repo . \
      --outdir output/audit/191_regles_et_sensibilite
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

FAMILLES = {"causal": {"causes", "triggers", "controls", "affects", "formedby"},
            "stratigraphique": {"overlies", "underlies"},
            "spatial": {"occursin", "partof"}}
CLASSES = ["entity_sub", "passage_perm", "relation_sub", "inversion"]


def norm(x):
    s = unicodedata.normalize("NFKC", str(x or "")).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def nrel(x):
    return norm(x).replace(" ", "")


def fam(r):
    r = nrel(r)
    for f, m in FAMILLES.items():
        if r in m:
            return f
    return r


def taux(k, n):
    return k / n if n else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def g(d, chemin):
    for k in chemin.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stress", default="output/run18/stress_test")
    ap.add_argument("--graphe", default="output/run18/kg_final/flat_run18_enforced_dedup.json")
    a = ap.parse_args()
    R = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(a.repo, x))
    outdir = R(a.outdir)
    if os.path.exists(outdir):
        sys.exit(f"{outdir} existe deja : aucune sortie n'est ecrasee.")
    os.makedirs(outdir)
    out = ["191_regles_et_sensibilite : lecture seule."]

    st = R(a.stress)
    ctrl = jsonl(os.path.join(st, "controls.jsonl"))
    juges = {}
    for j in ("llama", "mistral"):
        v = jsonl(os.path.join(st, j, "m4_verdicts.jsonl"))
        par = {}
        for r in v:
            i = r.get("m4_index")
            if i in par:
                sys.exit(f"m4_index {i} en double dans {j} : arret.")
            par[i] = r
        juges[j] = par
    # --- 1. jointure verifiee
    lignes = []
    for i, c in enumerate(ctrl):
        rec = {"i": i, "c": c}
        for j, par in juges.items():
            r = par.get(i)
            if r is None:
                sys.exit(f"{j} : m4_index {i} absent : arret.")
            if (norm(r["subject"]), nrel(r["relation"]), norm(r["object"])) != \
               (norm(c["subject"]), nrel(c["relation"]), norm(c["object"])):
                sys.exit(f"{j} : m4_index {i} ne correspond pas au controle {i} : arret.")
            rec[j] = r
        lignes.append(rec)
    out.append(f"jointure par m4_index verifiee sur le contenu : {len(lignes)}/{len(ctrl)} lignes, "
               f"0 discordance, pour llama et mistral")

    # --- 2. regles explicites
    def ev(rec, j):
        return str(g(rec[j], "evidence.verdict"))

    def bl(rec, j):
        return str(g(rec[j], "blind.verdict"))

    regles = {
        "phrase NOT_SUPPORTED llama": lambda r: ev(r, "llama") == "NOT_SUPPORTED",
        "phrase NOT_SUPPORTED mistral": lambda r: ev(r, "mistral") == "NOT_SUPPORTED",
        "phrase NOT_SUPPORTED, au moins un juge": lambda r: "NOT_SUPPORTED" in (ev(r, "llama"), ev(r, "mistral")),
        "phrase NOT_SUPPORTED, les deux juges": lambda r: ev(r, "llama") == ev(r, "mistral") == "NOT_SUPPORTED",
        "phrase NOT ou aveugle IMPLAUSIBLE, llama": lambda r: ev(r, "llama") == "NOT_SUPPORTED" or bl(r, "llama") == "IMPLAUSIBLE",
        "phrase NOT ou aveugle IMPLAUSIBLE, mistral": lambda r: ev(r, "mistral") == "NOT_SUPPORTED" or bl(r, "mistral") == "IMPLAUSIBLE",
        "phrase NOT ou aveugle IMPLAUSIBLE, au moins un juge":
            lambda r: any(ev(r, j) == "NOT_SUPPORTED" or bl(r, j) == "IMPLAUSIBLE" for j in ("llama", "mistral")),
        "phrase non SUPPORTED (NOT ou PARTIALLY), les deux juges":
            lambda r: ev(r, "llama") != "SUPPORTED" and ev(r, "mistral") != "SUPPORTED",
        "aveugle IMPLAUSIBLE, au moins un juge": lambda r: "IMPLAUSIBLE" in (bl(r, "llama"), bl(r, "mistral")),
    }
    # chiffres publies a reproduire
    publies = {}
    for j in ("llama", "mistral"):
        try:
            with open(os.path.join(st, j, "negatives_report.json"), encoding="utf-8") as f:
                nr = json.load(f)
            publies[f"negatives_report {j}"] = (nr["overall"]["tp"], nr["overall"]["fp"])
        except Exception as e:
            out.append(f"negatives_report {j} non lu : {e}")
    try:
        with open(os.path.join(st, "regles_decision.json"), encoding="utf-8") as f:
            rd = json.load(f)
        for nom, bloc in rd.items():
            if isinstance(bloc, dict) and "corrupted_flagged" in bloc:
                publies[f"regles_decision {nom}"] = (bloc["corrupted_flagged"], bloc["originals_flagged"])
    except Exception as e:
        out.append(f"regles_decision.json non lu : {e}")
    txt = os.path.join(st, "regles_decision.txt")
    if os.path.exists(txt):
        out.append(f"\n--- contenu de {txt}")
        with open(txt, encoding="utf-8", errors="replace") as f:
            out += [l.rstrip("\n") for l in f]

    pert = [r for r in lignes if r["c"]["label"] != "original"]
    orig = [r for r in lignes if r["c"]["label"] == "original"]
    out.append(f"\n=== 2. Regles recalculees (perturbes {len(pert)}, originaux {len(orig)}) ===")
    resultats = {}
    for nom, f in regles.items():
        k = sum(1 for r in pert if f(r))
        m = sum(1 for r in orig if f(r))
        resultats[nom] = (k, m)
        egal = [p for p, v in publies.items() if v == (k, m)]
        out.append(f"   {nom:<55} perturbes ecartes {k:>3}/{len(pert)}, originaux ecartes {m:>3}/{len(orig)}"
                   + (f"   = {', '.join(egal)}" if egal else ""))
    out.append("   chiffres publies non reproduits par ces regles : " +
               (", ".join(f"{p} {v}" for p, v in publies.items() if v not in resultats.values()) or "aucun"))

    # --- 3 et 4. taux par type, avec et sans cas ambigus
    origs_par_idx = {str(r["c"]["orig_index"]): r["c"] for r in orig}

    def ambigu(r):
        c = r["c"]
        o = origs_par_idx.get(str(c["orig_index"]))
        s, ob, p = norm(c["subject"]), norm(c["object"]), nrel(c["relation"])
        if s == ob:
            return "boucle sujet = objet"
        if o is None:
            return ""
        passage = norm(c.get("supporting_passage"))
        cl = c["corruption_class"]
        if cl == "entity_sub":
            nouv = [e for e, e0 in ((s, norm(o["subject"])), (ob, norm(o["object"]))) if e != e0]
            if nouv and all(e in passage for e in nouv):
                return "entite introduite presente dans le passage"
        if cl == "relation_sub" and fam(p) == fam(o["relation"]) and p != nrel(o["relation"]):
            return "relation de la meme famille"
        if cl == "passage_perm" and s in passage and ob in passage:
            return "sujet et objet dans le nouveau passage"
        return ""

    amb = Counter()
    lignes_csv = []
    for r in pert:
        r["amb"] = ambigu(r)
        if r["amb"]:
            amb[(r["c"]["corruption_class"], r["amb"])] += 1
    out.append("\n=== 3. Cas ambigus retenus pour la sensibilite ===")
    for (cl, m), n in sorted(amb.items()):
        out.append(f"   {cl:<14} {m} : {n}")

    for titre, garder in (("tous les perturbes", lambda r: True),
                          ("sans les cas ambigus", lambda r: not r["amb"])):
        out.append(f"\n=== 4. Taux par type, {titre} (Wilson 95 %) ===")
        for nom, f in regles.items():
            out.append(f"   -- {nom}")
            for cl in CLASSES:
                gr = [r for r in pert if r["c"]["corruption_class"] == cl and garder(r)]
                k = sum(1 for r in gr if f(r))
                lo, hi = wilson(k, len(gr))
                out.append(f"      {cl:<14} rejet {k:>3}/{len(gr):<3} = {taux(k, len(gr)):.3f} [{lo:.3f} ; {hi:.3f}]")
                lignes_csv.append([titre, nom, cl, len(gr), k, round(taux(k, len(gr)), 4), round(lo, 4), round(hi, 4)])
            gr = [r for r in pert if garder(r)]
            k = sum(1 for r in gr if f(r))
            lo, hi = wilson(k, len(gr))
            out.append(f"      {'tous':<14} rejet {k:>3}/{len(gr):<3} = {taux(k, len(gr)):.3f} [{lo:.3f} ; {hi:.3f}]")
            m = sum(1 for r in orig if not f(r))
            lo2, hi2 = wilson(m, len(orig))
            out.append(f"      {'originaux':<14} conservation {m}/{len(orig)} = {taux(m, len(orig)):.3f} [{lo2:.3f} ; {hi2:.3f}]")
            lignes_csv.append([titre, nom, "tous", len(gr), k, round(taux(k, len(gr)), 4), round(lo, 4), round(hi, 4)])
            lignes_csv.append([titre, nom, "originaux_conserves", len(orig), m, round(taux(m, len(orig)), 4),
                               round(lo2, 4), round(hi2, 4)])

    # --- 5. originaux contre graphe livre
    out.append("\n=== 5. Originaux du stress test et graphe livre ===")
    with open(R(a.graphe), encoding="utf-8") as f:
        graphe = json.load(f)
    if isinstance(graphe, dict):
        graphe = next(v for v in graphe.values() if isinstance(v, list))
    idx = defaultdict(list)
    for r in graphe:
        idx[(norm(r["subject"]), nrel(r["relation"]), norm(r["object"]))].append(r)
    concord = Counter()
    croise = defaultdict(Counter)
    sans = 0
    for r in orig:
        c = r["c"]
        m = idx.get((norm(c["subject"]), nrel(c["relation"]), norm(c["object"])), [])
        if len(m) != 1:
            sans += 1
            continue
        gv = str(g(m[0], "m4.evidence_verdict") or "")
        sv = f"{ev(r, 'llama')}|{ev(r, 'mistral')}"
        concord["identique" if gv == sv else "different"] += 1
        dec = str(g(m[0], "m4.decision"))
        for nom, f in regles.items():
            croise[nom][(dec, "ecarte" if f(r) else "garde")] += 1
    out.append(f"   originaux joints au graphe par cle de contenu : {len(orig) - sans}/{len(orig)} "
               f"({sans} sans correspondance unique)")
    out.append(f"   verdicts sur la phrase (llama|mistral), stress test contre graphe livre : {dict(concord)}")
    out.append("   decision du panel dans le graphe x regle sur l'original :")
    for nom, t in croise.items():
        out.append(f"      {nom:<55} " + ", ".join(f"{d} {e} {n}" for (d, e), n in sorted(t.items())))

    with open(os.path.join(outdir, "taux_par_type.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ensemble", "regle", "classe", "n", "k", "taux", "wilson_bas", "wilson_haut"])
        w.writerows(lignes_csv)
    with open(os.path.join(outdir, "cas_ambigus.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["m4_index", "classe", "motif", "sujet", "relation", "objet", "llama_phrase",
                    "mistral_phrase", "passage"])
        for r in pert:
            if r["amb"]:
                c = r["c"]
                w.writerow([r["i"], c["corruption_class"], r["amb"], c["subject"], c["relation"], c["object"],
                            ev(r, "llama"), ev(r, "mistral"), str(c.get("supporting_passage"))[:300]])
    with open(os.path.join(outdir, "_resume.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()