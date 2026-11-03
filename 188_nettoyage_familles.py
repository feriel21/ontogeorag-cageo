#!/usr/bin/env python3
"""188 — Nettoyage du graphe livre par regles fixees, puis regroupement des relations par familles.

LECTURE SEULE sur les sorties du run. CPU. Aucun modele. N'ecrit que dans --outdir
(refuse d'ecraser). Le graphe livre n'est pas modifie : le graphe nettoye est un NOUVEAU
fichier, et chaque relation retiree est listee avec sa regle.

ENTREES
  --graph     graphe_avec_provenance.json (sortie de 187 : graphe livre + sources localisees)
  --run       dossier du run (pass_a|b/verified_triples.jsonl) pour relire les phrases des
              autres extractions qui soutiennent une relation (optionnel)
  --familles  table entite -> famille (ex. configs/entity_families_run18.csv, colonnes entity,family)
  --reference configs/lb_reference_edges_original26.json (comparaison avant / apres seulement)

REGLES (fixees dans ce fichier ; chaque relation retiree porte la ou les regles)
  Une "phrase source" est une phrase de preuve localisee dans un article par 187 : la phrase
  affichee et celles des autres extractions qui soutiennent la relation.
  R1 bibliographie   TOUTES les phrases sources sont des entrees de bibliographie : section
                     deduite "references" (187) ou forme de reference (auteur avec initiales,
                     annee, puis volume "v." / "vol." / pages "p." "pp." ou "nn–nn").
  R2 etiquette       le sujet ou l'objet est une etiquette locale propre a un article, pas une
                     classe geologique (remarque de John sur "am1" et "unit a") :
                     code lettres+chiffres (am1, pm2, s1-s5, n1), "unit|facies|zone|... <lettre
                     ou numero>", "<classe> <numero ou lettre>" (mass transport deposit 5).
  R3 sujet absent    AUCUNE phrase source ne nomme le sujet : ni le nom, ni son mot principal
                     (dernier mot, singulier), ni un synonyme de sa famille. La relation n'est
                     alors pas soutenue par le texte qu'elle cite.
  Signales sans retrait (decision a prendre) :
  S1 lieu nomme      sujet ou objet contient un nom de lieu ou de formation (liste LIEUX
                     ci-dessous, etablie en lisant les 272 entites de run18 : A VALIDER).
                     Option --retirer-lieux pour les retirer.
  S2 commentaire     la phrase affichee commence par un commentaire du modele
                     ("the evidence for this claim...") au lieu d'une citation.
  S3 legende         la phrase affichee vient probablement d'une legende de figure (187).
  S4 pronom          sujet absent mais un pronom (it, its, they, this...) figure dans les trois
                     premiers mots d'une phrase source : le sujet est peut-etre nomme dans la
                     phrase precedente.

FAMILLES
  Chaque entite du graphe nettoye recoit sa famille (table --familles, apres minuscules et
  singulier simple) ; une entite absente de la table garde son nom (famille "hors table").
  Les relations sont regroupees par (famille du sujet, relation, famille de l'objet) :
  nombre de relations, de Tier 1, d'articles distincts, membres.

SORTIES
  graphe_nettoye.json, relations_retirees.tsv, relations_signalees.tsv,
  relations_par_famille.tsv, entites_hors_table.tsv, _resume.txt

USAGE (racine du worktree run18)
  python 188_nettoyage_familles.py --graph output/audit/187_provenance/graphe_avec_provenance.json \
      --run output/run18 --familles configs/entity_families_run18.csv \
      --reference configs/lb_reference_edges_original26.json --outdir output/audit/188_nettoyage
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# --- R2 : etiquettes locales -------------------------------------------------------------
CODE = re.compile(r"(?<![a-z0-9])[a-z]{1,4}[-_]?\d+[a-z]?(?:\s*[-–]\s*[a-z]{0,4}\d+)?(?![a-z0-9])")
LABEL_WORDS = r"unit|units|facies|zone|zones|interval|sequence|horizon|area|site|line|profile|well|core|block|blocks|" \
              r"basin|domain|lobe|package|subunit|mtc|mtcs|mtd|mtds|deposit|deposits|megaslide|megaslides|slide"
LABELLED = re.compile(r"(?<![a-z])(?:" + LABEL_WORDS + r")\s+(?:[a-h]|i{1,3}[ab]?|ia|ib|\d+[a-z]?)(?:\s+and\s+(?:[a-h]|\d+))?$")
NOT_LABEL = {"3d", "2d", "c14", "co2", "ch4"}

# --- S1 : lieux et formations nommes (lecture des entites de run18, a valider) -----------------
LIEUX = ["aleutian", "amapá", "amapa", "amazon", "bear island", "bering", "brazos", "cascadia", "ebro", "frio",
         "mississippi", "pakistan", "pearl river", "qiongdongnan", "wilkes land", "storegga", "lubina", "makran",
         "nankai", "gulf of mexico", "south china sea", "mediterranean", "niger"]
LIEU = re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(x) for x in LIEUX) + r")(?![a-z])")

# --- R1 : forme de reference bibliographique ------------------------------------------------
AUTHOR = re.compile(r"[a-zà-ÿ'\-]+,?\s+(?:[a-z]\.\s?){1,3}")
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}[a-z]?(?!\d)")
BIBMARK = re.compile(r"(?<![a-z])(?:v\.\s?\d+|vol\.\s?\d+|pp?\.\s?\d+|\d+\s?[–\-]\s?\d+\.\s*\"?$|"
                     r"doi[:\s]|bull\.|geol\.|j\.\s?geophys|mar\.\s?geol)")

ANAPHORE = re.compile(r"^\W*(?:[a-z,\-]+\s+){0,2}(?:it|its|they|their|these|this|such)(?![a-z])", re.I)
COMMENT = re.compile(r"^\W*(?:the evidence|the source text|the passage|the text|evidence for this claim|"
                     r"the following sentence)", re.I)

MTD_SYN = ["mass transport deposit", "mass-transport deposit", "mtd", "mass transport complex",
           "mass-transport complex", "mtc", "mass-wasting deposit", "mass-failed deposit", "mass failed deposit"]
EXTRA_SYN = {"seabed": ["seafloor", "sea floor", "sea-floor"], "seafloor": ["seabed", "sea floor"],
             "earthquake": ["seismic", "seismicity"], "debrite": ["debris flow", "debris-flow"],
             "debris flow": ["debrite", "debris-flow"]}


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def low(x):
    return " ".join(unicodedata.normalize("NFKC", str(x or "")).lower().replace("ﬁ", "fi").split())


def flat(x):
    return re.sub(r"[^a-z0-9]", "", low(x))


def sing(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "xes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "is", "us")) and len(w) > 3:
        return w[:-1]
    return w


def ent_key(e):
    return " ".join(sing(w) for w in low(e).split())


def is_label(e):
    e = low(e)
    codes = [m.group(0) for m in CODE.finditer(e) if m.group(0) not in NOT_LABEL]
    if codes:
        return "code " + ",".join(codes)
    if LABELLED.search(e):
        return "etiquette " + LABELLED.search(e).group(0)
    return ""


def is_bib(text, section):
    if section == "references":
        return True
    t = low(text)
    return bool(AUTHOR.search(t) and YEAR.search(t) and BIBMARK.search(t))


PROCESS_WORDS = {"development", "generation", "initiation", "formation", "propagation", "build-up", "buildup",
                 "loading", "change", "changes", "processes", "process", "features", "feature", "elements"}


def names(entity, fam_of, members):
    """Formes acceptees pour dire qu'une phrase nomme l'entite."""
    e = low(entity)
    out = {e, ent_key(e)}
    words = [w for w in re.split(r"[\s\-]+", e) if w]
    if words:
        out.add(sing(words[-1]))                       # mot principal
    m_of = re.match(r"^(.*?)\s+of\s+", e)
    if m_of:                                           # "block of undisturbed reflections" -> "block"
        out.add(sing(m_of.group(1).split()[-1]))
    core = [w for w in words if w not in PROCESS_WORDS]
    if len(core) >= 2:
        out.update(" ".join(core[i:i + 2]) for i in range(len(core) - 1))   # sous-groupes de 2 mots
    if core and len(core) < len(words):
        out.add(sing(core[-1]))                        # "pore pressure development" -> "pressure"
    k = ent_key(e)
    if k in ("mass transport deposit", "megaslide", "slide", "landslide", "submarine landslide") or \
            "masstransport" in flat(k):
        out.update(MTD_SYN)
    for base, syn in EXTRA_SYN.items():
        if base in k:
            out.update(syn)
    fams = {fam_of[k]} if k in fam_of else set()
    for te, tf in fam_of.items():                     # entite de la table contenue dans le nom
        if len(te) >= 6 and re.search(r"(?<![a-z])" + re.escape(te) + r"(?![a-z])", k):
            fams.add(tf)
    for fam in fams:
        for m in members.get(fam, []):
            out.add(m)
            mw = m.split()
            if mw:
                out.add(sing(mw[-1]))
    return {x for x in out if len(flat(x)) >= 3}


def mentioned(entity, texts, fam_of, members):
    forms = names(entity, fam_of, members)
    for t in texts:
        ft = flat(t) + "|" + flat(" ".join(sing(w) for w in re.split(r"[\s\-]+", low(t))))
        for f in forms:
            if flat(f) in ft or flat(sing(f)) in ft:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--familles", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--retirer-lieux", action="store_true")
    args = ap.parse_args()

    out = Path(args.outdir)
    if out.exists():
        die("le dossier existe deja, rien n'est ecrase : " + str(out))
    raw = json.load(open(args.graph, encoding="utf-8"))
    G = raw["triples"] if isinstance(raw, dict) else raw
    if not G or "provenance" not in G[0]:
        die("le graphe n'a pas de champ provenance : donner la sortie de 187 (graphe_avec_provenance.json)")

    fam_of, members = {}, defaultdict(list)
    with open(args.familles, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            e, fa = row.get("entity"), row.get("family")
            if e and fa and e != "entity":
                fam_of[ent_key(e)] = fa.strip()
                members[fa.strip()].append(low(e))

    ver = {}
    if args.run:
        for P in ("a", "b"):
            p = Path(args.run) / ("pass_" + P) / "verified_triples.jsonl"
            if p.exists():
                ver[P] = jl(p)

    full = {}                                          # 06 coupe la phrase affichee a 400 caracteres
    for P, V in ver.items():
        for v in V:
            ev = (v.get("_verification") or {}).get("evidence", "") or ""
            if len(ev) > 400:
                full.setdefault(ev[:400], ev)

    def source_texts(t):
        res = []
        for s in t["provenance"]["sources"]:
            o = s.get("origine", "")
            if o == "phrase affichee":
                ev = t.get("evidence") or ""
                res.append((full.get(ev, ev), s.get("section_deduite", "")))
            elif o.startswith("extraction ") and ver:
                P, li = o.split(" ", 1)[1].split(":")
                if P in ver and int(li) < len(ver[P]):
                    res.append(((ver[P][int(li)].get("_verification") or {}).get("evidence", ""),
                                s.get("section_deduite", "")))
        return res

    kept, removed, flagged = [], [], []
    for t in G:
        s, r, o = t.get("subject", ""), t.get("relation", ""), t.get("object", "")
        src = source_texts(t)
        texts = [x for x, _ in src if x]
        rules = []
        if src and all(is_bib(x, sec) for x, sec in src):
            rules.append("R1 bibliographie")
        lab = [x for x in (is_label(s) and "sujet " + is_label(s), is_label(o) and "objet " + is_label(o)) if x]
        if lab:
            rules.append("R2 etiquette (" + "; ".join(lab) + ")")
        flags = []
        if texts and not mentioned(s, texts, fam_of, members):
            if any(ANAPHORE.search(x) for x in texts):
                flags.append("S4 sujet repris par un pronom (it, they, this...)")
            else:
                rules.append("R3 sujet absent des phrases sources")
        lieu = [m.group(0) for e in (s, o) for m in LIEU.finditer(low(e))]
        if lieu:
            flags.append("S1 lieu nomme (" + ",".join(sorted(set(lieu))) + ")")
            if args.retirer_lieux:
                rules.append("S1 lieu nomme (retire par option)")
        if COMMENT.search(str(t.get("evidence") or "")):
            flags.append("S2 commentaire du modele")
        if any(x.get("legende_figure_probable") == "oui" for x in t["provenance"]["sources"]
               if x.get("origine") == "phrase affichee"):
            flags.append("S3 legende de figure probable")
        base = {"sujet": s, "relation": r, "objet": o, "tier": t.get("tier"),
                "decision_panel": (t.get("m4") or {}).get("decision", ""),
                "n_phrases_sources": len(src), "n_articles": t["provenance"].get("n_articles", 0),
                "articles": " ; ".join(sorted({x.get("reference", "")[:60] for x in t["provenance"]["sources"]})),
                "preuve": t.get("evidence", "")}
        if flags:
            flagged.append({**base, "signalements": " | ".join(flags), "retiree": "oui" if rules else ""})
        if rules:
            removed.append({**base, "regles": " | ".join(rules)})
        else:
            kept.append(t)

    # familles
    def fam(e):
        return fam_of.get(ent_key(e))
    out_of_table = Counter()
    groups = defaultdict(list)
    for t in kept:
        fs, fo = fam(t["subject"]), fam(t["object"])
        if not fs:
            out_of_table[t["subject"]] += 1
        if not fo and t["relation"] != "hasDescriptor":
            out_of_table[t["object"]] += 1
        t2 = dict(t)
        t2["famille_sujet"] = fs or t["subject"]
        t2["famille_objet"] = fo or t["object"]
        groups[(t2["famille_sujet"], t["relation"], t2["famille_objet"])].append(t2)
    fam_rows = []
    for (fs, r, fo), L in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        arts = {x.get("doc_id") for t in L for x in t["provenance"]["sources"]}
        fam_rows.append({"famille_sujet": fs, "relation": r, "famille_objet": fo, "n_relations": len(L),
                         "n_tier1": sum(t.get("tier") == 1 for t in L), "n_articles": len(arts),
                         "membres": " ; ".join(sorted("%s -> %s" % (t["subject"], t["object"]) for t in L))})

    # comparaison LB2019 avant / apres (lecture seule)
    ref_before = ref_after = None
    if args.reference:
        sys.path.insert(0, str(Path(".").resolve()))
        try:
            from pipeline.rag.constants import normalize_entity, normalize_relation
        except Exception:
            def normalize_entity(x):
                return low(x)

            def normalize_relation(x):
                return str(x)
        E = {(normalize_entity(e["subject"]), normalize_relation(e["relation"]), normalize_entity(e["object"]))
             for e in json.load(open(args.reference, encoding="utf-8"))["edges"]}

        def hits(L):
            return sorted("%s|%s|%s" % k for k in E if k in {(normalize_entity(t["subject"]),
                                                               normalize_relation(t["relation"]),
                                                               normalize_entity(t["object"])) for t in L})
        ref_before, ref_after = hits(G), hits(kept)

    out.mkdir(parents=True)
    clean = dict(raw) if isinstance(raw, dict) else {}
    clean["triples"] = [dict(t, famille_sujet=fam(t["subject"]) or t["subject"],
                             famille_objet=fam(t["object"]) or t["object"]) for t in kept]
    clean["meta_188"] = {"source": args.graph, "source_sha256": sha(args.graph), "regles": ["R1", "R2", "R3"] +
                         (["S1"] if args.retirer_lieux else []), "n_avant": len(G), "n_apres": len(kept)}
    json.dump(clean, open(out / "graphe_nettoye.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    def tsv(p, recs, cols=None):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols or (list(recs[0]) if recs else ["vide"]), delimiter="\t")
            w.writeheader()
            w.writerows(recs)
    tsv(out / "relations_retirees.tsv", removed)
    tsv(out / "relations_signalees.tsv", flagged)
    tsv(out / "relations_par_famille.tsv", fam_rows)
    tsv(out / "entites_hors_table.tsv", [{"entite": e, "n_relations": n} for e, n in out_of_table.most_common()],
        ["entite", "n_relations"])

    rule_c = Counter(x.split(" (")[0] for r in removed for x in r["regles"].split(" | "))
    by_tier = Counter((r["tier"], r["decision_panel"]) for r in removed)
    ent_before = {e for t in G for e in (t["subject"], t["object"])}
    ent_after = {e for t in kept for e in (t["subject"], t["object"])}
    L = ["=== 188 nettoyage par regles et familles ===",
         "graphe : %s (sha %s) ; familles : %s (sha %s, %d entites, %d familles)" % (
             args.graph, sha(args.graph)[:12], args.familles, sha(args.familles)[:12], len(fam_of), len(members)),
         "phrases des autres extractions relues : %s ; phrases affichees retrouvees en entier (coupees a 400) : %d" % (
             "oui" if ver else "non (phrase affichee seulement)",
             sum(1 for t in G if (t.get("evidence") or "") in full)),
         "", "-- retraits --",
         "relations : %d -> %d (retirees %d)" % (len(G), len(kept), len(removed)),
         "Tier 1 : %d -> %d" % (sum(t.get("tier") == 1 for t in G), sum(t.get("tier") == 1 for t in kept)),
         "entites : %d -> %d" % (len(ent_before), len(ent_after)),
         "par regle (une relation peut en avoir plusieurs) : %s" % dict(rule_c),
         "retirees par tier et decision du panel : %s" % dict(sorted(by_tier.items(), key=str)),
         "", "-- signalements (non retires sauf option) --",
         "%s" % dict(Counter(x.split(" (")[0] for r in flagged for x in r["signalements"].split(" | "))),
         "", "-- familles (graphe nettoye) --",
         "relations %d -> groupes (famille, relation, famille) %d" % (len(kept), len(fam_rows)),
         "groupes avec au moins 2 relations : %d (%d relations)" % (
             sum(1 for g in fam_rows if g["n_relations"] >= 2), sum(g["n_relations"] for g in fam_rows if g["n_relations"] >= 2)),
         "entites hors table (sujets, et objets hors hasDescriptor) : %d" % len(out_of_table),
         "", "-- groupes les plus fournis --"]
    for g in fam_rows[:15]:
        L.append("  %3d rel. (%d T1, %d art.)  %s --%s--> %s" % (g["n_relations"], g["n_tier1"], g["n_articles"],
                                                               g["famille_sujet"], g["relation"], g["famille_objet"]))
    if ref_before is not None:
        L += ["", "-- comparaison LB2019 (26 aretes, correspondance exacte) --",
              "avant : %d/26 ; apres : %d/26 ; perdues : %s" % (len(ref_before), len(ref_after),
                                                                 sorted(set(ref_before) - set(ref_after)))]
    L += ["", "R3 et S1 reposent sur des listes ecrites dans ce fichier : relire relations_retirees.tsv "
          "avant de citer un chiffre."]
    txt = "\n".join(L)
    (out / "_resume.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt + "\n[fait] " + str(out), flush=True)


if __name__ == "__main__":
    main()