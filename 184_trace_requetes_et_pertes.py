#!/usr/bin/env python3
"""184 — Requetes suspectes, suivi des pertes et provenance des preuves de run18.

LECTURE SEULE. CPU. Aucun modele. N'ecrit que dans --outdir (refuse d'ecraser).

CE QUE FAIT LE SCRIPT
  A. Requetes suspectes
     Recalcule, dans configs/descriptor_queries_run18.jsonl, les requetes dont le
     texte et l'ancre (focus), lus ensemble, nomment les deux arguments d'une
     arete de configs/lb_reference_edges_original26.json. Formes reconnues :
     terme exact, pluriel (+s, +es), tiret <-> espace, et pour
     "mass transport deposit" : mtd, mtds, mass-transport deposit(s).
     Le script s'arrete si ce n'est pas exactement les 16 requetes attendues.
     La classe de chaque requete (vocabulaire seul / relation suggeree) est une
     lecture humaine ecrite dans CLASSES ci-dessous, avec le motif qui la fonde.

  B. Suivi des pertes, triplet par triplet (passes a et b)
     Chaque ligne de pass_X/verified_triples.jsonl est rejouee, seule, a travers
     les fonctions de pipeline/04_clean_validate_run18.py (filtre de verdict,
     relation, regles de base, type, lexique), puis la table canonical_map_v5.json,
     le journal de 05a, la regle de 04c et la cle de 06. Les doublons sont suivis
     jusqu'a la relation qui les absorbe.
     Controles : les comptes rejoues doivent egaler cleaning_stats_v5.json,
     rejected_triples_v5.jsonl, canonical_triples_v5.jsonl, les fichiers de 04c
     et tiered_kg_run18.json. Chaque controle est affiche OK ou ECHEC.

     Objets ambigus : 04 remplace un descripteur hors vocabulaire par le premier
     descripteur d'un ensemble Python qui le contient ; l'ordre depend de
     PYTHONHASHSEED. Le choix reel du run est lu dans canonical_triples_v5.jsonl et
     le rejeu suit cet ordre (normalisations_ambigues.tsv).

  C. Devenir apres la fusion
     Pour chaque relation de tiered_kg_run18.json : decision du panel,
     quarantaine, presence et tier dans le graphe livre.

  D. Provenance
     La phrase de preuve de chaque extraction (_verification.evidence) est
     cherchee dans ses 5 passages, avec les regles de
     scripts/evaluation/graph_with_located_provenance.py (comparaison en
     minuscules alphanumeriques ; exact ; partial = une des trois fenetres de
     40 caracteres ; not_found). La phrase affichee dans le graphe livre est
     cherchee dans tout l'index.

  Une relation est "aussi obtenue par une requete non suspecte" si au moins une
  extraction d'une requete non suspecte arrive jusqu'a elle (gardee ou absorbee
  comme doublon) avec un verdict Qwen STRONG ou WEAK.

  E. Analyse de sensibilite (pas un nouveau run)
     Graphe livre moins les relations obtenues UNIQUEMENT par des requetes
     de classe R-* (relation suggeree) ; variante sans les 16. Correspondances exactes avec les 26 aretes avant et apres.

USAGE (racine du worktree run18, venv actif)
  cd ~/ontogeorag_run18 && export PYTHONPATH=$PWD
  python 184_trace_requetes_et_pertes.py --repo . --run output/run18 \
      --outdir output/audit/184_trace
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REF26 = "configs/lb_reference_edges_original26.json"
QUERIES = "configs/descriptor_queries_run18.jsonl"
QUERIES_SHA12 = "3dacffd9b7f7"
CLEAN = "pipeline/04_clean_validate_run18.py"
CLEAN_SHA12 = "e5a8db171df7"
WINDOW = 40

# Lecture humaine des 16 requetes : index (0-based) dans QUERIES -> (classe, motif).
#   R-reponse     : le sujet, la relation et l'objet attendu sont ecrits dans la question
#                   (question fermee, ou liste de reponses apres un tiret / un point d'interrogation)
#   R-presuppose  : la question presuppose la relation de reference, sans lister de reponse
#   R-candidat    : le sujet est propose comme un candidat parmi d'autres pour l'objet
#   V-vocabulaire : les deux termes apparaissent mais la question ne les relie pas
CLASSES = {
    60: ("R-reponse", "question fermee : Are MTDs characterized by <descripteur>"),
    61: ("R-reponse", "question fermee : Are MTDs characterized by <descripteur>"),
    62: ("R-reponse", "question fermee : Are MTDs characterized by <descripteur>"),
    63: ("R-reponse", "question fermee : Are MTDs characterized by <descripteur>"),
    64: ("R-reponse", "question fermee : Are MTDs characterized by <descripteur>"),
    65: ("R-reponse", "question fermee : Are MTDs characterized by <descripteur>"),
    101: ("R-presuppose", "transition des debris flows vers les turbidity currents presupposee"),
    160: ("R-candidat", "MTDs proposes parmi trois candidats pour 'massive'"),
    161: ("R-presuppose", "presuppose que des MTDs montrent un caractere massive"),
    162: ("V-vocabulaire", "'undeformed' qualifie 'sediments', pas 'slides'"),
    166: ("R-reponse", "cadres attendus listes apres le tiret"),
    167: ("R-reponse", "descripteurs attendus listes apres la question"),
    168: ("R-reponse", "descripteurs attendus listes apres le tiret"),
    169: ("R-reponse", "descripteurs attendus listes apres le tiret"),
    170: ("R-presuppose", "la question presuppose que slope failure produit des MTDs"),
    171: ("R-reponse", "cadres attendus listes apres le tiret"),
}


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------- A. requetes suspectes ------------------------------------------
def forms(term):
    t = term.lower()
    out = set()
    for b in {t, t.replace("-", " "), t.replace(" ", "-")}:
        out |= {b, b + "s", b + "es"}
    if t == "mass transport deposit":
        out |= {"mtd", "mtds", "mass-transport deposit", "mass-transport deposits"}
    return out


def hit(term, text):
    for f in sorted(forms(term), key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(f) + r"(?![a-z])", text):
            return f
    return None


def suspect_queries(Q, edges):
    res = defaultdict(list)
    for i, q in enumerate(Q):
        txt = (str(q.get("query", "")) + " || " + str(q.get("focus") or "")).lower().replace("_", " ")
        for s, r, o in edges:
            hs, ho = hit(s, txt), hit(o, txt)
            if hs and ho and hs != ho:
                res[i].append({"edge": (s, r, o), "vu_sujet": hs, "vu_objet": ho})
    return res


# ---------- D. localisation (regles de 145) --------------------------------
NFKC = False   # --nfkc : ligatures (ﬁ, ﬂ...) et formes de compatibilite ramenees a l'ASCII avant comparaison


def alnum(x):
    x = unicodedata.normalize("NFKC", str(x)) if NFKC else str(x)
    return re.sub(r"[^a-z0-9]", "", x.lower())


def locate(ev, chunk_alnum):
    e = alnum(ev)
    exact = [i for i, c in enumerate(chunk_alnum) if e and e in c]
    if exact:
        return "exact", exact
    if len(e) <= WINDOW:
        return ("vide" if not e else "not_found"), []
    mid = (len(e) - WINDOW) // 2
    wins = [e[:WINDOW], e[mid:mid + WINDOW], e[-WINDOW:]]
    partial = [i for i, c in enumerate(chunk_alnum) if any(w in c for w in wins)]
    return ("partial", partial) if partial else ("not_found", [])


def descriptor_order(M, ver, can, cmap):
    """Ordre des descripteurs qui reproduit les choix de 04 sur les objets ambigus.

    check_type_constraint (04) remplace un objet hors vocabulaire par le PREMIER
    descripteur de KNOWN_DESCRIPTORS qui le contient ou qu'il contient. KNOWN_DESCRIPTORS
    est un ensemble Python : son ordre de parcours depend de PYTHONHASHSEED, tire au
    hasard a chaque lancement. Quand plusieurs descripteurs conviennent (ex.
    "transparent to chaotic facies"), le choix de 04 n'est donc pas reproductible.
    Ici, le choix reel du run est lu dans canonical_triples_v5.jsonl (enregistrements
    apparies ligne a ligne, dans l'ordre), puis un ordre compatible est construit.
    """
    KD = sorted(M.KNOWN_DESCRIPTORS)

    def tnorm(t):
        return M.normalize_entity(t.get("target_norm", t.get("target", "")))

    def cands(tn):
        if tn in M.KNOWN_DESCRIPTORS:
            return []
        return [d_ for d_ in KD if d_ in tn or tn in d_]

    def sig(t):
        p_, v_ = t.get("_provenance") or {}, t.get("_verification") or {}
        return (p_.get("query"), tuple(p_.get("selected_chunks") or []), v_.get("evidence"), v_.get("verdict"))

    amb = {i: (tnorm(t), cands(tnorm(t))) for i, t in enumerate(ver)
           if normalize_relation_04(M, t) == "hasDescriptor" and len(cands(tnorm(t))) > 1}
    chosen, ptr, matched_all = {}, 0, True
    for c in can:
        ct = M.normalize_entity(c.get("target_norm", c.get("target", "")))
        while ptr < len(ver):
            t = ver[ptr]
            tn = tnorm(t)
            ok_t = ct == tn or ct == cmap.get(tn) or ct in cands(tn) or ct in {cmap.get(x) for x in cands(tn)}
            if sig(t) == sig(c) and ok_t:
                break
            ptr += 1
        if ptr == len(ver):
            matched_all = False
            break
        if ptr in amb and ct in amb[ptr][1]:
            chosen[ptr] = ct
        ptr += 1
    before = defaultdict(set)
    for i, w in chosen.items():
        for o in amb[i][1]:
            if o != w:
                before[o].add(w)          # w doit preceder o
    order, left, conflict = [], list(KD), False
    while left:
        ready = [x for x in left if not (before[x] & set(left))]
        if not ready:
            conflict, order = True, order + left
            break
        order.append(ready[0])
        left.remove(ready[0])
    return order, amb, chosen, matched_all, conflict


def normalize_relation_04(M, t):
    t = copy.deepcopy(t)
    M.apply_relation_mapping(t)
    return t.get("relation_norm", t.get("relation", ""))


def doc_of(cid):
    return str(cid).split("::")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--run", default="output/run18")
    ap.add_argument("--outdir", default="output/audit/184_trace")
    ap.add_argument("--nfkc", action="store_true",
                    help="normaliser les ligatures avant la recherche des preuves (regle 145 sinon)")
    ap.add_argument("--no-sha-check", action="store_true", help="pour les tests seulement")
    args = ap.parse_args()
    global NFKC
    NFKC = args.nfkc

    repo, run, out = Path(args.repo), Path(args.run), Path(args.outdir)
    if out.exists():
        die("le dossier existe deja, rien n'est ecrase : " + str(out))
    sys.path.insert(0, str(repo.resolve()))
    for f, s12 in ((QUERIES, QUERIES_SHA12), (CLEAN, CLEAN_SHA12)):
        if not (repo / f).exists():
            die("introuvable : " + f)
        if not args.no_sha_check and sha(repo / f)[:12] != s12:
            die("%s n'est pas celui de run18 (sha %s, attendu %s)" % (f, sha(repo / f)[:12], s12))

    from pipeline.rag.constants import KNOWN_DESCRIPTORS, KNOWN_SETTINGS, normalize_entity, normalize_relation
    M = load_module(repo / CLEAN, "clean18")
    F = load_module(repo / "pipeline/06_tiered_fusion.py", "fusion06")

    ref = json.load(open(repo / REF26, encoding="utf-8"))["edges"]
    edges = [(e["subject"].lower(), e["relation"], e["object"].lower()) for e in ref]
    ref_keys = {(normalize_entity(s), normalize_relation(r), normalize_entity(o)): (s, r, o) for s, r, o in edges}

    # ---------------- A
    Q = jl(repo / QUERIES)
    susp = suspect_queries(Q, edges)
    if sorted(susp) != sorted(CLASSES):
        die("requetes suspectes trouvees %s, attendues %s" % (sorted(susp), sorted(CLASSES)))
    qtext_to_idx = defaultdict(list)
    for i, q in enumerate(Q):
        qtext_to_idx[str(q.get("query", "")).strip()].append(i)
    susp_text = {str(Q[i]["query"]).strip(): i for i in susp}

    out.mkdir(parents=True)
    checks = []

    def check(name, ok, detail=""):
        checks.append(("OK    " if ok else "ECHEC ") + name + (("  [" + detail + "]") if detail else ""))

    with open(out / "requetes_suspectes.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["index_requete", "strategie", "focus", "requete", "classe", "motif",
                    "arete_sujet", "arete_relation", "arete_objet", "forme_vue_sujet", "forme_vue_objet"])
        for i in sorted(susp):
            for h in susp[i]:
                s, r, o = h["edge"]
                w.writerow([i, Q[i].get("strategy"), Q[i].get("focus"), Q[i].get("query"),
                            CLASSES[i][0], CLASSES[i][1], s, r, o, h["vu_sujet"], h["vu_objet"]])

    # lexique integre de 04 (meme construction que main() de 04)
    lexicon = ({M.normalize_entity(d) for d in KNOWN_DESCRIPTORS} | {M.normalize_entity(s) for s in KNOWN_SETTINGS}
               | {"mass transport deposit", "mtd", "turbidite", "debris flow", "slide", "slump", "hemipelagite",
                  "pelagite", "turbidity current", "slope failure", "submarine landslide", "earthquake",
                  "pore pressure", "sedimentation", "erosion", "deposition", "seafloor", "sediment",
                  "continental slope", "continental shelf"})
    lex04c = {str(x).strip().lower() for x in KNOWN_DESCRIPTORS}

    # ---------------- B
    rows = []                       # une ligne par extraction
    amb_rows = []                   # objets de hasDescriptor qui contiennent plusieurs descripteurs
    KD_SET = M.KNOWN_DESCRIPTORS
    enter_fusion = {}               # (pass, cle6) -> tier de l'enregistrement retenu
    for P in ("a", "b"):
        d = run / ("pass_" + P)
        raw = jl(d / "raw_triples.jsonl")
        ver = jl(d / "verified_triples.jsonl")
        same = len(raw) == len(ver) and all(
            (x.get("source"), x.get("relation"), x.get("target"), (x.get("_provenance") or {}).get("query"))
            == (y.get("source"), y.get("relation"), y.get("target"), (y.get("_provenance") or {}).get("query"))
            for x, y in zip(raw, ver))
        check("passe %s : verified = raw, ligne a ligne (%d)" % (P, len(ver)), same)
        stats = json.load(open(d / "cleaning_stats_v5.json", encoding="utf-8"))
        cmap = json.load(open(d / "canonical_map_v5.json", encoding="utf-8"))
        rlog = {}
        logp = d / "canonical_triples_v5.rulecanon_log.csv"
        if logp.exists():
            for r_ in csv.DictReader(open(logp, encoding="utf-8")):
                rlog[r_["from"]] = r_["to"]
        can_pre = jl(d / "canonical_triples_v5.jsonl")
        order, amb, chosen, matched_all, conflict = descriptor_order(M, ver, can_pre, cmap)
        M.KNOWN_DESCRIPTORS = order          # liste ordonnee : memes membres, ordre du run
        amb_rows.extend({"passe": P, "ligne": i, "objet_brut": ver[i].get("target"), "candidats": ";".join(c_),
                         "choix_du_run": chosen.get(i, ""),
                         "origine_du_choix": "lu dans canonical_triples_v5" if i in chosen else "non determinable",
                         "choix_rejoue": next((x for x in order if x in c_), "")}
                        for i, (tn_, c_) in sorted(amb.items()))
        check("passe %s : %d objet(s) ambigu(s), %d choix lus dans la sortie du run" % (P, len(amb), len(chosen)),
              matched_all and not conflict and len(chosen) == len(amb),
              ("appariement incomplet " if not matched_all else "") + ("conflit d'ordre" if conflict else "")
              + ("" if len(chosen) == len(amb) else "%d non determinable(s)" % (len(amb) - len(chosen))))
        rej_reason, seen1, seen2 = Counter(), {}, {}
        n_d1 = n_d2 = n_loop = 0
        survivors = []
        prow = []
        for li, t0 in enumerate(ver):
            t = copy.deepcopy(t0)
            prov = t.get("_provenance") or {}
            v = (t.get("_verification") or {})
            qtxt = str(prov.get("query", "")).strip()
            qidx = qtext_to_idx.get(qtxt, [None])[0]
            row = {"passe": P, "ligne": li, "index_requete": qidx, "requete": qtxt,
                   "strategie": prov.get("strategy", ""),
                   "suspecte": "oui" if qtxt in susp_text else "non",
                   "classe_requete": CLASSES[susp_text[qtxt]][0] if qtxt in susp_text else "",
                   "brut_sujet": t0.get("source"), "brut_relation": t0.get("relation"), "brut_objet": t0.get("target"),
                   "verdict_qwen": v.get("verdict", "MISSING"), "etape_sortie": "", "raison": "",
                   "cle_sujet": "", "cle_relation": "", "cle_objet": "", "absorbe_par_ligne": "",
                   "brut_arete_reference": "oui" if (normalize_entity(t0.get("source", "")),
                                                     normalize_relation(t0.get("relation", "")),
                                                     normalize_entity(t0.get("target", ""))) in ref_keys else ""}
            # preuve dans ses propres passages
            top = prov.get("top_chunks") or []
            st, idx = locate(v.get("evidence", ""), [alnum(c.get("text", "")) for c in top])
            row["preuve_dans_ses_passages"] = st
            row["preuve_chunks"] = ";".join(str(top[i].get("chunk_id")) for i in idx)
            row["preuve_articles"] = ";".join(sorted({doc_of(top[i].get("chunk_id")) for i in idx}))
            row["articles_recuperes"] = ";".join(sorted({doc_of(c) for c in prov.get("selected_chunks") or []}))
            prow.append(row)

            ok, reason = M.check_verification(t, "normal")
            if not ok:
                row.update(etape_sortie="04_verdict", raison=reason); rej_reason[reason] += 1; continue
            M.apply_relation_mapping(t)
            stop = False
            for fn in (M.check_basic, M.check_relation, M.check_type_constraint):
                ok, reason = fn(t)
                if not ok:
                    row.update(etape_sortie="04_validation", raison=reason); rej_reason[reason] += 1; stop = True; break
            if stop:
                continue
            ok, reason = M.check_lexicon_coverage_soft(t, lexicon)
            if not ok:
                row.update(etape_sortie="04_validation", raison=reason); rej_reason[reason] += 1; continue
            k1 = M.triple_key(t)
            M.apply_canonical_map([t], cmap)
            if k1 in seen1:
                n_d1 += 1
                row.update(etape_sortie="doublon_04", raison="meme cle avant fusion des noms",
                           absorbe_par_ligne=seen1[k1])
            else:
                seen1[k1] = li
                s_, o_ = (normalize_entity(t.get("source_norm", t.get("source", ""))),
                          normalize_entity(t.get("target_norm", t.get("target", ""))))
                if s_ == o_:
                    n_loop += 1
                    row.update(etape_sortie="boucle_04", raison="sujet = objet apres fusion des noms")
                    continue
                k2 = M.triple_key(t)
                if k2 in seen2:
                    n_d2 += 1
                    row.update(etape_sortie="doublon_04", raison="meme cle apres fusion des noms",
                               absorbe_par_ligne=seen2[k2])
                else:
                    seen2[k2] = li
                    survivors.append((li, t))
            # 05a (ne touche que les champs subject/object, comme le script)
            for kf in ("subject", "object"):
                if t.get(kf) and str(t[kf]).strip() in rlog:
                    t[kf] = rlog[str(t[kf]).strip()]
            k6 = F.triple_key(t)
            row["cle_sujet"], row["cle_relation"], row["cle_objet"] = k6
            if str(t.get("relation", "")).strip().lower() == "hasdescriptor" and \
                    str(t.get("object") or t.get("target") or "").strip().lower() not in lex04c:
                if not row["etape_sortie"]:
                    row.update(etape_sortie="04c", raison="descripteur hors des 40")
                continue
            if row["etape_sortie"]:
                continue                     # doublon : cle connue, pas d'enregistrement propre
            tier = F.verdict_to_tier(F.get_verdict(t))
            row["tier_06_de_l_extraction"] = tier
            if tier == 3:
                row.update(etape_sortie="06_tier3", raison="verdict %s -> tier 3, non garde" % F.get_verdict(t))
            key = (P, k6)
            if key not in enter_fusion or tier < enter_fusion[key]:
                enter_fusion[key] = tier
        rows.extend(prow)

        # controles de la passe
        exp_rej = Counter(stats.get("verif_rejected", {})) + Counter(stats.get("validation_rejected", {}))
        check("passe %s : rejets de 04 par raison = cleaning_stats" % P, rej_reason == exp_rej,
              "rejoue %s / fichier %s" % (dict(rej_reason), dict(exp_rej)) if rej_reason != exp_rej else "")
        n_rejf = len(jl(d / "rejected_triples_v5.jsonl"))
        check("passe %s : %d rejets = rejected_triples_v5.jsonl (%d)" % (P, sum(rej_reason.values()), n_rejf),
              sum(rej_reason.values()) == n_rejf)
        check("passe %s : doublons et boucles = cleaning_stats" % P,
              (n_d1, n_d2, n_loop) == (stats.get("duplicates_pass1"), stats.get("duplicates_pass2"),
                                       stats.get("self_loops_removed")),
              "rejoue %s / fichier %s" % ((n_d1, n_d2, n_loop), (stats.get("duplicates_pass1"),
                                          stats.get("duplicates_pass2"), stats.get("self_loops_removed"))))
        can = jl(d / "canonical_triples_v5.jsonl")
        check("passe %s : sorties de 04 = canonical_triples_v5.jsonl (%d)" % (P, len(can)),
              [M.triple_key(t) for _, t in survivors] == [M.triple_key(t) for t in can])
        enf = jl(d / "canonical_triples_v5.rulecanon.lexicon_enforced.jsonl")
        mine = Counter((r["cle_sujet"], r["cle_relation"], r["cle_objet"]) for r in prow
                       if r["etape_sortie"] in ("", "06_tier3"))
        check("passe %s : sorties de 04c = lexicon_enforced (%d)" % (P, len(enf)),
              mine == Counter(F.triple_key(t) for t in enf))
        check("passe %s : 05a a fait %d substitution(s)" % (P, len(rlog)), True)

    M.KNOWN_DESCRIPTORS = KD_SET

    # ---------------- C
    kg = json.load(open(run / "kg" / "tiered_kg_run18.json", encoding="utf-8"))["triples"]
    kg_by = {(r["subject"], r["relation"], r["object"]): r for r in kg}
    fus = {}
    for (P, k), tier in enter_fusion.items():
        fus[k] = min(tier, fus.get(k, 9))
    expect6 = {k for k, t in fus.items() if t <= 2}
    check("cles tier 1-2 rejouees = tiered_kg_run18.json (%d / %d)" % (len(expect6), len(kg_by)),
          expect6 == set(kg_by))

    def nk(s, r, o):
        return (str(s).strip().lower(), str(r).strip(), str(o).strip().lower())

    dec = {nk(d["subject"], d["relation"], d["object"]): d for d in jl(run / "m4_panel" / "m4_panel_decisions.jsonl")}
    quar = {nk(t.get("subject"), t.get("relation"), t.get("object")) for t in jl(run / "m4_panel" / "m4_quarantine.jsonl")}
    flat = json.load(open(run / "kg_final" / "flat_run18_enforced_dedup.json", encoding="utf-8"))
    flat = flat["triples"] if isinstance(flat, dict) else flat
    flat_by = {nk(t.get("subject"), t.get("relation"), t.get("object")): t for t in flat}
    check("panel : une decision par relation fusionnee (%d / %d)" % (len(dec), len(kg_by)),
          all(nk(*k) in dec for k in kg_by))
    check("graphe livre %d + quarantaine %d = fusion %d" % (len(flat_by), len(quar), len(kg_by)),
          len(flat_by) + len(quar) == len(kg_by))

    chunks = jl(run / "index" / "chunks.jsonl")
    ch_al = [alnum(c["text"]) for c in chunks]

    def final_status(k):
        if k not in kg_by:
            return "absent de la fusion"
        n = nk(*k)
        if n in quar:
            return "quarantaine"
        if n in flat_by:
            return "livre Tier %s" % flat_by[n].get("tier")
        return "retire apres le panel (04b)"

    for r in rows:
        k = (r["cle_sujet"], r["cle_relation"], r["cle_objet"])
        r["statut_final_de_la_relation"] = final_status(k) if r["cle_sujet"] else ""
        r["arete_reference"] = "oui" if k in ref_keys else ""

    # ---------------- relations finales
    by_key = defaultdict(list)
    for r in rows:
        if r["cle_sujet"] and r["etape_sortie"] in ("", "doublon_04") and \
                r["verdict_qwen"] in ("STRONG_SUPPORT", "WEAK_SUPPORT"):
            by_key[(r["cle_sujet"], r["cle_relation"], r["cle_objet"])].append(r)
    rel_rows = []
    for k, rec in kg_by.items():
        ex = by_key.get(k, [])
        ex_s = [r for r in ex if r["suspecte"] == "oui"]
        ex_n = [r for r in ex if r["suspecte"] == "non"]
        ex_r = [r for r in ex if r["classe_requete"].startswith("R-")]
        raw_n = [r for r in rows if r["suspecte"] == "non" and (r["cle_sujet"], r["cle_relation"], r["cle_objet"]) == k]
        st_ev, idx = locate(rec.get("evidence", ""), ch_al)
        # verdict STRONG absorbe comme doublon derriere un enregistrement retenu non STRONG (meme passe)
        strong_lost = any(r["etape_sortie"] == "doublon_04" and r["verdict_qwen"] == "STRONG_SUPPORT" and
                          any(k_["passe"] == r["passe"] and k_["ligne"] == r["absorbe_par_ligne"] and
                              k_["verdict_qwen"] != "STRONG_SUPPORT" for k_ in ex) for r in ex)
        docs_ev = sorted({r_ for r in ex if r["preuve_dans_ses_passages"] in ("exact", "partial")
                          for r_ in r["preuve_articles"].split(";") if r_})
        rel_rows.append({
            "sujet": k[0], "relation": k[1], "objet": k[2], "tier_06": rec.get("tier"), "origine": rec.get("origin"),
            "decision_panel": (dec.get(nk(*k)) or {}).get("m4_decision", ""), "statut_final": final_status(k),
            "arete_reference": "oui" if k in ref_keys else "",
            "requete_affichee_suspecte": "oui" if str(rec.get("query", "")).strip() in susp_text else "non",
            "n_extractions_retenues": len(ex), "n_par_requetes_suspectes": len(ex_s),
            "n_par_requetes_non_suspectes": len(ex_n),
            "provenance_requetes": ("uniquement suspectes" if ex_s and not ex_n else
                                    "suspectes et non suspectes" if ex_s and ex_n else "non suspectes"),
            "provenance_requetes_R": ("uniquement R" if ex_r and len(ex_r) == len(ex) else
                                      "R et autres" if ex_r else "sans R"),
            "non_suspectes_extraites_mais_ecartees": len(raw_n) - len(ex_n) if ex_s and not ex_n else "",
            "strong_absorbe_comme_doublon": "oui" if strong_lost and rec.get("tier") == 2 else "",
            "verdicts_extractions": ";".join(sorted({r["verdict_qwen"] for r in ex})),
            "articles_avec_preuve_localisee": ";".join(docs_ev), "n_articles_preuve": len(docs_ev),
            "preuve_affichee_dans_index": st_ev,
            "preuve_affichee_articles": ";".join(sorted({doc_of(chunks[i]["chunk_id"]) for i in idx})),
        })

    def tsv(path, recs, cols):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(recs)

    cols_rows = ["passe", "ligne", "index_requete", "suspecte", "classe_requete", "strategie", "requete",
                 "brut_sujet", "brut_relation", "brut_objet", "brut_arete_reference", "verdict_qwen", "etape_sortie", "raison",
                 "absorbe_par_ligne", "cle_sujet", "cle_relation", "cle_objet", "arete_reference",
                 "statut_final_de_la_relation", "preuve_dans_ses_passages", "preuve_chunks", "preuve_articles",
                 "articles_recuperes"]
    tsv(out / "suivi_extractions.tsv", rows, cols_rows)
    tsv(out / "extractions_requetes_suspectes.tsv", [r for r in rows if r["suspecte"] == "oui"], cols_rows)
    tsv(out / "relations_fusionnees.tsv", rel_rows, list(rel_rows[0]) if rel_rows else ["vide"])
    tsv(out / "normalisations_ambigues.tsv", amb_rows,
        ["passe", "ligne", "objet_brut", "candidats", "choix_du_run", "origine_du_choix", "choix_rejoue"])

    # ---------------- pertes par etape
    loss = Counter((r["passe"], r["etape_sortie"] or "entre_en_fusion", r["raison"]) for r in rows)
    with open(out / "pertes_par_etape.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["passe", "etape", "raison", "n_extractions"])
        for (P, e, r_), n in sorted(loss.items()):
            w.writerow([P, e, r_, n])

    # ---------------- E. sensibilite
    deliv = [x for x in rel_rows if x["statut_final"].startswith("livre")]
    only = [x for x in deliv if x["provenance_requetes_R"] == "uniquement R"]
    keep = [x for x in deliv if x["provenance_requetes_R"] != "uniquement R"]
    keep16 = [x for x in deliv if x["provenance_requetes"] != "uniquement suspectes"]

    def summary(lst):
        return {"relations": len(lst), "tier1": sum(1 for x in lst if x["statut_final"] == "livre Tier 1"),
                "aretes_reference_exactes": sorted({"%s|%s|%s" % (x["sujet"], x["relation"], x["objet"])
                                                    for x in lst if x["arete_reference"]})}
    sens = {"graphe_livre": summary(deliv), "sans_relations_uniquement_R": summary(keep),
            "retirees_uniquement_R": summary(only), "sans_relations_uniquement_16": summary(keep16),
            "avertissement": "analyse de sensibilite sur le graphe existant, pas un nouveau run : retirer des "
                             "requetes changerait aussi les doublons, la table des noms et le panel"}
    json.dump(sens, open(out / "sensibilite.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---------------- resume
    ex_susp = [r for r in rows if r["suspecte"] == "oui"]
    L = ["=== 184 requetes suspectes, pertes et provenance ===",
         "requetes suspectes : %d (classes %s)" % (len(susp), dict(Counter(c for c, _ in CLASSES.values()))),
         "aretes de reference nommees par une requete R-* : %d / 26" % len({h["edge"] for i in susp for h in susp[i]
                                                                           if CLASSES[i][0] != "V-vocabulaire"}),
         "", "-- controles --"] + checks + [
         "", "-- extractions des requetes suspectes --",
         "extractions : %d (passe a %d, passe b %d)" % (len(ex_susp), sum(r["passe"] == "a" for r in ex_susp),
                                                       sum(r["passe"] == "b" for r in ex_susp)),
         "etape de sortie : %s" % dict(Counter(r["etape_sortie"] or "entre_en_fusion" for r in ex_susp)),
         "extraction brute = arete de reference : %d ; apres nettoyage : %d" % (
             sum(1 for r in ex_susp if r["brut_arete_reference"]), sum(1 for r in ex_susp if r["arete_reference"])),
         "", "-- relations fusionnees touchees par une requete suspecte --",
         "%s" % dict(Counter((x["provenance_requetes"], x["statut_final"]) for x in rel_rows
                             if x["n_par_requetes_suspectes"])),
         "", "-- sensibilite (graphe livre) --",
         "livre : %d relations, Tier 1 %d, aretes exactes %d / 26" % (
             sens["graphe_livre"]["relations"], sens["graphe_livre"]["tier1"],
             len(sens["graphe_livre"]["aretes_reference_exactes"])),
         "sans les relations obtenues uniquement par des requetes R-* : %d, Tier 1 %d, aretes exactes %d / 26" % (
             sens["sans_relations_uniquement_R"]["relations"], sens["sans_relations_uniquement_R"]["tier1"],
             len(sens["sans_relations_uniquement_R"]["aretes_reference_exactes"])),
         "aretes perdues : %s" % sorted(set(sens["graphe_livre"]["aretes_reference_exactes"])
                                        - set(sens["sans_relations_uniquement_R"]["aretes_reference_exactes"])),
         "variante, sans les relations obtenues uniquement par les 16 : %d, aretes exactes %d / 26" % (
             sens["sans_relations_uniquement_16"]["relations"],
             len(sens["sans_relations_uniquement_16"]["aretes_reference_exactes"])),
         "", "-- normalisation des descripteurs dans 04 --",
         "objets contenant plusieurs descripteurs : %d (le choix de 04 depend de l'ordre d'un ensemble Python, "
         "donc de PYTHONHASHSEED ; choix du run lus dans canonical_triples_v5.jsonl, voir normalisations_ambigues.tsv)"
         % len(amb_rows),
         "", "-- pertes (toutes requetes) --",
         "%s" % dict(Counter(r["etape_sortie"] or "entre_en_fusion" for r in rows)),
         "relations en Tier 2 a la fusion alors qu'une extraction STRONG a ete absorbee comme doublon : %d" % sum(
             1 for x in rel_rows if x["strong_absorbe_comme_doublon"]),
         "", "-- provenance --",
         "preuve de l'extraction dans ses 5 passages : %s" % dict(Counter(r["preuve_dans_ses_passages"] for r in rows)),
         "preuve affichee des relations livrees, dans l'index : %s" % dict(Counter(x["preuve_affichee_dans_index"]
                                                                                for x in deliv)),
         "articles avec preuve localisee par relation livree : %s" % dict(sorted(Counter(
             x["n_articles_preuve"] for x in deliv).items())),
         ]
    n_fail = sum(c.startswith("ECHEC") for c in checks)
    L.append("\n%d controle(s) en echec%s" % (n_fail, " : NE PAS UTILISER les chiffres avant correction" if n_fail else ""))
    txt = "\n".join(L)
    (out / "_resume.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt + "\n[fait] " + str(out), flush=True)
    sys.exit(3 if n_fail else 0)


if __name__ == "__main__":
    main()