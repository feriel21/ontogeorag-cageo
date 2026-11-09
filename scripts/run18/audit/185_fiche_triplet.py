#!/usr/bin/env python3
"""185 — Fiche d'un triplet de run18, de la requete au graphe livre.

LECTURE SEULE. CPU. Aucun modele n'est appele (le tokenizer de Qwen peut etre
charge pour afficher le texte exact apres le gabarit de conversation).
N'ecrit que dans --outdir (refuse d'ecraser).

Chaque bloc de la fiche porte son origine :
  [TEXTE]     texte de l'article, tel qu'il est dans l'index (minuscules, SYNONYM_MAP de 49)
  [CONFIG]    fichier de configuration du run (requetes, schema)
  [CODE]      valeur produite ou modifiee par un script, avec le fichier et la fonction
  [MODELE]    sortie d'un LLM, lue dans un fichier du run
  [REGENERE]  non journalise par le run ; recalcule ici avec la meme fonction et les memes
              entrees ; la fiche dit comment la regeneration est controlee
  [INFERENCE] deduit, non enregistre par le run
  [MANQUANT]  aucun fichier du run ne le contient ; rien n'est reconstruit

USAGE (racine du worktree run18, venv actif)
  cd ~/ontogeorag_run18 && export PYTHONPATH=$PWD
  python 185_fiche_triplet.py --repo . --run output/run18 \
      --subject earthquake --relation triggers --object "slope failure" \
      --outdir output/audit/185_fiche_earthquake
  option : --tokenizer <dossier local de Qwen2.5-7B-Instruct> pour afficher le texte
           exact apres gabarit de conversation (sinon le script le cherche dans _steps.jsonl)
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

EXTRACT = "pipeline/02_extract_triples_run18.py"
EXTRACT_SHA12 = "7fdc6cc1a442"
CLEAN = "pipeline/04_clean_validate_run18.py"
CLEAN_SHA12 = "e5a8db171df7"
QUERIES = "configs/descriptor_queries_run18.jsonl"
SCHEMA = "configs/ontology_schema.json"
WINDOW = 40


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


NFKC = False   # --nfkc : ligatures (ﬁ, ﬂ...) et formes de compatibilite ramenees a l'ASCII avant comparaison


def alnum(x):
    x = unicodedata.normalize("NFKC", str(x)) if NFKC else str(x)
    return re.sub(r"[^a-z0-9]", "", x.lower())


def locate(ev, texts):
    e = alnum(ev)
    al = [alnum(t) for t in texts]
    ex = [i for i, c in enumerate(al) if e and e in c]
    if ex:
        return "exact", ex
    if len(e) <= WINDOW:
        return ("vide" if not e else "not_found"), []
    mid = (len(e) - WINDOW) // 2
    wins = [e[:WINDOW], e[mid:mid + WINDOW], e[-WINDOW:]]
    pa = [i for i, c in enumerate(al) if any(w in c for w in wins)]
    return ("partial", pa) if pa else ("not_found", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--run", default="output/run18")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--relation", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-chars", type=int, default=None, help="sinon lu dans _steps.jsonl, sinon 12000")
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
    sys.path.insert(0, str((repo / "m4").resolve()))
    for f, s12 in ((EXTRACT, EXTRACT_SHA12), (CLEAN, CLEAN_SHA12)):
        if not (repo / f).exists():
            die("introuvable : " + f)
        if not args.no_sha_check and sha(repo / f)[:12] != s12:
            die("%s n'est pas celui de run18 (sha %s)" % (f, sha(repo / f)[:12]))

    from pipeline.rag.constants import normalize_entity, normalize_relation
    E = load(repo / EXTRACT, "extract18")
    M = load(repo / CLEAN, "clean18")
    F = load(repo / "pipeline/06_tiered_fusion.py", "fusion06")
    V = load(repo / "pipeline/03_verify_triples.py", "verify03")
    import m4_verify as J

    key = (normalize_entity(args.subject), normalize_relation(args.relation), normalize_entity(args.object))
    md, missing = [], []

    def H(t):
        md.append("\n## " + t + "\n")

    def P(t=""):
        md.append(t)

    def block(label, txt):
        md.append("**%s**\n\n```\n%s\n```\n" % (label, txt))

    # ---------- relation dans la fusion
    kg = json.load(open(run / "kg/tiered_kg_run18.json", encoding="utf-8"))["triples"]
    recs = [r for r in kg if (r["subject"], r["relation"], r["object"]) == key]
    if len(recs) != 1:
        die("relation %s trouvee %d fois dans tiered_kg_run18.json" % (key, len(recs)))
    rec = recs[0]
    P("# Fiche run18 : %s — %s — %s\n" % key)
    P("Relation de la fusion : tier %s, origine `%s`, verdict retenu `%s`." % (rec.get("tier"), rec.get("origin"),
                                                                               rec.get("verdict")))
    P("Pour une relation presente dans les deux passes, 06 garde l'enregistrement de la passe a "
      "(boucle sur `idx_a`), donc la fiche suit la passe %s." % ("b" if rec.get("origin") == "iter_b" else "a"))
    Ppass = "b" if rec.get("origin") == "iter_b" else "a"
    d = run / ("pass_" + Ppass)

    # enregistrement exact que 06 a utilise : build_index sur lexicon_enforced
    enf = jl(d / "canonical_triples_v5.rulecanon.lexicon_enforced.jsonl")
    idx = F.build_index(enf)
    if key not in idx:
        die("cle absente de lexicon_enforced de la passe " + Ppass)
    t06 = idx[key]["raw"]
    prov = t06.get("_provenance") or {}
    ver6 = t06.get("_verification") or {}

    # ligne d'origine dans verified_triples (rejeu des normalisations de 04)
    verified = jl(d / "verified_triples.jsonl")
    raw = jl(d / "raw_triples.jsonl")
    cmap = json.load(open(d / "canonical_map_v5.json", encoding="utf-8"))
    # 04 remplace un descripteur hors vocabulaire par le PREMIER descripteur d'un ensemble
    # Python qui le contient : avec plusieurs candidats, le choix depend de PYTHONHASHSEED.
    # Pour retrouver la ligne, l'objet de la relation est place en tete de l'ordre.
    KD_SET = M.KNOWN_DESCRIPTORS
    M.KNOWN_DESCRIPTORS = ([key[2]] if key[2] in KD_SET else []) + sorted(x for x in KD_SET if x != key[2])
    cand = []
    for i, v in enumerate(verified):
        pv = v.get("_provenance") or {}
        if pv.get("query") != prov.get("query") or pv.get("selected_chunks") != prov.get("selected_chunks"):
            continue
        if (v.get("_verification") or {}).get("evidence") != ver6.get("evidence"):
            continue
        t = copy.deepcopy(v)
        M.apply_relation_mapping(t)
        M.check_type_constraint(t)
        M.apply_canonical_map([t], cmap)
        if F.triple_key(t) == key:
            cand.append(i)
    if not cand:
        die("ligne d'origine introuvable dans verified_triples.jsonl")
    li = cand[0]
    P("Ligne d'origine : passe %s, ligne %d de `verified_triples.jsonl` et de `raw_triples.jsonl` "
      "(%d ligne(s) candidate(s) ; 04 garde la premiere)." % (Ppass, li, len(cand)))
    v = verified[li]
    r0 = raw[li]
    if (r0.get("source"), r0.get("relation"), r0.get("target")) != (v.get("source"), v.get("relation"), v.get("target")):
        die("raw et verified ne concordent pas a la ligne %d" % li)

    # ---------- 1. requete
    H("1. La requete")
    Q = jl(repo / QUERIES)
    qi = [i for i, q in enumerate(Q) if str(q.get("query", "")).strip() == str(prov.get("query", "")).strip()]
    q = Q[qi[0]] if qi else {}
    block("[CONFIG] %s, ligne %s (0 = premiere)" % (QUERIES, qi[0] if qi else "?"), json.dumps(q, ensure_ascii=False))
    P("La strategie `%s` choisit le gabarit de prompt (`get_prompt`, 02) ; une strategie absente de `PROMPTS` "
      "prend le gabarit `descriptor`. [CODE]" % q.get("strategy"))

    # ---------- 2. passages
    H("2. Les cinq passages selectionnes")
    chunks = {c["chunk_id"]: c for c in jl(run / "index/chunks.jsonl")}
    sel = prov.get("selected_chunks") or []
    P("Ordre = ordre apres reranking (cross-encoder), puis `candidates[:5]`. [CODE]")
    P("Score BM25 du premier candidat : %s, seuil %s. [CODE]" % (prov.get("best_bm25"), prov.get("bm25_threshold")))
    P("[MANQUANT] Les scores BM25 et de reranking de chaque passage et les 15 autres candidats ne sont pas "
      "ecrits par 02.")
    missing.append("scores BM25 / reranking par passage et candidats 6 a 20 (non ecrits par 02)")
    for k, cid in enumerate(sel, 1):
        c = chunks.get(cid)
        if c is None:
            P("[MANQUANT] passage %s absent de l'index" % cid)
            missing.append("texte du passage " + cid)
            continue
        block("[TEXTE] passage %d : %s (article %s, %d caracteres)" % (k, cid, c.get("doc_id"), len(c["text"])),
              c["text"])
    P("Le texte est celui de l'index : minuscules, et `SYNONYM_MAP` de 49 deja appliquee "
      "(par ex. mtd -> mass transport deposit). [CODE]")

    # ---------- 3. prompt
    H("3. Le prompt envoye a Qwen")
    max_chars = args.max_chars
    stp = run / "_steps.jsonl"
    model_path = args.tokenizer
    if stp.exists():
        for s in jl(stp):
            if s.get("name") == "extract_pass_" + Ppass:
                a = s.get("args") or ""
                a = a if isinstance(a, str) else " ".join(map(str, a))
                m = re.search(r"--max-chars\s+(\d+)", a)
                if m and max_chars is None:
                    max_chars = int(m.group(1))
                m = re.search(r"--model\s+(\S+)", a)
                if m and model_path is None:
                    model_path = m.group(1)
    if max_chars is None:
        max_chars = 12000
        P("`--max-chars` non trouve dans `_steps.jsonl` : 12000 (valeur du job). [CONFIG]")
    sel_dicts = [{"text": chunks[c]["text"], "source_file": chunks[c].get("source_file", "unknown")}
                 for c in sel if c in chunks]
    ctx = E.concat_chunks(sel_dicts, max_chars=max_chars)
    schema = json.loads((repo / SCHEMA).read_text())
    relations = [r["name"] if isinstance(r, dict) else r for r in schema.get("relations", [])]
    types = [t["name"] if isinstance(t, dict) else t for t in schema.get("types", [])]
    kw = dict(chunk_text=ctx, query=prov.get("query", ""), relations=", ".join(relations), types=", ".join(types))
    if q.get("strategy") == "profile":
        foc = str(q.get("focus") or "")
        foc = foc.split("→")[0].strip() if "→" in foc else foc.strip()
        kw["focus_object"] = foc or "mass transport deposit"
    prompt = E.get_prompt(q.get("strategy", "descriptor"), **kw)
    ok_prev = ctx[:1500] == ((v.get("_provenance") or {}).get("context_preview") or "")
    P("[REGENERE] Le prompt n'est pas journalise. Il est recalcule avec `concat_chunks` et `get_prompt` de "
      "`%s`, les 5 passages ci-dessus, `max_chars=%d`, et les listes du schema." % (EXTRACT, max_chars))
    P("Controle : les 1 500 premiers caracteres du contexte regenere %s `context_preview` enregistre par 02."
      % ("sont IDENTIQUES a" if ok_prev else "DIFFERENT de"))
    P("En-tete de chaque passage : `[CHUNK i | %s]` — `source_file` vient de l'index ; s'il manque, le code met "
      "`unknown`. [CODE]" % ", ".join(sorted({d_["source_file"] for d_ in sel_dicts})))
    P("Listes du schema transmises : relations = `%s`, types = `%s` (lues dans les cles `relations` et `types`). "
      "[CODE]" % (", ".join(relations) or "(vide)", ", ".join(types) or "(vide)"))
    P("Longueur du prompt : %d caracteres. Envoye comme message utilisateur, sans message systeme "
      "(`generate = lambda prompt: _gen(\"\", prompt, ...)`). [CODE]" % len(prompt))
    block("[REGENERE] prompt (message utilisateur)", prompt)
    if model_path:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path)
            full = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True)
            head = full[:full.find(prompt[:80])] if prompt[:80] in full else full[:400]
            block("[REGENERE] texte ajoute par le gabarit de conversation de Qwen avant le message", head)
            P("Ce gabarit vient du tokenizer de Qwen (%s), pas du code du pipeline. [CODE]" % model_path)
        except Exception as e:
            P("Gabarit de conversation non affiche (%s)." % e)
    else:
        P("Gabarit de conversation de Qwen non affiche : passer `--tokenizer <dossier du modele>`.")
    P("Parametres de generation : temperature %s, `top_p=0.9` si temperature > 0, `repetition_penalty=1.1`, "
      "`max_new_tokens` = `--max-tokens` du job. [CODE]" % ("0.0" if Ppass == "a" else "0.3"))

    # ---------- 4. reponse brute
    H("4. La reponse brute de Qwen")
    P("[MANQUANT] 02 n'ecrit pas la reponse du modele : il ne garde que les triplets lus par `parse_response`. "
      "Aucun fichier du run ne la contient. Rien n'est reconstruit ici.")
    missing.append("reponse brute de Qwen a l'extraction (non ecrite par 02)")

    # ---------- 5. triplet apres parsing
    H("5. Le triplet apres parsing")
    vis = {k: r0.get(k) for k in ("source", "source_type", "relation", "target", "target_type", "salience")}
    block("raw_triples.jsonl, passe %s, ligne %d" % (Ppass, li), json.dumps(vis, ensure_ascii=False, indent=1))
    P("- `source`, `target`, `source_type`, `target_type` : ecrits par Qwen, recopies tels quels. [MODELE]")
    P("- `relation` : deja remplacee par `normalize_relation` (constants) dans 02 ; le libelle ecrit par Qwen "
      "n'est pas conserve. [CODE] [MANQUANT pour le libelle d'origine]")
    P("- `salience` : ecrite par Qwen, ou mise a `common` par 02 si absente pour un hasDescriptor ; le fichier "
      "ne dit pas lequel des deux. [MODELE ou CODE]")
    missing.append("libelle de relation ecrit par Qwen (remplace dans 02)")
    block("[CODE] _provenance (sans les textes)", json.dumps({k: v_ for k, v_ in (r0.get("_provenance") or {}).items()
                                                             if k not in ("context_preview",)}, ensure_ascii=False,
                                                            indent=1))

    # ---------- 6. normalisations
    H("6. Les normalisations appliquees")
    t = copy.deepcopy(v)
    tn0 = M.normalize_entity(v.get("target_norm", v.get("target", "")))
    amb0 = [x for x in sorted(KD_SET) if tn0 not in KD_SET and (x in tn0 or tn0 in x)]
    if len(amb0) > 1:
        P("Objet ambigu : `%s` contient plusieurs descripteurs (%s). 04 prend le premier d'un ensemble Python, "
          "dont l'ordre depend de PYTHONHASHSEED ; le run a retenu `%s`. Ce choix n'est pas reproductible. [CODE]"
          % (tn0, ", ".join(amb0), key[2]))
    rows = [("02 parse + normalize_relation", "%s | %s | %s" % (t.get("source"), t.get("relation"), t.get("target")))]
    M.apply_relation_mapping(t)
    rows.append(("04 apply_relation_mapping", "%s | %s | %s" % (t.get("source"), t.get("relation"), t.get("target"))))
    okv, reason_v = M.check_verification(copy.deepcopy(v), "normal")
    for fn in (M.check_basic, M.check_relation, M.check_type_constraint):
        ok_, r_ = fn(t)
        rows.append(("04 %s -> %s" % (fn.__name__, r_), "%s | %s | %s" % (t.get("source"), t.get("relation"),
                                                                           t.get("target"))))
    used = {k_: cmap[k_] for k_ in (M.normalize_entity(t.get("source", "")), M.normalize_entity(t.get("target", "")))
            if k_ in cmap}
    M.apply_canonical_map([t], cmap)
    rows.append(("04 canonical_map_v5 %s" % (used or "(aucune regle)"),
                 "%s | %s | %s" % (t.get("source"), t.get("relation"), t.get("target"))))
    logp = d / "canonical_triples_v5.rulecanon_log.csv"
    rlog = {}
    if logp.exists():
        import csv
        rlog = {x["from"]: x["to"] for x in csv.DictReader(open(logp, encoding="utf-8"))}
    hit5 = [kf for kf in ("subject", "object") if t.get(kf) and str(t[kf]).strip() in rlog]
    rows.append(("05a (lit subject/object ; champs presents : %s)" % [k_ for k_ in ("subject", "object") if k_ in t],
                 "substitution : %s" % (hit5 or "aucune")))
    rows.append(("06 normalize_entity / normalize_relation", "%s | %s | %s" % F.triple_key(t)))
    P("| Etape (fichier, fonction) | Valeur apres l'etape |\n|---|---|")
    for a_, b_ in rows:
        P("| %s | %s |" % (a_, b_))
    P("\nToutes ces valeurs sont produites par le code [CODE], a partir des noms ecrits par Qwen [MODELE].")
    try:
        enfkg = json.load(open(run / "kg_enforced/tiered_kg_enforced.json", encoding="utf-8"))
        allr = enfkg.get("tier1", []) + enfkg.get("tier2", []) if isinstance(enfkg, dict) else enfkg
        e4b = [x for x in allr if (normalize_entity(x.get("subject", "")), normalize_relation(x.get("relation", "")),
                                   normalize_entity(x.get("object", ""))) == key]
        if e4b:
            P("Types : Qwen `%s` / `%s` -> 04b `%s` / `%s` (TYPE_MAP de 04b). [MODELE] puis [CODE]" % (
                rec.get("subject_type"), rec.get("object_type"), e4b[0].get("subject_type"),
                e4b[0].get("object_type")))
    except FileNotFoundError:
        P("[MANQUANT] kg_enforced/tiered_kg_enforced.json")

    # ---------- 7. verification
    H("7. Le passage et le verdict qui ont permis de le garder")
    ver = v.get("_verification") or {}
    top = [c.get("text", "") for c in (v.get("_provenance") or {}).get("top_chunks", [])]
    block("[MODELE] verdict de Qwen (verified_triples.jsonl, ligne %d)" % li,
          json.dumps({k: ver.get(k) for k in ("verdict", "evidence", "reasoning", "chunks_checked", "model")},
                     ensure_ascii=False, indent=1))
    aud = d / "verification_audit.jsonl"
    araw = None
    if aud.exists():
        al = jl(aud)
        if li < len(al) and al[li].get("index") == li:
            araw = al[li].get("raw")
    if araw is not None:
        block("[MODELE] reponse brute de Qwen pour le passage retenu (verification_audit.jsonl, index %d)" % li, araw)
    else:
        P("[MANQUANT] reponse brute de la verification (verification_audit.jsonl absent ou sans champ raw).")
        missing.append("reponse brute de la verification")
    P("[MANQUANT] 03 ne journalise que la reponse du meilleur passage, pas les quatre autres, ni le numero du "
      "passage retenu. Le code garde le premier passage qui atteint le meilleur verdict (`score > best_score`).")
    missing.append("reponses de verification des 4 autres passages et numero du passage retenu")
    st, where = locate(ver.get("evidence", ""), top)
    P("[INFERENCE] La phrase `evidence` est retrouvee (%s) dans le(s) passage(s) %s. Regles du script 145 : "
      "comparaison en minuscules alphanumeriques ; partial = une des trois fenetres de 40 caracteres." % (
          st, [i + 1 for i in where] or "aucun"))
    for i in where[:1]:
        cid = ((v.get("_provenance") or {}).get("top_chunks") or [{}])[i].get("chunk_id")
        chunk_chars = 2000
        block("[TEXTE] passage %d (%s) tel que lu par le verificateur (coupe a %d caracteres)" % (i + 1, cid,
                                                                                                    chunk_chars),
              top[i][:chunk_chars])
        rel = normalize_relation(str(v.get("relation_norm", v.get("relation", ""))))
        user = V.COT_PROMPT.format(chunk_text=top[i][:chunk_chars], source=v.get("source_norm", v.get("source", "")),
                                   relation=rel, gloss=V.RELATION_GLOSSES.get(rel, "Subject %s Object" % rel)
                                   if hasattr(V, "RELATION_GLOSSES") else rel,
                                   target=v.get("target_norm", v.get("target", "")))
        block("[REGENERE] message systeme de la verification (03, SYSTEM)", V.SYSTEM)
        block("[REGENERE] message utilisateur de la verification pour ce passage (03, COT_PROMPT)", user)
    P("Decision de 04 sur le verdict : `%s` (politique normal : STRONG et WEAK gardes, NO_CHUNK et UNPARSEABLE "
      "gardes aussi). [CODE]" % reason_v)
    P("Tier a la fusion : `verdict_to_tier(%s)` = %s ; tier de la relation fusionnee = %s, origine `%s`. [CODE]" % (
        ver.get("verdict"), F.verdict_to_tier(ver.get("verdict", "")), rec.get("tier"), rec.get("origin")))

    # ---------- 8. panel et graphe livre
    H("8. Panel et graphe livre")
    passage_j = J.evidence_from_provenance(rec)
    block("[CODE] passage donne aux juges (m4_verify.evidence_from_provenance sur l'enregistrement fusionne ; "
          "06 a coupe evidence a 400 caracteres)", passage_j)

    def nk(s, r, o):
        return (str(s).strip().lower(), str(r).strip(), str(o).strip().lower())
    for jn in ("llama", "mistral"):
        p = run / "m4" / jn / "m4_verdicts.jsonl"
        if not p.exists():
            P("[MANQUANT] " + str(p))
            missing.append(str(p))
            continue
        jr = [x for x in jl(p) if nk(x["subject"], x["relation"], x["object"]) == nk(*key)]
        if not jr:
            P("[MANQUANT] relation absente de " + str(p))
            continue
        x = jr[0]
        block("[MODELE] %s, passe aveugle : verdict %s (reponse coupee a 800 caracteres par m4_verify)" % (
            jn, x["blind"].get("verdict")), x["blind"].get("raw", ""))
        block("[MODELE] %s, passe avec phrase : verdict %s (reponse coupee a 800 caracteres)" % (
            jn, x["evidence"].get("verdict")), x["evidence"].get("raw", ""))
    dec = [x for x in jl(run / "m4_panel/m4_panel_decisions.jsonl")
           if nk(x["subject"], x["relation"], x["object"]) == nk(*key)]
    if dec:
        P("Decision du panel : `%s`. [CODE]" % dec[0].get("m4_decision"))
    flat = json.load(open(run / "kg_final/flat_run18_enforced_dedup.json", encoding="utf-8"))
    flat = flat["triples"] if isinstance(flat, dict) else flat
    fr = [x for x in flat if nk(x.get("subject"), x.get("relation"), x.get("object")) == nk(*key)]
    if fr:
        block("[CODE] enregistrement livre (flat_run18_enforced_dedup.json)", json.dumps(fr[0], ensure_ascii=False,
                                                                                        indent=1))
    else:
        P("Relation absente du graphe livre (quarantaine ou retiree).")

    H("Ce qui manque dans les fichiers du run")
    for m_ in dict.fromkeys(missing):
        P("- " + m_)

    out.mkdir(parents=True)
    slug = re.sub(r"[^a-z0-9]+", "_", "_".join(key).lower()).strip("_")
    (out / ("fiche_%s.md" % slug)).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\n[fait] " + str(out / ("fiche_%s.md" % slug)))


if __name__ == "__main__":
    main()