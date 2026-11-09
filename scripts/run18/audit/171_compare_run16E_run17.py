#!/usr/bin/env python3
"""171 — Comparaison a l'identique du graphe livre run16E et du graphe run17.

POURQUOI
    Le mail de run17 donne la vue interne article_metrics (a ne pas citer).
    Avant de decider si run17 remplace run16E, il faut les memes mesures,
    calculees par le meme code, sur les deux graphes livres (flat_*_enforced_dedup).

QUOI (lecture seule, CPU, pas de modele)
    1. Taille : relations, Tier 1 / Tier 2, entites, types de relation,
       decisions M4, origine (both / pass_a / pass_b).
    2. Recouvrement : relations communes, perdues, nouvelles (cle normalisee
       sujet, relation, objet) ; changements de tier pour les communes.
    3. Rappel contre le graphe expert, 26 et 34 aretes, apparieur EXACT
       (borne basse, celle du papier) et apparieur sous-chaine, arete par arete :
       gagnee, perdue, gardee, absente.
    4. Pour chaque arete gagnee par run17 : les relations qui la portent,
       leur tier, leur phrase de preuve, les verdicts M4.
    5. Si --chunks est donne : localisation de la phrase de preuve dans l'index
       (meme regle que graph_with_located_provenance.py : exact / partial /
       not_found), et presence des deux arguments dans la phrase et dans
       le chunk localise (indicatif).

GARDE-FOU
    Le graphe A doit reproduire les chiffres publies de run16E :
    232 relations, Tier 1 160, Tier 2 72, rappel exact 13/26 et 18/34,
    et avec --chunks : exact 164, partial 64, not_found 4.
    Sinon le script s'arrete sans rien ecrire.

Les fonctions de rappel sont celles de pipeline/07_final_metrics.py,
importees sans modification. Le dossier --outdir ne doit pas exister.

USAGE (depuis la racine du worktree run17)
    python 171_compare_run16E_run17.py \
        --a $HOME/ontogeorag_D/output/run16E/kg_final/flat_run16E_enforced_dedup.json \
        --b $HOME/ontogeorag_run17/output/run17/kg_final/flat_run17_enforced_dedup.json \
        --chunks $HOME/ontogeorag_D/output/run16E/index/chunks.jsonl \
        --outdir $HOME/ontogeorag_run17/output/audit/171_compare
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

EXPECT_A = {"n": 232, "tier1": 160, "tier2": 72, "exact26": 13, "exact34": 18}
EXPECT_A_LOC = {"exact": 164, "partial": 64, "not_found": 4}
WINDOW = 40
REF26 = "configs/lb_reference_edges_original26.json"
REF34 = "configs/lb_reference_edges.json"


def die(msg):
    print("ARRET : " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def load_07(repo):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("m07", repo / "pipeline" / "07_final_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        from pipeline.rag.constants import normalize_relation
    except Exception:
        def normalize_relation(r):
            return (r or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return mod, normalize_relation


def load_graph(path):
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    for k in ("triples", "relations", "edges", "records", "items"):
        if isinstance(raw.get(k), list):
            return raw[k]
    die("liste de relations introuvable dans " + str(path))


def alnum(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def locate(ev, chunk_alnum):
    e = alnum(ev)
    exact = [i for i, c in enumerate(chunk_alnum) if e and e in c]
    if exact:
        return "exact", exact
    if len(e) <= WINDOW:
        return "not_found", []
    mid = (len(e) - WINDOW) // 2
    wins = [e[:WINDOW], e[mid:mid + WINDOW], e[-WINDOW:]]
    part = [i for i, c in enumerate(chunk_alnum) if any(w in c for w in wins)]
    return ("partial", part) if part else ("not_found", [])


def edge_table(m07, graph, ref_path):
    """Statut de chaque arete de reference : exact, sous-chaine seulement, absente."""
    t12 = [t for t in graph if t.get("tier", 3) <= 2]
    r = m07.recall(t12, ref_path=ref_path)
    sub = {tuple(e) for e in r["matched_edges"]}
    exa = {tuple(e) for e in r["lower_bound"]["matched_edges"]}
    refs = m07._load_reference_edges_from_json(ref_path)
    out = {}
    for e in refs:
        e = tuple(e)
        out[e] = "exact" if e in exa else ("substring_only" if e in sub else "absent")
    return out, r["lower_bound"]["hits"], r["hits"], len(refs)


def carriers(m07, normrel, graph, edge, exact_only):
    rs, rr, ro = edge
    rrn, rs_n, ro_n = normrel(rr), m07.norm(rs), m07.norm(ro)
    hits = []
    for t in graph:
        if t.get("tier", 3) > 2 or normrel(t.get("relation", "")) != rrn:
            continue
        ts, to = m07.norm(t.get("subject", "")), m07.norm(t.get("object", ""))
        ok = (ts == rs_n and to == ro_n) if exact_only else (rs_n in ts and ro_n in to)
        if ok:
            hits.append(t)
    return hits


def describe(m07, normrel, graph, chunks, chunk_alnum):
    key = lambda t: (m07.norm(t.get("subject", "")), normrel(t.get("relation", "")), m07.norm(t.get("object", "")))
    d = {
        "n": len(graph),
        "tiers": dict(sorted(Counter(t.get("tier") for t in graph).items(), key=lambda x: str(x[0]))),
        "entities": len({m07.norm(t.get("subject", "")) for t in graph} | {m07.norm(t.get("object", "")) for t in graph}),
        "distinct_keys": len({key(t) for t in graph}),
        "relations": dict(Counter(t.get("relation") for t in graph).most_common()),
        "origin": dict(Counter(t.get("origin") for t in graph).most_common()),
        "m4_decision": dict(Counter((t.get("m4") or {}).get("decision") for t in graph).most_common()),
        "args_in_evidence": dict(Counter(
            "both" if alnum(t.get("subject")) in alnum(t.get("evidence")) and alnum(t.get("object")) in alnum(t.get("evidence"))
            else "not_both" for t in graph)),
    }
    loc = {}
    if chunks is not None:
        mc, nd, chunk_both = Counter(), Counter(), Counter()
        for i, t in enumerate(graph):
            status, ids = locate(t.get("evidence") or "", chunk_alnum)
            docs = sorted({str(chunks[j]["doc_id"]) for j in ids})
            mc[status] += 1
            nd[len(docs)] += 1
            if ids:
                s, o = alnum(t.get("subject")), alnum(t.get("object"))
                chunk_both["both" if any(s in chunk_alnum[j] and o in chunk_alnum[j] for j in ids) else "not_both"] += 1
            loc[i] = (status, docs)
        d["located"] = dict(mc)
        d["documents_per_relation"] = {str(k): v for k, v in sorted(nd.items())}
        d["args_in_located_chunk"] = dict(chunk_both)
    return d, loc, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="graphe livre run16E (flat_run16E_enforced_dedup.json)")
    ap.add_argument("--b", required=True, help="graphe run17 (flat_run17_enforced_dedup.json)")
    ap.add_argument("--chunks", default=None, help="index chunks.jsonl de run16E (optionnel)")
    ap.add_argument("--repo", default=".", help="racine du depot (pipeline/ et configs/)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    if out.exists():
        die("le dossier existe deja, rien n'est ecrase : " + str(out))
    repo = Path(args.repo).resolve()
    m07, normrel = load_07(repo)
    import os
    os.chdir(repo)

    A, B = load_graph(args.a), load_graph(args.b)
    chunks = chunk_alnum = None
    if args.chunks:
        chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8") if l.strip()]
        if len(chunks) != 1758:
            die("chunks = %d, attendu 1758" % len(chunks))
        chunk_alnum = [alnum(c["text"]) for c in chunks]

    dA, locA, key = describe(m07, normrel, A, chunks, chunk_alnum)
    dB, locB, _ = describe(m07, normrel, B, chunks, chunk_alnum)
    e26A, x26A, s26A, n26 = edge_table(m07, A, REF26)
    e26B, x26B, s26B, _ = edge_table(m07, B, REF26)
    e34A, x34A, s34A, n34 = edge_table(m07, A, REF34)
    e34B, x34B, s34B, _ = edge_table(m07, B, REF34)

    got = {"n": dA["n"], "tier1": dA["tiers"].get(1, 0), "tier2": dA["tiers"].get(2, 0),
           "exact26": x26A, "exact34": x34A}
    if got != EXPECT_A:
        die("le graphe A ne reproduit pas run16E publie : %s au lieu de %s" % (got, EXPECT_A))
    if chunks is not None and dA["located"] != EXPECT_A_LOC:
        die("localisation A %s au lieu de %s" % (dA["located"], EXPECT_A_LOC))

    # recouvrement
    kA = {}
    for t in A:
        kA.setdefault(key(t), t)
    kB = {}
    for t in B:
        kB.setdefault(key(t), t)
    common = set(kA) & set(kB)
    trans = Counter("T%s->T%s" % (kA[k].get("tier"), kB[k].get("tier")) for k in common)
    lost = sorted(set(kA) - set(kB))
    new = sorted(set(kB) - set(kA))
    new_tiers = Counter(kB[k].get("tier") for k in new)

    out.mkdir(parents=True)

    def status_rows(eA, eB):
        rows = []
        for e in eA:
            a, b = eA[e], eB[e]
            if a == "absent" and b != "absent":
                change = "gagnee"
            elif a != "absent" and b == "absent":
                change = "perdue"
            elif a == "absent":
                change = "absente_des_deux"
            else:
                change = "gardee"
            rows.append([e[0], e[1], e[2], a, b, change])
        return rows

    rows26, rows34 = status_rows(e26A, e26B), status_rows(e34A, e34B)
    for name, rows in (("edges_26.tsv", rows26), ("edges_34.tsv", rows34)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["subject", "relation", "object", "run16E", "run17", "changement"])
            w.writerows(rows)

    # preuves des aretes gagnees ou en sous-chaine seulement (26 puis 34, sans doublon)
    idxB = {id(t): i for i, t in enumerate(B)}
    seen, ev_rows = set(), []
    for rows in (rows26, rows34):
        for s, r, o, a, b, ch in rows:
            e = (s, r, o)
            if e in seen or not (ch == "gagnee" or b == "substring_only"):
                continue
            seen.add(e)
            for t in carriers(m07, normrel, B, e, exact_only=(b == "exact")):
                m4 = t.get("m4") or {}
                st, docs = locB.get(idxB[id(t)], ("non_verifie", []))
                ev_rows.append([s, r, o, b, t.get("subject"), t.get("relation"), t.get("object"),
                                t.get("tier"), t.get("origin"), m4.get("decision"),
                                m4.get("blind_verdict"), m4.get("evidence_verdict"),
                                st, " | ".join(d[:60] for d in docs), t.get("evidence")])
    with open(out / "gained_edges_evidence.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_subject", "ref_relation", "ref_object", "match_run17",
                    "subject", "relation", "object", "tier", "origin", "m4_decision",
                    "m4_blind", "m4_evidence", "located", "documents", "evidence"])
        w.writerows(ev_rows)

    with open(out / "lost_relations.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["subject", "relation", "object", "tier_run16E", "evidence"])
        for k in lost:
            t = kA[k]
            w.writerow([t.get("subject"), t.get("relation"), t.get("object"), t.get("tier"), t.get("evidence")])

    report = {
        "inputs": {"a": [args.a, sha(args.a)], "b": [args.b, sha(args.b)],
                   "chunks": [args.chunks, sha(args.chunks)] if args.chunks else None},
        "guard_run16E": "OK (232, 160/72, 13/26, 18/34" + (", 164/64/4)" if chunks is not None else ")"),
        "run16E": dA, "run17": dB,
        "overlap": {"common": len(common), "only_run16E": len(lost), "only_run17": len(new),
                    "tier_transitions_common": dict(trans),
                    "only_run17_by_tier": dict(new_tiers)},
        "recall": {
            "26": {"run16E_exact": x26A, "run16E_substring": s26A,
                   "run17_exact": x26B, "run17_substring": s26B, "n": n26,
                   "changes": dict(Counter(r[5] for r in rows26))},
            "34": {"run16E_exact": x34A, "run16E_substring": s34A,
                   "run17_exact": x34B, "run17_substring": s34B, "n": n34,
                   "changes": dict(Counter(r[5] for r in rows34))},
        },
    }
    json.dump(report, open(out / "report.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)

    # resume lisible
    L = []
    L.append("=== 171 comparaison run16E | run17 (memes mesures, meme code) ===")
    L.append("garde-fou run16E : " + report["guard_run16E"])
    L.append("%-34s %10s %10s" % ("", "run16E", "run17"))
    L.append("%-34s %10d %10d" % ("relations", dA["n"], dB["n"]))
    L.append("%-34s %10d %10d" % ("Tier 1", dA["tiers"].get(1, 0), dB["tiers"].get(1, 0)))
    L.append("%-34s %10d %10d" % ("Tier 2", dA["tiers"].get(2, 0), dB["tiers"].get(2, 0)))
    L.append("%-34s %10d %10d" % ("entites", dA["entities"], dB["entities"]))
    L.append("%-34s %10s %10s" % ("rappel 26 exact (papier)", "%d/%d" % (x26A, n26), "%d/%d" % (x26B, n26)))
    L.append("%-34s %10s %10s" % ("rappel 26 sous-chaine", "%d/%d" % (s26A, n26), "%d/%d" % (s26B, n26)))
    L.append("%-34s %10s %10s" % ("rappel 34 exact (papier)", "%d/%d" % (x34A, n34), "%d/%d" % (x34B, n34)))
    L.append("%-34s %10s %10s" % ("rappel 34 sous-chaine", "%d/%d" % (s34A, n34), "%d/%d" % (s34B, n34)))
    L.append("%-34s %10s %10s" % ("phrase avec les 2 arguments",
                                  dA["args_in_evidence"].get("both", 0), dB["args_in_evidence"].get("both", 0)))
    if chunks is not None:
        for s in ("exact", "partial", "not_found"):
            L.append("%-34s %10d %10d" % ("phrase localisee : " + s, dA["located"].get(s, 0), dB["located"].get(s, 0)))
        L.append("%-34s %10s %10s" % ("chunk localise avec les 2 arguments",
                                      dA["args_in_located_chunk"].get("both", 0), dB["args_in_located_chunk"].get("both", 0)))
    L.append("recouvrement : communes %d, seulement run16E %d, seulement run17 %d" % (len(common), len(lost), len(new)))
    L.append("changements de tier (communes) : " + json.dumps(dict(trans)))
    L.append("nouvelles relations run17 par tier : " + json.dumps({str(k): v for k, v in new_tiers.items()}))
    L.append("aretes 26 : " + json.dumps(report["recall"]["26"]["changes"], ensure_ascii=False))
    for r in rows26:
        if r[5] in ("gagnee", "perdue") or r[4] == "substring_only":
            L.append("   %-8s %-15s %s --%s--> %s" % (r[5], r[4], r[0], r[1], r[2]))
    L.append("aretes 34 : " + json.dumps(report["recall"]["34"]["changes"], ensure_ascii=False))
    L.append("fichiers : report.json, edges_26.tsv, edges_34.tsv, gained_edges_evidence.tsv, lost_relations.tsv")
    L.append("[fait] " + str(out))
    txt = "\n".join(L)
    (out / "_resume.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt, flush=True)


if __name__ == "__main__":
    main()