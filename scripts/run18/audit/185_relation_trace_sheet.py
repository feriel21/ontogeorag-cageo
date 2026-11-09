#!/usr/bin/env python3
"""185 — Trace sheet of one run18 relation, from the query to the delivered graph.

READ-ONLY. CPU. No model is called (the Qwen tokenizer can be loaded to show the
exact text after the chat template).
Writes only into --outdir (refuses to overwrite).

Each block of the sheet states its origin:
  [TEXT]      text of the paper as stored in the index (lowercase, SYNONYM_MAP of 49)
  [CONFIG]    configuration file of the run (queries, schema)
  [CODE]      value produced or changed by a script, with the file and the function
  [MODEL]     output of an LLM, read from a file of the run
  [REGENERATED] not logged by the run; recomputed here with the same function and the
              same inputs; the sheet states how the regeneration is checked
  [INFERENCE] deduced, not recorded by the run
  [MISSING]   no file of the run holds it; nothing is rebuilt

USAGE (root of the run18 worktree, environment active)
  cd ~/ontogeorag_run18 && export PYTHONPATH=$PWD
  python 185_relation_trace_sheet.py --repo . --run output/run18 \
      --subject earthquake --relation triggers --object "slope failure" \
      --outdir output/audit/185_trace_earthquake
  option: --tokenizer <local folder of Qwen2.5-7B-Instruct> to show the exact text
          after the chat template (otherwise the script looks for it in _steps.jsonl)
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
    print("STOP: " + msg, file=sys.stderr, flush=True)
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


NFKC = False   # --nfkc: ligatures (ﬁ, ﬂ...) and compatibility forms are mapped to ASCII before comparison


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
        return ("empty" if not e else "not_found"), []
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
    ap.add_argument("--max-chars", type=int, default=None, help="otherwise read from _steps.jsonl, otherwise 12000")
    ap.add_argument("--nfkc", action="store_true",
                    help="normalise ligatures before searching the evidence (rule of 145 otherwise)")
    ap.add_argument("--no-sha-check", action="store_true", help="for tests only")
    args = ap.parse_args()
    global NFKC
    NFKC = args.nfkc

    repo, run, out = Path(args.repo), Path(args.run), Path(args.outdir)
    if out.exists():
        die("the folder already exists, nothing is overwritten: " + str(out))
    sys.path.insert(0, str(repo.resolve()))
    sys.path.insert(0, str((repo / "m4").resolve()))
    for f, s12 in ((EXTRACT, EXTRACT_SHA12), (CLEAN, CLEAN_SHA12)):
        if not (repo / f).exists():
            die("not found: " + f)
        if not args.no_sha_check and sha(repo / f)[:12] != s12:
            die("%s is not the run18 version (sha %s)" % (f, sha(repo / f)[:12]))

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

    # ---------- relation in the fusion
    kg = json.load(open(run / "kg/tiered_kg_run18.json", encoding="utf-8"))["triples"]
    recs = [r for r in kg if (r["subject"], r["relation"], r["object"]) == key]
    if len(recs) != 1:
        die("relation %s found %d times in tiered_kg_run18.json" % (key, len(recs)))
    rec = recs[0]
    P("# run18 trace sheet: %s — %s — %s\n" % key)
    P("Fused relation: tier %s, origin `%s`, kept verdict `%s`." % (rec.get("tier"), rec.get("origin"),
                                                                               rec.get("verdict")))
    P("For a relation found in both passes, 06 keeps the record of pass a "
      "(loop over `idx_a`), so the sheet follows pass %s." % ("b" if rec.get("origin") == "iter_b" else "a"))
    Ppass = "b" if rec.get("origin") == "iter_b" else "a"
    d = run / ("pass_" + Ppass)

    # exact record used by 06: build_index on lexicon_enforced
    enf = jl(d / "canonical_triples_v5.rulecanon.lexicon_enforced.jsonl")
    idx = F.build_index(enf)
    if key not in idx:
        die("key missing from lexicon_enforced of pass " + Ppass)
    t06 = idx[key]["raw"]
    prov = t06.get("_provenance") or {}
    ver6 = t06.get("_verification") or {}

    # original line in verified_triples (replay of the normalisations of 04)
    verified = jl(d / "verified_triples.jsonl")
    raw = jl(d / "raw_triples.jsonl")
    cmap = json.load(open(d / "canonical_map_v5.json", encoding="utf-8"))
    # 04 replaces an out-of-vocabulary descriptor by the FIRST descriptor of a Python set
    # that contains it: with several candidates, the choice depends on PYTHONHASHSEED.
    # To find the line again, the object of the relation is put first in the order.
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
        die("original line not found in verified_triples.jsonl")
    li = cand[0]
    P("Original line: pass %s, line %d of `verified_triples.jsonl` and `raw_triples.jsonl` "
      "(%d candidate line(s); 04 keeps the first)." % (Ppass, li, len(cand)))
    v = verified[li]
    r0 = raw[li]
    if (r0.get("source"), r0.get("relation"), r0.get("target")) != (v.get("source"), v.get("relation"), v.get("target")):
        die("raw and verified do not match at line %d" % li)

    # ---------- 1. query
    H("1. The query")
    Q = jl(repo / QUERIES)
    qi = [i for i, q in enumerate(Q) if str(q.get("query", "")).strip() == str(prov.get("query", "")).strip()]
    q = Q[qi[0]] if qi else {}
    block("[CONFIG] %s, line %s (0 = first)" % (QUERIES, qi[0] if qi else "?"), json.dumps(q, ensure_ascii=False))
    P("Strategy `%s` selects the prompt template (`get_prompt`, 02); a strategy missing from `PROMPTS` "
      "takes the `descriptor` template. [CODE]" % q.get("strategy"))

    # ---------- 2. passages
    H("2. The five selected passages")
    chunks = {c["chunk_id"]: c for c in jl(run / "index/chunks.jsonl")}
    sel = prov.get("selected_chunks") or []
    P("Order = order after reranking (cross-encoder), then `candidates[:5]`. [CODE]")
    P("BM25 score of the first candidate: %s, threshold %s. [CODE]" % (prov.get("best_bm25"), prov.get("bm25_threshold")))
    P("[MISSING] The BM25 and reranking scores of each passage and the 15 other candidates are not "
      "written by 02.")
    missing.append("BM25 / reranking scores per passage and candidates 6 to 20 (not written by 02)")
    for k, cid in enumerate(sel, 1):
        c = chunks.get(cid)
        if c is None:
            P("[MISSING] passage %s not in the index" % cid)
            missing.append("text of passage " + cid)
            continue
        block("[TEXT] passage %d: %s (paper %s, %d characters)" % (k, cid, c.get("doc_id"), len(c["text"])),
              c["text"])
    P("The text is the one of the index: lowercase, with `SYNONYM_MAP` of 49 already applied "
      "(e.g. mtd -> mass transport deposit). [CODE]")

    # ---------- 3. prompt
    H("3. The prompt sent to Qwen")
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
        P("`--max-chars` not found in `_steps.jsonl`: 12000 (value of the job). [CONFIG]")
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
    P("[REGENERATED] The prompt is not logged. We recompute it with `concat_chunks` and `get_prompt` of "
      "`%s`, the 5 passages above, `max_chars=%d`, and the lists of the schema." % (EXTRACT, max_chars))
    P("Check: the first 1,500 characters of the regenerated context %s the `context_preview` stored by 02."
      % ("are IDENTICAL to" if ok_prev else "DIFFER from"))
    P("Header of each passage: `[CHUNK i | %s]` — `source_file` comes from the index; when it is missing, the code writes "
      "`unknown`. [CODE]" % ", ".join(sorted({d_["source_file"] for d_ in sel_dicts})))
    P("Schema lists passed on: relations = `%s`, types = `%s` (read from the keys `relations` and `types`). "
      "[CODE]" % (", ".join(relations) or "(empty)", ", ".join(types) or "(empty)"))
    P("Prompt length: %d characters. Sent as a user message, without a system message "
      "(`generate = lambda prompt: _gen(\"\", prompt, ...)`). [CODE]" % len(prompt))
    block("[REGENERATED] prompt (user message)", prompt)
    if model_path:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path)
            full = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True)
            head = full[:full.find(prompt[:80])] if prompt[:80] in full else full[:400]
            block("[REGENERATED] text added by the Qwen chat template before the message", head)
            P("This template comes from the Qwen tokenizer (%s), not from the pipeline code. [CODE]" % model_path)
        except Exception as e:
            P("Chat template not shown (%s)." % e)
    else:
        P("Qwen chat template not shown: pass `--tokenizer <model folder>`.")
    P("Generation parameters: temperature %s, `top_p=0.9` when temperature > 0, `repetition_penalty=1.1`, "
      "`max_new_tokens` = `--max-tokens` of the job. [CODE]" % ("0.0" if Ppass == "a" else "0.3"))

    # ---------- 4. raw answer
    H("4. The raw answer of Qwen")
    P("[MISSING] 02 does not write the answer of the model: it keeps only the triples read by `parse_response`. "
      "No file of the run holds it. Nothing is rebuilt here.")
    missing.append("raw answer of Qwen at extraction (not written by 02)")

    # ---------- 5. triple after parsing
    H("5. The triple after parsing")
    vis = {k: r0.get(k) for k in ("source", "source_type", "relation", "target", "target_type", "salience")}
    block("raw_triples.jsonl, pass %s, line %d" % (Ppass, li), json.dumps(vis, ensure_ascii=False, indent=1))
    P("- `source`, `target`, `source_type`, `target_type`: written by Qwen, copied as they are. [MODEL]")
    P("- `relation`: already replaced by `normalize_relation` (constants) in 02; the label written by Qwen "
      "is not kept. [CODE] [MISSING for the original label]")
    P("- `salience`: written by Qwen, or set to `common` by 02 when missing for a hasDescriptor; the file "
      "does not say which. [MODEL or CODE]")
    missing.append("relation label written by Qwen (replaced in 02)")
    block("[CODE] _provenance (without the texts)", json.dumps({k: v_ for k, v_ in (r0.get("_provenance") or {}).items()
                                                             if k not in ("context_preview",)}, ensure_ascii=False,
                                                            indent=1))

    # ---------- 6. normalisations
    H("6. The normalisations applied")
    t = copy.deepcopy(v)
    tn0 = M.normalize_entity(v.get("target_norm", v.get("target", "")))
    amb0 = [x for x in sorted(KD_SET) if tn0 not in KD_SET and (x in tn0 or tn0 in x)]
    if len(amb0) > 1:
        P("Ambiguous object: `%s` contains several descriptors (%s). 04 takes the first item of a Python set, "
          "whose order depends on PYTHONHASHSEED; the run kept `%s`. This choice is not reproducible. [CODE]"
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
    rows.append(("04 canonical_map_v5 %s" % (used or "(no rule)"),
                 "%s | %s | %s" % (t.get("source"), t.get("relation"), t.get("target"))))
    logp = d / "canonical_triples_v5.rulecanon_log.csv"
    rlog = {}
    if logp.exists():
        import csv
        rlog = {x["from"]: x["to"] for x in csv.DictReader(open(logp, encoding="utf-8"))}
    hit5 = [kf for kf in ("subject", "object") if t.get(kf) and str(t[kf]).strip() in rlog]
    rows.append(("05a (reads subject/object; fields present: %s)" % [k_ for k_ in ("subject", "object") if k_ in t],
                 "substitution: %s" % (hit5 or "none")))
    rows.append(("06 normalize_entity / normalize_relation", "%s | %s | %s" % F.triple_key(t)))
    P("| Step (file, function) | Value after the step |\n|---|---|")
    for a_, b_ in rows:
        P("| %s | %s |" % (a_, b_))
    P("\nThe code produces all these values [CODE], from the names written by Qwen [MODEL].")
    try:
        enfkg = json.load(open(run / "kg_enforced/tiered_kg_enforced.json", encoding="utf-8"))
        allr = enfkg.get("tier1", []) + enfkg.get("tier2", []) if isinstance(enfkg, dict) else enfkg
        e4b = [x for x in allr if (normalize_entity(x.get("subject", "")), normalize_relation(x.get("relation", "")),
                                   normalize_entity(x.get("object", ""))) == key]
        if e4b:
            P("Types: Qwen `%s` / `%s` -> 04b `%s` / `%s` (TYPE_MAP of 04b). [MODEL] then [CODE]" % (
                rec.get("subject_type"), rec.get("object_type"), e4b[0].get("subject_type"),
                e4b[0].get("object_type")))
    except FileNotFoundError:
        P("[MISSING] kg_enforced/tiered_kg_enforced.json")

    # ---------- 7. verification
    H("7. The passage and the verdict that kept it")
    ver = v.get("_verification") or {}
    top = [c.get("text", "") for c in (v.get("_provenance") or {}).get("top_chunks", [])]
    block("[MODEL] verdict of Qwen (verified_triples.jsonl, line %d)" % li,
          json.dumps({k: ver.get(k) for k in ("verdict", "evidence", "reasoning", "chunks_checked", "model")},
                     ensure_ascii=False, indent=1))
    aud = d / "verification_audit.jsonl"
    araw = None
    if aud.exists():
        al = jl(aud)
        if li < len(al) and al[li].get("index") == li:
            araw = al[li].get("raw")
    if araw is not None:
        block("[MODEL] raw answer of Qwen for the kept passage (verification_audit.jsonl, index %d)" % li, araw)
    else:
        P("[MISSING] raw answer of the verification (verification_audit.jsonl missing or without a raw field).")
        missing.append("raw answer of the verification")
    P("[MISSING] 03 logs only the answer for the best passage, not the four others, nor the number of the "
      "kept passage. The code keeps the first passage that reaches the best verdict (`score > best_score`).")
    missing.append("verification answers for the 4 other passages and number of the kept passage")
    st, where = locate(ver.get("evidence", ""), top)
    P("[INFERENCE] The `evidence` sentence is found (%s) in passage(s) %s. Rules of script 145: "
      "comparison on lowercase alphanumeric characters; partial = one of the three 40-character windows." % (
          st, [i + 1 for i in where] or "none"))
    for i in where[:1]:
        cid = ((v.get("_provenance") or {}).get("top_chunks") or [{}])[i].get("chunk_id")
        chunk_chars = 2000
        block("[TEXT] passage %d (%s) as read by the verifier (cut at %d characters)" % (i + 1, cid,
                                                                                                    chunk_chars),
              top[i][:chunk_chars])
        rel = normalize_relation(str(v.get("relation_norm", v.get("relation", ""))))
        user = V.COT_PROMPT.format(chunk_text=top[i][:chunk_chars], source=v.get("source_norm", v.get("source", "")),
                                   relation=rel, gloss=V.RELATION_GLOSSES.get(rel, "Subject %s Object" % rel)
                                   if hasattr(V, "RELATION_GLOSSES") else rel,
                                   target=v.get("target_norm", v.get("target", "")))
        block("[REGENERATED] system message of the verification (03, SYSTEM)", V.SYSTEM)
        block("[REGENERATED] user message of the verification for this passage (03, COT_PROMPT)", user)
    P("Decision of 04 on the verdict: `%s` (normal policy: STRONG and WEAK kept, NO_CHUNK and UNPARSEABLE "
      "kept too). [CODE]" % reason_v)
    P("Tier at fusion: `verdict_to_tier(%s)` = %s; tier of the fused relation = %s, origin `%s`. [CODE]" % (
        ver.get("verdict"), F.verdict_to_tier(ver.get("verdict", "")), rec.get("tier"), rec.get("origin")))

    # ---------- 8. panel and delivered graph
    H("8. Panel and delivered graph")
    passage_j = J.evidence_from_provenance(rec)
    block("[CODE] passage given to the judges (m4_verify.evidence_from_provenance on the fused record; "
          "06 cut evidence at 400 characters)", passage_j)

    def nk(s, r, o):
        return (str(s).strip().lower(), str(r).strip(), str(o).strip().lower())
    for jn in ("llama", "mistral"):
        p = run / "m4" / jn / "m4_verdicts.jsonl"
        if not p.exists():
            P("[MISSING] " + str(p))
            missing.append(str(p))
            continue
        jr = [x for x in jl(p) if nk(x["subject"], x["relation"], x["object"]) == nk(*key)]
        if not jr:
            P("[MISSING] relation not in " + str(p))
            continue
        x = jr[0]
        block("[MODEL] %s, blind pass: verdict %s (answer cut at 800 characters by m4_verify)" % (
            jn, x["blind"].get("verdict")), x["blind"].get("raw", ""))
        block("[MODEL] %s, pass with sentence: verdict %s (answer cut at 800 characters)" % (
            jn, x["evidence"].get("verdict")), x["evidence"].get("raw", ""))
    dec = [x for x in jl(run / "m4_panel/m4_panel_decisions.jsonl")
           if nk(x["subject"], x["relation"], x["object"]) == nk(*key)]
    if dec:
        P("Panel decision: `%s`. [CODE]" % dec[0].get("m4_decision"))
    flat = json.load(open(run / "kg_final/flat_run18_enforced_dedup.json", encoding="utf-8"))
    flat = flat["triples"] if isinstance(flat, dict) else flat
    fr = [x for x in flat if nk(x.get("subject"), x.get("relation"), x.get("object")) == nk(*key)]
    if fr:
        block("[CODE] delivered record (flat_run18_enforced_dedup.json)", json.dumps(fr[0], ensure_ascii=False,
                                                                                        indent=1))
    else:
        P("Relation not in the delivered graph (quarantined or removed).")

    H("What the run files do not hold")
    for m_ in dict.fromkeys(missing):
        P("- " + m_)

    out.mkdir(parents=True)
    slug = re.sub(r"[^a-z0-9]+", "_", "_".join(key).lower()).strip("_")
    (out / ("trace_%s.md" % slug)).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\n[done] " + str(out / ("trace_%s.md" % slug)))


if __name__ == "__main__":
    main()