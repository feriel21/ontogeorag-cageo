#!/usr/bin/env python3
"""
58v2_freeze_benchmark.py — Sortir le benchmark du code, sans modifier le code

Corrige v1 : `norm()` supposait des triplets stricts. Au moins un des fichiers
benchmark porte des elements d'une autre forme (dicts, ou listes de longueur
!= 3). v2 accepte :
  - [s, r, o] et tout iterable de longueur >= 3 (les 3 premiers champs)
  - {"subject"/"source"/"s", "relation"/"r"/"rel", "object"/"target"/"o"}
et nomme le fichier fautif si la forme reste inconnue.

Sous-commande `inspect` ajoutee : affiche la forme reelle de chaque fichier
avant toute tentative de gel.

PRINCIPE INCHANGE
    LB_REFERENCE_EDGES reste code en dur dans 04 (le retirer casserait la
    reproductibilite de run15-frozen). On ne le retire pas : on le rend
    VERIFIABLE. `check` sort en code 1 a la moindre divergence.

Usage :
    python 58v2_freeze_benchmark.py inspect
    python 58v2_freeze_benchmark.py build --out configs/benchmark/benchmark_v1.json
    python 58v2_freeze_benchmark.py check --benchmark configs/benchmark/benchmark_v1.json
"""
import argparse, ast, hashlib, json, subprocess, sys, tempfile
from pathlib import Path

TAG = "run15-frozen"
S_KEYS = ("subject", "source", "s", "head", "from")
R_KEYS = ("relation", "rel", "r", "predicate", "p")
O_KEYS = ("object", "target", "o", "tail", "to")


def git_show(tag, path):
    out = subprocess.run(["git", "show", f"{tag}:{path}"],
                         capture_output=True, text=True, check=True).stdout
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
    f.write(out); f.close()
    return f.name


def ast_const(path, name):
    for n in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        hit = ((isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
               or (isinstance(n, ast.AnnAssign)
                   and isinstance(n.target, ast.Name) and n.target.id == name))
        if hit and getattr(n, "value", None) is not None:
            return ast.literal_eval(n.value)
    return None


def _pick(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def as_edge(x, where):
    """Ramene un element quelconque a (s, r, o) ou leve avec un message utile."""
    if isinstance(x, dict):
        s, r, o = _pick(x, S_KEYS), _pick(x, R_KEYS), _pick(x, O_KEYS)
        if None not in (s, r, o):
            return s, r, o
        raise ValueError(f"{where}: dict sans cles (s,r,o) reconnues -> {sorted(x)[:8]}")
    if isinstance(x, (list, tuple)):
        if len(x) >= 3:
            return x[0], x[1], x[2]
        raise ValueError(f"{where}: sequence de longueur {len(x)} -> {x}")
    raise ValueError(f"{where}: type inattendu {type(x).__name__} -> {str(x)[:80]}")


def norm(edges, where="<code>"):
    out = []
    for x in edges:
        s, r, o = as_edge(x, where)
        out.append([str(s).lower().strip(), str(r).strip(), str(o).lower().strip()])
    return sorted(out)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()


def raw_list(p):
    """Renvoie la liste d'aretes brute d'un fichier JSON, ou None."""
    if not Path(p).exists():
        return None
    d = json.load(open(p, encoding="utf-8"))
    if isinstance(d, dict):
        for k in ("edges", "reference_edges", "benchmark", "triples", "items"):
            if k in d and isinstance(d[k], list):
                return d[k]
        return None
    return d if isinstance(d, list) else None


def cmd_inspect(a):
    for label, path in a.files:
        lst = raw_list(path)
        if lst is None:
            print(f"  {label:16s} {path:44s} ABSENT ou forme non listee")
            continue
        first = lst[0] if lst else None
        kind = type(first).__name__
        extra = (sorted(first)[:8] if isinstance(first, dict)
                 else (f"len={len(first)}" if isinstance(first, (list, tuple)) else ""))
        print(f"  {label:16s} {path:44s} n={len(lst):3d}  premier={kind} {extra}")
        print(f"      {json.dumps(first, ensure_ascii=False)[:160]}")


def cmd_build(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"ERREUR : {out} existe deja. Refus d'ecraser.")
    out.parent.mkdir(parents=True, exist_ok=True)

    p04 = git_show(a.tag, "pipeline/04_clean_validate.py")
    in_code = norm(ast_const(p04, "LB_REFERENCE_EDGES") or [], f"{a.tag}:04")

    sets = {"lb26_from_code": {"edges": in_code, "n": len(in_code),
                               "source": f"{a.tag}:pipeline/04_clean_validate.py"}}
    skipped = {}
    for label, path in a.files:
        lst = raw_list(path)
        if lst is None:
            skipped[label] = f"{path} absent ou forme non reconnue"
            continue
        try:
            sets[label] = {"edges": norm(lst, path), "n": len(lst), "source": path}
        except ValueError as e:
            skipped[label] = str(e)

    for v in sets.values():
        v["sha256"] = sha(v["edges"])

    checks = {}
    s26 = {tuple(e) for e in sets["lb26_from_code"]["edges"]}
    if "lb26_original" in sets:
        checks["code_matches_lb26_original"] = (
            sets["lb26_from_code"]["sha256"] == sets["lb26_original"]["sha256"])
    if "extended34" in sets:
        s34 = {tuple(e) for e in sets["extended34"]["edges"]}
        checks["26_subset_of_34"] = s26 <= s34
        checks["n_added_by_extension"] = len(s34 - s26)
        checks["missing_from_34"] = sorted(list(e) for e in (s26 - s34))

    doc = {"_meta": {"produced_by": "58v2_freeze_benchmark.py", "code_tag": a.tag,
                     "usage": "Source de verite du benchmark. Cite par SHA dans le manuscrit.",
                     "skipped": skipped},
           "sets": sets, "consistency": checks}
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Ecrit : {out}")
    for k, v in sets.items():
        print(f"  {k:18s} n={v['n']:3d}  sha={v['sha256'][:12]}  <- {v['source']}")
    for k, v in skipped.items():
        print(f"  [ignore] {k}: {v}")
    for k, v in checks.items():
        print(f"  [check] {k} = {v}")


def cmd_check(a):
    doc = json.load(open(a.benchmark, encoding="utf-8"))
    frozen = doc["sets"]["lb26_from_code"]
    now = norm(ast_const(git_show(a.tag, "pipeline/04_clean_validate.py"),
                         "LB_REFERENCE_EDGES") or [], f"{a.tag}:04")
    ok = sha(now) == frozen["sha256"]
    print(f"benchmark gele     : {frozen['sha256'][:12]}  (n={frozen['n']})")
    print(f"constante @{a.tag} : {sha(now)[:12]}  (n={len(now)})")
    if ok:
        print("CONFORME — la constante en dur est identique au benchmark gele.")
    else:
        fs, ns = {tuple(e) for e in frozen["edges"]}, {tuple(e) for e in now}
        print("DIVERGENCE")
        for e in sorted(fs - ns):
            print("   manquante dans le code :", e)
        for e in sorted(ns - fs):
            print("   ajoutee dans le code   :", e)

    if a.emit_patch:
        Path(a.emit_patch).write_text(
            "\n# --- run16 : benchmark lu depuis le fichier gele, plus en dur ---\n"
            "import json as _json\nfrom pathlib import Path as _Path\n"
            "_BM = _json.load(open(_Path(__file__).resolve().parents[1]\n"
            "                      / 'configs/benchmark/benchmark_v1.json',\n"
            "                      encoding='utf-8'))\n"
            "LB_REFERENCE_EDGES = [tuple(e) for e in _BM['sets']['lb26_from_code']['edges']]\n",
            encoding="utf-8")
        print(f"\nPatch run16 ecrit dans {a.emit_patch} (branche separee, PAS run15-frozen).")
    sys.exit(0 if ok else 1)


def add_files(p):
    p.add_argument("--lb26", default="configs/lb_26_original.json")
    p.add_argument("--ext", default="configs/lb_extended_benchmark.json")
    p.add_argument("--dev", default="configs/lb_dev.json")
    p.add_argument("--test", default="configs/lb_test.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect"); i.set_defaults(fn=cmd_inspect); add_files(i)
    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build); add_files(b)
    b.add_argument("--out", required=True); b.add_argument("--tag", default=TAG)
    c = sub.add_parser("check"); c.set_defaults(fn=cmd_check)
    c.add_argument("--benchmark", required=True); c.add_argument("--tag", default=TAG)
    c.add_argument("--emit-patch", default=None); add_files(c)
    a = ap.parse_args()
    a.files = [("lb26_original", a.lb26), ("extended34", a.ext),
               ("dev", a.dev), ("test", a.test)]
    a.fn(a)