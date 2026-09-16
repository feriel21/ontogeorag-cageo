#!/usr/bin/env python3
"""Check that edited Python files differ from their originals only in comments and docstrings.

Comments are not part of the Python syntax tree, and docstrings are removed
before comparison. Two files are reported as equivalent when their syntax trees
are otherwise identical, so the edit cannot have changed what the code does.
Any other change, including to a printed message or a path, is reported.

Usage
    python tools/check_code_unchanged.py ORIGINAL.py EDITED.py
    python tools/check_code_unchanged.py --tree ORIGINAL_DIR EDITED_DIR
"""
import argparse
import ast
import sys
from pathlib import Path

DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, DOC_NODES) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def fingerprint(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    return ast.dump(strip_docstrings(tree), annotate_fields=True, include_attributes=False)


def compare(original, edited):
    try:
        return fingerprint(original) == fingerprint(edited)
    except SyntaxError as exc:
        print("SYNTAX ERROR  %s: %s" % (edited, exc))
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs=2)
    ap.add_argument("--tree", action="store_true", help="compare every .py file of two directories")
    args = ap.parse_args()
    a, b = Path(args.paths[0]), Path(args.paths[1])

    pairs = [(a, b)]
    if args.tree:
        pairs = [(p, b / p.relative_to(a)) for p in sorted(a.rglob("*.py"))]
    changed = 0
    for original, edited in pairs:
        if not edited.is_file():
            print("MISSING       " + str(edited))
            changed += 1
        elif compare(original, edited):
            print("equivalent    " + str(edited))
        else:
            print("CODE CHANGED  " + str(edited))
            changed += 1
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()