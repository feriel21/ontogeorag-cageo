#!/usr/bin/env python3
"""Check that an edited Python file does the same computation as its original.

Default (strict) mode
    The two syntax trees must be identical once docstrings are removed.
    Comments are not part of the syntax tree. Use this when only comments
    and docstrings were edited.

--allow-messages
    Also tolerates changes to human-readable text:
      * string literals inside print(...) calls,
      * help=, description= and metavar= texts of argparse,
      * the message of a raised exception,
      * print statements made only of literal text (added or removed).
    Every other string (paths, patterns, dictionary keys, file names) must
    stay identical, and so must every print that shows a computed value.

--allow-strings
    Also tolerates changes to the text of any string literal, including the literal
    parts of f-strings, file names, dictionary keys and column names. The expressions
    inside f-strings and every other node must stay identical. Use this when all the
    text of a script is translated. With --list-strings FILE, every changed string is
    written to FILE (tab-separated: file, original, edited) so that each change can be
    read; a string read from a file produced by another script must be checked there.

--allow-renames
    Also tolerates consistent renaming of identifiers (functions, variables,
    arguments) and of command-line flags together with their attribute on
    the parsed arguments (--racine / args.racine -> --root / args.root).
    A rename is accepted only if it is a one-to-one substitution throughout
    the file: merging or swapping two names is reported as a change.

Exit code 0 only if every compared pair is equivalent under the chosen mode.

Usage
    python tools/check_code_unchanged.py ORIGINAL.py EDITED.py
    python tools/check_code_unchanged.py --allow-messages --allow-renames ORIGINAL.py EDITED.py
    python tools/check_code_unchanged.py --tree ORIGINAL_DIR EDITED_DIR
"""
import argparse
import ast
import sys
from pathlib import Path

DOC_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
TEXT_KEYWORDS = {"help", "description", "metavar"}
BLANK = "<text>"


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, DOC_NODES) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]


def is_print(call):
    return isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "print"


def literal_only(nodes):
    """True if the expressions contain no name, call, attribute or subscript."""
    for expr in nodes:
        for sub in ast.walk(expr):
            if isinstance(sub, (ast.Name, ast.Call, ast.Attribute, ast.Subscript)):
                return False
    return True


def blank_strings(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            sub.value = BLANK
        elif isinstance(sub, ast.JoinedStr):
            sub.values = [ast.Constant(value=BLANK)]


def relax_messages(tree):
    # drop statements that only print literal text
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list):
                kept = [s for s in stmts
                        if not (isinstance(s, ast.Expr) and is_print(s.value)
                                and literal_only(s.value.args))]
                setattr(node, field, kept or [ast.Pass()])
    for node in ast.walk(tree):
        if is_print(node):
            for arg in node.args:
                blank_strings(arg)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in TEXT_KEYWORDS:
                    blank_strings(kw.value)
        elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            for arg in node.exc.args:
                blank_strings(arg)


def canonical_names(tree):
    """Replace identifiers and argparse flags by their order of first appearance."""
    flags = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    flags.add(arg.value[2:].replace("-", "_"))
    mapping = {}
    # keyword arguments are renamed only when the name is a parameter defined in the file,
    # so that a keyword of a library call (encoding=, indent=) can never be renamed
    params = {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}

    def canon(key):
        return mapping.setdefault(key, "id%d" % len(mapping))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = canon(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.name = canon(node.name)
        elif isinstance(node, ast.arg):
            node.arg = canon(node.arg)
        elif isinstance(node, ast.keyword) and node.arg in params:
            node.arg = canon(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            node.name = canon(node.name)
        elif isinstance(node, ast.Attribute) and node.attr in flags:
            node.attr = canon("flag:" + node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.startswith("--") and node.value[2:].replace("-", "_") in flags:
            node.value = canon("flag:" + node.value[2:].replace("-", "_"))
    return mapping


def string_constants(tree):
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def blank_all_strings(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = BLANK


def fingerprint(path, allow_messages, allow_renames, allow_strings=False, strings=None):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    strip_docstrings(tree)
    if allow_messages:
        relax_messages(tree)
    if allow_renames:
        canonical_names(tree)
    if allow_strings:
        if strings is not None:
            strings.extend(string_constants(tree))
        blank_all_strings(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs=2)
    ap.add_argument("--tree", action="store_true", help="compare every .py file of two directories")
    ap.add_argument("--allow-messages", action="store_true", help="tolerate translated or removed text")
    ap.add_argument("--allow-renames", action="store_true", help="tolerate consistent renaming")
    ap.add_argument("--allow-strings", action="store_true", help="tolerate changes to the text of any string")
    ap.add_argument("--list-strings", default=None, help="with --allow-strings, write the changed strings here")
    args = ap.parse_args()
    listed = []
    a, b = Path(args.paths[0]), Path(args.paths[1])

    pairs = [(a, b)]
    if args.tree:
        pairs = [(p, b / p.relative_to(a)) for p in sorted(a.rglob("*.py"))]
    changed = 0
    for original, edited in pairs:
        if not edited.is_file():
            print("MISSING       " + str(edited))
            changed += 1
            continue
        try:
            sa, sb = [], []
            same = (fingerprint(original, args.allow_messages, args.allow_renames, args.allow_strings, sa)
                    == fingerprint(edited, args.allow_messages, args.allow_renames, args.allow_strings, sb))
            if same and args.allow_strings:
                listed += [(str(edited), x, y) for x, y in zip(sa, sb) if x != y]
        except SyntaxError as exc:
            print("SYNTAX ERROR  %s: %s" % (edited, exc))
            same = False
        print(("equivalent    " if same else "CODE CHANGED  ") + str(edited))
        changed += not same
    if args.list_strings:
        seen = set()
        with open(args.list_strings, "w", encoding="utf-8") as f:
            f.write("file\toriginal\tedited\n")
            for row in listed:
                if row not in seen:
                    seen.add(row)
                    f.write("\t".join(x.replace("\t", "\\t").replace("\n", "\\n") for x in row) + "\n")
        print("%d changed strings written to %s" % (len(seen), args.list_strings))
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()