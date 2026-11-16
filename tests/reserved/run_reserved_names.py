#!/usr/bin/env python3
"""Keep semantic.c's PY_RESERVED list in sync with what codegen emits.

Compiles every corpus program (with and without --emit-trace), walks the
generated Python's AST, and requires every name the *compiler* introduces --
i.e. not a user dim/tensor/rank variable -- to be either '_'-prefixed, or
listed in PY_RESERVED. A new
codegen temporary that users could collide with fails this test until it is
added to the list.

Usage: run_reserved_names.py <edgecc>
"""
import ast
import glob
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reserved_list():
    src = open(os.path.join(REPO, "src", "semantic.c")).read()
    block = src[src.index("/* BEGIN PY_RESERVED */"):src.index("/* END PY_RESERVED */")]
    block = re.sub(r"/\*.*?\*/", "", block, flags=re.S)
    return set(re.findall(r'"([^"]+)"', block))


def user_names(edge_src):
    code = re.sub(r"//[^\n]*", "", edge_src)
    names = set(re.findall(r"\bdim\s+([A-Za-z_]\w*)", code))
    names |= set(re.findall(r"\btensor\s+([A-Za-z_]\w*)", code))
    for line in code.split(";"):
        if "::" not in line:
            continue
        for idx in re.findall(r"\w+\s*\[([^\]]*)\]", line.split("::")[0]):
            names |= set(re.findall(r"[A-Za-z_]\w*", idx))
    return names


def helper_globals(func):
    """Names a helper function reads from module or builtin scope: every name
    it loads that is neither a parameter nor bound inside it. Its locals cannot
    collide with user names, but these can: dims and loaded tensors are module
    globals, so a user dim `zip` would shadow the builtin the helper calls."""
    bound = {arg.arg for arg in ast.walk(func.args) if isinstance(arg, ast.arg)}
    loaded = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, ast.alias):
            bound.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return loaded - bound


class Collector(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_FunctionDef(self, node):
        self.names.add(node.name)
        if node.name.startswith("_edgecc"):
            self.names |= helper_globals(node)   # its locals cannot collide
            return
        self.generic_visit(node)

    def visit_Name(self, node):
        self.names.add(node.id)

    def visit_arg(self, node):
        self.names.add(node.arg)

    def visit_alias(self, node):
        self.names.add(node.asname or node.name.split(".")[0])


def main():
    edgecc = os.path.abspath(sys.argv[1])
    reserved = reserved_list()
    corpus = sorted(glob.glob(os.path.join(REPO, "examples", "*.edge"))
                    + glob.glob(os.path.join(REPO, "tests", "e2e", "*", "input.edge"))
                    + glob.glob(os.path.join(REPO, "tests", "e2e", "iterative", "*", "kernel.edge"))
                    + glob.glob(os.path.join(REPO, "tests", "reference", "kernels", "*.edge")))
    missing, compiled = {}, 0
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "gen.py")
        for path in corpus:
            src = open(path).read()
            users = user_names(src)
            for flags in ([], ["--emit-trace"]):
                p = subprocess.run([edgecc, path, "-o", out] + flags, capture_output=True, text=True)
                if p.returncode != 0:
                    continue
                compiled += 1
                c = Collector()
                c.visit(ast.parse(open(out).read()))
                for name in c.names - users:
                    if not name.startswith("_") and name not in reserved:
                        missing.setdefault(name, os.path.relpath(path, REPO))

    for name, where in sorted(missing.items()):
        print(f"  FAIL codegen emits '{name}' (e.g. for {where}) but it is not in PY_RESERVED (src/semantic.c)")
    print(f"reserved-name check: {compiled} generated programs, {len(missing)} unlisted names")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
