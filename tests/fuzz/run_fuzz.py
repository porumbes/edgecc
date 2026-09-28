#!/usr/bin/env python3
"""Token-mutation fuzzer for edgecc.

Mutates programs from the example/e2e/negative corpus and checks that edgecc
either rejects each one cleanly or emits Python that compiles (and, when the
program has no file I/O, runs without a codegen-bug exception).

Usage: run_fuzz.py <edgecc> [--cases N] [--seed S] [--only I]

Each case uses its own RNG seeded from "<seed>:<index>", so a failure is
replayed exactly with `--seed S --only I`.
"""
import argparse
import glob
import os
import random
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKEN_RE = re.compile(r"::|\.\d+|\w+|[^\s\w]|\s+")
JUNK = ["@", "#", "$", "-1", "999999999", "1e999", "\"x\"", "(", ")", "[", "]",
        ",", ";", ".", "=", "!", "::", ".0", ".3"]
# Exceptions that, raised by a generated program, point at a codegen bug
# rather than at legitimate program behavior (e.g. the generation cap).
CODEGEN_BUG_EXC = {"SyntaxError", "NameError", "UnboundLocalError",
                   "IndentationError", "AttributeError", "TypeError", "IndexError",
                   "KeyError", "ZeroDivisionError", "RecursionError"}
EDGECC_TIMEOUT = 10
RUN_TIMEOUT = 20


def load_corpus():
    """Return (valid programs, negative-test programs, vocabulary)."""
    valid = sorted(glob.glob(os.path.join(REPO, "examples", "*.edge"))
                   + glob.glob(os.path.join(REPO, "tests", "e2e", "*", "input.edge"))
                   + glob.glob(os.path.join(REPO, "tests", "reference", "kernels", "*.edge")))
    negative = sorted(glob.glob(os.path.join(REPO, "tests", "negative", "*", "input.edge")))
    read = lambda ps: [(os.path.relpath(p, REPO), open(p).read()) for p in ps]
    valid, negative = read(valid), read(negative)
    vocab = sorted({t for _, src in valid + negative for t in TOKEN_RE.findall(src) if not t.isspace()})
    return valid, negative, vocab + JUNK


def pick(valid, negative, rng):
    # Mutating a valid program is far likelier to reach codegen than mutating
    # an already-invalid one, so negatives are only an occasional source.
    return rng.choice(negative) if negative and rng.random() < 0.1 else rng.choice(valid)


def kind(tok):
    if re.fullmatch(r"\d+|\.\d+", tok):
        return "num"
    if re.fullmatch(r"\w+", tok):
        return "word"
    return "punct"


def mutate(src, vocab, rng):
    """Apply 1-3 token mutations to one line (biased toward Einsum lines)."""
    lines = src.split("\n")
    einsum = [i for i, l in enumerate(lines) if "::" in l]
    nonblank = [i for i, l in enumerate(lines) if l.strip()]
    li = rng.choice(einsum) if einsum and rng.random() < 0.7 else rng.choice(nonblank)
    toks = TOKEN_RE.findall(lines[li])
    for _ in range(rng.randint(1, 3)):
        spots = [i for i, t in enumerate(toks) if not t.isspace()]
        if not spots:
            break
        i = rng.choice(spots)
        op = rng.random()
        if op < 0.3:
            del toks[i]
        elif op < 0.6:
            toks.insert(i, " " + rng.choice(vocab) + " ")
        elif op < 0.9:
            # Mostly swap like for like (word/number/punct) so more mutants
            # still parse and reach semantic analysis and codegen.
            same = [t for t in vocab if kind(t) == kind(toks[i])]
            toks[i] = rng.choice(same if same and rng.random() < 0.7 else vocab)
        else:
            toks.insert(i, toks[i])
    lines[li] = "".join(toks)
    return "\n".join(lines), lines[li]


def check(edgecc, prog, workdir):
    """Return None if the oracle holds, else a failure description."""
    src = os.path.join(workdir, "case.edge")
    out = os.path.join(workdir, "case.py")
    with open(src, "w") as f:
        f.write(prog)
    if os.path.exists(out):
        os.remove(out)
    try:
        p = subprocess.run([edgecc, src, "-o", out], capture_output=True, encoding="utf-8", errors="replace",
                           text=True, timeout=EDGECC_TIMEOUT, cwd=workdir)
    except subprocess.TimeoutExpired:
        return "edgecc timed out"
    if "runtime error" in p.stderr:
        return "sanitizer: " + p.stderr.strip()[-300:]
    if p.returncode == 1:
        if "error" not in p.stderr.lower():
            return "rejected without an error message: " + p.stderr.strip()[-300:]
        return None
    if p.returncode != 0:
        return f"edgecc exited {p.returncode}: " + p.stderr.strip()[-300:]
    try:
        with open(out, "rb") as f:
            code = f.read().decode("utf-8")
        compile(code, out, "exec")
    except (UnicodeDecodeError, SyntaxError, ValueError) as e:
        return f"generated Python does not compile: {type(e).__name__}: {e}"
    if "load(" in prog or "save:" in prog:
        return None  # needs input files; compile check only
    try:
        r = subprocess.run([sys.executable, out], capture_output=True, encoding="utf-8", errors="replace",
                           text=True, timeout=RUN_TIMEOUT, cwd=workdir)
    except subprocess.TimeoutExpired:
        return None  # a legal program may simply be slow
    if r.returncode < 0:
        return f"generated program killed by signal {-r.returncode}"
    if r.returncode != 0:
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        exc = last.split(":", 1)[0].strip()
        if exc in CODEGEN_BUG_EXC:
            return "generated program raised " + last
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edgecc")
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--seed", default="1")
    ap.add_argument("--only", type=int, help="replay a single case index")
    args = ap.parse_args()
    edgecc = os.path.abspath(args.edgecc)  # cases run with cwd = a temp dir

    valid, negative, vocab = load_corpus()
    indices = [args.only] if args.only is not None else range(args.cases)
    failures, accepted = [], 0
    with tempfile.TemporaryDirectory() as workdir:
        for i in indices:
            rng = random.Random(f"{args.seed}:{i}")
            origin, src = pick(valid, negative, rng)
            prog, line = mutate(src, vocab, rng)
            why = check(edgecc, prog, workdir)
            if why:
                failures.append((i, origin, line, why))
            elif os.path.exists(os.path.join(workdir, "case.py")):
                accepted += 1
            if args.only is not None:
                print(prog)

    n = len(indices)
    for i, origin, line, why in failures:
        print(f"  FAIL case {i} (from {origin}): {why}")
        print(f"       mutated line: {line.strip()}")
        print(f"       replay: run_fuzz.py <edgecc> --seed {args.seed} --only {i}")
    print(f"{n - len(failures)}/{n} fuzz cases passed "
          f"(seed {args.seed}; {accepted} compiled, {n - accepted - len(failures)} rejected cleanly)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
