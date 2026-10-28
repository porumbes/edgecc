#!/usr/bin/env python3
"""End-to-end test driver for edgecc."""
import os
import subprocess
import sys
import tempfile

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

def run_test(test_dir, edgecc_bin):
    name = os.path.basename(test_dir)
    input_file = os.path.join(test_dir, "input.edge")
    expected_output = os.path.join(test_dir, "expected_output.txt")

    if not os.path.exists(input_file) or not os.path.exists(expected_output):
        return None

    # Tests that load/save .mtx files require scipy; skip cleanly if absent.
    if os.path.exists(os.path.join(test_dir, "needs_scipy")) and not HAVE_SCIPY:
        print(f"  SKIP {name}: scipy not installed")
        return None

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp_py = tmp.name

    try:
        result = subprocess.run(
            [edgecc_bin, input_file, "-o", tmp_py],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print(f"  FAIL {name}: edgecc failed: {result.stderr.strip()}")
            return False

        # Optional argv override: an `args` file supplies positional input paths
        # to the generated program. The token {DATA} resolves to the repo-root
        # data/ directory, so tests can source graphs from data/suitesparse/.
        extra_args = []
        args_file = os.path.join(test_dir, "args")
        if os.path.exists(args_file):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            data_dir = os.path.join(repo_root, "data")
            with open(args_file) as f:
                for tok in f.read().split():
                    if "{DATA}" in tok and not os.path.isdir(data_dir):
                        print(f"  FAIL {name}: cannot resolve {{DATA}} (no {data_dir})")
                        return False
                    extra_args.append(tok.replace("{DATA}", data_dir))

        result = subprocess.run(
            [sys.executable, tmp_py, *extra_args],
            capture_output=True, text=True, timeout=20, cwd=test_dir
        )
        if result.returncode != 0:
            print(f"  FAIL {name}: python failed: {result.stderr.strip()}")
            return False

        with open(expected_output) as f:
            expected = f.read().strip()
        actual = result.stdout.strip()

        if actual != expected:
            print(f"  FAIL {name}: expected '{expected}', got '{actual}'")
            return False

        # Optional repeatability check: a `run_twice` marker re-runs the program's
        # __main__ in-process (output captured), then calls run() again on the
        # same input objects and requires the same result, so run() must not
        # keep state in its arguments.
        if os.path.exists(os.path.join(test_dir, "run_twice")):
            snippet = (
                "import contextlib, io, runpy, sys\n"
                "path = sys.argv[1]\n"
                "sys.argv = sys.argv[1:]\n"
                "with contextlib.redirect_stdout(io.StringIO()):\n"
                "    g = runpy.run_path(path, run_name='__main__')\n"
                "m = g['__edgecc_meta__']\n"
                "first = g[m['output']]\n"
                "second = g['run'](*[g[n] for n in m['inputs']])\n"
                "if first != second:\n"
                "    sys.exit('run() is not repeatable: %r then %r' % (first, second))\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", snippet, tmp_py, *extra_args],
                capture_output=True, text=True, timeout=20, cwd=test_dir
            )
            if result.returncode != 0:
                print(f"  FAIL {name}: {result.stderr.strip()}")
                return False

        # Round-trip check: re-read a saved .mtx and compare to expected values.
        rt = os.path.join(test_dir, "roundtrip_check")
        if os.path.exists(rt) and HAVE_SCIPY:
            from scipy.io import mmread
            saved = os.path.join(test_dir, "result.mtx")
            if not os.path.exists(saved):
                print(f"  FAIL {name}: expected saved result.mtx not found")
                return False
            m = mmread(saved)
            arr = m.toarray() if hasattr(m, "toarray") else m
            vals = arr.flatten().tolist()
            with open(rt) as f:
                want = [float(x) for x in f.read().split()]
            os.unlink(saved)
            if vals != want:
                print(f"  FAIL {name}: round-tripped {vals} != expected {want}")
                return False

        print(f"  PASS {name}")
        return True
    finally:
        os.unlink(tmp_py)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    edgecc_bin = sys.argv[1] if len(sys.argv) > 1 else "edgecc"

    passed = 0
    failed = 0
    for entry in sorted(os.listdir(script_dir)):
        test_dir = os.path.join(script_dir, entry)
        if not os.path.isdir(test_dir):
            continue
        result = run_test(test_dir, edgecc_bin)
        if result is None:
            continue
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n{passed}/{passed + failed} e2e tests passed")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
