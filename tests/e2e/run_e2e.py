#!/usr/bin/env python3
"""End-to-end test driver for edgecc."""
import os
import subprocess
import sys
import tempfile

def run_test(test_dir, edgecc_bin):
    name = os.path.basename(test_dir)
    input_file = os.path.join(test_dir, "input.edge")
    expected_output = os.path.join(test_dir, "expected_output.txt")

    if not os.path.exists(input_file) or not os.path.exists(expected_output):
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

        result = subprocess.run(
            [sys.executable, tmp_py],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print(f"  FAIL {name}: python failed: {result.stderr.strip()}")
            return False

        with open(expected_output) as f:
            expected = f.read().strip()
        actual = result.stdout.strip()

        if actual == expected:
            print(f"  PASS {name}")
            return True
        else:
            print(f"  FAIL {name}: expected '{expected}', got '{actual}'")
            return False
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
