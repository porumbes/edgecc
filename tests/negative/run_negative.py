#!/usr/bin/env python3
"""Negative test driver for edgecc."""
import os
import subprocess
import sys

def run_test(test_dir, edgecc_bin):
    name = os.path.basename(test_dir)
    input_file = os.path.join(test_dir, "input.edge")
    expected_errors = os.path.join(test_dir, "expected_errors.txt")

    if not os.path.exists(input_file) or not os.path.exists(expected_errors):
        return None

    result = subprocess.run(
        [edgecc_bin, input_file, "-o", "/dev/null"],
        capture_output=True, text=True, timeout=10
    )

    if result.returncode == 0:
        print(f"  FAIL {name}: expected failure but edgecc succeeded")
        return False

    stderr = result.stderr
    with open(expected_errors) as f:
        expected = [line.strip() for line in f if line.strip()]

    all_found = True
    for pattern in expected:
        if pattern not in stderr:
            print(f"  FAIL {name}: expected error '{pattern}' not found in stderr")
            all_found = False

    if all_found:
        print(f"  PASS {name}")
        return True
    return False


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

    print(f"\n{passed}/{passed + failed} negative tests passed")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
