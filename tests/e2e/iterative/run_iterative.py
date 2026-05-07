#!/usr/bin/env python3
"""Iterative end-to-end test driver for edgecc.

Reads each subdirectory under tests/e2e/iterative/ that contains a kernel.edge
and a driver.yaml, compiles the kernel, imports the resulting Python module,
drives mod.run(...) in a loop until the watched tensor stabilizes, and
compares the final state against expected values.
"""
import argparse
import importlib.util
import os
import subprocess
import sys

try:
    import yaml
except ImportError:
    print("SKIP: PyYAML not installed; iterative tests skipped.", file=sys.stderr)
    sys.exit(0)


def deep_equal(a, b):
    """Recursive comparison; handles inf == inf as True (Python's default)."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def elementwise(op, a, b):
    """Apply binary op element-wise to nested lists of equal shape."""
    if isinstance(a, list):
        return [elementwise(op, x, y) for x, y in zip(a, b)]
    if op == "min":     return min(a, b)
    if op == "max":     return max(a, b)
    if op == "add":     return a + b
    if op == "or":      return bool(a) or bool(b)
    if op == "and":     return bool(a) and bool(b)
    raise ValueError(f"unknown combine op: {op}")


def apply_feedback(prev, new, combine):
    if combine == "replace":
        return new
    return elementwise(combine, prev, new)


def import_module_from_file(path):
    spec = importlib.util.spec_from_file_location("kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_test(test_dir, edgecc_bin):
    name = os.path.basename(test_dir)
    kernel_edge = os.path.join(test_dir, "kernel.edge")
    driver_yaml = os.path.join(test_dir, "driver.yaml")
    if not (os.path.exists(kernel_edge) and os.path.exists(driver_yaml)):
        return None

    out_py = os.path.join(test_dir, "_kernel_generated.py")
    try:
        result = subprocess.run(
            [edgecc_bin, kernel_edge, "-o", out_py],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"  FAIL {name}: edgecc failed: {result.stderr.strip()}")
            return False

        mod = import_module_from_file(out_py)
        meta = getattr(mod, "__edgecc_meta__", None)
        if not meta:
            print(f"  FAIL {name}: kernel missing __edgecc_meta__")
            return False

        with open(driver_yaml) as f:
            cfg = yaml.safe_load(f)

        inputs = dict(cfg["inputs"])
        feedback_to = cfg["feedback"]["to"]
        feedback_combine = cfg["feedback"].get("combine", "replace")
        watched = cfg["termination"]["on"]
        max_iter = int(cfg["max_iterations"])

        stabilized = False
        for it in range(max_iter):
            snapshot = inputs[watched]
            output = mod.run(*[inputs[name] for name in meta["inputs"]])
            inputs[feedback_to] = apply_feedback(
                inputs[feedback_to], output, feedback_combine,
            )
            if deep_equal(inputs[watched], snapshot) and it > 0:
                stabilized = True
                break

        if not stabilized:
            print(f"  FAIL {name}: did not stabilize within {max_iter} iterations")
            return False

        for tensor_name, expected in cfg["expected"]["final"].items():
            actual = inputs.get(tensor_name)
            if actual is None and tensor_name == meta["output"]:
                actual = output
            if not deep_equal(actual, expected):
                print(f"  FAIL {name}: {tensor_name} mismatch")
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")
                return False

        print(f"  PASS {name}")
        return True
    finally:
        if os.path.exists(out_py):
            os.unlink(out_py)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--tests-dir", required=True)
    args = parser.parse_args()

    edgecc_bin = os.path.join(args.build_dir, "edgecc")
    passed = 0
    failed = 0
    found_any = False
    for entry in sorted(os.listdir(args.tests_dir)):
        d = os.path.join(args.tests_dir, entry)
        if not os.path.isdir(d):
            continue
        result = run_test(d, edgecc_bin)
        if result is None:
            continue
        found_any = True
        if result:
            passed += 1
        else:
            failed += 1

    if not found_any:
        print("SKIP: no iterative tests found.", file=sys.stderr)
        sys.exit(0)

    print(f"\n{passed}/{passed + failed} iterative e2e tests passed")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
