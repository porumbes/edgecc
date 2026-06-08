#!/usr/bin/env python3
"""Trace test driver: compile a kernel with --emit-trace, run it, validate JSON."""
import json
import os
import subprocess
import sys
import tempfile

def run_test(test_dir, edgecc_bin):
    name = os.path.basename(test_dir)
    src = os.path.join(test_dir, "input.edge")
    if not os.path.exists(src):
        return None
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as t:
        tmp_py = t.name
    # edgecc names the trace after the source-file stem (input.edge -> input.trace.json).
    stem = os.path.splitext(os.path.basename(src))[0]
    trace_path = os.path.join(test_dir, f"{stem}.trace.json")
    try:
        r = subprocess.run([edgecc_bin, src, "-o", tmp_py, "--emit-trace"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print(f"  FAIL {name}: edgecc failed: {r.stderr.strip()}")
            return False
        r = subprocess.run([sys.executable, tmp_py],
                           capture_output=True, text=True, timeout=10, cwd=test_dir)
        if r.returncode != 0:
            print(f"  FAIL {name}: run failed: {r.stderr.strip()}")
            return False
        if not os.path.exists(trace_path):
            print(f"  FAIL {name}: no trace file written")
            return False
        with open(trace_path) as f:
            tr = json.load(f)
        # Generic structural checks (kernel-agnostic).
        if tr.get("schema") != "edgecc-trace/1":
            print(f"  FAIL {name}: bad schema {tr.get('schema')}"); return False
        watch = tr.get("watch")
        gts = tr.get("gen_tensors", [])
        if not gts or watch not in gts:
            print(f"  FAIL {name}: bad metadata gen_tensors={gts} watch={watch}")
            return False
        if "empty" not in tr or watch not in tr["empty"]:
            print(f"  FAIL {name}: missing empty map {tr.get('empty')}"); return False
        steps = tr.get("steps", [])
        if len(steps) < 2:
            print(f"  FAIL {name}: too few steps ({len(steps)})"); return False
        if steps[0]["gen"] != 0:
            print(f"  FAIL {name}: first step is not gen 0"); return False
        # Watched tensor must be stable at the end (converged).
        if steps[-1]["tensors"][watch] != steps[-2]["tensors"][watch]:
            print(f"  FAIL {name}: watched tensor not stable at end"); return False
        # Optional per-fixture final-value assertion.
        exp_path = os.path.join(test_dir, "expected_final.json")
        if os.path.exists(exp_path):
            with open(exp_path) as f:
                expected = json.load(f)
            got = steps[-1]["tensors"][watch]
            if got != expected:
                print(f"  FAIL {name}: final {watch}={got} != expected {expected}")
                return False
        print(f"  PASS {name}")
        return True
    finally:
        os.unlink(tmp_py)
        if os.path.exists(trace_path):
            os.unlink(trace_path)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    edgecc_bin = sys.argv[1] if len(sys.argv) > 1 else "edgecc"
    passed = failed = 0
    for entry in sorted(os.listdir(script_dir)):
        d = os.path.join(script_dir, entry)
        if not os.path.isdir(d):
            continue
        res = run_test(d, edgecc_bin)
        if res is None:
            continue
        passed += res
        failed += (not res)
    print(f"\n{passed}/{passed + failed} trace tests passed")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
