#!/usr/bin/env python3
"""Viz tests. Pure-function tests always run; full rendering and graph-loading
checks are skipped if matplotlib/networkx/scipy are unavailable."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, TOOLS)

import edgecc_viz as viz   # must import WITHOUT importing matplotlib/networkx

results = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'} {name}")
    results.append(bool(cond))

def test_load_trace():
    tr = viz.load_trace(os.path.join(HERE, "sample", "trace.json"))
    check("load_trace schema", tr["schema"] == "edgecc-trace/1")
    check("load_trace steps", len(tr["steps"]) == 4)

def test_active_edges():
    # adjacency 0->1->2->3 ; prev F lit at node 1, curr F lit at node 2
    adj = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    prev = {"F": [0.0, 1.0, 0.0, 0.0]}
    curr = {"F": [0.0, 0.0, 1.0, 0.0]}
    edges = viz.active_edges(adj, prev, curr, "F")
    check("active_edges finds 1->2", (1, 2) in edges)
    check("active_edges excludes 0->1", (0, 1) not in edges)

def test_bad_schema():
    try:
        viz.validate_trace({"schema": "nope"})
        check("validate_trace rejects bad schema", False)
    except ValueError:
        check("validate_trace rejects bad schema", True)

def test_render_frames():
    try:
        import matplotlib  # noqa: F401
        import networkx    # noqa: F401
        import scipy       # noqa: F401
    except ImportError:
        print("  SKIP test_render_frames: matplotlib/networkx/scipy not installed")
        return
    import tempfile, glob
    out = tempfile.mkdtemp()
    viz.main([os.path.join(HERE, "sample", "trace.json"),
              "--graph", os.path.join(HERE, "sample", "graph.mtx"),
              "-o", out])
    frames = sorted(glob.glob(os.path.join(out, "gen_*.png")))
    check("render produced 4 frames", len(frames) == 4)
    check("frames non-empty", all(os.path.getsize(f) > 0 for f in frames))

def test_size_mismatch():
    import tempfile
    g = tempfile.NamedTemporaryFile("w", suffix=".mtx", delete=False)
    g.write("%%MatrixMarket matrix coordinate real general\n3 3 2\n1 2 1.0\n2 3 1.0\n")
    g.close()
    try:
        viz.main([os.path.join(HERE, "sample", "trace.json"),
                  "--graph", g.name, "-o", "/tmp/x"])
        check("size mismatch raises", False)
    except ValueError:
        check("size mismatch raises", True)
    except ImportError:
        print("  SKIP test_size_mismatch: scipy not installed")

def test_bad_color():
    try:
        viz.main([os.path.join(HERE, "sample", "trace.json"),
                  "--graph", os.path.join(HERE, "sample", "graph.mtx"),
                  "--color", "ZZZ", "-o", "/tmp/x"])
        check("bad --color raises", False)
    except ValueError:
        check("bad --color raises", True)
    except ImportError:
        print("  SKIP test_bad_color: scipy not installed")

test_load_trace()
test_active_edges()
test_bad_schema()
test_render_frames()
test_size_mismatch()
test_bad_color()

print(f"\n{sum(results)}/{len(results)} viz checks passed")
sys.exit(1 if not all(results) else 0)
