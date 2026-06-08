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

def test_active_edges_weighted():
    # min-plus: empty=999; source node 0 holds real distance 0.0 and IS active.
    adj = [[0, 1, 4, 0], [0, 0, 2, 6], [0, 0, 0, 3], [0, 0, 0, 0]]
    prev = {"D": [0.0, 999.0, 999.0, 999.0]}
    curr = {"D": [0.0, 1.0, 4.0, 999.0]}
    edges = viz.active_edges(adj, prev, curr, "D", empty=999.0)
    check("weighted active_edges lights source 0->1", (0, 1) in edges)
    check("weighted active_edges lights source 0->2", (0, 2) in edges)
    # Same with the wrong empty (0.0) would treat the source as inactive:
    edges0 = viz.active_edges(adj, prev, curr, "D", empty=0.0)
    check("empty=0.0 misses the source", (0, 1) not in edges0)

def test_active_edges_2d_guard():
    adj = [[0, 1], [0, 0]]
    prev = {"T": [[1.0, 0.0], [0.0, 1.0]]}
    curr = {"T": [[1.0, 1.0], [0.0, 1.0]]}
    check("active_edges returns empty for 2-D", viz.active_edges(adj, prev, curr, "T") == set())

def test_render_weighted_with_intermediate():
    try:
        import matplotlib, networkx, scipy  # noqa: F401
    except ImportError:
        print("  SKIP test_render_weighted_with_intermediate: deps not installed")
        return
    import tempfile, glob
    out = tempfile.mkdtemp()
    # Bellman-Ford-style: intermediate T absent at gen 0; weighted distances; empty=999.
    viz.main([os.path.join(HERE, "sample_bf", "trace.json"),
              "--graph", os.path.join(HERE, "sample_bf", "graph.mtx"), "-o", out])
    frames = sorted(glob.glob(os.path.join(out, "gen_*.png")))
    check("BF render produced 4 frames", len(frames) == 4)
    check("BF frames non-empty", all(os.path.getsize(f) > 0 for f in frames))

def test_render_2d_color_raises():
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("  SKIP test_render_2d_color_raises: scipy not installed")
        return
    try:
        viz.main([os.path.join(HERE, "sample_2d", "trace.json"),
                  "--graph", os.path.join(HERE, "sample_2d", "graph.mtx"), "-o", "/tmp/x2d"])
        check("2-D --color raises cleanly", False)
    except ValueError:
        check("2-D --color raises cleanly", True)

def test_shell_nlist_basic():
    steps = [
        {"gen": 0, "tensors": {"D": [0.0, 99.0, 99.0, 99.0]}},
        {"gen": 1, "tensors": {"D": [0.0,  1.0,  1.0, 99.0]}},
        {"gen": 2, "tensors": {"D": [0.0,  1.0,  1.0,  2.0]}},
    ]
    nlist = viz.shell_nlist(steps, "D", 99.0, 4)
    check("shell_nlist groups by first-reached gen", nlist == [[0], [1, 2], [3]])

def test_shell_nlist_unreached():
    steps = [
        {"gen": 0, "tensors": {"D": [0.0, 99.0, 99.0, 99.0]}},
        {"gen": 1, "tensors": {"D": [0.0,  1.0, 99.0, 99.0]}},
    ]
    nlist = viz.shell_nlist(steps, "D", 99.0, 4)
    check("shell_nlist puts unreached in outer ring", nlist == [[0], [1], [2, 3]])

def test_shell_nlist_sample_reachability():
    tr = viz.load_trace(os.path.join(HERE, "sample", "trace.json"))
    nlist = viz.shell_nlist(tr["steps"], "P", 0.0, 4)
    check("shell_nlist on sample P", nlist == [[0], [1], [2], [3]])

def test_render_shell_layout():
    try:
        import matplotlib, networkx, scipy  # noqa: F401
    except ImportError:
        print("  SKIP test_render_shell_layout: deps not installed")
        return
    import tempfile, glob
    out = tempfile.mkdtemp()
    viz.main([os.path.join(HERE, "sample", "trace.json"),
              "--graph", os.path.join(HERE, "sample", "graph.mtx"),
              "--layout", "shell", "-o", out])
    frames = sorted(glob.glob(os.path.join(out, "gen_*.png")))
    check("shell render produced 4 frames", len(frames) == 4)
    check("shell frames non-empty", all(os.path.getsize(f) > 0 for f in frames))

def test_first_reached_basic():
    steps = [
        {"gen": 0, "tensors": {"D": [0.0, 99.0, 99.0, 99.0]}},
        {"gen": 1, "tensors": {"D": [0.0,  1.0,  1.0, 99.0]}},
        {"gen": 2, "tensors": {"D": [0.0,  1.0,  1.0, 99.0]}},
    ]
    check("_first_reached per-node gens", viz._first_reached(steps, "D", 99.0, 4) == [0, 1, 1, None])

def test_render_more_layouts():
    try:
        import matplotlib, networkx, scipy  # noqa: F401
    except ImportError:
        print("  SKIP test_render_more_layouts: deps not installed")
        return
    import tempfile, glob
    for kind in ("kamada_kawai", "spectral", "multipartite"):
        out = tempfile.mkdtemp()
        viz.main([os.path.join(HERE, "sample", "trace.json"),
                  "--graph", os.path.join(HERE, "sample", "graph.mtx"),
                  "--layout", kind, "-o", out])
        frames = sorted(glob.glob(os.path.join(out, "gen_*.png")))
        check(f"{kind} render produced 4 frames", len(frames) == 4)

def test_layout_disconnected_ok():
    # kamada_kawai on a disconnected graph (node 0 isolated): networkx may handle
    # it (no raise) or raise -> fall back; either way the render must succeed.
    try:
        import matplotlib, networkx, scipy  # noqa: F401
    except ImportError:
        print("  SKIP test_layout_disconnected_ok: deps not installed")
        return
    import tempfile, glob
    g = tempfile.NamedTemporaryFile("w", suffix=".mtx", delete=False)
    g.write("%%MatrixMarket matrix coordinate real general\n4 4 2\n2 3 1.0\n3 4 1.0\n")
    g.close()
    out = tempfile.mkdtemp()
    viz.main([os.path.join(HERE, "sample", "trace.json"),
              "--graph", g.name, "--layout", "kamada_kawai", "-o", out])
    frames = sorted(glob.glob(os.path.join(out, "gen_*.png")))
    check("disconnected render still succeeds", len(frames) == 4)

def test_layout_fallback_on_error():
    # Deterministically exercise the fallback wrapper: force the chosen layout to
    # raise (independent of networkx's version-specific disconnected handling) and
    # assert the render still succeeds via spring, with a stderr warning.
    try:
        import matplotlib, networkx as nx, scipy  # noqa: F401
    except ImportError:
        print("  SKIP test_layout_fallback_on_error: deps not installed")
        return
    import tempfile, glob, io, contextlib
    def boom(*a, **k):
        raise RuntimeError("forced layout failure")
    orig = nx.kamada_kawai_layout
    nx.kamada_kawai_layout = boom
    try:
        out = tempfile.mkdtemp()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            viz.main([os.path.join(HERE, "sample", "trace.json"),
                      "--graph", os.path.join(HERE, "sample", "graph.mtx"),
                      "--layout", "kamada_kawai", "-o", out])
    finally:
        nx.kamada_kawai_layout = orig
    frames = sorted(glob.glob(os.path.join(out, "gen_*.png")))
    check("fallback render produced 4 frames", len(frames) == 4)
    check("fallback warning emitted", "falling back to spring" in err.getvalue())

test_load_trace()
test_active_edges()
test_bad_schema()
test_render_frames()
test_size_mismatch()
test_bad_color()
test_active_edges_weighted()
test_active_edges_2d_guard()
test_render_weighted_with_intermediate()
test_render_2d_color_raises()
test_shell_nlist_basic()
test_shell_nlist_unreached()
test_shell_nlist_sample_reachability()
test_render_shell_layout()
test_first_reached_basic()
test_render_more_layouts()
test_layout_disconnected_ok()
test_layout_fallback_on_error()

print(f"\n{sum(results)}/{len(results)} viz checks passed")
sys.exit(1 if not all(results) else 0)
