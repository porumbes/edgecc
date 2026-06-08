# Viz Shell Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--layout shell` option to `tools/edgecc_viz.py` that places nodes in concentric rings by the generation they were first reached, so a BFS/SSSP/generational wavefront renders as expanding rings.

**Architecture:** A new pure helper `shell_nlist(steps, color, empty, n)` computes the ring assignment from the trace (no graph traversal). `_layout` gains an `nlist` parameter and calls `nx.shell_layout` when `kind == "shell"`; `render_all` computes the nlist once and passes it. The render loop (edges, active-edge glow, node coloring, panels) is otherwise untouched.

**Tech Stack:** Python 3, networkx (lazily imported), matplotlib (lazily imported), the existing `tests/viz/run_viz_tests.py` harness.

**Spec:** `docs/superpowers/specs/2026-06-08-edgecc-viz-shell-layout-design.md`

---

## File Structure

**Modify:**
- `tools/edgecc_viz.py` — add `shell_nlist` (pure); extend `_layout` with `nlist`; wire shell into `render_all`; mention `shell` in usage strings.
- `tests/viz/run_viz_tests.py` — pure `shell_nlist` tests + a `--layout shell` render smoke test.
- `docs/visualization.md` — document `--layout shell`.

No new files; the feature is a focused addition to the existing renderer.

---

## Task 1: `shell_nlist` pure helper

**Files:**
- Modify: `tools/edgecc_viz.py` (add `shell_nlist` near the other pure helpers, after `_node_colors`/`_finite_range`, before `_layout`)
- Test: `tests/viz/run_viz_tests.py`

- [ ] **Step 1: Write the failing pure-function tests**

In `tests/viz/run_viz_tests.py`, add these three test functions (place them after `test_active_edges_2d_guard`) and register their calls in the bottom run-list (see Step 5):

```python
def test_shell_nlist_basic():
    # nodes first become non-empty at generations 0, 1, 1, 2 (empty sentinel = 99.0)
    steps = [
        {"gen": 0, "tensors": {"D": [0.0, 99.0, 99.0, 99.0]}},
        {"gen": 1, "tensors": {"D": [0.0,  1.0,  1.0, 99.0]}},
        {"gen": 2, "tensors": {"D": [0.0,  1.0,  1.0,  2.0]}},
    ]
    nlist = viz.shell_nlist(steps, "D", 99.0, 4)
    check("shell_nlist groups by first-reached gen", nlist == [[0], [1, 2], [3]])

def test_shell_nlist_unreached():
    # node 2 and 3 never leave empty -> appended outer ring
    steps = [
        {"gen": 0, "tensors": {"D": [0.0, 99.0, 99.0, 99.0]}},
        {"gen": 1, "tensors": {"D": [0.0,  1.0, 99.0, 99.0]}},
    ]
    nlist = viz.shell_nlist(steps, "D", 99.0, 4)
    check("shell_nlist puts unreached in outer ring", nlist == [[0], [1], [2, 3]])

def test_shell_nlist_sample_reachability():
    # sample trace watches P (boolean reachability): first-reached rule still
    # yields one ring per generation, NOT the degenerate 2 rings of final-value.
    tr = viz.load_trace(os.path.join(HERE, "sample", "trace.json"))
    nlist = viz.shell_nlist(tr["steps"], "P", 0.0, 4)
    check("shell_nlist on sample P", nlist == [[0], [1], [2], [3]])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd build && ctest -R viz_tests --output-on-failure 2>&1 | tail -20` (or directly: `python3 tests/viz/run_viz_tests.py`)
Expected: FAIL — `AttributeError: module 'edgecc_viz' has no attribute 'shell_nlist'`.

- [ ] **Step 3: Implement `shell_nlist`**

In `tools/edgecc_viz.py`, add (after `_finite_range`, before `_layout`):

```python
def shell_nlist(steps, color, empty, n):
    """Ring assignment for the shell layout: shell k = node indices whose `color`
    tensor first became non-empty at generation index k (walking `steps` in order);
    a trailing ring holds nodes never reached. Pure (no networkx/matplotlib)."""
    first = [None] * n
    for gi, step in enumerate(steps):
        vals = step.get("tensors", {}).get(color)
        if not vals or isinstance(vals[0], list):   # tensor absent this gen, or 2-D
            continue
        for node in range(min(n, len(vals))):
            if first[node] is None and vals[node] != empty:
                first[node] = gi
    buckets, unreached = {}, []
    for node in range(n):
        if first[node] is None:
            unreached.append(node)
        else:
            buckets.setdefault(first[node], []).append(node)
    shells = [buckets[g] for g in sorted(buckets)]
    if unreached:
        shells.append(unreached)
    return shells
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 tests/viz/run_viz_tests.py 2>&1 | tail -6`
Expected: all checks PASS, including the three new `shell_nlist` checks; final line `N/N viz checks passed`.

- [ ] **Step 5: Register the new tests in the run-list**

In `tests/viz/run_viz_tests.py`, the bottom of the file calls each test in sequence (e.g. `test_active_edges_2d_guard()`). Add these three calls alongside them (before the final `print(...)`):

```python
test_shell_nlist_basic()
test_shell_nlist_unreached()
test_shell_nlist_sample_reachability()
```

(If you already added them in Step 1, just confirm they are present in the call-list, not only defined.)

- [ ] **Step 6: Commit**

```bash
git add tools/edgecc_viz.py tests/viz/run_viz_tests.py
git commit -m "feat(viz): shell_nlist — concentric rings by first-reached generation"
```

---

## Task 2: Wire shell layout into the renderer

**Files:**
- Modify: `tools/edgecc_viz.py` (`_layout` signature + body ~lines 115-126; `render_all` layout call ~line 136; usage string ~line 6)
- Test: `tests/viz/run_viz_tests.py`

- [ ] **Step 1: Write the failing render smoke test**

In `tests/viz/run_viz_tests.py`, add (after `test_render_2d_color_raises`) and register the call in the run-list:

```python
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
```

Add the call `test_render_shell_layout()` to the bottom run-list.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/viz/run_viz_tests.py 2>&1 | tail -8`
Expected: FAIL on the shell render — with the current `_layout`, `kind == "shell"` falls into the `else` branch and renders `circular` (4 frames *would* be produced, so this check may PASS spuriously). To make the test meaningfully fail first, temporarily confirm shell is unsupported by asserting positions differ; **simpler and sufficient:** proceed to Step 3 (the implementation), since the real behavioral coverage is `shell_nlist` (Task 1) — this smoke test guards that `--layout shell` runs end-to-end without error. If it already "passes" via the circular fallback, that is acceptable; Step 3 makes it use the true shell layout.

- [ ] **Step 3: Extend `_layout` to accept and use `nlist`**

In `tools/edgecc_viz.py`, replace `_layout` (currently lines ~115-126):

```python
def _layout(adj, kind, seed=1):
    import networkx as nx
    n = len(adj)
    directed = any(adj[i][j] != adj[j][i] for i in range(n) for j in range(n))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i][j] != 0.0:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, seed=seed) if kind == "spring" else nx.circular_layout(G)
    return G, pos
```

with:

```python
def _layout(adj, kind, seed=1, nlist=None):
    import networkx as nx
    n = len(adj)
    directed = any(adj[i][j] != adj[j][i] for i in range(n) for j in range(n))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i][j] != 0.0:
                G.add_edge(i, j)
    if kind == "shell":
        pos = nx.shell_layout(G, nlist=nlist)
    elif kind == "spring":
        pos = nx.spring_layout(G, seed=seed)
    else:
        pos = nx.circular_layout(G)
    return G, pos
```

- [ ] **Step 4: Compute the nlist in `render_all` and pass it**

In `tools/edgecc_viz.py`, in `render_all`, replace the layout call (currently line ~136):

```python
    G, pos = _layout(adj, layout_kind)
```

with:

```python
    nlist = (shell_nlist(tr["steps"], color, empty, len(adj))
             if layout_kind == "shell" else None)
    G, pos = _layout(adj, layout_kind, nlist=nlist)
```

(`render_all` already receives `tr`, `color`, and `empty` — no signature change.)

- [ ] **Step 5: Mention `shell` in the usage string**

In `tools/edgecc_viz.py`, the module docstring usage line (~line 6) reads:

```
        [--color TENSOR] [--layout spring] [--theme dark]
```

Change `[--layout spring]` to `[--layout spring|circular|shell]`.

- [ ] **Step 6: Run the viz tests**

Run: `python3 tests/viz/run_viz_tests.py 2>&1 | tail -6`
Expected: all PASS including `shell render produced 4 frames` and `shell frames non-empty`.

- [ ] **Step 7: Manual end-to-end check on a real graph**

Run:
```bash
cd /Users/porumbes/Dropbox/Home/Serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc
mkdir -p /tmp/shell_check && cp data/suitesparse/karate.mtx /tmp/shell_check/graph.mtx
cp tests/e2e/seed_sssp_karate/input.edge /tmp/shell_check/k.edge
./build/edgecc /tmp/shell_check/k.edge -o /tmp/shell_check/k.py --emit-trace
( cd /tmp/shell_check && python3 k.py graph.mtx >/dev/null && \
  python3 "$OLDPWD/tools/edgecc_viz.py" k.trace.json --graph graph.mtx --layout shell -o frames )
ls /tmp/shell_check/frames/
```
Expected: `gen_000.png … gen_004.png` (5 frames; karate SSSP from node 0 stabilizes in 5 generations). The trace file is `k.trace.json` because the kernel source was copied to `k.edge` (trace name = source-file stem).

- [ ] **Step 8: Commit**

```bash
git add tools/edgecc_viz.py tests/viz/run_viz_tests.py
git commit -m "feat(viz): --layout shell renders the wavefront as expanding rings"
```

---

## Task 3: Documentation

**Files:**
- Modify: `docs/visualization.md` (the "Useful flags" section)

- [ ] **Step 1: Document `--layout shell`**

In `docs/visualization.md`, the "Useful flags" section currently lists:

```markdown
- `--layout circular` — node layout (default: `spring`).
```

Replace that line with:

```markdown
- `--layout circular` / `--layout shell` — node layout (default: `spring`).
  `shell` places nodes in concentric rings by the generation each was first
  reached (source at the center, unreached nodes in a dim outer ring), so a
  BFS/SSSP wavefront reads as expanding rings — much clearer than `spring` for
  graphs past ~60 nodes. Works on any generational trace with a 1-D watched/
  `--color` tensor.
```

- [ ] **Step 2: Run the full suite**

Run: `cd build && cmake .. >/dev/null && make >/dev/null && ctest --output-on-failure 2>&1 | tail -6`
Expected: 100% pass — `viz_tests` (now including the shell checks) and all pre-existing targets.

- [ ] **Step 3: Commit**

```bash
git add docs/visualization.md
git commit -m "docs: document --layout shell in the visualization guide"
```

---

## Self-Review Notes

- **Spec coverage:** shelling rule (first-reached generation) → `shell_nlist` (Task 1); CLI `--layout shell` → Task 2 Steps 3-5; layout threading (nlist into `_layout`) → Task 2 Steps 3-4; never-reached → outer ring → `shell_nlist` + tested in Task 1 Step 1; minimal `render_all` integration → Task 2 Step 4; 1-D guard inherited (no code) → covered by existing `test_render_2d_color_raises`; pure helper + render smoke test → Tasks 1 & 2; docs → Task 3. All covered.
- **No new flags / behavior unchanged for spring & circular:** `_layout`'s `nlist` defaults to `None` and only `kind == "shell"` uses it; the spring/circular branches are byte-identical.
- **Type/name consistency:** `shell_nlist(steps, color, empty, n)` defined in Task 1 is called with exactly those args in Task 2 Step 4 (`shell_nlist(tr["steps"], color, empty, len(adj))`); `_layout(..., nlist=None)` signature matches its `nlist=nlist` call site.
- **Task 2 Step 2 caveat is explicit:** the smoke test may pass via the circular fallback before Step 3; that's called out rather than hidden, and the real behavioral coverage lives in the `shell_nlist` pure tests (Task 1).
- **Non-goal honored:** within-ring ordering is left to networkx defaults; not implemented.
