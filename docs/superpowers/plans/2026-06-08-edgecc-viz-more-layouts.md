# Viz More Layouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `multipartite`, `kamada_kawai`, and `spectral` to `tools/edgecc_viz.py`'s `--layout`, with a spring fallback (and stderr warning) when a layout fails on a disconnected graph.

**Architecture:** Extract a pure `_first_reached` helper from `shell_nlist` (per-node first-reached generation), reused by `shell` (rings) and `multipartite` (layered columns). Extend `_layout`'s dispatch with the three layouts, wrapping every non-spring layout in a try/except that warns and falls back to spring. `render_all` passes the grouping; CLI `choices` and the usage string list all six layouts.

**Tech Stack:** Python 3, networkx 3.1 (lazily imported), matplotlib (lazily imported), `tests/viz/run_viz_tests.py`.

**Spec:** `docs/superpowers/specs/2026-06-08-edgecc-viz-more-layouts-design.md`

---

## File Structure

**Modify:**
- `tools/edgecc_viz.py` — add `_first_reached`; refactor `shell_nlist` to use it; extend `_layout` (3 layouts + fallback + `first` param); pass `first` in `render_all`; `choices=` + usage string.
- `tests/viz/run_viz_tests.py` — pure `_first_reached` test; render smoke tests for the 3 layouts; a fallback test.
- `docs/visualization.md` — list all six layouts.

No new files.

---

## Task 1: Extract `_first_reached`, refactor `shell_nlist`

**Files:**
- Modify: `tools/edgecc_viz.py` (the `shell_nlist` function, currently ~lines 115-133)
- Test: `tests/viz/run_viz_tests.py`

- [ ] **Step 1: Write the failing pure test**

In `tests/viz/run_viz_tests.py`, add (after `test_shell_nlist_sample_reachability`) and register the call in the bottom run-list:

```python
def test_first_reached_basic():
    # nodes first non-empty at gens 0, 1, 1, and never (empty sentinel = 99.0)
    steps = [
        {"gen": 0, "tensors": {"D": [0.0, 99.0, 99.0, 99.0]}},
        {"gen": 1, "tensors": {"D": [0.0,  1.0,  1.0, 99.0]}},
        {"gen": 2, "tensors": {"D": [0.0,  1.0,  1.0, 99.0]}},
    ]
    check("_first_reached per-node gens", viz._first_reached(steps, "D", 99.0, 4) == [0, 1, 1, None])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/viz/run_viz_tests.py 2>&1 | tail -8`
Expected: FAIL — `AttributeError: module 'edgecc_viz' has no attribute '_first_reached'`.

- [ ] **Step 3: Add `_first_reached` and refactor `shell_nlist`**

In `tools/edgecc_viz.py`, the current `shell_nlist` is:

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

Replace it with the extracted helper plus the refactored `shell_nlist`:

```python
def _first_reached(steps, color, empty, n):
    """Per-node first-reached generation index, or None if never reached: the
    generation at which `color`'s value first left `empty`, walking `steps` in
    order. Pure (no networkx/matplotlib)."""
    first = [None] * n
    for gi, step in enumerate(steps):
        vals = step.get("tensors", {}).get(color)
        if not vals or isinstance(vals[0], list):   # tensor absent this gen, or 2-D
            continue
        for node in range(min(n, len(vals))):
            if first[node] is None and vals[node] != empty:
                first[node] = gi
    return first


def shell_nlist(steps, color, empty, n):
    """Ring assignment for the shell layout: shell k = node indices whose `color`
    tensor first became non-empty at generation index k; a trailing ring holds
    nodes never reached. Pure (no networkx/matplotlib)."""
    first = _first_reached(steps, color, empty, n)
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

- [ ] **Step 4: Run to verify it passes (and shell_nlist regression holds)**

Run: `python3 tests/viz/run_viz_tests.py 2>&1 | tail -6`
Expected: all PASS — the new `_first_reached` check, and the existing `shell_nlist groups by first-reached gen` / `unreached in outer ring` / `on sample P` checks still pass (they guard the refactor).

- [ ] **Step 5: Commit**

```bash
git add tools/edgecc_viz.py tests/viz/run_viz_tests.py
git commit -m "refactor(viz): extract _first_reached; shell_nlist builds on it"
```

---

## Task 2: Add the three layouts + spring fallback

**Files:**
- Modify: `tools/edgecc_viz.py` (`_layout` ~lines 115-126 post-Task-1 numbering; `render_all` layout call; `parse_args` `--layout`; usage docstring ~line 6)
- Test: `tests/viz/run_viz_tests.py`

- [ ] **Step 1: Write the failing render + fallback tests**

In `tests/viz/run_viz_tests.py`, add (after `test_render_shell_layout`) and register both calls in the run-list:

```python
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

def test_layout_fallback_disconnected():
    try:
        import matplotlib, networkx, scipy  # noqa: F401
    except ImportError:
        print("  SKIP test_layout_fallback_disconnected: deps not installed")
        return
    import tempfile, glob, io, contextlib
    # 4x4 graph with node 0 (0-indexed) isolated -> disconnected -> kamada_kawai
    # cannot build a finite distance matrix and raises; the wrapper must fall back.
    g = tempfile.NamedTemporaryFile("w", suffix=".mtx", delete=False)
    g.write("%%MatrixMarket matrix coordinate real general\n4 4 2\n2 3 1.0\n3 4 1.0\n")
    g.close()
    out = tempfile.mkdtemp()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        viz.main([os.path.join(HERE, "sample", "trace.json"),
                  "--graph", g.name, "--layout", "kamada_kawai", "-o", out])
    frames = sorted(glob.glob(os.path.join(out, "gen_*.png")))
    check("disconnected render still succeeds", len(frames) == 4)
    check("fallback warning emitted", "falling back to spring" in err.getvalue())
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 tests/viz/run_viz_tests.py 2>&1 | tail -12`
Expected: FAIL — `--layout` rejects `kamada_kawai`/`spectral`/`multipartite` via the current `choices=["spring","circular","shell"]` (argparse SystemExit), so the new render tests error.

- [ ] **Step 3: Extend `_layout` with the three layouts + fallback**

In `tools/edgecc_viz.py`, the current `_layout` (post-Task-1) is:

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

Replace it with:

```python
def _layout(adj, kind, seed=1, nlist=None, first=None):
    import sys
    import networkx as nx
    n = len(adj)
    directed = any(adj[i][j] != adj[j][i] for i in range(n) for j in range(n))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i][j] != 0.0:
                G.add_edge(i, j)
    if kind == "spring":
        return G, nx.spring_layout(G, seed=seed)
    try:
        if kind == "circular":
            pos = nx.circular_layout(G)
        elif kind == "shell":
            pos = nx.shell_layout(G, nlist=nlist)
        elif kind == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif kind == "spectral":
            pos = nx.spectral_layout(G)
        elif kind == "multipartite":
            reached = [g for g in (first or []) if g is not None]
            outer = (max(reached) + 1) if reached else 0
            for node in range(n):
                fr = first[node] if first else None
                G.nodes[node]["layer"] = fr if fr is not None else outer
            pos = nx.multipartite_layout(G, subset_key="layer")
        else:
            pos = nx.circular_layout(G)
        return G, pos
    except Exception as e:                       # disconnected graph, etc.
        print(f"edgecc_viz: {kind} layout failed ({e.__class__.__name__}: {e}), "
              f"falling back to spring", file=sys.stderr)
        return G, nx.spring_layout(G, seed=seed)
```

- [ ] **Step 4: Pass `first` from `render_all`**

In `tools/edgecc_viz.py`, in `render_all`, the current layout block (post-shell-feature) is:

```python
    nlist = (shell_nlist(tr["steps"], color, empty, len(adj))
             if layout_kind == "shell" else None)
    G, pos = _layout(adj, layout_kind, nlist=nlist)
```

Replace with:

```python
    nlist = (shell_nlist(tr["steps"], color, empty, len(adj))
             if layout_kind == "shell" else None)
    first = (_first_reached(tr["steps"], color, empty, len(adj))
             if layout_kind == "multipartite" else None)
    G, pos = _layout(adj, layout_kind, nlist=nlist, first=first)
```

- [ ] **Step 5: Update CLI `choices` and usage string**

In `tools/edgecc_viz.py`, change the `--layout` argument:

```python
    p.add_argument("--layout", default="spring", choices=["spring", "circular", "shell"])
```

to:

```python
    p.add_argument("--layout", default="spring",
                   choices=["spring", "circular", "shell",
                            "multipartite", "kamada_kawai", "spectral"])
```

And the module docstring usage line, currently:

```
        [--color TENSOR] [--layout spring|circular|shell] [--theme dark]
```

to:

```
        [--color TENSOR] [--layout spring|circular|shell|multipartite|kamada_kawai|spectral] [--theme dark]
```

- [ ] **Step 6: Run the viz tests**

Run: `python3 tests/viz/run_viz_tests.py 2>&1 | tail -8`
Expected: all PASS, including `kamada_kawai/spectral/multipartite render produced 4 frames`, `disconnected render still succeeds`, and `fallback warning emitted`.

(If `fallback warning emitted` fails, networkx did not raise on the isolated-node fixture in this version. Confirm with `python3 -c "import networkx as nx; G=nx.Graph(); G.add_nodes_from(range(4)); G.add_edges_from([(1,2),(2,3)]); nx.kamada_kawai_layout(G)"` — it should raise. If it does not, the fallback wrapper is still correct; adjust the fixture to one that does raise.)

- [ ] **Step 7: Manual end-to-end check on a real graph**

Run (multipartite on dolphins; reuses the trace from earlier work if present, else regenerate):
```bash
cd /Users/porumbes/Dropbox/Home/Serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc
mkdir -p /tmp/ml_check && cp data/suitesparse/dolphins.mtx /tmp/ml_check/graph.mtx
cp tests/e2e/seed_sssp_karate/input.edge /tmp/ml_check/d.edge
./build/edgecc /tmp/ml_check/d.edge -o /tmp/ml_check/d.py --emit-trace
( cd /tmp/ml_check && python3 d.py graph.mtx >/dev/null && \
  for L in multipartite kamada_kawai spectral; do
    python3 "$OLDPWD/tools/edgecc_viz.py" d.trace.json --graph graph.mtx --layout $L -o frames_$L && \
    echo "$L: $(ls frames_$L | wc -l | tr -d ' ') frames"; done )
```
Expected: `multipartite: 8 frames`, `kamada_kawai: 8 frames`, `spectral: 8 frames` (dolphins SSSP from node 0 = 8 generations).

- [ ] **Step 8: Commit**

```bash
git add tools/edgecc_viz.py tests/viz/run_viz_tests.py
git commit -m "feat(viz): --layout multipartite/kamada_kawai/spectral with spring fallback"
```

---

## Task 3: Documentation

**Files:**
- Modify: `docs/visualization.md` (the `--layout` flag line)

- [ ] **Step 1: Document the new layouts**

In `docs/visualization.md`, the `--layout` bullet currently reads:

```markdown
- `--layout circular` / `--layout shell` — node layout (default: `spring`).
  `shell` places nodes in concentric rings by the generation each was first
  reached (source at the center, unreached nodes in a dim outer ring), so a
  BFS/SSSP wavefront reads as expanding rings — much clearer than `spring` for
  graphs past ~60 nodes. Works on any generational trace with a 1-D watched/
  `--color` tensor.
```

Append, right after that bullet:

```markdown
- More layouts (all networkx, no extra deps; any that fails on a disconnected
  graph falls back to `spring` with a stderr warning):
  - `--layout multipartite` — layered columns by first-reached generation (the
    layered cousin of `shell`; the wavefront marches left→right).
  - `--layout kamada_kawai` — stress-based force layout, usually cleaner than
    `spring` for medium graphs.
  - `--layout spectral` — Laplacian layout that separates communities.
```

- [ ] **Step 2: Run the full suite**

Run: `cd build && cmake .. >/dev/null && make >/dev/null && ctest --output-on-failure 2>&1 | tail -6`
Expected: 100% pass — `viz_tests` (now with the new layout + fallback checks) and all pre-existing targets.

- [ ] **Step 3: Commit**

```bash
git add docs/visualization.md
git commit -m "docs: document multipartite/kamada_kawai/spectral layouts"
```

---

## Self-Review Notes

- **Spec coverage:** three layouts → Task 2 Step 3; shared `_first_reached` + `shell_nlist` refactor → Task 1; `multipartite` layer attr from `first` (None → outer layer) → Task 2 Step 3 (`outer` computed as `max(reached)+1`); spring fallback + stderr warning → Task 2 Step 3 (try/except) + tested in Task 2 Step 1; `render_all` passes grouping → Task 2 Step 4; CLI choices + usage → Task 2 Step 5; tests (pure + render smoke + fallback) → Tasks 1 & 2; docs → Task 3. All covered.
- **spring unchanged & unwrapped:** the `kind == "spring"` early return is outside the try, so the default path is byte-identical and the fallback can't recurse.
- **Type/name consistency:** `_first_reached(steps, color, empty, n)` returns `list[int|None]`, called identically in `shell_nlist`, `render_all`, and the test; `_layout(adj, kind, seed=1, nlist=None, first=None)` signature matches the `render_all` call (`nlist=nlist, first=first`); the warning string `"... falling back to spring"` matches the test's substring assertion.
- **Fallback test honesty:** Step 6 includes a verification note in case networkx doesn't raise on the chosen fixture, so the engineer isn't stuck on a version-dependent assumption.
- **Non-goal honored:** no within-layer ordering optimization; `arf` not added.
