# edgecc viz shell layout — design

**Date:** 2026-06-08
**Status:** Draft for review
**Scope:** Add a distance-shell (`--layout shell`) layout to `tools/edgecc_viz.py` that
positions nodes in concentric rings by the generation at which they were first reached,
so a BFS/SSSP/generational wavefront renders as expanding rings. Active-edge filtering and
additional layouts are out of scope (see Future Work).

## Motivation

`tools/edgecc_viz.py` renders a generational trace to per-generation PNG frames: a graph
panel (nodes colored by the watched tensor, cyan "active" edges) beside per-tensor heatmap
panels. It ships two layouts: `spring` (default) and `circular`. Both place nodes by graph
topology, so on anything past ~60 nodes the graph panel becomes a hairball and the
*wavefront* — the whole point of the per-generation view — is hard to read.

A **shell layout** keyed to distance makes the wavefront self-documenting: the source sits
at the center, each successive ring is the set of nodes reached one generation later, and
the animation fills inner→outer. A throwaway prototype on the dolphins SSSP trace confirmed
this reads far more clearly than spring (distance is encoded by both radius and brightness),
at ~50 lines reusing the existing renderer.

## Shelling rule: first-reached generation

A node's shell index = **the generation at which its watched-tensor value first became
non-empty**. Walking the trace's per-generation snapshots of the watched (or `--color`)
tensor in order, for each node record the first generation index where its value `!= empty`:

- **Shell 0** = nodes non-empty at generation 0 (the seed) → center.
- **Shell k** = nodes first reached at generation k.
- **Outermost ring** = nodes never reached (still `empty`/non-finite at the last step).

This single rule is meaningful across all semirings the tool sees:
- unit-weight SSSP / BFS levels → rings are hop-distance (arrival generation == hop count),
- boolean reachability BFS → rings are BFS depth (avoids the degenerate "all reached nodes
  share value 1.0 → 2 rings" that a shell-by-final-value rule would produce),
- weighted SSSP → rings are Bellman-Ford relaxation rounds.

It is computed purely from data already in the trace — no graph traversal.

## CLI surface

Add `shell` as a third accepted value of the existing `--layout` flag (joining `spring`,
`circular`). No new flags.

```
python3 tools/edgecc_viz.py kernel.trace.json --graph g.mtx --layout shell -o frames/
```

Node coloring is unchanged: nodes still color by the watched/`--color` tensor's **current**
value each generation; only node **positions** change. The shell layout needs a 1-D
per-node color tensor — `main()` already rejects a 2-D `--color` before rendering, so shell
inherits that guard with no new code.

## Architecture

The change is confined to `tools/edgecc_viz.py` plus `tests/viz/run_viz_tests.py`.

### Component 1 — `shell_nlist(steps, color, empty, n)` (new pure function)

- **Input:** the trace `steps` list, the color/watched tensor name, that tensor's `empty`
  value, and node count `n`.
- **Output:** an `nlist` — a `list[list[int]]` of node indices, innermost ring first, with
  a trailing ring of never-reached nodes appended only if any exist.
- **Logic:** for each generation index in order, for each node not yet assigned, if the
  tensor value at that node `!= empty` (and the tensor is present and 1-D this step), assign
  the node to the shell for that generation index. After all generations, any unassigned
  node goes in the final ring. Empty intermediate shells (a generation that reached no new
  node) are omitted so `shell_layout` gets only non-empty rings.
- **Purity:** no matplotlib/networkx import; importable and testable without plotting deps
  (consistent with the existing pure helpers `load_trace`, `active_edges`, `_node_colors`).

### Component 2 — `_layout(adj, kind, ...)` (extended)

Extend the existing `_layout` to accept the precomputed `nlist` (default `None`) and, when
`kind == "shell"`, build the graph as today and return
`nx.shell_layout(G, nlist=nlist)`. `spring`/`circular` ignore `nlist` and behave exactly as
before.

### Component 3 — `render_all` (minimal integration)

When `layout_kind == "shell"`, `render_all` calls `shell_nlist(...)` once (it already holds
the trace, the color tensor name, and `empty`) and passes the result into `_layout`. The
layout is computed once and reused for every frame, so node positions are fixed and nodes
merely light up as the wave reaches their ring. The rest of the render loop — edge drawing,
the cyan active-edge glow, `_node_colors` brightness, the T/D heatmap panels — is untouched.

### Within-ring ordering

Nodes are placed by `shell_layout`'s default even spacing in list order; ordering to reduce
edge crossings is an explicit **non-goal**.

## Edge cases

- **Never-reached nodes** (common on directed graphs, e.g. celegansneural seeded
  peripherally) → grouped in the dim outermost ring (their `empty` value already maps to
  0 brightness via `_node_colors`). Honest: present but never reached. If no node is
  unreached, there is no outer ring.
- **Everything reached at generation 0** (a fully-seeded tensor) → a single shell; one ring.
  Valid, if uninteresting.
- **A shell with one node** (e.g. the lone source in shell 0) → centered, as the prototype
  showed.
- **2-D watched tensor** → already rejected by `main()`'s existing 1-D check; `shell_nlist`
  is never reached with 2-D data. (Defensive: a step where the tensor is 2-D or absent is
  skipped during the walk.)
- **`--color` overrides the watched tensor** → shells are computed from the same tensor
  being colored, so rings always match the node colors.

## Testing

Following `tests/viz/run_viz_tests.py`'s split — pure-function tests always run; rendering
checks skip when matplotlib/networkx/scipy are absent.

**Pure-function tests (always run), exercising `shell_nlist`:**
- A synthetic `steps` sequence where nodes first become non-empty at generations 0,1,1,2
  → asserts `nlist == [[0], [1, 2], [3]]`.
- A node that is never non-empty → lands in the appended outermost ring.
- Seed nodes non-empty at generation 0 → shell 0.
- One realistic case driven by the existing `tests/viz/sample/` trace.

**Rendering smoke test (skips without deps):**
- `--layout shell` on the sample trace produces one PNG per generation without error
  (mirrors the existing spring/circular render check), and the count equals the number of
  generations.

No change to existing tests; `--layout spring`/`circular` output is unchanged.

## Future work

- **`--active-edges-only` edge filter** — draw only the cyan active edges (hide the rest) to
  declutter dense graphs; orthogonal to layout, deferred to its own design.
- **More layouts** — `kamada_kawai` (better default force layout), `spectral` (community
  structure); pure-networkx, no new deps.
- **Within-ring ordering** to minimize edge crossings.

## References

- Visualization quickstart: `docs/visualization.md`
- Graphics design (the original viz spec): `docs/superpowers/specs/2026-06-08-edgecc-graphics-design.md`
- Renderer + pure helpers: `tools/edgecc_viz.py` (`_layout`, `render_all`, `active_edges`,
  `_node_colors`); tests: `tests/viz/run_viz_tests.py`
- Seeded-SSSP kernels that produce shell-friendly traces: the `seed(...)` construct
  (`docs/superpowers/specs/2026-06-08-edgecc-seed-vector-design.md`); example run
  `tests/e2e/seed_sssp_karate/`
