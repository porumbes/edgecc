# edgecc viz: more layouts (multipartite, kamada_kawai, spectral) — design

**Date:** 2026-06-08
**Status:** Draft for review
**Scope:** Add three networkx layouts to `tools/edgecc_viz.py`'s `--layout` flag —
`multipartite` (layered by first-reached generation), `kamada_kawai` (cleaner force
layout), and `spectral` (community structure) — with a spring fallback on layout failure.
`arf` and an active-edge filter are out of scope (see Future Work).

## Motivation

`tools/edgecc_viz.py` renders a per-generation trace to frames and currently offers three
layouts: `spring` (default), `circular`, and `shell` (concentric rings by first-reached
generation). networkx (already a dependency, v3.1) ships several more layouts for free.
Three add distinct value for viewing edgecc's BFS/SSSP/generational wavefronts:

- **`multipartite`** — the layered cousin of `shell`: columns by first-reached generation, so
  the wavefront reads as marching stages left→right (vs. `shell`'s expanding rings).
- **`kamada_kawai`** — stress-majorization force layout; usually cleaner than `spring`
  (fewer crossings, deterministic) for ≤few-hundred-node graphs. A better general default.
- **`spectral`** — graph-Laplacian eigenvector positions; separates communities (karate's
  two factions, dolphins' two pods).

(`arf` was considered and dropped: marginal benefit over `spring` for our sizes. Layouts
that don't fit — `bipartite`, `planar`, `spiral`, `random`, `fruchterman_reingold` (a
`spring` alias), `rescale` (a helper) — are excluded.)

## Layouts and the shared grouping

`--layout` gains three accepted values; the flag now allows
`spring | circular | shell | multipartite | kamada_kawai | spectral`.

- **`kamada_kawai`** → `nx.kamada_kawai_layout(G)`.
- **`spectral`** → `nx.spectral_layout(G)`.
- **`multipartite`** → assign each node a `layer` attribute equal to its first-reached
  generation (never-reached nodes → a single trailing layer), then
  `nx.multipartite_layout(G, subset_key="layer")`.

**Shared grouping (refactor).** Extract a pure helper:

```
_first_reached(steps, color, empty, n) -> list[int | None]
```

returning, per node index, the generation index at which its `color` tensor first became
non-empty (walking `steps` in order), or `None` if never. This is the grouping logic
currently inline in `shell_nlist`. After the refactor:

- `shell_nlist` builds its rings from `_first_reached` (group reached nodes by value,
  sorted; append a trailing ring of the `None` nodes if any). **No behavior change** — it
  must return the same nlist as today.
- `multipartite` builds the `layer` attribute from `_first_reached`: `layer = first[node]`
  for reached nodes; never-reached nodes get `max_reached_layer + 1` (so they share one
  trailing layer; if every node is reached there is no extra layer).

Rings (`shell`) and columns (`multipartite`) therefore derive from one tested function.

## Failure fallback

`kamada_kawai` and `spectral` can raise or degenerate on disconnected graphs (e.g.
`celegansneural` seeded peripherally leaves unreachable nodes, giving infinite
shortest-path distances that break `kamada_kawai`). To keep the tool robust for
exploratory use on arbitrary graphs:

In `_layout`, compute the chosen layout's positions inside a `try`. On any `Exception`,
print a clear warning to stderr and fall back to `spring`:

```
edgecc_viz: <kind> layout failed (disconnected graph?), falling back to spring
```

then use `nx.spring_layout(G, seed=seed)`. This wraps every non-`spring` layout uniformly;
`spring` is the fallback and is not itself wrapped (cannot recurse). The render always
succeeds and produces frames.

## Architecture

The change is confined to `tools/edgecc_viz.py` plus `tests/viz/run_viz_tests.py` and
`docs/visualization.md`.

### `tools/edgecc_viz.py`

- **`_first_reached(steps, color, empty, n)`** — new pure helper (no networkx/matplotlib),
  placed beside the other pure helpers. Skips a step where the tensor is absent or 2-D
  (`not vals or isinstance(vals[0], list)`), bounds the node loop by `min(n, len(vals))`.
- **`shell_nlist`** — refactored to call `_first_reached` and assemble rings from it
  (same output as before).
- **`_layout(adj, kind, seed=1, nlist=None, first=None)`** — extended dispatch with two
  optional grouping params, each consumed by exactly one layout (`nlist` by `shell`,
  `first` by `multipartite`); other layouts ignore both:
  - build `G` as today;
  - select positions: `spring` → `spring_layout(seed)`; `circular` → `circular_layout`;
    `shell` → `shell_layout(nlist=nlist)`; `kamada_kawai` → `kamada_kawai_layout`;
    `spectral` → `spectral_layout`; `multipartite` → set each node's `layer` attribute from
    `first` (reached node → `first[node]`; `None` → `max_reached + 1`) then
    `multipartite_layout(subset_key="layer")`;
  - the non-`spring` selection runs inside a `try/except` that warns and falls back to
    `spring_layout(seed)`.
- **`render_all`** — compute the grouping for whichever layout needs it and pass it:
  `nlist = shell_nlist(...)` for `shell`; `first = _first_reached(...)` for `multipartite`;
  call `_layout(adj, layout_kind, nlist=nlist, first=first)` (each `None` unless its layout
  is active). The render loop is otherwise unchanged.
- **Usage string + argparse `choices`** — list all six layouts.

Note: node coloring still requires a 1-D watched/`--color` tensor (enforced by `main()`);
the new layouts inherit that guard. `multipartite`/`shell` need the grouping, which comes
from that same 1-D tensor.

## Testing

Following `tests/viz/run_viz_tests.py`'s split (pure-function tests always run; rendering
checks skip without matplotlib/networkx/scipy):

**Pure-function tests (always run):**
- `_first_reached` on a synthetic `steps` sequence where nodes first become non-empty at
  generations 0, 1, 1, 2 with one never-reached node → asserts the per-node list,
  e.g. `[0, 1, 1, None]`.
- Regression: `shell_nlist` on the same/synthetic and sample traces still returns the
  pre-refactor rings (e.g. sample `P` → `[[0],[1],[2],[3]]`).

**Render smoke tests (skip without deps):**
- `--layout kamada_kawai`, `--layout spectral`, `--layout multipartite` each produce one
  PNG per generation on the sample trace (4 frames) without error.

**Fallback test (skips without deps):**
- Render `--layout kamada_kawai` on a **disconnected** graph (a small `.mtx` with an
  isolated node, written to a temp file) → still produces frames. Without the fallback,
  `kamada_kawai` would raise on the infinite distances; the test proves the fallback path.

No change to existing tests; `spring`/`circular`/`shell` behavior is unchanged.

## Documentation

Update the `--layout` line in `docs/visualization.md` to list all six values with one-line
notes for the three new ones:
- `multipartite` — layered columns by first-reached generation (the layered cousin of
  `shell`).
- `kamada_kawai` — cleaner force layout than `spring` for medium graphs.
- `spectral` — Laplacian layout that separates communities.

## Future work

- **`--active-edges-only`** edge filter — the higher-value lever for dense graphs
  (the hairball is an edge problem, not node placement); orthogonal to layout.
- **`arf`** and other force layouts if a future need appears.
- **Within-ring / within-layer ordering** to reduce edge crossings.

## References

- Shell layout design: `docs/superpowers/specs/2026-06-08-edgecc-viz-shell-layout-design.md`
- Graphics design (original viz spec): `docs/superpowers/specs/2026-06-08-edgecc-graphics-design.md`
- Renderer + pure helpers: `tools/edgecc_viz.py` (`_layout`, `render_all`, `shell_nlist`,
  `active_edges`); tests: `tests/viz/run_viz_tests.py`; quickstart: `docs/visualization.md`
