# edgecc Graphics — Per-Generation Visual Stepper — Design

**Date:** 2026-06-08
**Status:** Approved (brainstorming complete; ready for implementation plan)

## Goal

A **per-generation visual stepper** for EDGE's iterative kernels: watch the graph,
the tensor values, and the frontiers evolve as a generational computation
proceeds. Primary purpose is understanding/debugging a kernel (#1); the same
frames serve teaching/animation (#3) and, later, paper figures (#2).

## Decisions (and rationale)

1. **Per-generation granularity, first.** One frame per iterative generation
   (e.g. each BFS step shows the frontier after that generation). Cheap — snapshot
   the generational tensors once per outer loop. Per-scalar (inner-loop) stepping
   is deferred but the trace schema is **designed to extend to it** (explicit
   "keep the door open" constraint).

2. **Frame sequence (PNG per generation), first.** Render one image per
   generation to a folder. Simplest, scriptable, no GUI; each frame doubles as a
   paper figure and as the substrate for a future animation/interactive stepper.

3. **Frame content: graph + tensor panel.** A node-link graph (left) plus a raw
   tensor-value panel (right) — the "debugger dashboard" view that literally shows
   the data values next to the graph.

4. **Compiler emits a trace file; standalone renderer consumes it.** Three
   decoupled stages so the compiler never depends on a plotting stack:
   - `edgecc --emit-trace` instruments **generational** kernels to write a JSON
     trace at runtime (stdlib `json` only — no heavy deps to get a trace).
   - `tools/edgecc_viz.py` (lazy matplotlib + networkx, skip-if-missing) reads
     `{trace.json + adjacency .mtx}` and emits PNG frames.
   - The **trace file is the contract** — each side is independently testable, and
     it is the seam where per-scalar tracing slots in later.

5. **Generational kernels only.** The first cut traces/renders only kernels that
   actually have generations to watch (BFS, reachability, Bellman-Ford, SSSP,
   widest-path). Non-generational kernels (GEMM, single push) are out of scope.

6. **Dark/neon theme, with a theme abstraction.** Ship the dark look (black bg,
   cream-ramp nodes, cyan active edges) — best for watching/debugging. The palette
   is a swappable object (`THEMES["dark"]`); only `dark` is implemented now, but
   the structure and the `--theme` flag leave room for print/light themes later.

## Architecture & data flow

```
.edge --(edgecc --emit-trace)--> prog.py  --(run)-->  trace.json
                                                          │
                          adjacency.mtx ──────────────────┤
                                                          ▼
                              tools/edgecc_viz.py (lazy matplotlib + networkx)
                                                          ▼
                                  frames/gen_000.png, gen_001.png, …
```

- **Trace capture** lives in generated Python; needs only the stdlib.
- **Rendering** is a separate dev tool with the heavy deps, lazily imported and
  skipped-if-missing (same pattern as scipy/PyYAML).
- Decoupling means the compiler stays viz-free and the renderer is reusable and
  independently testable.

## Trace format (the contract)

JSON, schema versioned and designed for forward extension:

```json
{
  "schema": "edgecc-trace/1",
  "kernel": "reachability_bfs",
  "dims": { "N": 34 },
  "gen_tensors": ["F", "P"],
  "watch": "P",
  "steps": [
    { "kind": "generation", "gen": 0, "tensors": { "F": [/* len N */], "P": [/* len N */] } },
    { "kind": "generation", "gen": 1, "tensors": { "F": [/* len N */], "P": [/* len N */] } }
  ]
}
```

- `steps` is an **ordered, tagged** list. Today every entry is
  `"kind": "generation"`. Per-scalar tracing later interleaves
  `{"kind": "write", "tensor": "F", "index": [d], "value": v}` entries between
  generation boundaries — same file, same renderer loop, finer mode.
- Each generation snapshots **all generational tensors** (the values that change).
- The static adjacency `G` is **not** duplicated into the trace; the renderer takes
  the `.mtx` separately, keeping the trace small.
- Consecutive `generation` entries give the renderer prev/next state for the
  active-edge (cyan) layer.

## Trace capture (codegen, behind `--emit-trace`)

- New CLI flag `edgecc --emit-trace`. **Without it, generated code is
  byte-identical to today** — zero impact on existing programs/tests.
- With it, on a **generational** kernel, codegen instruments the gen loop to
  append a snapshot of each generational tensor per iteration into a `_trace`
  list, and writes `<output>.trace.json` when the program runs.
- Capture uses only the stdlib `json` — running a traced kernel needs no extra
  dependencies.
- `--emit-trace` on a **non-generational** kernel: clear stderr warning, no trace
  emitted (out of scope this cut).
- `__edgecc_meta__` gains a `"trace"` block (kernel name, `gen_tensors`, `watch`)
  so the renderer can self-configure.

## Renderer (`tools/edgecc_viz.py`)

Standalone dev tool; **lazy** matplotlib + networkx, clear error if absent.

```
python tools/edgecc_viz.py reachability_bfs.trace.json --graph karate.mtx -o frames/ \
       [--color P] [--layout spring] [--theme dark]
```

- **Graph** (left): node-link from the adjacency `.mtx`; layout computed **once**
  (networkx `spring_layout`, seeded) and reused so nodes don't jump between frames.
  Directed vs undirected inferred from matrix symmetry. All edges drawn faint gray;
  **active edges this generation drawn cyan with a glow** (multi-stroke / path
  effects). Nodes colored by the `--color` tensor (default = `watch`) on the dark
  cream-ramp.
- **Active edges**: computed by the renderer from `{adjacency, prev snapshot,
  current snapshot}` as `adjacency ∧ (source was active) ∧ (dest changed)`.
- **Tensor panel** (right): each generational tensor rendered as a labeled heatmap
  chip (vector = column strip, matrix = grid), cream→black ramp.
- **Theme**: a palette object `THEMES["dark"]` (bg / node-cmap / active-edge /
  edge colors). Only `dark` implemented; `--theme` accepts it now, structured so
  print/light themes add later.
- **Output**: `frames/gen_000.png …`, one per generation, fixed canvas size.

## Error handling

| Condition | Stage | Behavior |
|---|---|---|
| `--emit-trace` on a non-generational kernel | compile | Clear warning; compile succeeds, no trace block |
| Trace file path unwritable at runtime | run | `OSError` with the path |
| Trace JSON missing/malformed/unknown `schema` | render | Clear error naming file + expected `edgecc-trace/1` |
| `--graph` `.mtx` missing | render | `FileNotFoundError` with the path |
| Graph node count ≠ trace `dims.N` | render | `ValueError` naming both sizes |
| `--color` names a tensor not in `gen_tensors` | render | `ValueError` listing available tensors |
| matplotlib/networkx not installed | render | `ImportError` with `pip install matplotlib networkx` |
| `--theme` other than `dark` | render | `ValueError` listing available themes (`dark`) |

Compile/runtime errors stay on the dependency-free capture side; all
rendering/plotting errors are isolated to the tool.

## Testing

- **Capture (no plotting deps):**
  - Compile a small BFS kernel with `--emit-trace`, run it, assert the produced
    `trace.json` is valid `edgecc-trace/1` with the right generation count and the
    expected per-generation frontier/visited values (pure data assertion — runs
    everywhere).
  - Regression: compiling the same kernel *without* `--emit-trace` is byte-identical
    to today.
  - Negative: `--emit-trace` on a non-generational kernel emits the warning and no
    trace.
- **Renderer (skip-if-deps-missing, like scipy/PyYAML):**
  - Run `edgecc_viz.py` on a committed sample `{trace.json + karate.mtx}`; assert
    the expected number of non-empty PNG frames (existence + size > 0; no pixel
    comparison).
  - Unit: the active-edge computation tested directly on a tiny hand-built trace,
    independent of rendering.
  - Negative: size-mismatch graph, bad `--color`, malformed trace → the errors above.

## Deferred / future work

- **Per-scalar (inner-loop) stepping** — finer `"kind": "write"` events + a
  renderer mode; schema and codegen instrumentation point chosen to accommodate it.
- **Interactive stepper** — slider/keyboard scrubbing over the frames.
- **Animation export** — assemble frames into GIF/MP4 (`--gif`).
- **Contact-sheet / filmstrip** — tile all generations into one image.
- **Light/print themes** — additional palette objects for publication figures.
- **Non-generational kernels** — single-frame input-graph + result view.

## Scope boundaries restated

- Per-generation only; generational kernels only; PNG frame sequence; graph +
  tensor-panel content; dark theme; trace capture behind `--emit-trace`
  (default-off, byte-identical otherwise); renderer is a standalone lazy-deps tool.
- No per-scalar stepping, no interactive UI, no animation, no non-generational
  kernels, no print theme in this cut.
