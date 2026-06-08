# Visualizing generational kernels

edgecc can record a **per-generation trace** of a generational (iterative) kernel
and render it to a sequence of PNG frames — one per generation — showing the graph
(nodes colored by a watched tensor, cyan "active" edges that propagated this
generation) beside a heatmap panel of the generational tensors.

## The three moving parts

1. **Compile with trace capture.** The `--emit-trace` flag adds per-generation JSON
   capture to the generated Python. It is **default-off and only affects
   generational kernels** (a warning is printed and ignored otherwise); without it
   the generated Python is byte-identical to before.

   ```bash
   ./build/edgecc kernel.edge -o k.py --emit-trace
   ```

2. **Run it.** Running the generated program writes `<source-stem>.trace.json` in
   the current directory. The stem comes from the *source* `.edge` file, not the
   `-o` output — so `input.edge` produces `input.trace.json`. Trace capture uses
   only the stdlib `json`.

   ```bash
   python3 k.py            # writes <source-stem>.trace.json
   ```

3. **Render to frames.** Point the renderer at the trace plus the adjacency matrix
   used as the graph; it emits one PNG per generation (`gen_000.png`, `gen_001.png`, …).

   ```bash
   python3 tools/edgecc_viz.py <trace>.json --graph <adjacency>.mtx -o frames/
   ```

### Useful flags

- `--color F` — color nodes by a different tensor (default: the `stop:`-watched one).
- `--layout circular` / `--layout shell` — node layout (default: `spring`).
  `shell` places nodes in concentric rings by the generation each was first
  reached (source at the center, unreached nodes in a dim outer ring), so a
  BFS/SSSP wavefront reads as expanding rings — much clearer than `spring` for
  graphs past ~60 nodes. Works on any generational trace with a 1-D watched/
  `--color` tensor.
- `--theme dark` — only the `dark` theme ships today; the flag is structured for
  print themes later.

### Dependencies

Frame rendering lazily imports **matplotlib**, **networkx**, and **scipy**. If they
are not present:

```bash
pip install --user matplotlib networkx scipy
```

(Trace *capture* in step 2 needs none of these — only the renderer does.)

## Worked examples

Run these from the repo root.

### Bellman–Ford (weighted, min-plus)

```bash
cd /Users/porumbes/Dropbox/Home/Serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc
mkdir -p /tmp/viz_bf && cp tests/trace/bellman_ford/graph.mtx /tmp/viz_bf/
./build/edgecc tests/trace/bellman_ford/input.edge -o /tmp/viz_bf/bf.py --emit-trace
( cd /tmp/viz_bf && python3 bf.py && python3 "$OLDPWD/tools/edgecc_viz.py" input.trace.json --graph graph.mtx -o frames )
open /tmp/viz_bf/frames/        # macOS: opens the folder of gen_000.png … gen_004.png
```

### Reachability BFS (boolean frontier)

```bash
mkdir -p /tmp/viz_bfs && cp tests/trace/reachability_bfs/graph.mtx /tmp/viz_bfs/
./build/edgecc tests/trace/reachability_bfs/input.edge -o /tmp/viz_bfs/bfs.py --emit-trace
( cd /tmp/viz_bfs && python3 bfs.py && python3 "$OLDPWD/tools/edgecc_viz.py" input.trace.json --graph graph.mtx -o frames )
open /tmp/viz_bfs/frames/
```

> The `$OLDPWD` trick lets the inner subshell `cd` into the scratch dir (so the
> trace lands there) while still reaching back to the repo's `tools/edgecc_viz.py`.

## Notes

- **Active edges** are computed from consecutive generation snapshots as
  `adjacency ∧ source-active ∧ dest-changed`, so the cyan edges show what actually
  propagated each generation.
- The renderer is **empty-aware**: the per-tensor `empty` map in the trace lets it
  distinguish "active" from empty across semirings — e.g. the min-plus source
  distance `0.0` is active even though the tensor's empty value is `inf`.
- `inf`-empty kernels write JSON `Infinity` (Python `json`'s non-strict extension);
  the renderer reads it back fine.

## Scope

Generational kernels only. Per-scalar (inner-loop) stepping, interactive scrubbing,
animation export, contact-sheet/filmstrip, and print themes are deferred. See the
design spec: `docs/superpowers/specs/2026-06-08-edgecc-graphics-design.md`.
