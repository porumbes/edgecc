# edgecc roadmap & feature status

**Last updated:** 2026-06-08

Legend:  ✅ implemented · 🚧 partial · ❌ planned

## Status at a glance

| Capability | Status |
|---|:---:|
| **Language** | |
| Einsum expressions with Map / Reduce semirings | ✅ |
| Compute ops — `add` `sub` `mul` `min` `max` `and` `or` `take_left` | ✅ |
| Merge ops — 9 of 16 (`passthrough` `nopass` `intersect` `take_left[_only]` `take_right[_only]` `xor` `union`) | 🚧 |
| Unary complement `!` (the paper's `¬`) — per-operand existence complement | ✅ |
| `<<` update op · compute labels · multi-operation expressions | ❌ |
| Anonymous tensors / sub-expressions · user-defined unary funcs · user-defined ops | ❌ |
| **Iteration** | |
| Generational ranks (`gen`, `stop: stable`) | ✅ |
| Cascades with ≥1 cross-generational tensor | ✅ |
| Nested cascades / multiple generational ranks | ❌ |
| **Graph algorithms** | |
| SSSP / Bellman–Ford (min-plus) | ✅ |
| Reachability BFS | ✅ |
| Paper-faithful 3-step BFS (`ANY`/parent, occupancy stop; unary `¬`=`!` ✅, multi-action fusion ✅) | 🚧 |
| **Data & I/O** | |
| Inline `init` literals | ✅ |
| Matrix Market `load` / `save` (densify-on-load) | ✅ |
| Inferred dims (`dim N = *`) | ✅ |
| `seed(source, value)` source vectors (auto-size to loaded graph) | ✅ |
| Multi-source seed · 1-D vector `.mtx` load · diagonal/`I` preprocessing | ❌ |
| Sparse tensor storage + sparse compute (CSR/CSC/COO) | ❌ |
| Non-`.mtx` formats · EDGE-native input CLI | ❌ |
| **Visualization** | |
| `--emit-trace` → per-generation PNG frames (graph + tensor panels, active edges) | ✅ |
| Layouts — `spring` `circular` `shell` `multipartite` `kamada_kawai` `spectral` | ✅ |
| `--active-edges-only` edge filter | ❌ |
| Per-scalar stepping · scrubbing · animation export · print themes | ❌ |
| **Test data** | |
| SuiteSparse: vendored graphs + stdlib fetch script | ✅ |
| Expanded test set · one-kernel × many-graphs sweep runner | ❌ |
| **Tooling** | |
| CLI (`--dump-ir`, `--emit-trace`) | ✅ |
| VS Code: syntax highlighting + static autocomplete | ✅ |
| VS Code Tier-2 (context-aware) · Neovim port | ❌ |
| Compiler optimizations | ❌ |

---

## Detail

### Implemented ✅
- **Pipeline:** `.edge` → Lexer → Parser → AST → Semantic → IR → Python codegen (pure nested loops).
- **Semirings:** `map(compute, merge) reduce(compute, merge)` over the 8 compute ops and 9 merge ops above.
- **Unary complement `!`** (the paper's `¬`): per-operand existence complement; paper-faithful BFS masking `A . !B :: map(take_left, intersect)`, equivalent to the `take_left_only` encoding (`examples/push_bfs_neg.edge`).
- **Multi-action (fused) Einsums** `(A .1 B) .2 C`: paper-faithful one-line BFS gather/mask (`examples/push_bfs_fused.edge`, `pull_bfs_fused.edge`); transient-scalar intermediate, optional `over=` rank tags.
- **Generational ranks** (`gen`) with `stop: stable(T)`; internal double-buffered loop in `run()`.
- **Cascades:** multiple Einsums per generation, one or more cross-generational tensors plus intra-generation intermediates (Bellman–Ford, reachability BFS).
- **File I/O:** Matrix Market `load`/`save`, inferred dims `dim N = *`, argv path overrides; densify-on-load (compute stays dense).
- **`seed(source=I, value=V)`** source vectors sized to a loaded graph (unblocks BFS/SSSP on real graphs).
- **SuiteSparse data:** `data/suitesparse/` (vendored karate/lesmis/GD98_a) + `tools/fetch_suitesparse.py` (manifest + sha256).
- **Visualization:** `tools/edgecc_viz.py` with six layouts; non-spring layouts fall back to spring on failure.
- **Tooling/tests:** CLI flags; VS Code extension (Tier 1); 11 ctest targets (unit, e2e, negative, iterative, trace, viz, fetch-helper).

### Partial 🚧
- **Merge ops:** 9 of 16 shipped; the 7 `(No,No)→Yes` operators (NOR, equivalence, NAND, ¬→, ¬←, ⇐, ⇒) are not implemented.
- **BFS (paper-faithful):** masking / pruned / depth BFS work, and unary `¬` (as `!`) now expresses the mask directly (`A . !B :: map(take_left, intersect)`); the full 3-step formulation still needs multi-output return, `ANY`/parent-pointer, and occupancy (`‖F‖≡0`) termination.

### Planned ❌ (by area)
- **Language:** `<<` update op; compute labels; general N-ary nesting & matrix-chain (both-actions-reduce) beyond the two-action fused form; nested cascades / multiple generational ranks; anonymous tensors; unary functions; user-defined compute ops; the 7 remaining merge ops.
- **Seeding & data:** multi-source `seed` (keyword syntax already extensible); 1-D vector `.mtx` loading; diagonal/reflexive-adjacency (`+I`) preprocessing; iterative-driver graph loading; one-kernel-over-many-graphs sweep runner; expanded committed SuiteSparse test set.
- **File I/O:** sparse tensor storage + sparse compute (CSR/CSC/COO); non-`.mtx` formats; EDGE-native input-path CLI.
- **Visualization:** `--active-edges-only` edge filter (highest-value next addition for dense graphs); `arf` layout; within-ring/layer crossing minimization; per-scalar (inner-loop) stepping; interactive scrubbing; animation export (GIF/MP4); contact-sheet/filmstrip; light/print themes; non-generational-kernel viz.
- **Tooling:** compiler optimizations of any kind; VS Code Tier-2 (context-aware completions); Neovim port.

> Design specs and implementation plans for shipped and in-flight features live under
> `docs/superpowers/specs/` and `docs/superpowers/plans/`.
