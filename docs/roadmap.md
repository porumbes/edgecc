# edgecc roadmap & feature status

**Last updated:** 2026-10-28

Legend:  ✅ implemented · 🚧 partial · ❌ planned

## Status at a glance

| Capability | Status |
|---|:---:|
| **Language** | |
| Einsum expressions with Map / Reduce semirings | ✅ |
| Compute ops — `add` `sub` `mul` `min` `max` `and` `or` `take_left` | ✅ |
| Merge ops — 9 of 16 (`passthrough` `nopass` `intersect` `take_left[_only]` `take_right[_only]` `xor` `union`) | 🚧 |
| Unary complement `!` (the paper's `¬`) — per-operand existence complement | ✅ |
| Two-action fused Einsums `(A .1 B) .2 C` | ✅ |
| `<<` update op · compute labels · general N-ary nesting · matrix-chain | ❌ |
| Anonymous tensors / sub-expressions · user-defined unary funcs · user-defined ops | ❌ |
| **Iteration** | |
| Generational ranks (`gen`): every generation stored, as in the paper | ✅ |
| Stop predicates: occupancy `\|\|F[i+1]\|\| == 0`, equality, `stable(X)` sugar | ✅ |
| Cascades with ≥1 cross-generational tensor | ✅ |
| Nested cascades / multiple generational ranks | ❌ |
| **Graph algorithms** | |
| SSSP / Bellman–Ford (min-plus) | ✅ |
| Reachability BFS | ✅ |
| Paper-faithful 3-step BFS (`ANY`/parent; unary `¬`=`!` ✅, multi-action fusion ✅, occupancy stop ✅) | 🚧 |
| **Data & I/O** | |
| Inline `init` literals | ✅ |
| Matrix Market `load` / `save` (densify-on-load) | ✅ |
| Inferred dims (`dim N = *`) | ✅ |
| `seed(source, value)` source vectors (auto-size to loaded graph) | ✅ |
| Multi-source seed · 1-D / N-D load (today `load` must be 2-D) · diagonal/`I` preprocessing | ❌ |
| Sparse tensor storage + sparse compute (CSR/CSC/COO) | ❌ |
| Non-`.mtx` formats · EDGE-native input CLI | ❌ |
| **Visualization** | |
| `--emit-trace` → per-generation PNG frames (graph + tensor panels, active edges) | ✅ |
| Layouts — `spring` `circular` `shell` `multipartite` `kamada_kawai` `spectral` | ✅ |
| `--active-edges-only` edge filter | ❌ |
| Per-scalar stepping · scrubbing · animation export · print themes | ❌ |
| **Test data** | |
| SuiteSparse: vendored graphs + stdlib fetch script | ✅ |
| Independent-reference suite (kernels × vendored + random graphs vs scipy) | ✅ |
| Expanded test set · user-facing one-kernel × many-graphs sweep runner | ❌ |
| **Tooling** | |
| CLI (`--dump-ir`, `--emit-trace`) | ✅ |
| Pre-merge gate `tools/premerge.sh` (ctest + UBSan + mutation fuzz) | ✅ |
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
- **Generational ranks** (`gen`) store every generation (paper semantics; `run()` returns the full watched tensor); stop predicates `||F[i+1]|| == 0`, `X[i+1] == X[i]`, `stable(X)`. Demonstrator: `examples/bfs_paper.edge`.
- **Cascades:** multiple Einsums per generation, one or more cross-generational tensors plus intra-generation intermediates (Bellman–Ford, reachability BFS).
- **File I/O:** Matrix Market `load`/`save`, inferred dims `dim N = *`, argv path overrides; densify-on-load (compute stays dense).
- **`seed(source=I, value=V)`** source vectors sized to a loaded graph (unblocks BFS/SSSP on real graphs).
- **SuiteSparse data:** `data/suitesparse/` (vendored karate/lesmis/GD98_a) + `tools/fetch_suitesparse.py` (manifest + sha256).
- **Visualization:** `tools/edgecc_viz.py` with six layouts; non-spring layouts fall back to spring on failure.
- **Robustness (DFT gaps, batch A):** negative literals; lexer errors fail the compile; no silent caps (any number of `run()` inputs, up to 64 Einsums, hard errors past the other limits); argv/`-h`/`file_inputs` share `init` order; `load` targets rank-checked (2-D); warning for read-but-never-initialized tensors.
- **Intermediates out of the ABI (DFT gaps, batch B):** `run()` takes only read-but-never-written tensors; written tensors start empty inside `run()` (fixes `dim N = *` + intermediate); no `init` on, or read before, a written tensor; an Einsum may not read its own non-generational output; non-generational writes rejected in generational programs.
- **Tooling/tests:** CLI flags; VS Code extension (Tier 1); 14 ctest targets (unit, e2e, negative, iterative, trace, viz, fetch-helper, reference, fuzz, reserved-names); `tools/premerge.sh` gate (normal + UBSan suites, long mutation fuzz) required before merging.

### Partial 🚧
- **Merge ops:** 9 of 16 shipped; the 7 `(No,No)→Yes` operators (NOR, equivalence, NAND, ¬→, ¬←, ⇐, ⇒) are not implemented.
- **BFS (paper-faithful):** masking / pruned / depth BFS work, and unary `¬` (as `!`) now expresses the mask directly (`A . !B :: map(take_left, intersect)`); the full 3-step formulation still needs multi-output return, and `ANY`/parent-pointer.

### Planned ❌ (by area)
- **Language:** hoisting loop-invariant Einsums out of a generational loop; `<<` update op; compute labels; general N-ary nesting & matrix-chain (both-actions-reduce) beyond the two-action fused form; nested cascades / multiple generational ranks; anonymous tensors; unary functions; user-defined compute ops; the 7 remaining merge ops.
- **Seeding & data:** multi-source `seed` (keyword syntax already extensible); 1-D vector `.mtx` loading; diagonal/reflexive-adjacency (`+I`) preprocessing; iterative-driver graph loading; one-kernel-over-many-graphs sweep runner; expanded committed SuiteSparse test set.
- **File I/O:** sparse tensor storage + sparse compute (CSR/CSC/COO); non-`.mtx` formats; EDGE-native input-path CLI.
- **Visualization:** `--active-edges-only` edge filter (highest-value next addition for dense graphs); `arf` layout; within-ring/layer crossing minimization; per-scalar (inner-loop) stepping; interactive scrubbing; animation export (GIF/MP4); contact-sheet/filmstrip; light/print themes; non-generational-kernel viz.
- **Tooling:** compiler optimizations of any kind; liveness-based reclamation of generational storage (ping-pong); VS Code Tier-2 (context-aware completions); Neovim port.

### Known issues & tech debt
Open items found in review but not fixed. Add new ones here; delete an entry in the
commit that fixes it. (Origin in parentheses.)

- **Semantics to settle**
  - Two Einsums writing the same non-generational tensor accumulate into one buffer
    (`B = A.A; B = A.A` gives 2·A²). The paper means several Einsums covering different
    regions of the output; edgecc keeps the old accumulate behaviour for now. (batch B
    final review)
  - The self-read rule (E4) matches by tensor name, regardless of index pattern. So an
    elementwise `B[i] = B[i] . A[i]`, which would be loop-order safe, is rejected too.
    (batch B final review)
  - Generational `run()` returns the watched tensor with generation 0 being the caller's
    own list object (aliasing, not a copy). Mutating the result mutates the caller's
    input. (batch B final review)
- **Test gaps**
  - E1 (no `init` on a written tensor) is only tested with a literal init. `load(...)` and
    `seed(...)` take the same `has_init` path and were checked by hand only. (batch B)
  - No e2e test combines an intermediate with `save:` of the returned tensor (needs a
    `roundtrip_check`); checked by hand only. (batch B)
  - The e2e `run_twice` check compares results with `!=`, so it would fail on NaN
    outputs. (batch B)
- **Code cleanup** (`src/codegen.c`, `src/semantic.c`)
  - Three linear name-search helpers (`name_in_tnl`, `name_in_list`, the loop inside
    `add_input_name`); the generational and non-generational input filters repeat the
    same keep/free compaction loop. (batch B)
  - The once-per-tensor dedupe arrays for E2/E3/E4 are capped at `MAX_TENSORS`; past the
    cap a message could repeat (unreachable today). (batch B)

> Design specs and implementation plans for shipped and in-flight features live under
> `docs/superpowers/specs/` and `docs/superpowers/plans/`.
