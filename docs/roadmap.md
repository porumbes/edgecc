# edgecc roadmap & feature status

**Last updated:** 2026-11-16

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
| Matrix Market `load` / `save` (densify-on-load) · `.npy` `load` / `save` for any rank | ✅ |
| Inferred dims (`dim N = *`) | ✅ |
| Multi-output `run()` (`output: A, B;`) · several `save:` directives | ✅ |
| `seed(source, value)` source vectors (auto-size to loaded graph) | ✅ |
| Multi-source seed · diagonal/`I` preprocessing | ❌ |
| Sparse tensor storage + sparse compute (CSR/CSC/COO) | ❌ |
| Formats beyond `.mtx`/`.npy` · general data loading in EDGE · EDGE-native input CLI | ❌ |
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
- **Generational ranks** (`gen`) store every generation (paper semantics; `run()` returns by default the full watched tensor); stop predicates `||F[i+1]|| == 0`, `X[i+1] == X[i]`, `stable(X)`. Demonstrator: `examples/bfs_paper.edge`.
- **Cascades:** multiple Einsums per generation, one or more cross-generational tensors plus intra-generation intermediates (Bellman–Ford, reachability BFS).
- **.npy load/save (DFT gaps G8/G9, batch D):** any-rank .npy load and save, format by extension, load-time shape and dtype check (real floats only) for .mtx and .npy.
- **File I/O:** Matrix Market `load`/`save`, inferred dims `dim N = *`, argv path overrides; densify-on-load (compute stays dense).
- **`seed(source=I, value=V)`** source vectors sized to a loaded graph (unblocks BFS/SSSP on real graphs).
- **SuiteSparse data:** `data/suitesparse/` (vendored karate/lesmis/GD98_a) + `tools/fetch_suitesparse.py` (manifest + sha256).
- **Visualization:** `tools/edgecc_viz.py` with six layouts; non-spring layouts fall back to spring on failure.
- **Robustness (DFT gaps, batch A):** negative literals; lexer errors fail the compile; no silent caps (any number of `run()` inputs, up to 64 Einsums, hard errors past the other limits); argv/`-h`/`file_inputs` share `init` order; `load` targets rank-checked (2-D); warning for read-but-never-initialized tensors.
- **Intermediates out of the ABI (DFT gaps, batch B):** `run()` takes only read-but-never-written tensors; written tensors start empty inside `run()` (fixes `dim N = *` + intermediate); no `init` on, or read before, a written tensor; an Einsum may not read its own non-generational output; non-generational writes rejected in generational programs.
- **Multi-output `run()` (DFT gap G3, batch C):** `output: B, Z;` returns a tuple in listed order; `save:` may appear once per output; meta `"outputs"`.
- **Tooling/tests:** CLI flags; VS Code extension (Tier 1); 14 ctest targets (unit, e2e, negative, iterative, trace, viz, fetch-helper, reference, fuzz, reserved-names); `tools/premerge.sh` gate (normal + UBSan suites, long mutation fuzz) required before merging.

### Partial 🚧
- **Merge ops:** 9 of 16 shipped; the 7 `(No,No)→Yes` operators (NOR, equivalence, NAND, ¬→, ¬←, ⇐, ⇒) are not implemented.
- **BFS (paper-faithful):** masking / pruned / depth BFS work, and unary `¬` (as `!`) now expresses the mask directly (`A . !B :: map(take_left, intersect)`); the full 3-step formulation still needs `ANY`/parent-pointer.

### Planned ❌ (by area)
- **Language:** hoisting loop-invariant Einsums out of a generational loop; `<<` update op; compute labels; general N-ary nesting & matrix-chain (both-actions-reduce) beyond the two-action fused form; nested cascades / multiple generational ranks; anonymous tensors; unary functions; user-defined compute ops; the 7 remaining merge ops.
- **Seeding & data:** multi-source `seed` (keyword syntax already extensible); diagonal/reflexive-adjacency (`+I`) preprocessing; iterative-driver graph loading; one-kernel-over-many-graphs sweep runner; expanded committed SuiteSparse test set.
- **File I/O:** sparse tensor storage + sparse compute (CSR/CSC/COO); formats beyond `.mtx`/`.npy`; EDGE-native input-path CLI.
- **General data loading in EDGE:** how EDGE should load data in general (formats, a standard or utility library of loaders, user-supplied loaders, mapping a file's layout onto a tensor's ranks); batch D's `.npy` support is a stopgap.
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
- **Semantics / diagnostics (batch C)**
  - When every `output:` entry has an error, `num_outputs == 0` suppresses all `save:`
    diagnostics; a `save:` of a bad output name reports both "unknown tensor" and "not an
    output" (mild pile-on). No test covers an output-error-then-save interaction. (batch C)
  - Pre-existing: `__main__` rebinds cross-generational outputs to their generation lists,
    so a `run_twice` second `run()` on a generational kernel would get a wrong
    generation-0 input; more outputs widen it. No test hits it today. (batch C)
  - More test gaps (batch C): no unit test for a generational tensor in `output:`, or for
    the >1-data-rank `save:` check with several outputs; the trace harness
    (`tests/trace/run_trace_tests.py`) never checks `dims.N`, so `watch_not_output` proves
    no crash but not the `N` value (which `tools/edgecc_viz.py` consumes);
    `test_codegen_saves_follow_prints` cannot fail for its named risk (it could assert that
    `B, Z = run(A)` precedes the saves).
  - VS Code `directives.test.edge` has no negative assertions (e.g. that `// output:` stays a
    comment, or that `output` inside an Einsum is not a directive). (batch C)
  - `src/semantic.c` caps `num_outputs` / `num_saves` with `< MAX_TENSORS` guards that silently
    drop entries past 64; unreachable today (outputs are distinct declared tensors, saves are
    distinct outputs), recorded for consistency with the batch B `MAX_TENSORS` entry. (batch C)
  - The VS Code directive pattern's lookahead `(?=\s*:)` also matches before `::`; unreachable
    today because `output`/`save`/`stop` are lexer keywords and cannot name a tensor. (batch C)
  - The `save:` duplicate-path check compares strings literally, so `"b.mtx"` and `"./b.mtx"` are
    both accepted and the second overwrites the first. (batch C)
  - Pre-existing: a `save:` path equal to a `load(...)` path overwrites the input file. (batch C)
  - Pre-existing: `__main__` prints `inf` as `inf`, which is not a valid Python literal (the spec
    says each printed line is a Python literal); a consumer parsing stdout must special-case it.
    (batch C)
  - Semantic tests: `test_semantic_unsupported_extension` checks `errors[0]` rather than counting
    matching errors; no test for a case-mismatched `.NPY` or `x.npy.mtx` (correct by reading:
    case-sensitive suffix). (batch D)
  - An unsupported-extension `save:` of a non-output reports only the extension error (first error
    wins). (batch D)
  - `seed()` may pick a pin tensor that `__main__` never loads (a loaded tensor no Einsum reads):
    `emit_seed_init` (`src/codegen.c` ~805) then emits `len(V)` and fails with a `NameError`.
    Pre-existing (reproduces on main with `.mtx`). (batch D)
  - The committed FFT oracle (`npy_fft_split_n4`) is N = 4 (`Whi` all zero, H == TWO), so it cannot
    catch an H/TWO axis swap or a dropped `Whi` term; N = 8 needs approximate comparison in the
    harness. (batch D)
  - Set aside by the final reviewer (all pre-existing): a ragged literal init on a repeated inferred
    dim truncates silently; paths containing `"` or `\` are emitted verbatim into Python string
    literals; the too-many-argv check runs after the loads; a zero-data-rank `T[gen I]` `.npy` load
    is accepted. (batch D)
  - `len_expr` (and the older fixed-buffer length expressions) truncate without a closing `)` for
    names over ~245 characters. (batch D)
  - The reserved-names collector would miss a helper-local name that is also read as a global (no
    current helper does this). (batch D)
  - `run_e2e.py`: a malformed shape in a `roundtrip_check` line (e.g. `z.npy (2 2): …`) raises an
    uncaught `SyntaxError` from `ast.literal_eval` instead of a clean FAIL; a bare `!` line in
    `expected_runtime_error.txt` forbids the empty string and so always fails. No e2e loads a uint
    `.npy` (bool and int32 are covered). (batch D)

> Design specs and implementation plans for shipped and in-flight features live under
> `docs/superpowers/specs/` and `docs/superpowers/plans/`.
