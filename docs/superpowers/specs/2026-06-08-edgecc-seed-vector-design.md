# edgecc seed-vector construct — design

**Date:** 2026-06-08
**Status:** Draft for review
**Scope:** A new `seed(...)` init form that sizes a source vector to a loaded graph, so
BFS/SSSP kernels run end-to-end on arbitrarily-sized loaded graphs. Single-source only;
multi-source, 1-D-vector `.mtx` loading, and iterative-driver changes are out of scope
(see Future Work).

## Motivation

edgecc can `load(...)` an N×N graph whose dimension `N` is inferred at runtime
(`dim N = *`). BFS/SSSP kernels also need a **source vector** of length `N` (e.g.
`[0.0, inf, inf, …]` for single-source shortest paths). Today that vector is an inline
literal of fixed length, which cannot match an arbitrarily-sized loaded graph — the
generated runtime shape assertion (`if len(F) != N: raise ValueError(...)`,
`src/codegen.c:427-456`) rejects the mismatch. This blocks running BFS/SSSP on the
SuiteSparse test graphs (see the SuiteSparse test-set work merged 2026-06-08).

### Why a construct, not a loaded `.mtx`

`load(...)` densifies absent Matrix Market entries to `0.0` (the densify-on-load design).
A loaded seed therefore works for a BFS boolean seed (`source=1.0`, rest `0.0`) but
**cannot express an SSSP min-plus seed** (`source=0.0`, rest `inf`) — the absent entries
would densify to `0.0`, not `inf`, making every node look like distance 0. A construct
that fills with the *tensor's own `empty` value* sidesteps this entirely and serves both
semiring families with one mechanism.

## The `seed(...)` construct

A new **init form**, a peer of inline-literal init and `load(...)` init:

```
init D = seed(source=0, value=0.0);
```

**Semantics:** build a 1-D vector whose length is the tensor's (inferred or fixed)
dimension `N`, **filled with the tensor's declared `empty` value**, with the element at
0-based index `source` set to `value`.

- SSSP: `tensor D[...] : float, empty = inf;` → `seed(source=0, value=0.0)` →
  `[0.0, inf, inf, …]`.
- BFS: `tensor F[...] : float, empty = 0.0;` → `seed(source=0, value=1.0)` →
  `[1.0, 0.0, 0.0, …]`.

**Syntax:** keyword arguments `source=<int>` and `value=<number>`, both required (no
defaults — the correct `value` differs per semiring). `value` accepts floats and `inf`
(already lexable; used in `empty = inf` today). Keyword form is chosen so multi-source can
be added later as `seed(sources=[...], value=...)` or a map form **without a grammar
break**.

**Constraints:**
- The seed tensor must be **rank-1** (1-D). Using `seed` on a 2-D tensor is a compile-time
  error.
- `source` must be a non-negative integer literal (compile-time check). Its upper bound
  cannot be checked at compile time when `N` is inferred, so a runtime guard is emitted.
- Length `N` is normally inferred and **must be pinned by a loaded tensor**. This reuses
  the existing pin check (`src/semantic.c:655-670`): a `seed` on an inferred dim with no
  loaded graph to size it fails with `inferred dim '…' is not pinned by any load(...)
  tensor`. A fixed dim (`dim N = 4;`) also works (`[empty]*N` with the constant).

## Architecture

The change threads a new init kind through the pipeline (lex/parse/AST → semantic → IR →
codegen `__main__`); the einsum/loop lowering and the `run()` body are untouched.

### 1. Surface → AST (`src/lexer.l`, `src/parser.y`, `src/ast.h/c`)

- Add keyword `seed` to the lexer.
- Grammar: the init RHS gains a production
  `seed ( source = INT , value = number )` alongside the existing literal-list and
  `load ( STRING )` productions. `number` covers signed floats and `inf`.
- Add an init-kind `INIT_SEED` to the AST, carrying `{ int source; double value; }`
  (mirrors how the `load` init carries its path). Constructor + destructor follow the
  existing init-node pattern.

### 2. Semantic (`src/semantic.c/h`)

- Validate the seed tensor is rank-1; else error `seed: tensor '<name>' must be 1-D`.
- Validate `source >= 0`; else error `seed: source index must be non-negative`.
- No new dim logic: the seed tensor's dimension flows through the existing inferred-dim
  pinning check unchanged.

### 3. Codegen (`src/codegen.c`) — materialization in `__main__`

A generational tensor's `init` is materialized in `__main__` and passed to `run()`
(confirmed: Bellman-Ford emits `D = [..]; D = run(W, D)`; loaded inputs in
`__edgecc_meta__["file_inputs"]` order are constructed first). The seed replaces the inline
literal at that same site. Reusing the dim-pinning helper that already powers
`emit_inferred_dim_derivation` (`src/codegen.c:406-425`) to find the pinning input and axis,
emit:

```python
D = [math.inf] * len(W)          # [<D.empty>] * <len of pinning input on its axis>
if not (0 <= 0 < len(W)):        # runtime guard: source in range
    raise ValueError("edgecc: seed source 0 out of range [0, %d) for 'D'" % len(W))
D[0] = 0.0                       # D[<source>] = <value>
D = run(W, D)
```

- `<D.empty>` is the tensor's declared empty (`math.inf`, `0.0`, …) rendered exactly as
  elsewhere in codegen.
- The length expression is `len(<pin>)` for a dim pinned on axis 0, `len(<pin>[0])` for
  axis 1, or the integer constant for a fixed dim.
- The seed init is emitted in the init phase (after loaded inputs), so the pinning input
  is already in scope.

### 4. IR (`src/ir.h/c`)

Inits flow through the IR as `IR_TENSOR_INIT` (codegen reads the init from the IR, not the
AST), so `IRTensorInit` gains the seed fields (`is_seed`, `seed_source`, `seed_value`) and
`emit_tensor_inits` copies them from the AST — mirroring how it already copies `is_load`/
`path`. The **einsum/loop lowering is unchanged** (no new loop, load, map, reduce, or store
instructions), and `--dump-ir` is otherwise unaffected. To keep the diff small and avoid
touching every existing `is_load` read site, the new kind is an **additive `is_seed` flag**
(a literal init is `is_load == 0 && is_seed == 0`) rather than replacing `is_load` with an
enum.

## Testing strategy

### Primary end-to-end test (the unblock)

A generational SSSP kernel reading a loaded graph + seed, run standalone:

```
dim N = *;
tensor W[R=N, C=N]   : float, empty = 0.0;     // adjacency (load densifies absent->0.0)
tensor T[gen I, R=N] : float, empty = inf;     // relaxed candidates
tensor D[gen I, R=N] : float, empty = inf;     // distances (seeded, watched)
init W = load("graph.mtx");
init D = seed(source=0, value=0.0);            // D[0]=0, rest inf; sized to N=len(W)
T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);
stop: stable(D);
```

On **karate** (a pattern matrix → weights `1.0`) this computes BFS hop-distance from node 0.
New e2e test `tests/e2e/seed_sssp_karate/` with `input.edge`, `args`
(`{DATA}/suitesparse/karate.mtx`), `needs_scipy`, and `expected_output.txt`. The expected
output is generated by edgecc and independently cross-checked against
`scipy.sparse.csgraph.shortest_path` (unweighted BFS distances from node 0) before being
committed — the same oracle discipline used for the SuiteSparse tests. `W` keeps
`empty = 0.0` (matches densified absents so the `intersect` map excludes non-edges); `T`/`D`
use `empty = inf` (the `min` reduce identity — see the reduce-empty constraint noted in
CLAUDE.md).

### Unit tests (per-stage, project convention)

- `tests/unit/test_lexer.c`: `seed` produces the new keyword token.
- `tests/unit/test_parser.c`: `init D = seed(source=0, value=0.0)` builds an `INIT_SEED`
  node with `source==0`, `value==0.0`.
- `tests/unit/test_semantic.c`: a rank-1 seed validates; a 2-D seed is rejected; a negative
  `source` is rejected.
- `tests/unit/test_codegen.c`: the emitted `__main__` contains the `[empty]*len(...)` fill,
  the range guard, and the `D[source]=value` assignment.

### Negative tests (`tests/negative/`)

- `seed_2d/` → expects `seed: tensor '<name>' must be 1-D`.
- `seed_unpinned/` → a `seed` on an inferred dim with no loaded tensor; expects the existing
  `inferred dim '<name>' is not pinned by any load(...) tensor`.

### Docs

- CLAUDE.md EDGE-syntax section documents the `seed(...)` init form.
- Update the SuiteSparse memory note to record that seed-vector seeding now exists.

## Error handling

- **Compile time:** non-rank-1 seed tensor; negative `source`; unpinned inferred dim (via
  the existing check).
- **Runtime:** the emitted guard raises `ValueError` with a clear message if `source` is
  outside `[0, N)` once `N` is known (e.g. a source index larger than the loaded graph).

## Future work

- **Multi-source seeds** — `seed(sources=[i, j], value=v)` or an index→value map; the
  keyword syntax is chosen to extend without a grammar break.
- **1-D vector `.mtx` loading** — flatten an N×1/1×N loaded matrix to 1-D so a real seed
  *file* can also be used (only valid for the `0.0`-default/BFS case, given densify-on-load).
- **Iterative-driver graph loading** — teach `tests/e2e/iterative/` to load graphs and size
  feedback vectors; currently it uses inline literals only.

## References

- SuiteSparse test-set design: `docs/superpowers/specs/2026-06-08-edgecc-suitesparse-test-set-design.md`
- File I/O design: `docs/superpowers/specs/2026-06-07-edgecc-file-io-design.md`
- Generational ranks design: `docs/superpowers/specs/2026-06-01-edgecc-generational-ranks-a-design.md`
- Working min-reduce + inf-empty pattern: `tests/trace/bellman_ford/input.edge`
- Inferred-dim derivation / shape assert: `src/codegen.c:406-456`; pin check:
  `src/semantic.c:655-670`
