# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is edgecc?

edgecc is a compiler/transpiler for the EDGE language, a declarative index-driven language that extends Einstein summation (Einsum) notation with alternative semirings. EDGE expressions describe tensor computations using Map (binary op on paired elements) and Reduce (aggregation). The compiler targets Python — generating pure nested for-loops with scalar operations.

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

Requires: C compiler (C99), CMake >= 3.15, Flex, Bison, Python 3.

Note: macOS ships Bison 2.3. The parser.y uses void* casts in the %union to stay compatible. If upgrading to Bison 3.x, the %union can use actual struct types.

## Run

```bash
./build/edgecc input.edge -o output.py
python3 output.py
```

## Test

```bash
cd build && ctest --output-on-failure
```

This runs:
- **Unit tests** per pipeline stage: `test_lexer`, `test_parser`, `test_semantic`, `test_ir`, `test_codegen`
- **E2E tests** (`tests/e2e/run_e2e.py`): compiles `.edge` files, runs generated Python, compares stdout against expected output
- **Negative tests** (`tests/negative/run_negative.py`): invalid programs that must produce specific error messages
- **Iterative E2E tests** (`tests/e2e/iterative/run_iterative.py`): drives a generated kernel in a loop with feedback until a watched tensor stabilizes (e.g., BFS). Requires PyYAML (`pip install pyyaml`); skipped if missing.

Run a single unit test: `./build/test_lexer`

## Architecture

Pipeline: `.edge source → Lexer (Flex) → Parser (Bison) → AST → Semantic Analysis → IR → Python Codegen → .py`

- `src/lexer.l` — Flex tokenizer (keywords, identifiers, literals, operators, `//` comments).
- `src/parser.y` — Bison grammar, builds AST via constructors in ast.h. Uses void* casts for Bison 2.3 compat.
- `src/ast.h/c` — AST node types (tagged union structs), constructors, destructors. Forward typedefs + struct definitions for C99 compat. Defines `MergeOp` (9-value enum) alongside `OperatorType`.
- `src/semantic.h/c` — Validates index consistency, resolves dimensions, classifies indices as free (output) vs reduction (inner loop). Produces a SymbolTable. Provides `compute_op_identity` (identity per built-in compute op) and emits a `nopass` warning to stderr.
- `src/ir.h/c` — Flat VM-style instruction list (LOOP_BEGIN, LOAD, MAP, REDUCE, STORE, LOOP_END). Lowered from AST + SymbolTable. Each MAP/REDUCE carries `merge_op`, `identity`, and operand empty values for codegen.
- `src/codegen.h/c` — Walks IR and emits Python wrapped in `def run(<inputs>) -> <output>` plus a `__main__` block. The inner-loop body emits the general merge skeleton (existence checks → merge guard → compute → store-or-empty). No special-case fast path — all merge configurations go through the same emitter.
- `src/main.c` — CLI driver: reads file, runs all pipeline stages, writes output. Supports `--dump-ir` to print the IR (including merge keyword and identity values) to stdout for debugging.

## EDGE Syntax (v2)

```
dim M = 4;
tensor Z[R=M, C=N] : float, empty = 0.0;
init A = [[1, 2], [3, 4]];
Z[m,n] = A[k,m] . B[k,n] :: map(mul, intersect) reduce(add, union);
```

### Generational (iterative) ranks

A single Einsum may iterate via a **generational rank** declared with `gen` (no
dimension). The Einsum reads generation `i` and writes generation `i+1`; the
loop runs until the watched tensor is stable:

```
dim N = 4;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
init F = [1.0, 0.0, 0.0, 0.0];           // generation 0
F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);
stop: stable(F);
```

Generated `run()` loops internally with a **fresh (empty) output slice each
generation** (the paper's default populate; no cross-generation accumulation),
double-buffered as `F_prev`/`F_next`, and returns the final state. A fixed
safety cap (`__EDGECC_MAX_GEN__ = 1000`) raises `RuntimeError` if the watched
tensor never stabilizes. Limitations: one generational tensor / one
generational rank, offsets `i` and `i+1` only, `stable` stop only, single
Einsum, no `<<` accumulation. See `docs/superpowers/specs/2026-06-01-edgecc-generational-ranks-a-design.md`.

**Cascades (multiple Einsums per generation).** A generation may run a *sequence*
of Einsums in source order, sharing one generational rank. Exactly one
**cross-generational** tensor is written at `i+1` (it carries the `init`, is
double-buffered, and is returned); any number of **intra-generation
intermediates** are written and read at `i` (no `init`, recomputed fresh each
generation). Bellman-Ford shortest paths:

```
dim N = 4;
tensor W[R=N, C=N]   : float, empty = inf;
tensor T[gen I, R=N] : float, empty = inf;   // intermediate (relaxed candidates)
tensor D[gen I, R=N] : float, empty = inf;   // cross-generational (distances)
init W = [[inf,1.0,4.0,inf],[inf,inf,2.0,6.0],[inf,inf,inf,3.0],[inf,inf,inf,inf]];
init D = [0.0, inf, inf, inf];
T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);   // relax
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);      // combine
stop: stable(D);
```

The cross-generation combine is just another Einsum (`map(min, union)`), so no
special accumulation operator is needed. An Einsum with no reduction rank (the
combine) still writes `reduce(...)`; with zero reduction indices it passes the
map value through. See `docs/superpowers/specs/2026-06-01-edgecc-generational-cascades-b-design.md`.

**Multiple cross-generational tensors (C1).** A cascade may write **more than one**
cross-generational tensor (each declared with `gen`, written at `i+1`, carrying an
`init`). A later Einsum may read an earlier one's just-written `i+1` value
(*next-reads-next*). `run()` returns the single tensor named by `stop: stable(...)`;
the others are computed and double-buffered each generation but not returned. Naive
reachability BFS — frontier `F` (recomputed, unpruned) and visited set `P`
(accumulated by union, watched because it is monotone):

```
F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);   // gather neighbors
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);        // accumulate visited
stop: stable(P);
```

See `docs/superpowers/specs/2026-06-02-edgecc-multi-cross-gen-tensors-c1-design.md`.

Rank variables in expressions bind to declared rank names by position. Built-in compute operators: `add`, `sub`, `mul`, `min`, `max`, `and`, `or`, `take_left` (map-only — emits its left operand verbatim; pair with the `take_left_only` or `intersect` merge for a value-preserving mask; using it as a `reduce` operator is an error). Built-in merge operators (optional second arg, default `passthrough`): `passthrough`, `nopass`, `intersect`, `take_left`, `take_left_only`, `take_right`, `take_right_only`, `xor`, `union`. Omitting the merge argument preserves v1 source-program semantics. Unary complement: prefix a map operand with `!` (the paper's `¬`) to use its existence complement — `!X` is present exactly where `X` is empty (supplying a `1.0` true marker there). Paper-faithful masking is `A . !B :: map(take_left, intersect)`, equivalent to the `map(_, take_left_only)` encoding. `!` is operand-only (not on the output) and keys off empty-vs-non-empty, so it works for any declared type.

### File input/output

A tensor may be initialized from a file instead of inline literals, and the
output may be written to a file:

```
dim N = *;                                  // dimension inferred from the loaded file
tensor G[R=N, C=N] : float, empty = 0.0;
tensor F[R=N]      : float, empty = 0.0;
tensor Z[R=N]      : float, empty = 0.0;
init G = load("graph.mtx");                 // dense-materialized from Matrix Market
init F = [1.0, 0.0, 0.0, 0.0];              // inline literals still work
Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);
save: Z -> "result.mtx";                    // written after run()
```

`load("path")` reads a Matrix Market (`.mtx`) file and **densifies** it into the
normal nested-list representation (the compute stays dense). `dim N = *;` infers
the size from a loaded tensor at runtime; an inferred dim must be pinned by at
least one `load(...)` tensor (else a compile-time error). A literal init on an
inferred-dim tensor is not shape-checked at compile time; its length is verified
against the loaded shape at runtime. `save: T -> "path";` writes tensor `T` after
`run()` (1-D results are stored as a column). Generated programs accept
positional argv overrides for the file-backed inputs (in
`__edgecc_meta__["file_inputs"]` order); run with `-h` for usage. See
`examples/load_push.edge`.

**Runtime dependency:** `load`/`save` require `scipy` (`pip install scipy`),
lazily imported only when used. The e2e harness skips `.mtx` tests when scipy is
absent (mark such tests with a `needs_scipy` file). See the design spec:
`docs/superpowers/specs/2026-06-07-edgecc-file-io-design.md`.

**Real test graphs.** `data/suitesparse/` holds a curated set of small
[SuiteSparse](https://sparse.tamu.edu) graphs. The three smallest
(`karate.mtx` 34, `lesmis.mtx` 77 weighted, `GD98_a.mtx` 38 directed) are
vendored; the rest are fetch-on-demand and gitignored. Download with
`python3 tools/fetch_suitesparse.py <name>` (or `--all`); the script (stdlib
only) reads `data/suitesparse/manifest.json`, pulls the Matrix Market tarball,
extracts the `.mtx`, and verifies its sha256. E2e tests source these via the
`{DATA}` token in a test's `args` file (resolved to repo-root `data/` and passed
as positional argv to the generated program). The current tests use the
diagonal-of-G² trick (`Z[i] = G[i,k].G[k,i]`) so output is a small,
hand-verifiable length-N vector and no seed vector is needed; see
`tests/e2e/load_karate_degree/`. Note a real semiring subtlety surfaced by the
weighted case: the reduce accumulator seeds from the **output tensor's `empty`**,
so for `reduce(min, ...)` the output must be declared `empty = inf` (the min
identity) while a `load`-ed operand keeps `empty = 0.0` to match densified
absents — see `tests/e2e/load_lesmis_minplus/`. A one-kernel-over-many-graphs
data-sweep runner is future work.

**Seed/source vectors.** A 1-D tensor may be initialized with
`init D = seed(source=I, value=V);` — it builds a length-N vector (N inferred
from the loaded graph that pins the dim) filled with the tensor's own declared
`empty` value, with 0-based index `I` set to `V`. Filling with `empty` makes one
construct serve both families: SSSP uses `empty = inf` + `seed(source=0,
value=0.0)`; BFS uses `empty = 0.0` + `seed(source=0, value=1.0)`. This unblocks
BFS/SSSP on arbitrarily-sized loaded graphs (the source vector auto-sizes to the
graph instead of being a fixed-length literal). The seed is materialized in the
generated `__main__` and passed to `run()`; a runtime guard rejects an
out-of-range source index. The seed tensor must be 1-D — a generational tensor
`D[gen I, R=N]` counts as 1-D (the `gen` axis is not a data dimension) — and its
dim must be pinned by a `load(...)`. Single-source only for now (the keyword
syntax is reserved for a future multi-source form). See
`tests/e2e/seed_sssp_karate/` (SSSP/BFS on loaded karate).

### Visualization (per-generation trace + renderer)

Compile a **generational** kernel with `--emit-trace` to record a per-generation
JSON trace, then render it to PNG frames:

```
./build/edgecc kernel.edge -o kernel.py --emit-trace
python3 kernel.py                       # writes <kernel-stem>.trace.json
python3 tools/edgecc_viz.py kernel.trace.json --graph graph.mtx -o frames/
```

`--emit-trace` is **default-off and only affects generational kernels** (a clear
warning is printed and ignored otherwise); without it the generated Python is
byte-identical to before. Trace capture uses only the stdlib `json`. The trace
(`edgecc-trace/1`) records every generational tensor's value per generation,
plus a gen-0 seed snapshot and a per-tensor `empty` map (so the renderer tells
"active" from empty across semirings — e.g. the min-plus source distance `0.0`
is active even though its empty value is `inf`). The schema is an ordered, tagged
`steps` list so finer (per-scalar) tracing can be added later. The trace file is
named after the source-file stem (`reachability_bfs.edge` →
`reachability_bfs.trace.json`). Note: `inf`-empty kernels write JSON `Infinity`
(Python `json`'s non-strict extension); the renderer reads it back fine.

`tools/edgecc_viz.py` reads `{trace.json + adjacency .mtx}` and emits one frame
per generation: the graph (nodes colored by the watched tensor; **cyan "active"
edges** that propagated this generation, computed as `adjacency ∧ source-active ∧
dest-changed` from consecutive snapshots) beside a heatmap panel of the
generational tensors. It lazily imports `matplotlib`, `networkx`, and `scipy`
(`pip install matplotlib networkx scipy`). Only the `dark` theme ships; `--theme`
is structured for print themes later. Tests: `trace_tests` (capture, no plotting
deps) and `viz_tests` (pure functions always; rendering checks skip when deps are
absent). See `docs/superpowers/specs/2026-06-08-edgecc-graphics-design.md`. For a
runnable quickstart with Bellman-Ford and reachability-BFS recipes plus flag notes,
see `docs/visualization.md`.

## Generated Python ABI

```python
import math

S = 4
D = 4

__edgecc_meta__ = {"inputs": ["G", "F"], "output": "F1"}

def run(G, F):
    F1 = ...
    for d in range(D):
        ...
    return F1

if __name__ == "__main__":
    G = ...
    F = ...
    F1 = run(G, F)
    print(F1)
```

The `if __name__ == "__main__"` block constructs inputs from `init` declarations and prints the output. Iterative drivers import the module and call `run()` repeatedly with feedback applied between iterations (see `tests/e2e/iterative/run_iterative.py`).

When a program uses `load`/`save`, the module also defines `_edgecc_load`/`_edgecc_save` helpers (scipy, lazily imported), `__edgecc_meta__` gains a `"file_inputs"` list, and inferred dims (`dim N = *`) are derived inside `run()` from the input shapes (e.g. `N = len(G)`) rather than emitted as module constants.

## Iterative Tests

Tests under `tests/e2e/iterative/<name>/` consist of `kernel.edge` plus `driver.yaml`. The driver YAML has this shape:

```yaml
inputs:
  G: [...]
  F: [0.0, .inf, .inf, .inf]

feedback:
  to: F                # input slot updated by run()'s output each iteration
  combine: min         # one of: replace (default), min, max, add, or, and

termination:
  type: stable         # only "stable" supported in v2
  watch: F             # watched tensor; iteration stops when its value is unchanged

max_iterations: 10

expected:
  final:
    F: [0.0, 1.0, 1.0, 2.0]
```

Note: the field is `watch:` (not `on:`) because YAML 1.1 (PyYAML's default) parses bare `on` as boolean `True`.

Iterative tests are wired into ctest as the `e2e_iterative` target. Missing PyYAML causes only iterative tests to be skipped; everything else runs.

## VS Code extension

Lives in `editors/vscode/`. Provides syntax highlighting and static-list autocompletion for `.edge` files.

Build: `cd editors/vscode && npm install && npm run package`
Install: `code --install-extension edgecc-0.1.0.vsix`
Test: `cd editors/vscode && npm test` (full suite) or `npm run test:grammar` (grammar only, fast)
Develop: open `editors/vscode/` in VS Code and press F5 (Extension Development Host).
Requires Node.js 18+.

Coverage is Tier 1 only — static dropdowns for compute / merge / type / empty-value slots; no parsing, no diagnostics, no go-to-definition. Tier 2 features (context-aware completions) and Neovim port are deferred to future work.

## v2 Scope Limitations

Not yet supported: multi-operation expressions with compute labels, the `<<` update compute operator, the remaining gaps for full 3-step BFS (multi-output return, `ANY` compute, occupancy (`‖F‖≡0`) termination; unary `¬` is now supported as `!`, see EDGE Syntax above), and nested cascades / multiple generational ranks — single-Einsum generational ranks and cascades with one **or more** cross-generational tensors (e.g. Bellman-Ford; naive reachability BFS) are supported; see "Generational (iterative) ranks", anonymous tensors / parenthesized sub-expressions, unary functions, the 7 (No,No)→Yes merge operators (NOR, equivalence, NAND, ¬→, ¬←, ⇐, ⇒), sparse tensor storage, optimizations of any kind, user-defined compute operators. File-based init is now supported for Matrix Market (`.mtx`) via `load`/`save` (see "File input/output"); it **densifies on load** — sparse *compute* (CSR/CSC/COO iteration in the generated loops) remains part of the deferred sparse-tensor-storage work, as does loading non-`.mtx` formats and a proper EDGE-native CLI for input paths. Per-generation visualization is available via `--emit-trace` + `tools/edgecc_viz.py` (generational kernels only; see "Visualization"); per-scalar (inner-loop) stepping, interactive scrubbing, animation export, contact-sheet/filmstrip, and print themes are deferred.
