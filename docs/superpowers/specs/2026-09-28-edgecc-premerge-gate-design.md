# edgecc pre-merge gate — design

**Date:** 2026-09-28
**Status:** approved (process review, 2026-09-28)

## Motivation

Verifying the multi-action-Einsums branch before merge exposed gaps in how we
test features:

1. **Hand-built, tiny, self-referential tests.** All fused e2e tests used one
   6-node graph, and the differential oracle compared edgecc against edgecc
   (fused vs cascade), which share codegen helpers. A throwaway scipy
   comparison on 43 graphs exercised two fused forms the branch never tested.
2. **Negative tests only cover what the plan author imagined.** Unknown *merge*
   names were rejected; unknown *compute* names silently compiled to broken
   Python (and, in fused codegen, leaked uninitialized stack bytes). A
   throwaway token-mutation fuzzer found it in seconds.
3. **No sanitizer run.** (ASan hangs on the dev machine; UBSan works.)
4. **No finish step.** The branch sat complete-but-unmerged for 3.5 months.

There is no CI/remote, so every check must be one local command.

## Components

### 1. Mutation fuzzer — `tests/fuzz/run_fuzz.py` (ctest `fuzz_tests`)

Token-level mutations of a corpus (`examples/*.edge`, `tests/e2e/*/input.edge`,
`tests/negative/*/input.edge`). Mutations are biased toward Einsum statement
lines (`::`) and draw replacement tokens from the corpus vocabulary plus a few
junk tokens. 1–3 mutations per case (delete / insert / replace / duplicate).

Oracle — every case must end in exactly one of:

- **Clean rejection:** exit code 1, stderr mentions `error`, no sanitizer
  `runtime error`, no timeout.
- **Valid output:** exit code 0, output is UTF-8 and `compile()`s as Python.
  If the program has no file I/O (`load`/`save`), it is also run (timeout);
  it fails the oracle only if it dies with an exception that indicates a
  codegen bug (`SyntaxError`, `NameError`, `UnboundLocalError`,
  `IndentationError`, `AttributeError`) or by a signal. Other exceptions
  (e.g. the generational cap's `RuntimeError`) are legitimate program
  behavior.

Deterministic by default (fixed seed, 1000 cases, a few seconds);
`--cases N --seed S` for long runs. Failures print the seed, case index, and
the mutated statement for reproduction.

### 2. Independent-reference suite — `tests/reference/run_reference.py` (ctest `reference_tests`)

Generated kernels checked against **independent** implementations
(scipy.sparse.csgraph / numpy) — not against other edgecc forms.

- Kernels live in `tests/reference/kernels/*.edge`, written with
  `dim N = *` + `load(...)` + `seed(...)` so one kernel runs on any graph.
  The harness compiles once, imports the module, and calls `run()` directly
  with inputs ordered by `__edgecc_meta__["inputs"]`.
- Graphs: the three vendored SuiteSparse graphs + seeded random directed
  weighted graphs (sizes 1–25, several densities). Each case declares whether
  it wants the weighted graph or its 0/1 pattern, and may cap graph size (for
  cubic kernels).
- Initial cases: push/pull fused BFS, cascade BFS (`!`), depth BFS (hop
  counts), Bellman-Ford cascade SSSP, fused SSSP (both paren sides),
  wavefront SSSP, widest path (max-min), diag(G²), transitive closure.
- Requires scipy; skips cleanly (exit 0, `SKIP`) without it, like the e2e
  `needs_scipy` tests.

### 3. `tools/premerge.sh` — the gate

One command, exit non-zero on any failure, prints a summary:

1. Normal build (`build/`) + full `ctest`.
2. UBSan build (`build-ubsan/`, `-fsanitize=undefined
   -fno-sanitize-recover=undefined`) + full `ctest`.
3. Long fuzz against the UBSan binary with a fresh random seed (printed).

### 4. Process documentation — CLAUDE.md "Adding a feature"

The spec → plan → subagent implementation flow, the required test kinds, and
the plan-template rules: an *unknown-value* negative test for every new named
slot; an independent-reference check on at least one real graph for new
semantics; a final **Finish** task (gate → reconcile spec deviations → tick
plan boxes → merge → update memory, same session).

## Deviations and findings (recorded at Finish)

Building the gate found four pre-existing compiler bugs; per the triage rule
(small → failing test, then fix) all were fixed on this branch:

1. **Generational multi-dim allocation** (found by the reference suite's
   transitive-closure kernel): cross-gen `_next` buffers and intermediates were
   sized by size-matching DIM_DEFs, so a second axis on an inferred dim was
   emitted as `0` → `IndexError` on every loaded-graph 2-D generational kernel.
   Dim names are now threaded through `IRGenAlloc`.
2. **Identifier collisions** (fuzzer): dims/tensors/rank variables named like
   Python keywords, builtins used by codegen, or codegen temporaries produced
   broken Python. Chose **rejection** (`PY_RESERVED` + `_` prefix) over
   mangling — small, keeps generated code readable; mangling remains possible
   later. Added a *reserved-name sync test* (not in the original design) so the
   list can't silently drift from codegen.
3. **Rank variable shadowing a dim/tensor** (fuzzer): `A[k,N]` with dim `N` →
   rejected.
4. **Declarations-only program** (fuzzer): emitted an empty `def run():` →
   rejected ("no Einsum statement").

Other deviations: the fuzz oracle also counts `TypeError`/`IndexError` from a
generated program as codegen bugs (both surfaced real bugs above); the fuzz
corpus vocabulary includes comment tokens, which is how the lexer's
byte-at-a-time message for non-ASCII input (e.g. the paper's `¬`) showed up —
cosmetic, recorded as future work, not fixed. `ctest` gained three targets
(`fuzz_tests`, `reference_tests`, `reserved_names_tests`).

**Pre-merge review** (independent reviewer, then fixed with tests): the
`<T>_prev`/`_next` check also applies to dim names and rank variables; a dim
and tensor may not share a name; the *generational* rank variable is exempt
from name checks (it is never emitted); the gate fails fast if numpy/scipy/
PyYAML are missing (their suites would otherwise "pass" by skipping); the
reference harness reports a comparison crash as a FAIL line. Known limits,
not fixed: only ~8% of fuzz mutants reach codegen (most die in the parser or
semantic analysis — ~400 codegen'd programs per 5000-case gate run); the
reserved-name sync test does not inspect globals referenced inside
`_edgecc*` helper bodies (today only `float`, a lexer keyword, so safe).

## Out of scope

A Python reference *interpreter* for EDGE semantics (strongest oracle; its own
project). CI. Fetch-on-demand SuiteSparse graphs in the reference suite.
