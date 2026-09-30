# edgecc paper-faithful generational tensors — design

**Date:** 2026-09-30
**Status:** approved in brainstorming (2026-09-28 – 2026-09-30); pending written-spec review
**Branch:** `feature/paper-faithful-generations`

## Motivation

While designing multi-output `run()`, we found that edgecc's generational
semantics are shaped by an optimization rather than by the EDGE paper:

- The paper defines a generational tensor as its **whole history**: adding
  the rank `I` "collaps[es] the F1, F2, … tensors in the unrolled cascade
  into a single 2-D tensor, whose generational rank, I, is unbounded"
  (`tex/bfs-example/07-iteration.tex`). Intermediates carry `I` too
  (`T^{I, D}` in the full BFS specification).
- The paper names ping-ponging as a compiler **optimization** found by
  liveness analysis ("only two copies of the frontier are needed at any
  given time, leading to software optimizations like … 'ping-ponging'").
- The stop condition "simply indicates the shape of I, which may not be
  known until runtime"; formally the generational rank variable set is
  `{ i | i ∈ [0, I) ∧ ¬◇(i) }`. The paper's BFS stops on occupancy,
  `◇: ‖F_{i+1}‖ ≡ 0`.

edgecc today double-buffers generational tensors (`_prev`/`_next`), discards
intermediates each generation, returns only the final slice of the watched
tensor, and supports only its own `stop: stable(X)`. Principle adopted: edgecc
implements what the paper specifies; optimizations come later. This branch
makes generational storage and stopping paper-faithful. Multi-output `run()`
follows as a separate piece.

## Semantics

**Generational tensors hold every generation.** `F[gen I, R=N]` denotes all
generations. Generation 0 is supplied by `init` (literal, `seed(...)`, or
`load(...)`, unchanged: it provides the generation-0 slice). Every new slice
starts empty (paper: "All output tensors are initially empty").

**One iteration `i`** runs the cascade in source order: intermediates are
written at `i` (`T[i, d]`), cross-generational tensors at `i+1`
(`F[i+1, d]`); next-reads-next (`P[i+1] = P[i] . F[i+1]`) works as today.

**Compute-then-check.** After the whole cascade for iteration `i`, the stop
predicate is evaluated. The generation that satisfies it is **kept**; then
the loop ends. If the last iteration is `k`, cross-generational tensors hold
generations `0…k+1` and intermediates `0…k`.

**Stop predicates** (exactly one per program; `and`/`or` combinations are
future work):

| Form | Meaning |
|---|---|
| `stop: \|\|X[g]\|\| == 0;` | occupancy: generation `g` of `X` has no non-empty coordinate (w.r.t. `X`'s declared `empty`) |
| `stop: X[g] == X[h];` | generations `g` and `h` of the same tensor `X` are elementwise equal |
| `stop: stable(X);` | sugar for `X[i+1] == X[i]` (keeps the 28 existing programs compiling); desugared in semantic analysis, since only there is the generation variable's name known |

`g`, `h` ∈ {`i`, `i+1`}, where `i` is the program's generational rank
variable. A reference must name a generation that exists when the predicate
is checked: `i` or `i+1` for a cross-generational tensor, `i` only for an
intermediate.

**`run()`** returns the tensor named in the stop predicate, **in full** (a
list of generations). Non-generational programs are unchanged. Choosing
which tensors `run()` returns is the job of the follow-on multi-output piece.

**New rule:** the `gen` rank must be the **first** rank of its tensor (the
paper's convention; all 149 generational declarations in the corpus comply).
Storage is then `F[i][…]`. Can be lifted later if a real use appears.

**Unchanged:** the 1000-generation safety cap (`RuntimeError`); single
assignment per generation.

## Generated code

Paper BFS (occupancy stop, intermediate `T`, cross-generational `F`, `P`):

```python
def run(G, F, P):
    F = [F]                    # generational tensor; the argument is generation 0
    P = [P]
    T = []                     # intermediate: no generation 0 from init
    _i = 0
    while True:
        T.append([math.inf] * N)     # T[_i]    starts empty
        F.append([math.inf] * N)     # F[_i+1]  starts empty
        P.append([0.0] * N)          # P[_i+1]  starts empty
        for d in range(N): ...       # T[_i][d]   from G[s][d], F[_i][s]
        for d in range(N): ...       # F[_i+1][d] from T[_i][d], P[_i][d]
        for d in range(N): ...       # P[_i+1][d] from P[_i][d], F[_i+1][d]
        if _edgecc_occupancy(F[_i + 1], math.inf) == 0:
            break                    # compute-then-check: F[_i+1] is kept
        _i += 1
        if _i >= __EDGECC_MAX_GEN__:
            raise RuntimeError("generational loop did not stop")
    return F                         # every generation
```

- `_prev`/`_next` and the buffer swap are removed. Generational reads and
  writes index by the counter: `F[_i]`, `F[_i + 1]`, `T[_i]`.
- Each iteration first appends one empty slice per generational tensor
  written in it (allocated by dim name, per axis — as fixed on the gate
  branch), so every read of a slice written earlier in the same iteration
  finds it.
- Generation 0 is the caller's own list; it is never written (outputs go to
  new slices), so no copy is made.
- Equality compiles to list equality (`P[_i + 1] == P[_i]`). Occupancy uses
  `_edgecc_occupancy(slice, empty)`, emitted only when used (like
  `_edgecc_load`); it counts entries `!= empty` in a slice of any rank.
- `__main__` prints the full tensor. `save:` of a generational tensor writes
  the full tensor (a 1-D generational tensor saves as a generations × N
  matrix), consistent with `run()`.
- `--emit-trace` keeps the `edgecc-trace/1` schema so `tools/edgecc_viz.py`
  works unchanged; the trace is built after the loop from the stored
  generations instead of per-generation snapshots.
- Memory is O(generations × tensor size) — fine for this pure-Python backend
  at our graph sizes. Reclaiming it is the paper's liveness optimization,
  future work.

## Validation

New semantic errors (each with a negative test):

| Situation | Error (substring) |
|---|---|
| `gen` rank not first (`X[R=N, gen I]`) | generational rank 'I' must be the first rank of 'X' |
| predicate tensor unknown or not generational | stop predicate: 'X' is not a generational tensor |
| generation offset not `i`/`i+1` | stop predicate: generation must be 'i' or 'i+1' |
| intermediate referenced at `i+1` | intermediate 'T' has only generation 'i' when the stop predicate is checked |
| `==` between different tensors | stop predicate: '==' compares two generations of the same tensor |
| `X[i] == X[i]` | stop predicate compares a generation with itself |
| occupancy compared to non-zero | only '\|\|X[g]\|\| == 0' is supported |
| predicate uses another generation variable (`F[j+1]` vs Einsums' `i`) | stop predicate uses 'j', but the generational rank variable is 'i' |
| unknown predicate function (`stop: settled(P);`) | unknown stop predicate 'settled' (expected 'stable') |

Existing rules stay: a generational Einsum requires `stop:`; `stop:` without
a generational Einsum is an error.

Lexer: new `||` and `==` tokens; a lone `|` is a lexer error.

**Retired scaffolding:** `_prev`/`_next` no longer exist in generated code, so
the `<T>_prev`/`<T>_next` name reservation added on the gate branch is removed
(`is_gen_buffer_name`, negative tests `reserved_gen_suffix` and
`reserved_gen_suffix_dim_var`). The reserved-name sync test confirms codegen no
longer emits those names. New codegen names (`_i`, `_edgecc_occupancy`) are
`_`-prefixed and already reserved.

## Testing

- **Unit, per stage, test-first:** lexer (`||`, `==`); parser (three
  predicate forms); semantic (every error above plus positives; `stable(X)`
  resolves to the same predicate as `X[i+1] == X[i]`); IR (predicate on the gen
  loop); codegen (no `_prev`/`_next`, per-iteration appends, occupancy helper
  only when used). Existing codegen assertions on `F_next = …` /
  `state = F_next[d]` are rewritten to the new shape.
- **Migration of the 28 generational programs:** a one-off script
  regenerates each `expected_output.txt` only after asserting that the new
  output's **last generation equals the old expected output exactly** —
  proving the change is purely representational for existing programs. One
  small BFS history is also verified by hand.
- **Independent history checks** (`tests/reference/`):
  - paper BFS (occupancy stop): support of `F[g]` = vertices at hop distance
    exactly `g` (scipy); last generation empty; number of generations =
    source eccentricity + 2;
  - naive / stable-stop BFS: support of `P[g]` = vertices within distance `g`;
  - existing kernels compare their last generation, `out[-1]`.
- **Paper demonstrator:** `examples/bfs_paper.edge` = the paper's equation
  (bfsi) as written — `T`/`F`/`P` cascade, `min` (for `ANY`, unsupported),
  `!P`, `stop: ||F[i+1]|| == 0;` — with an e2e test and a reference case.
- Trace tests pass unchanged (schema compatibility); the fuzzer picks up the
  new syntax through the examples; reserved-name sync test passes.
- **Finish:** `tools/premerge.sh`; CLAUDE.md "Generational (iterative)
  ranks" rewritten; roadmap (occupancy termination → done); merge `--no-ff`
  from a worktree; memory updated.

## Questions for the paper authors

1. **Stop timing.** With `◇: ‖F_{i+1}‖ ≡ 0`, is the generation that first
   satisfies `◇` part of `F`? The rank-variable-set definition excludes the
   `i` where `◇(i)` holds, but evaluating `◇(i)` requires computing
   `F_{i+1}`. edgecc keeps it (compute-then-check, as in the paper's draft
   programming-model comment).
2. **`G^{I, S, D}`.** The full BFS specification declares the graph with a
   generational rank although `G` never changes. Typo, or intentional (e.g.
   dynamic graphs)? edgecc keeps `G` non-generational.

## Out of scope

Multi-output `run()` (next piece); `and`/`or` predicate combinations and
other predicate forms; `ANY`; `<<` update; `gen` rank in non-first position;
liveness-based memory reclamation (ping-pong) as an optimization pass.
