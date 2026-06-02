# edgecc Design Spec — Generational Cascades, Sub-project B

EDGE language compiler — extending in-language iteration from a **single** Einsum (Sub-project A) to a **cascade**: a sequence of Einsums that run in order within each generation, sharing one generational rank, with intermediate tensors flowing between them. This makes algorithms like **Bellman-Ford** expressible entirely in EDGE source.

This is **Sub-project B** of the generational-ranks effort. It depends on A (merged 2026-06-01) and builds directly on its machinery (generational rank, `i`/`i+1` offsets, `stop: stable(...)`, fresh-slice populate, the double-buffered loop, the safety cap).

## Background

A (`2026-06-01-edgecc-generational-ranks-a-design.md`) added a single-Einsum generational loop: one Einsum reads generation `i`, writes `i+1`, and the compiler emits a double-buffered fixed-point loop (`F_prev`/`F_next`, fresh empty slice each generation, `stable` termination, `__EDGECC_MAX_GEN__ = 1000` cap). Index classification (free vs reduction) is computed once and stored on the `SymbolTable`; IR lowering reads it; codegen emits one Einsum body inside the loop.

Many iterative algorithms need **multiple** Einsums per step. Bellman-Ford relaxes candidate distances (`T = W ⊕ D`, a Map+Reduce) and then keeps the smaller of the old and new distance (`D' = min(D, T)`, a Map). Expressing this needs a cascade: two Einsums sharing the generational rank, with `T` flowing from the first to the second within a generation.

Critically, the cross-generation "keep the smaller" combine is **just another Einsum** (`map(min, union)`). So neither the deferred `<<` ("update") compute operator nor any special accumulation populate is required — the cascade expresses accumulation explicitly. The canonical 3-step BFS would additionally need separately-deferred features (unary `¬`, a take-left compute op, `ANY`, occupancy termination), so Bellman-Ford — which needs only operators we already have — is B's demonstrator.

## Scope

**In scope:**
- A **cascade**: a sequence of two or more Einsum statements that all execute, in source order, inside one generational loop.
- Exactly **one cross-generational tensor**: the single generational tensor written at offset `i+1` (read at `i`). It carries the `init` (generation 0), is double-buffered `prev`/`next`, and is the value `run()` returns.
- Any number of **intra-generation intermediate tensors**: generational tensors written and read at offset `i` (never `i+1`). Paper-faithful notation — they carry the `gen` rank (e.g. `T[gen I, R=N]`, written `T[i,d]`). They have no `init`, are allocated fresh each generation, and are neither `run()` parameters nor returned.
- **SSA ordering** within a generation: Einsums run top-to-bottom; an intermediate must be written by an earlier Einsum before a later one reads it.
- **Per-Einsum index classification**: each Einsum classifies its own free/reduction indices (the current global single-classification is replaced).
- Einsums with **no reduction rank** (e.g. the Bellman-Ford combine `D[i+1,d] = D[i,d] . T[i,d]`): handled with the existing mandatory `map(...) reduce(...)` grammar — a `reduce` over zero reduction indices passes the map value through (`min(empty, v) = v`). No grammar change.
- One shared **generational rank** across the whole cascade.
- **Bellman-Ford** as the in-language e2e demonstrator (verified output `[0.0, 1.0, 3.0, 6.0]` on the reference graph).
- Preserving all existing behavior: non-iterative programs (e.g. GEMM) and A's single-Einsum generational programs (a cascade of length 1) compile and run unchanged.

**Out of scope (deferred):**
- The `<<` ("update") compute operator (its own future feature; even published BFS uses `OR`, not `<<`).
- **Multiple** cross-generational tensors — required for the full 3-step BFS (`F` and `P`); deferred together with the operators BFS also needs.
- Unary functions (`¬P`), the `ANY` compute op, and occupancy (`‖F‖ ≡ 0`) termination.
- Nested cascades / multiple generational ranks (`i` and `j`).
- Non-iterative multi-Einsum programs (a cascade with no generational rank) — undefined; B engages the cascade machinery only for generational programs.
- Everything already deferred in A/v2 (sparse storage, optimizations, user-defined compute, anonymous tensors, file-based init, the 7 (No,No)→Yes merges).

## Approach

**The generation loop wraps the entire cascade.** Every Einsum runs each generation, in source order. The only state carried across generations is the one cross-generational tensor (double-buffered). Every other written tensor (intermediates) is fresh each generation; inputs are parameters.

**Three tensor roles, classified automatically** from declarations + how each tensor is used across the cascade:

| Role | How identified | Storage | In `run()` |
|---|---|---|---|
| Cross-generational | gen-rank tensor written at offset `i+1` (exactly one) | `prev`/`next` double buffer | gen-0 param **and** return value |
| Intra-gen intermediate | gen-rank tensor written at offset `i` only | one buffer, fresh each generation | neither |
| Input | never written by any Einsum | the input value | parameter |

**Accumulation is explicit, not a populate trick.** A's fresh-slice populate is unchanged: each generation, the cross-gen `next` slice and every intermediate start empty and are filled by their Einsums. "Keeping the previous value" happens only because an Einsum *reads* the previous generation as an operand (`D[i+1,d] = D[i,d] . T[i,d]`). This is why no `<<` / seed-slot mechanism is needed.

**A is a special case.** A single-Einsum generational program is a length-1 cascade with one cross-generational tensor and no intermediates. The B refactor must keep A's programs (reachability, transitive closure, connected components) and the non-iterative path byte-for-byte equivalent.

## Surface language

Bellman-Ford, fully in EDGE:
```
dim N = 4;

tensor W[R=N, C=N]   : float, empty = inf;   // edge weights; inf = no edge
tensor T[gen I, R=N] : float, empty = inf;   // intra-generation intermediate (relaxed candidates)
tensor D[gen I, R=N] : float, empty = inf;   // cross-generational (distances)

init W = [[inf, 1.0, 4.0, inf], [inf, inf, 2.0, 6.0], [inf, inf, inf, 3.0], [inf, inf, inf, inf]];
init D = [0.0, inf, inf, inf];               // generation 0 (source = node 0)

T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);   // relax
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);      // combine
stop: stable(D);
```
→ prints `[0.0, 1.0, 3.0, 6.0]`

No new tokens or grammar productions: `gen`, `stop`, `stable`, `i+1` coordinates, and multiple statements all already parse. The novelty is entirely in semantic/IR/codegen handling a *sequence* of Einsums.

## Component changes

### Lexer (`lexer.l`) / Parser (`parser.y`) / AST (`ast.h/c`)
**No changes.** The grammar already parses multiple `einsum_stmt`s into multiple `NODE_EINSUM` statements; the tokens and nodes from A suffice.

### Semantic analysis (`semantic.h/c`)
The current code calls `analyze_einsum` once per `NODE_EINSUM` but stores a single global free/reduction classification and single `is_iterative`/`gen_tensor`/`gen_rank_var` on the `SymbolTable` (last Einsum wins). B replaces this with cascade-aware, per-Einsum handling:

- **Per-Einsum classification.** Free/reduction indices are computed per Einsum (not stored once globally). Store a classification record per Einsum (e.g. a parallel array on the result, indexed by Einsum order) for IR to consume, or recompute deterministically during lowering. Either way, the global `free_indices`/`reduction_indices` single-set contract is removed.
- **Cascade-level generational facts.** Across all Einsums: identify the generational rank variable (shared); identify the **cross-generational tensor** (gen-rank tensor written at offset `i+1`) and require **exactly one**; identify **intermediates** (gen-rank tensors written at offset `i`). A tensor written at `i+1` with no `init`, or more than one tensor written at `i+1`, is an error.
- **Roles & validation.**
  - Cross-gen tensor: must have an `init` (generation 0); is the returned tensor; `stop: stable(...)` must watch it.
  - Intermediate: must **not** have an `init`; must be written before it is read (SSA ordering across the cascade); read/write only at offset `i`.
  - Input: never written; resolved as today.
- **Offset rules** (generalizing A): the cross-gen tensor is written at `i+1`/read at `i`; intermediates are written and read at `i`; any other offset is an error.
- The generational rank variable is excluded from every Einsum's index classification (it is the time axis).

New `SymbolTable` fields (replacing/augmenting A's): the cross-generational tensor name, the gen-rank variable, the ordered list of intermediate tensor names, and per-Einsum classification access. A's `is_iterative`, `gen_tensor`, `stop_watch` remain (with `gen_tensor` now meaning specifically the cross-generational tensor).

### IR (`ir.h/c`)
- **One generational loop wraps all Einsums.** `IR_GEN_LOOP_BEGIN` is emitted once before the first Einsum's spatial loops; each Einsum emits its own spatial `LOOP_BEGIN … LOAD/MAP/REDUCE/STORE … LOOP_END` using **its own** free/reduction classification; `IR_GEN_LOOP_END` is emitted once after the last Einsum. (A emitted the gen loop around the single Einsum; this lifts it to wrap the sequence.)
- **Coordinate stripping for all gen-rank tensors.** Any access to a generational tensor (cross-gen or intermediate) has its generational coordinate stripped (buffers store one generation's slice). This already exists for the cross-gen tensor in A; it now also applies to intermediates.
- **Buffer-role tags.** `IRLoad.gen_read` / `IRStore.gen_write` are set **only for the cross-generational tensor** (drives `prev`/`next` naming in codegen). Intermediate accesses are lowered with stripped coordinates but `gen_read`/`gen_write` = 0, so codegen emits their plain (fresh-buffer) name. Inputs are unchanged.
- **Fresh-allocation list.** `IR_GEN_LOOP_BEGIN` carries the set of per-generation fresh allocations: the cross-generational `next` slice **and** each intermediate (name + spatial dims + empty value). Codegen emits these at the top of each generation.

### Codegen (`codegen.c`)
- **I/O discovery for cascades.** `run()` parameters = inputs (tensors never written) **plus** the cross-generational tensor (its generation-0 value). Returned = the cross-generational tensor. Intermediates (written **and** read) are excluded from both. (A's `collect_io` assumed one store = the output; this generalizes to multiple stores with role-based selection.)
- **Generation-loop scaffolding** (generalizing A):
  ```python
  def run(W, D):
      D_prev = D
      _it = 0
      while True:
          D_next = <fresh empty slice>     # cross-gen next
          T = <fresh empty slice>          # each intermediate
          ...Einsum 1 body... # relax  → T,      reads D_prev / W
          ...Einsum 2 body... # combine → D_next, reads D_prev / T
          if D_next == D_prev:
              break
          D_prev = D_next
          _it += 1
          if _it >= __EDGECC_MAX_GEN__:
              raise RuntimeError("generational loop did not stabilize")
      return D_prev
  ```
- **Skeleton naming** is unchanged from A and already handles the three roles correctly: cross-gen read → `D_prev`, cross-gen write → `D_next` (driven by `gen_read`/`gen_write`); everything else (intermediates, inputs) → plain name. Intermediates resolve to their fresh per-generation buffer.
- The cross-generational tensor's `run()` output-alloc and `__main__` dead-alloc are skipped exactly as in A; intermediates are never allocated at `run()` top level or in `__main__` (no `init`, not parameters) — only fresh inside the loop.

### Docs (`CLAUDE.md`)
- Extend the "Generational (iterative) ranks" subsection with cascades: multiple Einsums per generation, the one cross-generational tensor + intra-generation intermediates, the Bellman-Ford example.
- Update the deferred list: move single-cross-gen cascades out; keep `<<`, multiple cross-generational tensors / full BFS, unary `¬`, `ANY`, occupancy stop, and nested cascades deferred.

## Testing

- **Unit** (extend each stage's `test_*.c`):
  - `test_semantic`: per-Einsum classification for a 2-Einsum cascade; cross-gen vs intermediate role detection; negative cases (two tensors written at `i+1`; an intermediate given an `init`; reading an intermediate before it is written; `stop` watching an intermediate).
  - `test_ir`: a single `IR_GEN_LOOP_BEGIN`/`END` pair wrapping **two** Einsum bodies; the intermediate's accesses have stripped coordinates with `gen_read`/`gen_write` = 0; the fresh-allocation list contains the cross-gen `next` and the intermediate.
  - `test_codegen`: `run(W, D)` signature; `D_prev = D`; fresh `D_next` and `T` per generation; the combine Einsum reads `D_prev`/`T` and writes `D_next`; `return D_prev`.
  - Regression: A's single-Einsum generational codegen and the non-iterative GEMM codegen are unchanged.
- **E2E:** new `tests/e2e/bellman_ford/` (`input.edge` + `expected_output.txt` = `[0.0, 1.0, 3.0, 6.0]`), run through the plain e2e harness (the loop is internal). The three A examples and all existing e2e cases continue to pass.
- **Negative** (`tests/negative/`): a cascade with two `i+1`-written tensors (`more than one cross-generational tensor`); an intermediate with an `init`; an intermediate read before written.
- Add a Bellman-Ford program to `examples/`.

## Scope guardrails (restated)

One cross-generational tensor; one generational rank; offsets `i`/`i+1` only; `stable` stop; fresh-slice populate; existing compute/merge operators only; fixed iteration cap (1000). Deferred: `<<`, multiple cross-generational tensors (full BFS), unary `¬`, `ANY`, occupancy stop, nested cascades, non-iterative multi-Einsum programs.
