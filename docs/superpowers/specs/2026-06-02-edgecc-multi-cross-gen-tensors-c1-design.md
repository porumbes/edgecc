# edgecc Design Spec — Multiple Cross-Generational Tensors (Sub-project C1)

EDGE language compiler — generalizing a generational **cascade** from *exactly one*
cross-generational tensor (Sub-project B) to *N* of them, including the
**next-reads-next** dependency where a later Einsum in a generation reads the
just-written `i+1` value of an earlier Einsum's cross-generational tensor.

This is **Sub-project C1**, the structural spine of the larger "full 3-step BFS"
milestone. It depends on B (`2026-06-01-edgecc-generational-cascades-b-design.md`,
merged 2026-06-01) and reuses its machinery: the generational rank, `i`/`i+1`
offsets, `stop: stable(...)`, fresh-slice populate, the double-buffered loop, and
the `__EDGECC_MAX_GEN__ = 1000` safety cap.

## Background

B added cascades with **exactly one** cross-generational tensor: the single
generational tensor written at `i+1` (read at `i`), carrying the `init`,
double-buffered `prev`/`next`, and returned by `run()`. Any number of
*intra-generation intermediates* (gen-rank tensors written and read at `i`) may
flow between Einsums, but only one tensor crosses the generation boundary. The
"exactly one" rule is enforced at `semantic.c:286–292`; generational inputs are
required to be read at offset `0` at `semantic.c:308–316`.

Breadth-first search needs **two** cross-generational tensors — a frontier `F`
and a visited set `P` — both written at `i+1` and read at `i`, with the visited
update reading the *freshly computed* frontier (`P[i+1] = P[i] · F[i+1]`). C1
delivers exactly that structural generalization, and nothing more: it is purely
about supporting N cross-generational tensors and the next-reads-next edge. The
masking that makes BFS efficient (`¬P` / take-left) is deferred to C2.

## What C1 unlocks: naive reachability BFS

C1 alone — with **no** new operators — expresses a correct (if unpruned)
reachability BFS, by watching the monotone visited set `P` instead of relying on
the frontier emptying:

```
dim N = 6;

tensor G[R=N, C=N]   : float, empty = 0.0;   // adjacency (Boolean as 1.0/0.0)
tensor F[gen I, R=N] : float, empty = 0.0;   // frontier      (cross-generational)
tensor P[gen I, R=N] : float, empty = 0.0;   // visited set   (cross-generational)

init G = [[0.0,1.0,1.0,0.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,1.0,0.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,0.0]];
init F = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];     // generation 0: source = node 0
init P = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];     // generation 0: source already visited

F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);  // gather neighbors
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union);                          // accumulate visited
stop: stable(P);
```

`F` is recomputed each generation as the neighbor set of the current frontier;
because it is **not** pruned by visited nodes, it may oscillate, but it always
stays within the reachable set. `P` accumulates frontiers by union, so it grows
monotonically, is bounded by the reachable set, and therefore stabilizes once all
reachable nodes are in it. `stop: stable(P)` terminates on that, and `P` is the
reachability answer (`[1,1,1,1,1,1]` for this fully-reachable graph).

This is a genuine algorithm, so C1 ships with a real e2e test rather than a
contrived two-tensor program. C2's masking turns this into the efficient
push-BFS and enables depth-valued variants.

## Scope

**In scope:**
- **N ≥ 1 cross-generational tensors**: every gen-rank tensor written at offset
  `i+1`. Each carries an `init` (generation 0), is double-buffered `prev`/`next`,
  and is read at `i` (and optionally at `i+1`, see next item). B's "exactly one"
  rule is removed; the new rule is "≥ 1, each with an `init`".
- **Next-reads-next reads**: a cross-generational tensor may be read at offset
  `i+1`, but only if an **earlier** Einsum in the cascade already wrote it at
  `i+1` (SSA ordering, mirroring B's intermediate rule). Such a read resolves to
  the tensor's `_next` buffer.
- **Intra-generation intermediates** (B): unchanged — gen-rank tensors written
  and read at `i` only, no `init`, fresh each generation, never returned.
- **Per-Einsum index classification** (B): unchanged.
- **Single-valued return**: `run()` returns the one tensor named by
  `stop: stable(<T>)`. All cross-generational tensors are computed and
  double-buffered each generation; only the watched one is handed back. (Tuple /
  multi-output return is intentionally deferred — see "Out of scope".)
- **Naive reachability BFS** as the in-language e2e demonstrator.
- **Negative-test reconciliation** (see its own section): audit existing
  negative/regression tests and delete or tighten the ones that defended B's
  now-removed single-cross-gen limitation.
- Preserving all existing behavior **byte-for-byte**: non-iterative programs
  (GEMM), A's single-Einsum generational programs, and B's single-cross-gen
  cascades (Bellman-Ford) compile to identical Python.

**Out of scope (deferred):**
- Masking: unary `¬` and/or the `take_left_only` merge used as a `¬`-mask, and
  the take-left compute operator — all **C2**.
- Depth-valued BFS (carrying actual depths through the mask) — **C2/later**.
- Multi-output return shape (`return (F, P)`, list-valued `__edgecc_meta__`
  output, iterative-driver multi-output contract) — added later, when a program
  genuinely needs two outputs.
- Multi-action / parenthesized Einsums (the paper's `(G·F)·¬P` with compute
  labels), the `ANY` compute op, occupancy (`‖F‖ ≡ 0`) termination, more than one
  generational rank, the `<<` update operator.
- Everything already deferred in A/B/v2.

## Approach

The generation loop still wraps the entire cascade; every Einsum runs each
generation in source order. The change is that **more than one tensor may cross
the generation boundary**, and a later Einsum may read the `_next` buffer of one
written by an earlier Einsum in the same generation.

**Three tensor roles (B), with the cross-generational role now a set:**

| Role | How identified | Storage | In `run()` |
|---|---|---|---|
| Cross-generational | gen-rank tensor written at offset `i+1` (one **or more**) | `prev`/`next` double buffer, one pair each | gen-0 param each; **only the watched one** returned |
| Intra-gen intermediate | gen-rank tensor written at offset `i` only | one buffer, fresh each generation | neither |
| Input | never written | the input value | parameter |

**Buffer resolution by access offset.** Within a generation, a read of a
cross-generational tensor resolves by its offset: read at `i` → `_prev`
(previous generation), read at `i+1` → `_next` (this generation, already written
by an earlier Einsum). Writes always target `_next`. This generalizes B, where
cross-gen reads were always at `i` (always `_prev`).

**Backward compatibility by special-casing N == 1.** When exactly one tensor is
cross-generational, codegen emits B's existing epilogue verbatim
(`if T_next == T_prev: break` / `T_prev = T_next` / `return T_prev`), so
Bellman-Ford and A's programs are byte-for-byte unchanged. The N > 1 epilogue is
new code: swap **every** cross-gen `_prev = _next` each generation, gate the
break on **only** the watched tensor, and `return <watched>_prev`.

## Surface language

No new tokens or grammar productions. `gen`, `stop`, `stable`, `i+1` coordinates
(already legal on both inputs and outputs via the shared `ident_list` rule at
`parser.y:335,347`), and multiple statements all already parse. Two cross-gen
tensors and an `F[i+1,d]` read on the right-hand side are new *uses* of existing
syntax. The novelty is entirely in semantic/IR/codegen.

## Component changes

### Lexer (`lexer.l`) / Parser (`parser.y`) / AST (`ast.h/c`)
**No changes.** `TensorRef` already carries `gen_axis`/`gen_offset`
(`ast.h:63–64`) for inputs and output alike, and the parser already populates
them for inputs (`parser.y:317–319`).

### Semantic analysis (`semantic.h/c`)
- **Cross-generational set, not scalar.** Replace `char *gen_tensor`
  (`semantic.h:53`) with an ordered list `char *gen_tensors[MAX_TENSORS]` +
  `int num_gen_tensors`. At `semantic.c:286–292`, replace the "more than one →
  error" branch with append-if-new (dedup by name). Keep `gen_rank_var`,
  `intermediates[]`, `stop_watch`, and per-Einsum `einsum_class[]` as in B.
- **Each cross-gen tensor must have an `init`** (generation 0), validated for
  every member of the set (generalizing B's single-tensor check at
  `semantic.c:520–525`).
- **Relax the offset-0-input rule** (`semantic.c:308–316`). A read of a
  cross-generational tensor at offset `+1` is permitted **iff** that tensor is
  written at `+1` by an **earlier** Einsum in the cascade. Enforce this in
  `validate_cascade_ordering` (`semantic.c:423–459`) alongside the existing
  intermediate SSA check: track which cross-gen tensors have been written at
  `i+1` so far; an `i+1` read of one not yet written (including an Einsum reading
  its own `i+1` output) is an error
  (`"cross-generational tensor '%s' read at i+1 before it is written"`). Reads at
  `+1` with offset > 1, or `+N` reads of non-cross-gen tensors, remain errors.
- **Stop watch must name a cross-generational tensor** (already required); it
  must be a member of the set.
- New/changed `SymbolTable` fields: `gen_tensors[]` + `num_gen_tensors` replace
  `gen_tensor`. `is_iterative`, `gen_rank_var`, `intermediates[]`,
  `num_intermediates`, `stop_watch`, `einsum_class[]` are unchanged.

### IR (`ir.h/c`)
- **Cross-gen allocation list.** Replace `IRGenAlloc cross_gen` (`ir.h:89`) with
  `IRGenAlloc cross_gens[MAX_TENSORS]` + `int num_cross_gens`. Populate it at
  `ir.c:203–212` by looping over `symtab->gen_tensors[]` and calling
  `fill_gen_alloc` for each (spatial dims, empty value, name). The fresh-alloc
  list per generation is then every cross-gen `_next` slice **plus** every
  intermediate.
- **Tri-state `gen_read`.** Today `IRLoad.gen_read` is a bool meaning "cross-gen,
  read at `i` → `_prev`". Generalize so a load records whether it is a cross-gen
  read and at which offset: `0` = not cross-gen (plain name), `1` = cross-gen at
  `i` (`_prev`), `2` = cross-gen at `i+1` (`_next`). Set it at `ir.c:126,135` by
  checking membership in the cross-gen set **and** the operand's `gen_offset`.
  `IRStore.gen_write` (`ir.c:170`) stays a bool but is now set true for **any**
  cross-gen tensor's store, not just the single one.
- **Coordinate stripping** (`ir.c:124–125,133–134,167–168`) is unchanged — it
  already strips the gen axis for every gen-rank tensor by `gen_axis`.

### Codegen (`codegen.c`)
- **I/O discovery for N cross-gen tensors.** `run()` parameters = inputs (never
  written) **plus** every cross-gen tensor's generation-0 value (each is a
  parameter). The single returned value is the **watched** tensor
  (`symtab->stop_watch`). Generalize `collect_io` / the iterative override at
  `codegen.c:432–447`: filter intermediates from inputs (as today), add each
  cross-gen tensor as a parameter, and set the return to the watched tensor.
- **Per-tensor `prev`/`next` naming** (`codegen.c:311–323`). Drive operand names
  off the tri-state `gen_read`: `2` → `<name>_next`, `1` → `<name>_prev`, `0` →
  plain. Stores with `gen_write` → `<name>_next`. This makes
  `P[i+1] = P[i] · F[i+1]` emit `P_next = ... P_prev ... F_next ...`.
- **Fresh slices for every cross-gen `_next` and every intermediate** at the top
  of each generation (generalizing B, which emitted one cross-gen `_next`).
- **Epilogue, special-cased on count** (`codegen.c:552–595`):
  - **N == 1:** emit B's exact text — `if T_next == T_prev: break` /
    `T_prev = T_next` / cap check / `return T_prev`. Byte-for-byte unchanged.
  - **N > 1:** compute the watched tensor's stability into a flag, swap **every**
    cross-gen `_prev = _next`, then `if <flag>: break`, cap check, and
    `return <watched>_prev`. Sketch:
    ```python
    def run(G, F, P):
        F_prev = F
        P_prev = P
        _it = 0
        while True:
            F_next = [0.0] * N          # fresh, every cross-gen _next
            P_next = [0.0] * N
            ...Einsum 1 body...  # F_next = gather(F_prev, G)
            ...Einsum 2 body...  # P_next = combine(P_prev, F_next)
            _done = P_next == P_prev    # watched tensor only
            F_prev = F_next             # swap ALL cross-gen buffers
            P_prev = P_next
            if _done:
                break
            _it += 1
            if _it >= __EDGECC_MAX_GEN__:
                raise RuntimeError("generational loop did not stabilize")
        return P_prev                   # watched tensor
    ```
- **`__main__` / ABI.** `__edgecc_meta__["inputs"]` lists real inputs **and**
  each cross-gen tensor's gen-0 slot; `"output"` is the single watched tensor
  name (still a string). `__main__` constructs every input + each cross-gen
  gen-0 value from `init`, calls `run(...)`, and prints the single returned
  value. Each cross-gen tensor's `run()`-top output-alloc and `__main__`
  dead-alloc are skipped exactly as B did for the one cross-gen tensor.

### Docs (`CLAUDE.md`)
- Extend the "Cascades" subsection: a cascade may now write **more than one**
  cross-generational tensor; a later Einsum may read an earlier one's `i+1`
  value (next-reads-next); `run()` returns the `stop`-watched tensor. Add the
  naive reachability-BFS example.
- Update the deferred list: move "multiple cross-generational tensors" out of
  deferred; keep masking (`¬` / take-left), depth-valued BFS, multi-output
  return, multi-action Einsums, `ANY`, occupancy stop, `<<`, and >1 generational
  rank deferred.

## Negative-test reconciliation

C1 removes a limitation B enforced, so some tests that exist **only** to defend
that limitation are now wrong and must be retired or tightened. This is a
first-class task, not an afterthought. Audit `tests/negative/` and the
generational unit tests and classify each:

- **Delete (obsolete scaffolding):** any negative test asserting "more than one
  cross-generational tensor → error" (e.g. the `more than one cross-generational
  tensor` case). C1 makes such programs valid; the test must be removed, and its
  `.edge` fixture repurposed or deleted.
- **Tighten (narrower rule survives):** any negative test asserting "generational
  input read at `i+1` → error". The blanket error is gone; the remaining error is
  only "`i+1` read of a cross-gen tensor **not yet written** by an earlier
  Einsum". Rewrite the fixture so it triggers the narrower condition (e.g. an
  Einsum reading its own `i+1` output, or reading another cross-gen tensor's
  `i+1` before that tensor is written) and update the expected message.
- **Keep (still valid):** offset > 1 writes/reads, `i+1` write with no `init`,
  intermediate with an `init`, intermediate read before written, `stop` watching
  a non-cross-gen tensor — all unchanged. Add a new negative: an `i+1` read of a
  cross-gen tensor before it is written at `i+1`.

Record the classification (kept / tightened / deleted, with the reason) in the
implementation plan so the suite stays an honest record of current language
rules, not a museum of removed limitations.

## Testing

- **Unit** (extend each stage's `test_*.c`):
  - `test_semantic`: a 2-cross-gen-tensor cascade classifies both `F` and `P` as
    cross-generational; the `i+1` read of `F` in the `P` Einsum is accepted
    because `F` is written earlier; negatives — two-cross-gen is **no longer** an
    error (assert it parses/validates), an `i+1` read before write **is** an
    error, a cross-gen tensor with no `init` is an error.
  - `test_ir`: a single `IR_GEN_LOOP_BEGIN`/`END` pair; `num_cross_gens == 2`;
    the fresh-alloc list contains both `F` and `P`; the `F` load inside the `P`
    Einsum has tri-state `gen_read == 2` (next), the `P` load has `gen_read == 1`
    (prev); both stores have `gen_write == 1`.
  - `test_codegen`: `run(G, F, P)` signature; fresh `F_next`/`P_next` per
    generation; `P_next` Einsum reads `P_prev` and `F_next`; the N > 1 epilogue
    swaps both buffers and gates on `P`; `return P_prev`. **Regression:** A's
    single-Einsum generational codegen, B's Bellman-Ford codegen, and the
    non-iterative GEMM codegen are byte-for-byte unchanged (N == 1 path).
- **E2E:** new `tests/e2e/reachability_bfs/` (`input.edge` + `expected_output.txt`),
  the naive reachability BFS above, run through the plain e2e harness (the loop
  is internal). All existing e2e cases continue to pass unchanged.
- **Negative** (`tests/negative/`): per the reconciliation section — delete the
  two-cross-gen error case, tighten/replace the `i+1`-input case, add the
  `i+1`-read-before-write case.
- Add the reachability-BFS program to `examples/`.

## Scope guardrails (restated)

N ≥ 1 cross-generational tensors; one generational rank; offsets `i`/`i+1` only;
`i+1` reads allowed only after an earlier `i+1` write (SSA); single watched
return value; `stable` stop; fresh-slice populate; existing compute/merge
operators only; fixed iteration cap (1000); N == 1 output byte-for-byte
unchanged. Deferred: masking (`¬` / take-left), depth-valued BFS, multi-output
return, multi-action Einsums, `ANY`, occupancy stop, `<<`, > 1 generational rank,
and everything previously deferred in A/B/v2.
