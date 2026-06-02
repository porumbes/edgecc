# edgecc Design Spec — Generational Ranks, Sub-project A

EDGE language compiler — bringing **iteration into the EDGE source**. Today, iteration lives entirely outside the language: the YAML/Python harness (`tests/e2e/iterative/`) feeds a kernel's output back as input and stops when a watched tensor is stable. This spec adds a **generational (iterative) rank** to a *single* Einsum so the compiler emits the fixed-point loop itself.

This is **Sub-project A** of a two-part effort. A delivers single-Einsum generational loops; the follow-on **Sub-project B** (multi-Einsum cascades over shared generational ranks — real 3-step BFS) depends on A and is out of scope here.

## Background

The EDGE paper (`tex/bfs-example/07-iteration.tex`, `tex/6-syntax.tex`) expresses iteration by adding a special **generational rank** `I` to the iteration space. A tensor that changes from iteration to iteration gains that rank; you read generation `i` and write generation `i+1` (`F[i+1,d] = G[s,d]·F[i,s] :: …`). Generation 0 is initialized, and a **diamond stopping predicate** (`◇: …`) controls when the loop ends. Adding a generational rank collapses the unrolled `F1, F2, F3, …` cascade into one tensor whose generational dimension is unbounded — its size is not known until runtime.

The compiler today (v2) handles exactly **one** Einsum per program (`EinsumData` carries one output + two inputs, no notion of `i`). The default **populate** action — placement of reduced values into the output tensor — is already implemented: the REDUCE→STORE tail of the inner-loop skeleton (`codegen.c:343-371`) performs the reduction **in place in the output slot**, reading `state = Z[coords]`, merging via the reduce operator, and writing back, once per reduction-rank step. There is no separate reduce temporary; the output slot is the running accumulator, and its starting value is the tensor's `empty_val` (set once when the output is allocated).

## Scope

**In scope:**
- A generational rank declared per-tensor with an explicit `gen` keyword and **no dimension**: `tensor F[gen I, R=N] : float, empty = 0.0;`.
- Index arithmetic in tensor refs: read at `F[i, s]` (offset 0) and write at `F[i+1, d]` (offset +1).
- Generation-0 initialization via the existing `init` statement: `init F = […];` provides the generation-0 slice.
- A stop clause `stop: stable(F);` — the loop ends when the watched (generational) tensor is unchanged from generation `i` to `i+1`.
- **Fresh-slice populate** across generations: at the start of each generation the output slice is allocated **empty**; only this generation's contributions land in it. (See Approach — this is the literal paper default; it is *not* accumulation.)
- Codegen of a double-buffered fixed-point loop inside `run()`, returning the final state. Internal iteration replaces the external harness loop for this class of program.
- A **hard iteration cap** (`__EDGECC_MAX_GEN__ = 1000`, fixed for A) that raises `RuntimeError` if the watched tensor has not stabilized — fails loud rather than hanging or returning a half-converged answer.
- A new in-language e2e demonstrator: **boolean reachability / transitive closure** over a reflexive adjacency matrix (runs through the *plain* e2e harness, since the loop is now internal).
- Unit, e2e, and negative tests across all pipeline stages; `--dump-ir` support for the new IR instructions; CLAUDE.md documentation.

**Out of scope (deferred):**
- **Sub-project B:** multi-Einsum cascades, intermediate tensors flowing within a generation, SSA-per-Einsum ordering (the real 3-step BFS with the `P` visited tensor).
- The `<<` **update / accumulation populate** (seed generation `i+1` from generation `i`). This is what bfs_simple/bfs_disconnected need to accumulate distances; under fresh-slice populate they remain external-harness tests until `<<` or B lands.
- More than one generational tensor or more than one generational rank per program.
- Generational offsets other than read `i` / write `i+1` (e.g. `i+2`, write-at-`i`).
- Stop conditions other than `stable` (e.g. the paper's occupancy predicate `‖F‖ ≡ 0`, epsilon convergence).
- A source-configurable or runtime iteration cap.
- Everything already deferred in v2 (sparse storage, optimizations, user-defined compute, anonymous tensors, unary functions, the 7 (No,No)→Yes merges, file-based init).

## Approach

**`<<` is buffer initialization, not a new operator.** Because reduction is performed in place in the output slot, the *entire* difference between fresh-slice and `<<` accumulation is what the slot contains before a generation's reduction begins:
- **Fresh slice (A):** `F_next` allocated to `empty_val` at the top of each generation → first `state` read is empty → the generation accumulates only its own contributions over the reduction rank.
- **`<<` accumulation (deferred):** `F_next` seeded from `F_prev` → first `state` read is the previous generation's value → the *same* reduce operator merges this generation's contributions on top of the carried-forward state.

A implements only fresh slice. No new compute/merge operator and no new code path for reduction is introduced; the change is the buffer-management wrapper plus the `empty`-init of `F_next` each generation.

**Why reachability, not BFS, is A's demonstrator.** Under fresh-slice populate a single-Einsum BFS does *not* accumulate distances — `F` propagates a wavefront and drains to empty (`[0,inf,inf,inf]→[inf,1,1,inf]→[inf,inf,2,2]→…`). Reachability over a **reflexive** adjacency matrix *does* reach a fixed point under fresh slice: because `G[d,d]` is set, each generation re-derives the previous frontier and adds one hop, so `F` grows monotonically and goes stable at the reachable set:

```
F[0] = {0}  →  {0,1,2}  →  {0,1,2,3}  →  {0,1,2,3}  (stable)
```

**Double-buffered, bulk-synchronous.** Reads come from `F_prev` (generation `i`), writes go to `F_next` (generation `i+1`); they are distinct buffers, so each generation is computed bulk-synchronously (Jacobi-style), matching the harness's `F_new = run(F); F = F_new` shape. The non-iterative codegen path is untouched; generational support is an additive branch keyed off a SymbolTable flag.

## Surface language

```
dim N = 4;

tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;

init F = [1.0, 0.0, 0.0, 0.0];   // generation 0

F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);
stop: stable(F);
```

Three additions to v2 syntax:
1. A `gen <NAME>` rank entry inside a tensor's bracket — generational, carries no `=dim`.
2. Index arithmetic `i` / `i+1` in tensor-ref coordinates (existing rule unchanged: lowercase rank vars bind to declared rank names by position; the generational rank var binds to the `gen` rank).
3. A `stop: stable(<tensor>);` statement.

## Component changes

### Lexer (`lexer.l`)
- New keywords (matched before the `IDENT` rule): `gen`, `stop`, `stable`.
- Add `+` to the single-char token set (`[=\.\,\;\:\[\]\(\)+]`). The `1` in `i+1` is already `TOK_INT_LIT`.

### Parser (`parser.y`)
- `rank_pair_list`: new alternative `TOK_GEN TOK_IDENT` → a `RankPair` with `is_generational = 1` and `dim_name = NULL`.
- Tensor-ref coordinates: a coordinate becomes either a bare rank var *or* `TOK_IDENT '+' TOK_INT_LIT`. The offset is captured structurally (see AST). For A only `+1` is accepted; other offsets are a semantic error.
- New `stop_stmt`: `TOK_STOP ':' TOK_STABLE '(' TOK_IDENT ')' ';'`, added to the `stmt` alternatives.

### AST (`ast.h/c`)
- `RankPair`: add `int is_generational;` (with `dim_name == NULL` when set).
- `TensorRef`: add `int gen_axis;` (index of the generational coordinate, −1 if none) and `int gen_offset;` (0 = read generation `i`, 1 = write generation `i+1`). Base names remain in `rank_vars`.
- New `NODE_STOP` node `{ char *watch_tensor; }` (only `stable` for A). Constructor + destructor.

### Semantic analysis (`semantic.h/c`)
Validations (each gets a negative test):
- Exactly **one** generational tensor and **one** generational rank.
- The generational tensor appears as **write offset +1** in the output and **read offset 0** in an input, same tensor name; any other offset is rejected.
- `stop: stable(T)` must name the generational tensor.
- The generational rank is excluded from the free-vs-reduction index classification (it is the time axis, not an inner loop); spatial index classification is unchanged.

SymbolTable records: `is_iterative`, generational tensor name, generational-rank name, stop-watch tensor, and the spatial shape used to allocate the fresh slice.

### IR (`ir.h/c`)
- New `IR_GEN_LOOP_BEGIN` / `IR_GEN_LOOP_END` bracketing the existing spatial `LOOP_BEGIN … STORE … LOOP_END`. Carries: generational tensor name, stop-watch name, and the safety cap.
- The generational coordinate is **stripped** during lowering: `F[i,s]` → an `IR_LOAD` with coords `[s]` tagged `gen_read`; `F[i+1,d]` → an `IR_STORE` with coords `[d]` tagged `gen_write`. Add `int gen_read;` to `IRLoad` and `int gen_write;` to `IRStore`.
- `ir_dump` (`--dump-ir`) extended to print the new instructions and tags.

### Codegen (`codegen.c`)
Non-iterative path unchanged. Iterative branch (keyed off the SymbolTable flag, surfaced through the IR) emits:

```python
__EDGECC_MAX_GEN__ = 1000   # safety cap (fixed for A)

def run(G, F):              # input order follows operand order (G left, F right); F = generation 0
    F_prev = F
    _it = 0
    while True:
        F_next = [0.0] * N            # fresh empty slice (emit_alloc, spatial dims only)
        for s in range(N):
            for d in range(N):
                ...                   # body: reads F_prev / G, populates F_next via reduce op
        if F_next == F_prev:          # stable: Python nested-list ==, inf == inf is True
            break
        F_prev = F_next
        _it += 1
        if _it >= __EDGECC_MAX_GEN__:
            raise RuntimeError("generational loop did not stabilize")
    return F_prev
```

- Generational-tensor **reads → `F_prev`**, **writes → `F_next`**. The skeleton's `state = F_next[d]` starts at `empty_val` (fresh slice), so accumulation happens only across the reduction rank within a generation — exactly as the existing default populate does.
- The iterative branch **owns buffer allocation**; it does **not** emit the normal output-alloc, because `F` arrives as the generation-0 parameter (and `F` is therefore both an input and the returned output).
- Stable comparison uses Python `==` on nested lists (handles `inf == inf` as `True`).
- `__main__` builds `G` and `F` from `init`, calls `run`, prints the final `F` — same machinery as today.

### Docs (`CLAUDE.md`)
- New syntax + a "Generational ranks" subsection.
- ABI note: iterative kernels loop internally and return the final state.
- Move single-Einsum generational ranks **out** of the deferred list; note that cascades (B) and the `<<` update populate remain deferred.
- (Optional follow-up, not required for A) add `gen`/`stop`/`stable` to the VS Code grammar + completions.

## Testing

- **Unit** (extend each stage's `test_*.c`):
  - `test_lexer`: `gen`, `stop`, `stable`, `+` tokens.
  - `test_parser`: generational rank declaration, `i+1` coordinate, `stop` statement.
  - `test_semantic`: generational classification + all negative validations.
  - `test_ir`: `IR_GEN_LOOP_BEGIN/END`, `gen_read`/`gen_write` tags, gen-coordinate stripping.
  - `test_codegen`: while-loop emission, fresh `F_next` alloc, stable compare, cap.
- **E2E:** new `tests/e2e/reachability/` (`.edge` + `.expected`), run through the **plain** e2e harness (internal loop). Expected output = the transitive-closure reachable set.
- **Negative** (`tests/negative/`): stop on a non-generational tensor; generational tensor used without offsets; bad offset (`i+2`, write-at-`i`); two generational tensors; generational rank given a dimension.

## Scope guardrails (restated)

One generational tensor, one generational rank, offsets `i` / `i+1` only, `stable` stop only, single Einsum, fresh-slice populate (no `<<` / accumulation), fixed iteration cap of 1000.
