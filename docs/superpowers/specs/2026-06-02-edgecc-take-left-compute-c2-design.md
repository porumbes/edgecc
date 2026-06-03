# edgecc Design Spec — Take-Left Compute Operator (Sub-project C2)

EDGE language compiler — add a **take-left compute operator** (`take_left`, the
paper's `←`) that returns its left operand's value unchanged. Paired with the
existing `take_left_only` (or `intersect`) merge, it expresses a **value-preserving
mask** — "emit the left value wherever the merge admits it, regardless of what
that value is."

This is **Sub-project C2** of the "full 3-step BFS" milestone. It builds on C1
(multiple cross-generational tensors, merged 2026-06-02). It does **not** depend
on C1 mechanically — `take_left` is a self-contained compute operator usable in
any Einsum — but its motivating use is the C1-style frontier mask.

## Background

C1 showed that pruned/efficient BFS is already expressible today: the existing
`take_left_only` merge (truth table "left ∧ ¬right") encodes the `¬visited` mask,
so `F[i+1,d] = C[i,d] . P[i,d] :: map(or, take_left_only)` keeps gathered
candidates `C` only where `P` is empty. Verified working: pruned push-BFS
(`[1]×6`) and depth-valued BFS (`[0,1,1,2,2,3]`).

But that mask leans on a **trick**: `map(or, take_left_only)` relies on the `or`
compute passing the left value through — which only works because the masked
values happen to be truthy (≥1, or depth ≥1). The `or` silently drops a masked
value of `0.0` (`0.0 or <empty>` evaluates to `<empty>`), and corrupts whenever
the operand's empty value is not falsy-friendly. The paper's primitive for this
is the **take-left compute operator** `←`: it emits the left value *as-is*, so
the mask is correct for any value type. C2 adds it.

The current compute operators are `add, sub, mul, min, max, and, or`
(`OperatorType` in `ast.h`); merges include `take_left_only` already. There is no
compute operator that returns an operand unchanged — that is the single missing
primitive.

## Scope

**In scope:**
- A new compute operator **`take_left`** (`OperatorType` value `OP_TAKE_LEFT`)
  that, as a **map** operator, produces its left operand's value unchanged. In
  the generated Python the map computed value is simply the left operand
  expression.
- **Map-only.** Using `take_left` as a **reduce** operator is a semantic error.
  Reduce folds `state = take_left(state, val) = state`, which collapses to the
  reduction identity constant and is meaningless. The error message names it a
  map-only operator.
- The canonical value-preserving mask `map(take_left, take_left_only)` and the
  filter form `map(take_left, intersect)` (the paper's `←(∩)`).
- An **efficient push-BFS** example/e2e rewritten with `map(take_left, take_left_only)`.
- A focused test proving necessity: masking a tensor whose meaningful value is
  `0.0` — `take_left` preserves it where `map(or, take_left_only)` drops it.
- Docs: add `take_left` to the compute-operator list (noting map-only).

**Out of scope (deferred):**
- `take_right` compute operator (trivially symmetric; not needed — YAGNI).
- Unary `¬` / unary functions (paper-faithful `¬P` notation; adds no capability
  over the `take_left_only` encoding).
- Multi-action / fused Einsums (`(G·F)·¬P` in one statement).
- The `ANY` compute operator, coordinate-valued outputs (parent-pointer BFS),
  occupancy (`‖F‖≡0`) termination, `<<` update — all unchanged from prior scope.

## Surface language

No new tokens or grammar productions. A compute operator is already parsed as an
identifier in the `map(<compute>, <merge>)` / `reduce(<compute>, <merge>)`
position via `operator_from_str` (`parser.y:272,274`). `take_left` becomes a new
recognized name there. Example mask Einsum:

```
F[i+1, d] = C[i,d] . P[i,d] :: map(take_left, take_left_only) reduce(min, union);
```

`take_left` is the *compute*; `take_left_only` is the *merge*. They share a name
but occupy different argument slots — this matches the paper's `←(take_left_only)`
reading. (The `reduce(min, union)` here has no reduction index, so it passes the
map value through, as with the Bellman-Ford combine.)

## Component changes

### Lexer (`lexer.l`) / Parser (`parser.y`)
**No changes.** Compute-operator names are arbitrary identifiers resolved by
`operator_from_str`; the grammar is untouched.

### AST (`ast.h` / `ast.c`)
- `ast.h`: add `OP_TAKE_LEFT` to the `OperatorType` enum (after `OP_OR`).
- `ast.c`: `operator_from_str` maps `"take_left"` → `OP_TAKE_LEFT`; `operator_to_str`
  adds `case OP_TAKE_LEFT: return "take_left";`.

### Semantic analysis (`semantic.h` / `semantic.c`)
- `compute_op_identity`: add an explicit `case OP_TAKE_LEFT:` so the switch stays
  exhaustive. Its return value is never consumed (see the reduce guard), so it
  returns a defined placeholder (`0.0`).
- **Reduce guard.** During Einsum analysis, if the **reduce** operator is
  `OP_TAKE_LEFT`, emit an error:
  `"take_left is a map-only compute operator; it cannot be used as a reduce operator"`.
  Using `take_left` as the **map** operator is always allowed. (Determine the
  exact site by where `reduce_op` is available during `analyze_einsum`; the IR
  must not be lowered for a program that fails this check, consistent with other
  semantic errors halting codegen in `main.c`.)

### IR (`ir.h` / `ir.c`)
**No changes.** `OperatorType` flows through `IRMap.op` / `IRReduce.op`
unchanged; `OP_TAKE_LEFT` is carried like any other compute op.

### Codegen (`codegen.c`)
- `compute_op_pyexpr`: add `case OP_TAKE_LEFT: snprintf(out, cap, "%s", lhs); break;`
  — the computed value is exactly the left operand expression. This is the entire
  runtime behavior. The surrounding merge skeleton already decides *whether* to
  emit (via the merge op) and supplies the correct left-operand value; `take_left`
  only chooses *what* value to store when emission happens.

### Docs (`CLAUDE.md`)
- Add `take_left` to the built-in compute-operator list, annotated **map-only**.
- Update the deferred/limitations list: the take-left compute op is now available;
  keep unary `¬`, multi-action Einsums, `ANY`, occupancy stop, and `<<` deferred.

## Testing

- **Unit** (`tests/unit/`):
  - `test_semantic`: `take_left` is accepted as a map operator (a program using
    `map(take_left, take_left_only)` validates with 0 errors); `reduce(take_left)`
    produces an error.
  - `test_codegen`: an Einsum with `map(take_left, ...)` emits the left operand as
    the computed value. The map skeleton names the left operand `a_op`, so the
    generated body contains `map_val = a_op` (assert against the actual emitted
    text).
  - (Optional) `test_parser`/AST round-trip: `operator_from_str("take_left")` and
    `operator_to_str(OP_TAKE_LEFT)` agree.
- **E2E** (`tests/e2e/`):
  - `push_bfs_takeleft/`: efficient push-BFS using `map(take_left, take_left_only)`
    for the mask; expected reachability output. (Same graph as `reachability_bfs`
    so the result is comparable: `[1.0]×6`.)
  - `take_left_zero_preserve/`: the necessity test — a small program that masks a
    tensor containing a meaningful `0.0` value; `take_left` preserves the `0.0`
    where the `or` trick would drop it. Expected output makes the preserved `0.0`
    observable.
- **Negative** (`tests/negative/`): `reduce_take_left/` — a program using
  `reduce(take_left, ...)`; expected error mentions "map-only".
- Add the efficient push-BFS program to `examples/`.

## Scope guardrails (restated)

One new compute operator `take_left`; map-only (reduce is an error); no grammar
or IR changes; emits the left operand verbatim in codegen. Deferred: `take_right`,
unary `¬`, multi-action Einsums, `ANY`, coordinate-valued outputs, occupancy
termination, `<<`.
