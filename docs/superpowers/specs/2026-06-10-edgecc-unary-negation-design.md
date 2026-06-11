# edgecc unary negation (`!`) — design

**Date:** 2026-06-10
**Status:** design — approved to write up; implementation plan to follow
**Milestone:** paper-faithful 3-step BFS (first sub-piece; continues the A/B/C line)

## Summary

Add a unary **complement** operator, surfaced as `!`, applied as a per-operand
prefix on either operand of an Einsum map:

```edge
// today's masking trick (push_bfs.edge):
F[i+1,d] = C[i,d] . P[i,d]  :: map(take_left, take_left_only) reduce(or, union);
// paper-faithful with unary !:
F[i+1,d] = C[i,d] . !P[i,d] :: map(take_left, intersect)      reduce(or, union);
```

`!P` is the paper's `¬P`. It lets BFS gather-and-mask read the way the paper
writes it (filter out already-visited nodes by intersecting with `¬P`) instead
of relying on the `take_left_only` merge encoding. This is the first of the
remaining sub-pieces of paper-faithful BFS; the others (multi-action Einsums,
multi-output return, `ANY`/parent pointers, occupancy termination) are separate
specs.

## Motivation

The paper writes BFS's masked frontier as `(G·F)·¬P` (`tex/7-examples.tex`), and
the appendix notes that all 16 merge operators can be expressed with a unary
complement plus a simpler merge. edgecc can already express masked BFS, but only
via the `take_left_only` merge — a correct-but-indirect encoding. Unary `!` is
the missing *notation* that closes the gap between what the compiler accepts and
how the paper expresses the algorithm. It is also the primitive the eventual
fused `(G·F)·!P` multi-action statement will build on.

## Surface syntax

- **Token:** `!` (ASCII). It is unused in both the EDGE grammar (the paper uses
  `¬`/`\neg` exclusively) and edgecc's lexer/parser, so it is free.
- **Placement:** prefix on a map **operand** only — `op . !op` / `!op . op` /
  `!op . !op`. Not allowed on the output (LHS) tensor ref.
- **Glyph note (VS Code, future):** the editor extension may *render* a typed `!`
  as `¬` for paper fidelity. Out of scope here; tracked as extension future-work.
- The paper glyph `¬` is **not** accepted as an alias in this pass (ASCII-only,
  YAGNI). Easy to add later if wanted.

## Semantics: existence-flip

edgecc's data model conflates existence with the empty sentinel: a coordinate
"exists" iff its value `!= empty` (`codegen.c`: `x_exists = x_val != empty`).
The formal map semantics (`tex/semantics/3-map.tex`) define merge as a function
of **existence Booleans only** — `Merge(∃_A, ∃_B) → 𝔹` — with compute acting on
values separately, plus a second rule that omits the result if the computed
value lands in the output's empty space.

Unary `!` lives in that **existence space**. For an operand `X` carrying `!`,
substitute into the unmodified map definition:

- **effective existence** `ε_X := ¬∃_X(cs)` — present exactly where `X` is empty;
- **effective value** `ν_X := ε_X ? τ : eˣ`, where `τ` is the operand type's
  "true/present" marker (`1.0`).

Everything downstream (the merge guard, compute, the maptmp-omission rule) is
unchanged; it just sees `(ε_X, ν_X)` in place of `(∃_X, dv_X)`.

Consequences:
- `A · !B :: map(c, intersect)` ⇒ guard `∃_A ∧ ¬∃_B` = keep where A present and B
  absent. This is **exactly** `take_left_only`. That equivalence is the appendix's
  (`appendix/0-merge.tex:126`) and is our regression oracle (see Testing).
- For value-preserving masks, pair `!` with `take_left` or `and` so the result's
  value comes from the *other* operand and `τ` is never observed in the output.
  (`τ` only needs to be truthy and non-empty; its magnitude is immaterial in the
  intended uses.)

### Why not the §5.2 "take-left merge" reading

The tutorial (`bfs-example/05-merge-compute.tex`) defines `¬B` as sugar for a
degenerate map `(B·1) :: ∧ ¬(←)` — a **take-left merge** with `¬` as the
*compute*. Traced through the formal definition in edgecc's existence-conflated
model, that reading makes `¬B` **empty everywhere**: where B exists it is "true",
so `¬(true)=false=empty` is omitted; where B is absent the take-left guard is
false. The §5.2 sugar is only well-defined in the paper's *dense-Boolean* model,
where every coordinate exists and existence ≠ value. Since edgecc has no
"exists-but-false" representation, we take the appendix's existence-based reading
instead — which is equivalent in the dense-Boolean world and correct here.

This is documented because it looks like a deviation from the paper but is in
fact the faithful realization of the appendix's named-operator semantics under
edgecc's data model. **No existing code is wrong:** the merge machinery is
already correctly existence-based, and the C2 `take_left`/`take_left_only` work
took the same correct path. `!` is a clean additive change, not a fix.

### Type / empty interaction

`!` keys off *empty vs non-empty*, so it works on any declared type — including
the `float, empty = 0.0` Boolean-ish tensors the real BFS examples use (`P` in
`push_bfs.edge`). It is **not** restricted to declared-`boolean` tensors. `τ`
is `1.0`. `!` is intended for set-/Boolean-valued tensors whose `empty`
represents "absent/false"; a tensor declared `empty = 1.0` then negated is
degenerate (τ would equal its empty) — documented as a limitation, not guarded.

## Implementation

A per-operand `negate` flag threaded through the pipeline; the only behavioral
change is two lines in the inner-loop emitter.

1. **Lexer** (`src/lexer.l`): add `!` to the punctuation char class on line 51
   (`[=\.\,\;\:\[\]\(\)+*!]`) so it returns as a single-char token.

2. **AST** (`src/ast.h/c`): add `int negate;` to `TensorRef` (default 0,
   initialized in `tensor_ref_new`). The output ref keeps `negate = 0`.

3. **Parser** (`src/parser.y`): introduce a `map_operand` nonterminal used for
   the two operand positions in `einsum_stmt`:
   ```
   map_operand: tensor_ref            { $$ = $1; }
              | '!' tensor_ref        { ((TensorRef*)$2)->negate = 1; $$ = $2; }
   ```
   The einsum rule becomes `tensor_ref '=' map_operand '.' map_operand ::…`.
   Output position stays `tensor_ref`, so `!` on the LHS is a syntax error.

4. **Semantic** (`src/semantic.c`): no new validation required — `!` is type- and
   merge-agnostic. (Optional, low priority: a `nopass`-style note if `!` is used
   on a tensor with `empty = 1.0`, since τ collides; defer unless trivial.)

5. **IR** (`src/ir.h/c`): add `int negate;` to `IRLoad`; populate from
   `input1.negate` / `input2.negate` at `ir.c:170-183`. Surface it in
   `--dump-ir` (e.g. `LOAD(r1, P [neg], …)`).

6. **Codegen** (`src/codegen.c`, `emit_general_skeleton`): for an operand whose
   load has `negate`, emit existence-flip and τ:
   ```python
   # normal:           x_exists = x_val != eX ;  x_op = x_val if x_exists else eX
   # negated:          x_exists = x_val == eX ;  x_op = 1.0   if x_exists else eX
   ```
   No change to `emit_merge_guard` or `compute_op_pyexpr`. Both operands handled
   independently so `!a . !b` works.

## Testing

- **Differential oracle (the key test):** for representative computes/merges,
  `Z = A . !B :: map(c, intersect)` must produce **byte-identical** output to
  `Z = A . B :: map(c, take_left_only)`. Add an e2e pair asserting this across
  the four existence combinations (mirrors the `A=[1,2,0,0] B=[3,0,5,0]` check
  done during design → `[0,2,0,0]`).
- **Unit:** lexer emits the `!` token; parser sets `negate` on the right operand
  and rejects `!Z[...] = …`; IR carries `negate`; codegen emits the flipped
  existence line. A `--dump-ir` golden for a negated operand.
- **Paper-faithful BFS e2e:** add `examples/push_bfs_neg.edge` — `push_bfs.edge`
  with the mask rewritten as `C . !P :: map(take_left, intersect)` — and assert
  it matches `push_bfs`'s expected output (same graph, same result).
- **Negative:** `!` on the output ref → syntax error with a clear message.
- **Regression:** all existing e2e/unit/negative/iterative/trace/viz suites stay
  green; non-`!` programs are byte-identical to before (the emitter only branches
  when `negate` is set).

## Scope / limitations

- `!` applies only to a **map operand**, one prefix per operand. No `!` on the
  output, no `!` on a whole parenthesized sub-expression (that needs multi-action
  Einsums — separate spec).
- ASCII `!` only; no `¬` alias yet.
- `τ = 1.0`; intended for tensors whose `empty` means absent/false.
- Single-operation Einsums only (unchanged from v2). Fusing `(G·F)·!P` into one
  statement is the *next* sub-piece and depends on this one.

## Future work

- Multi-action / parenthesized Einsums to write the fully fused paper BFS.
- `¬` glyph as an accepted alias; VS Code rendering of `!` as `¬`.
- VS Code grammar/autocomplete awareness of the `!` prefix.

## Reconciliation with existing tests/docs

- Update `CLAUDE.md` (EDGE Syntax v2: built-in operators / scope limitations) to
  document `!` and remove unary `¬` from the "not yet supported" list.
- Update `docs/roadmap.md`: unary `¬` moves from ❌ to ✅ under BFS.
- `push_bfs.edge`'s comment about the `take_left_only` trick stays valid (it is
  still a supported encoding); the new `push_bfs_neg.edge` shows the `!` form
  beside it.
