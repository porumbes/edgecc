# edgecc multi-action (parenthesized) Einsums — design

**Date:** 2026-06-10
**Status:** design complete — pending user review, then implementation plan
**Milestone:** paper-faithful 3-step BFS (next sub-piece after unary `!`; the biggest piece)

## Summary

Fuse a parenthesized binary sub-expression and an outer binary action into a
single Einsum statement, so the paper's `(G·F)·¬P` masked-frontier gather can be
written in one line instead of a 3-statement cascade:

```edge
F[i+1,d] = (G[s,d] .1 F[i,s]) .2 !P[i,d]
  :: map.1(and,intersect) reduce.1(or,union) map.2(and,intersect);
```

This is the largest remaining piece of paper-faithful BFS. It turns the flat
binary `LOAD/LOAD/MAP/REDUCE/STORE` pipeline into a two-action expression with a
transient intermediate.

## Scope (locked)

**Minimal: one parenthesized binary sub-expression as one operand of an outer
binary action — `(A .1 B) .2 C` and its mirror `C .2 (A .1 B)`.** Two actions,
three operands, one paren group. Each action has its own map and an optional
reduce. Covers both BFS variants:
- **push** `(G·F)·!P` — inner gathers (map+reduce), outer masks (map only)
- **pull** `(G·!P)·F` — inner masks (map only), outer gathers (map+reduce)

Out of scope: general N-ary trees (`((A.B).C).D`, `(A.B).(C.D)`), loop-order
mapping choices, map rank tags. See "Future work."

## Section 1 — Surface syntax & semantics

### Grammar

```edge
// push
F[i+1,d] = (G[s,d] .1 F[i,s]) .2 !P[i,d]
  :: map.1(and,intersect) reduce.1(or,union) map.2(and,intersect);
// pull
F[i+1,d] = (G[s,d] .1 !P[i,d]) .2 F[i,s]
  :: map.1(and,intersect) map.2(and,intersect) reduce.2(or,union);
```

- The multi-action form requires **both** parentheses (delimit the inner
  sub-expression) and **`.N` labels** on operators (`.1` = inner, `.2` = outer).
  Specs are `map.N(...)` / `reduce.N(...)`, bound to operators **by label**, so
  spec order is free. Requiring both keeps the grammar precedence-free.
- The paren group may be either outer operand (`(A.1 B).2 C` or `C.2 (A.1 B)`);
  the `.1` label always identifies the inner action regardless of side.
- `reduce.N` is present **iff** action N reduces (see consistency check).
- **Single-action Einsums are unchanged**: plain `.`, mandatory `reduce(...)`,
  byte-identical generated code. The labeled/fused form is purely additive.
- Lexing: the float rule requires a leading digit (`0.1`) and operands never
  begin with a digit, so `.1` tokenizes unambiguously as `.` + `1`, and plain `.`
  remains the single-action operator.

### Reduce rank tags (`over=`) — optional, validated, uniform

A reduce may name the rank it collapses: `reduce.N(op[, merge][, over=RANK])`.
- **Optional** everywhere — including single-action `reduce(op[, merge][,
  over=RANK])` — so the language is uniform.
- **Inferred when omitted** (the default; see inference below).
- **Validated when present:** the named rank must be one the structure actually
  reduces for that action; a mismatch is a compile error. This catches
  mis-parenthesization (which would otherwise silently reduce the wrong rank and
  produce a wrong-but-valid program).

`over=` is well-defined (a reduce collapses exactly one rank) and load-bearing
only as validation/documentation — codegen derives the same rank either way.

### Maps are not tagged (documented divergence from the paper)

The paper tags maps too (`∧¹_s`, `∧²_d`). edgecc deliberately does **not**:
- **Redundant.** The merge guard already checks each operand's existence at its
  own projected coordinates (`a_exists = a_val != empty_a`, etc.), which strictly
  subsumes "merge over rank X." Feeding a map `over=s` vs `over=d` would generate
  identical code — it selects nothing.
- **Often ill-defined.** The map-over rank is plural for elementwise
  (`A[s,d].B[s,d]` → both `s` and `d`) and empty for outer products
  (`A[s].B[d]` → none). A required map tag would force a meaningless choice.

Only the *reduced* rank changes behavior (a dimension collapses), so only
`reduce.N` carries `over=`. Explicit per-action ranks belong to a future
loop-order-mapping-choice feature, where they would actually change behavior.

### Index inference (no paren index annotation)

Inference is over **spatial indices only**; the generational index `i` is
excluded (it is the generation counter — double-buffered `_prev`/`_next`, driven
by the gen loop — not a data-loop dimension). The intermediate is a
per-generation transient carrying generation `i` implicitly, which is why the
sets below read `{d}` where the paper writes `(…)_{i,d}`.

The compiler always computes:
- `intermediate = (inner-operand idx) ∩ (outer-operand idx ∪ output idx)`
- `action-1 reduces = (inner-operand idx) − intermediate`
- `action-2 reduces = (intermediate ∪ outer-operand idx) − output idx`

Intuition: an index belongs in the intermediate iff action 1 *produces* it and
something downstream (the outer operand or the final output) still *needs* it. An
index action 1 produces but nothing downstream needs is reduced inside action 1.
An index needed downstream but not produced by action 1 enters through the outer
operand directly. Action 2 reduces whatever the final output does not keep.

This **reuses the existing per-action index classifier** — run it twice, with the
inferred intermediate set standing in as action 1's "output." When `over=` tags
are present, they are checked against `action-N reduces`.

### Consistency / validation (semantic errors, clear messages)

- `reduce.N` present ⟺ `action-N reduces` is non-empty. Else: "action 2 reduces
  `s` but has no `reduce.2`" / "`reduce.1` given but action 1 reduces nothing."
- `over=RANK` (when present) ∈ `action-N reduces`. Else: "`reduce.1 over=s`: rank
  `s` is not reduced by action 1 (it reduces `{}`)."
- Structural soundness: `(intermediate ∪ outer) − action-2-reduces == output`
  and `inner − action-1-reduces == intermediate`. Else: "actions reduce to
  `{...}` but output is `{d}`."
- Existing single-action checks unchanged.

### Intermediate empty value

The fused intermediate inherits the **output tensor's empty value** (uniform-
semiring assumption). Correct for all Boolean BFS cases (empty `0.0`); mixed-empty
fusion within a single statement is future work.

### Worked verification against the paper

#### Push-BFS — paper eqn (41)
```
F_{i+1,d} = (G_{s,d} ·¹ F_{i,s})_{i,d} ·² ¬P_{i,d}
  :: ∧¹_s AND(∩)  ∨¹_s OR(∪)  ∧²_d AND(∩)
```

| set | indices |
|---|---|
| inner = `G[s,d]` ∪ `F[i,s]` | `{s,d}` ∪ `{s}` = `{s,d}` |
| outer = `¬P[i,d]` | `{d}` |
| output = `F[i+1,d]` | `{d}` |

- `intermediate = {s,d} ∩ ({d} ∪ {d}) = {d}` → with `i`: `{i,d}` ✓
- `action-1 reduces = {s,d} − {d} = {s}` ✓ (`∨¹_s`)
- `action-2 reduces = ({d} ∪ {d}) − {d} = {}` ✓ (no `∨²`)

#### Pull-BFS — paper eqn (83)
```
F_{i+1,d} = (G_{s,d} ·¹ ¬P_{i,d}) ·² F_{i,s}
  :: ∧¹_d AND(∩)  ∧²_s AND(∩)  ∨²_s OR(∪)
```

| set | indices |
|---|---|
| inner = `G[s,d]` ∪ `¬P[i,d]` | `{s,d}` ∪ `{d}` = `{s,d}` |
| outer = `F[i,s]` | `{s}` |
| output = `F[i+1,d]` | `{d}` |

- `intermediate = {s,d} ∩ ({s} ∪ {d}) = {s,d}` ✓ (recovers the index the paper
  leaves unwritten)
- `action-1 reduces = {s,d} − {s,d} = {}` ✓ (no `∨¹`)
- `action-2 reduces = ({s,d} ∪ {s}) − {d} = {s}` ✓ (`∨²_s`)

Both recover the paper's action structure exactly.

## Section 2 — IR & codegen

### The intermediate is a transient scalar

Because the loop nest covers every index, at the point where action 1 finishes
and action 2 runs, the intermediate is a single value. **No temp tensor is
allocated** — just a scalar `_inter` whose existence is `_inter != empty`
(empty = output's empty). Its value feeds action 2's merge/compute like any
operand.

### Loop-nest structure (forced by which action reduces)

Reductions are associative/commutative, so loop *order* never changes the result;
only *which action owns which reduction loop* matters, and that is forced by the
parens + which `reduce.N` is present.

**push** — action 1 reduces `s`; its `s`-loop must close before action 2:
```python
for d in range(N):                  # free index (output)
    _acc1 = <reduce.1 identity>
    for s in range(N):              # action-1 reduction
        # action 1 map+reduce — reuse emit_general_skeleton (existence→merge→compute)
        m1 = MAP.1(G[s][d], F_prev[s])
        _acc1 = REDUCE.1(_acc1, m1)
    _inter = _acc1                  # scalar; exists iff _inter != empty
    # action 2 map-only at point d:
    F_next[d] = MAP.2(_inter, !P_prev[d])   # existence→merge→compute→store
```

**pull** — action 2 reduces `s`; action 1 is map-only *inside* the shared `s`-loop:
```python
for d in range(N):                  # free index (output)
    _acc2 = <reduce.2 identity>
    for s in range(N):              # action-2 reduction
        _inter = MAP.1(G[s][d], !P_prev[d])  # action 1 map-only → scalar per (s,d)
        m2 = MAP.2(_inter, F_prev[s])        # action 2 map
        _acc2 = REDUCE.2(_acc2, m2)
    F_next[d] = _acc2
```

General placement rule (covers both, and matrix-chain as a bonus): emit the
loop nest free-indices-outer, reduction-indices-inner; an action that reduces a
rank has its accumulator initialized before that rank's loop and finalized after;
an action whose inputs are all available inside a loop is emitted there; the
producer (action 1) is emitted at the innermost level where its result is needed.

### IR shape

Add **one structured IR node** for a fused two-action Einsum rather than
stretching the flat 5-instruction window (which is brittle). It carries:
- output ref (coords, empty, gen-write);
- three operand refs (inner-left, inner-right, outer), each with coords,
  empty value, `negate` flag, and `gen_read` kind;
- per-action: map (op, merge) and optional reduce (op, merge, identity);
- the index classification: free indices, action-1 reduce indices, action-2
  reduce indices, intermediate indices, and which outer side holds the paren.

A dedicated codegen emitter expands this node into the loop nest above, calling
the existing `emit_general_skeleton` (or a refactored shared helper) for each
action's existence/merge/compute body so the merge logic stays DRY. The
single-action lowering and emitter are untouched.

### AST / parser / lexer

- **Lexer:** no new tokens strictly required — `.1` is `.`+`1`; `over` is a soft
  keyword (a plain identifier checked in the parser action), so it does not
  reserve `over` as a name.
- **AST:** extend the Einsum node (or add a sibling fused-einsum node) to hold
  two operators with labels, three operands, and two action specs each with
  optional reduce + optional `over` rank. Single-action construction is
  preserved.
- **Parser:** add the labeled multi-action production and the `map.N`/`reduce.N`
  spec list; extend the reduce arg grammar to accept an optional trailing
  `, over=IDENT` (after the optional merge arg). LALR(1) distinguishes `, IDENT`
  (merge) from `, IDENT '=' IDENT` (over tag) on one-token lookahead.

## Error handling

All validation is at compile time (semantic stage) with specific messages, listed
under "Consistency / validation" above. Surface (parser) errors: missing label on
a multi-action operator, a `.N` spec with no matching operator, malformed
`over=`. Runtime behavior is unchanged (no new runtime checks).

## Testing

- **Unit — lexer:** `.1`/`.2` tokenize; `over` is usable as a tensor/rank name
  outside the tag position (soft keyword).
- **Unit — parser:** labeled operators + tagged/untagged reduces build the right
  AST (both push and pull shapes; paren on either side).
- **Unit — semantic:** each validation error fires (missing `reduce.N`; `over=`
  naming a non-reduced rank; shape mismatch).
- **Unit — IR:** the fused node carries both actions, three operands, and the
  computed index classification (intermediate/free/reduce sets) for push and pull.
- **Unit — codegen:** generated push and pull code match the structures above
  (accumulator placement, `_inter` plumbing, store).
- **E2E — differential oracle (key):** `examples/push_bfs_fused.edge` (one fused
  statement) must produce **byte-identical** output to the existing
  `examples/push_bfs.edge` (3-statement cascade) on the same graph. Add
  `examples/pull_bfs_fused.edge` and assert it yields the same reachability set.
- **E2E — `over=` round-trip:** the same kernel with and without `over=` tags
  produces identical output.
- **Negative:** missing `reduce.N` when an action reduces; `over=` on a
  non-reduced rank; structural shape mismatch; unlabeled multi-action operator.
- **Regression:** all existing single-action e2e/unit/negative/iterative/trace/viz
  suites stay green; non-fused programs are byte-identical (the new emitter only
  runs for the fused node).

## Scope / limitations

- One paren group, two actions, three operands; paren on either outer side.
- `over=` optional + validated on any reduce; maps untagged.
- Intermediate inherits the output's empty value (uniform semiring).
- Both-actions-reduce (matrix-chain `(A·B)·C` contracting twice) falls out of the
  same emitter; we add one bonus test, but push/pull BFS are the guaranteed
  targets.
- No general N-ary nesting, no loop-order mapping choices, no map rank tags.

## Future work

- General N-ary / nested expression trees with recursive lowering.
- Loop-order mapping choices (the paper's push→pull derivation) — the coherent
  home for explicit per-action rank/loop-order annotations.
- Mixed-empty fused intermediates across differing semirings.

## Reconciliation with existing tests/docs

- `CLAUDE.md`: document the multi-action form under EDGE Syntax; move
  "multi-operation expressions" out of the v2 "not yet supported" list (note
  compute labels / `<<` remain unsupported).
- `docs/roadmap.md`: update the paper-faithful BFS row — multi-action done; add an
  Implemented bullet; the remaining BFS gaps become multi-output return,
  `ANY`/parent, occupancy stop.
- Keep the cascade `examples/push_bfs.edge` as the differential oracle beside the
  new fused examples.
