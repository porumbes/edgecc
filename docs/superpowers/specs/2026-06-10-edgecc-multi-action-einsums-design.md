# edgecc multi-action (parenthesized) Einsums — design

**Date:** 2026-06-10
**Status:** in design (Section 1 approved; Section 2 IR/codegen in progress)
**Milestone:** paper-faithful 3-step BFS (next sub-piece after unary `!`; the biggest piece)

## Summary

Fuse a parenthesized binary sub-expression and an outer binary action into a
single Einsum statement, so the paper's `(G·F)·¬P` masked-frontier gather can be
written in one line instead of a 3-statement cascade:

```edge
F[i+1,d] = (G[s,d] .1 F[i,s]) .2 !P[i,d]
  :: map.1(and,intersect) reduce.1(or,union) map.2(and,intersect);
```

## Scope (locked)

**Minimal: one parenthesized binary sub-expression as one operand of an outer
binary action — `(A . B) . C` (and its mirror `C . (A . B)`).** Two actions
total, three operands, one paren group. Each action has its own map and an
optional reduce. This covers both BFS variants:
- **push** `(G·F)·!P` — inner action gathers (map+reduce), outer masks (map only)
- **pull** `(G·!P)·F` — inner masks (map only), outer gathers (map+reduce)

General N-ary expression trees (`((A.B).C).D`, `(A.B).(C.D)`, …) are out of scope
for this pass.

## Section 1 — Surface syntax & semantics (approved)

### Grammar

```edge
F[i+1,d] = (G[s,d] .1 F[i,s]) .2 !P[i,d]
  :: map.1(and,intersect) reduce.1(or,union) map.2(and,intersect);
```

- Operators are labeled `.1` / `.2`; action specs are `map.N(op[,merge])` and
  `reduce.N(op[,merge])`, bound to operators **by label** (so spec order is free).
- The `.N` suffix gives a clean symmetry between the `.N` operator and its
  `map.N`/`reduce.N` specs, mirroring the paper's ¹² superscripts.
- `reduce.N` is **optional** in multi-action: present iff action N reduces.
- **Single-action Einsums are unchanged**: plain `.`, mandatory `reduce(...)`,
  byte-identical generated code. The labeled form is additive.
- Lexing: the float rule requires a leading digit (`0.1`), and operands never
  begin with a digit, so `.1` tokenizes unambiguously as `.` + `1` and plain `.`
  stays the single-action operator.

### Index inference (no explicit index annotation)

Inference is over **spatial indices only**; the generational index `i` is
excluded (it is the generation counter — double-buffered `_prev`/`_next`, driven
by the gen loop — not a data-loop dimension). The intermediate is a
per-generation transient that carries generation `i` implicitly, which is why the
sets below read `{d}` where the paper writes `(…)_{i,d}`.

Rules:
- `intermediate = (inner-operand idx) ∩ (outer-operand idx ∪ output idx)`
- `action-1 reduces = (inner-operand idx) − intermediate`
- `action-2 reduces = (intermediate ∪ outer-operand idx) − output idx`

Intuition: an index belongs in the intermediate iff action 1 *produces* it and
something downstream (the outer operand or the final output) still *needs* it. An
index action 1 produces but nothing downstream needs is reduced inside action 1.
An index needed downstream but not produced by action 1 enters through the outer
operand directly. Action 2 then reduces whatever the final output does not keep.

This **reuses the existing per-action index classifier** — run it twice, with the
inferred intermediate set standing in as action 1's "output."

### Consistency check (semantic error)

`reduce.N` must be present **iff** action N has reduction indices. Mismatches are
compile errors, e.g. "action 1 reduces `s` but has no `reduce.1`" or "`reduce.2`
given but action 2 reduces nothing."

### Intermediate empty value

The fused intermediate inherits the **output tensor's empty value** (uniform-
semiring assumption). Correct for all Boolean BFS cases (empty `0.0`); mixed-empty
fusion across a single statement is future work.

### Worked verification against the paper

#### Push-BFS — paper eqn (41)
```
F_{i+1,d} = (G_{s,d} ·¹ F_{i,s})_{i,d} ·² ¬P_{i,d}
  :: ∧¹_s AND(∩)  ∨¹_s OR(∪)  ∧²_d AND(∩)
```
Paper: action 1 maps & reduces over `s`; action 2 maps over `d`, no reduce;
intermediate `{i,d}`.

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
Paper: action 1 maps over `d`, no reduce; action 2 maps & reduces over `s`. (The
paper leaves the intermediate index unannotated here — inference must recover it.)

| set | indices |
|---|---|
| inner = `G[s,d]` ∪ `¬P[i,d]` | `{s,d}` ∪ `{d}` = `{s,d}` |
| outer = `F[i,s]` | `{s}` |
| output = `F[i+1,d]` | `{d}` |

- `intermediate = {s,d} ∩ ({s} ∪ {d}) = {s,d}` ✓ (recovers the unwritten index)
- `action-1 reduces = {s,d} − {s,d} = {}` ✓ (no `∨¹`)
- `action-2 reduces = ({s,d} ∪ {s}) − {d} = {s}` ✓ (`∨²_s`)

Both recover the paper's action structure exactly.

### Open question carried into Section 2: rank tagging

The index-set inference above recovers *which indices each action reduces*. What
it does not yet pin down is the **loop-nest order** — which rank is the outer
loop vs the inner loop — which is exactly what the paper's per-action rank
subscripts (`∧¹_s`, `∧²_d`) encode. For the two BFS cases there is a single
reduction index (`s`), so order is forced; but in general (and to match the
paper) we will likely need to **tag ranks explicitly** on the actions. Section 2
resolves this.

## Section 2 — IR & codegen (in progress)

_To be designed next: the two-action loop-nest structure (push: action-1 `s`-loop
closes before action 2; pull: action 1 is map-only inside the shared `s`-loop,
action 2 reduces), the IR shape that replaces the flat
LOAD/LOAD/MAP/REDUCE/STORE pattern, and whether/how ranks are tagged._
