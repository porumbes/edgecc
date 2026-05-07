# edgecc v2 Design Spec — Merge Operators

EDGE language compiler — second pass. Adds merge operators on Map and Reduce actions to v1's surface language, an iterative test harness for BFS-flavored workloads, and a uniform Python codegen ABI exposing the kernel as a callable function.

## Background

v1 (see `2026-03-31-edgecc-v1-design.md`) compiles a single Einsum expression with one Map and one Reduce into nested-loop Python. v1 walks the entire iteration space and applies compute at every point, equivalent to a *pass-through* merge in the formal EDGE semantics — every coordinate is admitted regardless of whether tensor values are empty.

The EDGE language defines a **merge operator** as a binary Boolean function over two existence flags. For Map, the flags are "is operand A non-empty at this coordinate?" and "is operand B non-empty at this coordinate?". For Reduce, the flags are "is the reduction state non-empty?" and "is the maptmp defined at this iteration point?". The merge operator decides whether compute runs at all, and (in Reduce) whether accumulated state should be invalidated.

There are 16 possible merge operators — one per truth table on two Boolean inputs. The paper enumerates and names them in the merge appendix. v2 supports a 9-operator subset.

## Scope

**In scope:**
- Surface syntax extension allowing `map(compute_op, merge_op)` and `reduce(compute_op, merge_op)`. When the merge operator is omitted, the default is `passthrough`, preserving v1 source-program semantics.
- Nine merge operators — those whose truth-table output on (No, No) is False:

  | `.edge` keyword | Paper symbol | Truth table (No,No / No,Yes / Yes,No / Yes,Yes) |
  |---|---|---|
  | `passthrough` (default) | 𝟙 | T / T / T / T |
  | `nopass` | 𝔽 | F / F / F / F |
  | `intersect` | ∩ | F / F / F / T |
  | `take_left` | ← | F / F / T / T |
  | `take_left_only` | ⊖_l | F / F / T / F |
  | `take_right` | → | F / T / F / T |
  | `take_right_only` | ⊖_r | F / T / F / F |
  | `xor` | ⊕ | F / T / T / F |
  | `union` | ∪ | F / T / T / T |

- Codegen for all 9 operators in dense Python, using a uniform skeleton that mirrors the formal Map/Reduce semantics step-for-step.
- Reduce-state initialization to the compute op's identity (hardcoded per built-in: `add`/`sub` → 0, `mul` → 1, `min` → +∞, `max` → −∞, `and` → true, `or` → false). For v1's compute set, identity coincides with the natural empty value, so v1 programs continue to behave identically.
- A new iterative test harness driven by `driver.yaml` files. Drives a generated kernel until a watched tensor stabilizes (or a max-iteration cap is reached) and compares the final state against expected values. Feedback supports element-wise combining (e.g., `min` over previous and new frontier) so workloads like BFS, which naturally preserve previously-discovered values across iterations, can be expressed without v2-out-of-scope features (cascades, update operator).
- A canonical BFS gather-step kernel as the iterative-harness reference workload, including a disconnected-graph variant.
- Generated Python ABI change: kernel is exposed as `def run(<inputs>) -> <output>`, with an `__edgecc_meta__` dict for harness introspection. A `__main__` block constructs inputs from `init` declarations and prints the output, preserving the script-execution behavior v1 tests rely on.

**Out of scope (deferred):**
- The 7 (No,No)→Yes merge operators (NOR, equivalence, NAND, ¬→, ¬←, ⇐, ⇒). They require compute to be invoked on two empty operands and the result checked against the output's empty value; the user's compute must define behavior on empty pairs.
- Multi-operation expressions with compute labels (`\bigwedge^1`, `\bigwedge^2`).
- Generational/iterative ranks (`F_{i+1, d}`).
- Anonymous tensors / parenthesized sub-expressions.
- Unary functions (e.g., `¬B`).
- Sparse tensor storage. v2 keeps dense iteration with runtime existence checks.
- Optimizations of any kind (no live-set tracking, no constant folding, no loop reordering).
- User-defined compute operators. Compute set stays at v1's seven built-ins.
- Termination types beyond `stable` in the iterative harness (e.g., `predicate`, `epsilon`).

## Approach

- **Dense everywhere.** Generated Python continues to walk the full iteration space. Merge is implemented as a runtime existence check at each loop iteration: `a_exists := A[...] != empty_A`, `b_exists := B[...] != empty_B`. We do *not* construct sparse iteration sets or filter loops.
- **Merge is an IR field, not a new instruction.** `MAP` and `REDUCE` IR instructions gain a `merge_op` field plus the empty/identity values needed at codegen time.
- **One codegen path.** Codegen always emits the general merge skeleton. There is no special-cased pass-through fast path: the same skeleton handles all 9 merge operators. The pass-through case generates conditionals that always evaluate True, kept verbose for uniformity rather than collapsed.
- **Reduce-state lifetime.** State is the output tensor itself. Output is initialized at the start of `run()` to the compute op's identity (= empty value, for v1's compute set). The merge truth table decides per iteration whether to update state via compute, leave it alone, or reset it to the output's empty value.
- **Backwards compatibility through behavior, not bytes.** v1 source programs (no merge keyword) compile under v2 and produce identical *runtime output* — the same numbers come out of `print(...)`. The exact emitted Python source differs (verbose skeleton instead of v1's inline single-line form). v1 e2e tests, which compare stdout, continue to pass; v1 codegen unit tests, which compare emitted strings, are regenerated.

## Surface Syntax

Grammar extension is a single rule:

```
action_args  : compute_op
             | compute_op ',' merge_op
```

Both `map` and `reduce` adopt this rule. Examples:

```
// v1 source — unchanged, defaults to passthrough merge:
Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);

// v2 — explicit merge for BFS gather step:
F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);

// v2 — masked degree count:
MD[s] = G[s,d] . Mask[s] :: map(and, intersect) reduce(add, union);
```

Operator names listed in the Scope table are recognized as keywords by the lexer.

## Compiler Pipeline Changes

### Lexer

Add tokens for the 9 merge operator keywords: `passthrough`, `nopass`, `intersect`, `take_left`, `take_left_only`, `take_right`, `take_right_only`, `xor`, `union`.

### Parser

Extend the action production rule to accept an optional second comma-separated merge operator argument. Build the merge field on the existing action AST node.

### AST

Add an enum and a field on the existing action node:

```c
typedef enum {
    MERGE_PASSTHROUGH,    /* default; also encodes "absent in source" */
    MERGE_NOPASS,
    MERGE_INTERSECT,
    MERGE_TAKE_LEFT,
    MERGE_TAKE_LEFT_ONLY,
    MERGE_TAKE_RIGHT,
    MERGE_TAKE_RIGHT_ONLY,
    MERGE_XOR,
    MERGE_UNION,
} MergeOp;

/* in ActionNode (existing struct) */
ComputeOp compute_op;        /* existing */
MergeOp   merge_op;          /* NEW; defaults to MERGE_PASSTHROUGH */
```

### Semantic Analysis

Three additions:

1. **Merge keyword validation.** If the parser produced an unknown identifier in the merge slot, fail with: `unknown merge operator '<name>' (expected one of: passthrough, nopass, intersect, take_left, take_left_only, take_right, take_right_only, xor, union)`.

2. **Identity table.** Each built-in compute op has a fixed identity used at IR-lowering time:

   | Compute op | Identity (float) | Identity (int) | Identity (bool) |
   |---|---|---|---|
   | `add`, `sub` | 0.0 | 0 | n/a |
   | `mul` | 1.0 | 1 | n/a |
   | `min` | +∞ | INT_MAX | n/a |
   | `max` | −∞ | INT_MIN | n/a |
   | `and` | n/a | n/a | true |
   | `or` | n/a | n/a | false |

   Existing v1 type-compatibility checks (e.g., `and`/`or` only on boolean tensors) are unchanged.

3. **`nopass` warning.** When `nopass` is used as a merge operator, emit on stderr: `warning: 'nopass' merge always produces empty output for this action; the action has no effect`. Non-fatal; the program still compiles.

There is no semantic check between the merge operator and the compute operator. Merge is purely Boolean; it operates on existence metadata only.

### IR

Two additions on existing instruction kinds; no new instruction kinds.

`MAP` instruction gains:
- `merge_op: MergeOp`
- `empty_a: literal` — empty value of the left operand
- `empty_b: literal` — empty value of the right operand

`REDUCE` instruction gains:
- `merge_op: MergeOp`
- `identity: literal` — compute op's identity, looked up at semantic time
- `empty_z: literal` — output tensor's empty value, used for the b_s existence check

The truth-table evaluation is implemented in codegen, not IR. IR carries the *what*; codegen handles the *how*.

A canonical 9-entry truth-table reference, used by both codegen and IR-level unit tests:

```c
/* one nibble per operator: bit i = output for input pair (i>>1, i&1)
 * where bit value 1 means "exists" — i.e. (NoNo, NoYes, YesNo, YesYes) */
static const uint8_t MERGE_TRUTH_TABLE[9] = {
    [MERGE_PASSTHROUGH]      = 0b1111,
    [MERGE_NOPASS]           = 0b0000,
    [MERGE_INTERSECT]        = 0b1000,
    [MERGE_TAKE_LEFT]        = 0b1100,
    [MERGE_TAKE_LEFT_ONLY]   = 0b0100,
    [MERGE_TAKE_RIGHT]       = 0b1010,
    [MERGE_TAKE_RIGHT_ONLY]  = 0b0010,
    [MERGE_XOR]              = 0b0110,
    [MERGE_UNION]            = 0b1110,
};
```

Codegen consults this table when emitting the existence-check expression; unit tests assert the table matches the keyword → emitted-expression mapping below.

### `--dump-ir` extension

When printing `MAP` or `REDUCE`, append the merge operator name and (for REDUCE) the identity. Example output:

```
LOOP_BEGIN d
  LOOP_BEGIN s
    LOAD A[s][d] -> tA
    LOAD B[s] -> tB
    MAP add intersect (eA=0.0, eB=inf) tA, tB -> tmap
    REDUCE min union (id=inf, eZ=inf) tmap -> Z[d]
  LOOP_END s
LOOP_END d
```

### Codegen

There is exactly one emitter path. v1's inline LOAD+MAP+REDUCE+STORE combiner is removed.

#### Generated Python ABI

```python
# Generated by edgecc
import math

# --- dimension constants (module scope) ---
S = 4
D = 4

# --- introspection metadata for test harnesses and tooling ---
__edgecc_meta__ = {
    "inputs":  ["G", "F"],
    "output":  "F1",
}

def run(G, F):
    F1 = [math.inf] * D       # output allocated to identity (= empty)
    for d in range(D):
        for s in range(S):
            # ... general merge skeleton ...
            pass
    return F1

# --- script entry: construct inputs from init blocks, call run, print ---
if __name__ == "__main__":
    G = [[0,1,1,0],[0,0,1,1],[0,0,0,1],[0,0,0,0]]
    F = [0, math.inf, math.inf, math.inf]
    F1 = run(G, F)
    print(F1)
```

Codegen rules:
- **Inputs** = tensors named on the RHS of the Einsum.
- **Output** = the tensor named on the LHS of the Einsum.
- **Function signature**: positional parameters in declaration order, matching `__edgecc_meta__["inputs"]`.
- **Module-level dimension constants** stay at module scope so they are accessible inside `run()` without being passed.
- **Init statements** lift into the `if __name__ == "__main__"` block.
- **`import math`** is added when any tensor's empty value is `inf` (or any compute identity is `inf`).

#### General merge skeleton (innermost loop body)

For an Einsum `Z[free] = A[*] . B[*] :: map(c_map, m_map) reduce(c_red, m_red);`:

```python
# --- LOAD operands ---
a_val = A[<a_indices>]
b_val = B[<b_indices>]

# --- MAP existence + merge ---
a_exists = a_val != <empty_a>
b_exists = b_val != <empty_b>
if <T_map_expr(a_exists, b_exists)>:
    a_op = a_val if a_exists else <empty_a>
    b_op = b_val if b_exists else <empty_b>
    map_val = <compute_map>(a_op, b_op)
    map_emitted = map_val != <empty_z>
else:
    map_emitted = False
    map_val = <empty_z>

# --- REDUCE existence + merge ---
state = Z[<z_indices>]
b_s = state != <empty_z>
b_m = map_emitted
if <T_red_expr(b_s, b_m)>:
    val_for_red = map_val if map_emitted else <empty_z>
    new_state = <compute_red>(state, val_for_red)
    Z[<z_indices>] = new_state if new_state != <empty_z> else <empty_z>
else:
    Z[<z_indices>] = <empty_z>
```

Substitutions:
- `<empty_a>`, `<empty_b>`, `<empty_z>` are literals from IR; `inf` becomes `math.inf`.
- `<compute_map>`, `<compute_red>` map to Python expressions: `add` → `a + b`, `mul` → `a * b`, `min` → `min(a, b)`, `max` → `max(a, b)`, `sub` → `a - b`, `and` → `a and b`, `or` → `a or b`.
- `<T_map_expr>` and `<T_red_expr>` substitute per the table below.

Per-operator truth-table → Python expression:

| Merge op | Emitted expression (Map: substitute `a_exists, b_exists`; Reduce: `b_s, b_m`) |
|---|---|
| `passthrough` | `True` |
| `nopass` | `False` |
| `intersect` | `a_exists and b_exists` |
| `take_left` | `a_exists` |
| `take_left_only` | `a_exists and not b_exists` |
| `take_right` | `b_exists` |
| `take_right_only` | `b_exists and not a_exists` |
| `xor` | `a_exists != b_exists` |
| `union` | `a_exists or b_exists` |

The skeleton intentionally retains conditionals even when they always evaluate True (`passthrough`) or False (`nopass`). Constant folding is a future optimization.

#### Worked example: BFS gather step

Source:
```edge
dim S = 4;
dim D = 4;
tensor F1[R=D] : float, empty = inf;
tensor G[R=S, C=D] : float, empty = 0.0;
tensor F[R=S]  : float, empty = inf;
init G = [[0,1,1,0],[0,0,1,1],[0,0,0,1],[0,0,0,0]];
init F = [0, inf, inf, inf];
F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);
```

Generated Python:
```python
# Generated by edgecc
import math

S = 4
D = 4

__edgecc_meta__ = {
    "inputs":  ["G", "F"],
    "output":  "F1",
}

def run(G, F):
    F1 = [math.inf] * D
    for d in range(D):
        for s in range(S):
            a_val = G[s][d]
            b_val = F[s]
            a_exists = a_val != 0.0
            b_exists = b_val != math.inf
            if a_exists and b_exists:                  # T_map: intersect
                a_op = a_val if a_exists else 0.0
                b_op = b_val if b_exists else math.inf
                map_val = a_op + b_op
                map_emitted = map_val != math.inf
            else:
                map_emitted = False
                map_val = math.inf
            state = F1[d]
            b_s = state != math.inf
            b_m = map_emitted
            if b_s or b_m:                             # T_red: union
                val_for_red = map_val if map_emitted else math.inf
                new_state = min(state, val_for_red)
                F1[d] = new_state if new_state != math.inf else math.inf
            else:
                F1[d] = math.inf
    return F1

if __name__ == "__main__":
    G = [[0,1,1,0],[0,0,1,1],[0,0,0,1],[0,0,0,0]]
    F = [0, math.inf, math.inf, math.inf]
    F1 = run(G, F)
    print(F1)
```

#### Empty value comparison

- Floats: `value != 0.0`, `value != math.inf` use Python `!=`. `math.inf != math.inf` is False (correct). NaN is never produced.
- Integers: plain `!=`.
- Booleans: plain `!=`.
- No epsilon tolerance. Exact comparison only.

## Iterative Test Harness

### Layout

```
tests/e2e/iterative/
├── bfs_simple/
│   ├── kernel.edge
│   └── driver.yaml
├── bfs_disconnected/
│   ├── kernel.edge
│   └── driver.yaml
└── (additional iterative tests as written)
tests/e2e/run_iterative.py
```

### `driver.yaml` schema

```yaml
inputs:                            # initial values for input tensors
  <tensor_name>: <list_or_nested_list>
  ...                              # one entry per tensor in __edgecc_meta__["inputs"]

feedback:
  to: <input_tensor_name>          # input slot updated by the output each iteration
  combine: <combine_op>            # how output is merged into the input slot;
                                   # one of: replace (default), min, max, add, or, and
                                   # "replace" overwrites the input slot with the output
                                   # all others apply the named op element-wise between
                                   # the previous input value and the new output value

termination:
  type: stable                     # only "stable" supported in v2
  on: <tensor_name>                # tensor to watch (typically the feedback target)

max_iterations: <int>              # hard cap; if reached without stability, the test
                                   # fails with a non-zero exit code

expected:
  final:                           # tensor name → expected list (compared exactly)
    <tensor_name>: <list_or_nested_list>
```

PyYAML's `.inf` literal loads as `math.inf`, allowing initial values to match the kernel's `empty = inf` declarations.

The `combine` op exists to model algorithms that naturally preserve previously-discovered values across iterations. BFS, for example, uses `combine: min` so that vertex depths once written are not clobbered when a later iteration's output is `inf` at that index. `combine: replace` (the default) is appropriate for algorithms where each iteration produces a complete, self-contained successor state.

The element-wise combine ops correspond to the same names in the v2 compute set: `min`, `max`, `add`, `or`, `and`. Their identities (used as the comparison element when the previous-input value is empty) are the same as the codegen identities listed earlier in this spec.

### Runner (`run_iterative.py`)

For each subdirectory in `tests/e2e/iterative/`:

1. Compile `kernel.edge` → `kernel.py` via `./build/edgecc`.
2. Load `driver.yaml`.
3. Import `kernel.py`; read `__edgecc_meta__` to learn input/output names.
4. Initialize an `inputs` dict from `driver.yaml["inputs"]`.
5. Loop up to `max_iterations`:
   - Snapshot the watched tensor.
   - Call `mod.run(*[inputs[name] for name in meta["inputs"]])` → `output`.
   - Apply feedback per `feedback.combine`:
     - `replace` (default): `inputs[feedback["to"]] = output`
     - `min` / `max` / `add` / `or` / `and`: element-wise apply the op between `inputs[feedback["to"]]` (previous) and `output` (new), recursing into nested lists to handle multi-rank tensors.
   - Termination check: if the watched tensor equals its snapshot, break.
6. After loop exit, compare each tensor in `expected.final` against the current value; report pass/fail with diff on mismatch.

Errors at any step (compile failure, import failure, exhausting `max_iterations` without stability, mismatched final tensor) print a clear message and exit non-zero. Each is its own distinguishable error message; the runner exits with code 1 on any failure.

### Sample test: `bfs_simple/`

`kernel.edge`:
```
dim S = 4;
dim D = 4;
tensor F1[R=D] : float, empty = inf;
tensor G[R=S, C=D] : float, empty = 0.0;
tensor F[R=S]  : float, empty = inf;
init G = [[0,1,1,0],[0,0,1,1],[0,0,0,1],[0,0,0,0]];
init F = [0, inf, inf, inf];
F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);
```

`driver.yaml`:
```yaml
inputs:
  G: [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
  F: [0, .inf, .inf, .inf]

feedback:
  to: F
  combine: min                     # preserves discovered depths across iterations

termination:
  type: stable
  on: F

max_iterations: 10

expected:
  final:
    F: [0, 1, 1, 2]
```

Without `combine: min`, the seed `F[0] = 0` would be overwritten by the kernel's output `F1[0] = inf` (vertex 0 has no incoming edges in this graph) and BFS would diverge. The element-wise `min` combiner is what allows iterative BFS to be expressed without v2-out-of-scope features.

A second test, `bfs_disconnected/`, exercises the harness against a graph with unreachable vertices, verifying that stabilization handles `inf` entries that legitimately remain in the final state. It uses the same `combine: min` feedback.

## Testing Strategy

### Unit tests (per pipeline stage)

- `test_lexer`: tokenizes the 9 merge keywords. Existing v1 token tests untouched.
- `test_parser`: parses `map(mul, intersect)`, `reduce(min, union)`, `map(mul)` (default merge). Verifies the AST has the correct `MergeOp` value.
- `test_semantic`: validates merge keyword recognition; asserts identity is correctly looked up per compute op; asserts unknown merge identifier raises the expected error.
- `test_ir`: lowers AST with merge into IR; asserts MAP/REDUCE instructions carry merge_op, identity, and empty values; verifies each operator's truth-table value matches the canonical table.
- `test_codegen`: golden-string tests. Old v1 goldens are regenerated to the new general-skeleton + callable form. New goldens added for each of the 9 merge operators on a minimal kernel.

### Single-shot E2E tests (extend existing framework)

Under `tests/e2e/` alongside existing v1 tests:

- `merge_intersect_gemm.edge` — masked GEMM using `map(mul, intersect)`.
- `merge_union_min.edge` — single BFS gather with `map(add, intersect) reduce(min, union)`.
- `merge_take_left.edge` — element-wise pass-through with `map(mul, take_left)`, exercising substitution-with-empty for B.

Existing v1 E2E tests (`gemm.edge`, `dotproduct.edge`, `gemm_broadcast.edge`) continue running. Their generated Python now goes through the v2 callable wrapper, which prints identical output.

### Negative tests (extend existing framework)

Under `tests/negative/`:

- `bad_merge_op.edge` — `map(mul, frobnicate)`. Expected error: `unknown merge operator 'frobnicate' (expected one of: ...)`.
- `too_many_args.edge` — `map(mul, intersect, union)`. Expected error: parser-level "too many arguments to map(...)".
- `merge_no_compute.edge` — `map(, intersect)`. Expected error: parser-level missing compute op.

### Iterative E2E tests (new framework)

Run via `tests/e2e/run_iterative.py`. Initial test set: `bfs_simple/` and `bfs_disconnected/`.

### CTest wiring

`CMakeLists.txt` gains:

```cmake
add_test(
    NAME e2e_iterative
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/e2e/run_iterative.py
            --build-dir ${CMAKE_BINARY_DIR}
            --tests-dir ${CMAKE_SOURCE_DIR}/tests/e2e/iterative
)
```

`ctest --output-on-failure` runs alongside existing tests. Missing PyYAML causes only iterative tests to be skipped.

## Build & Dependencies

- **Build dependencies:** unchanged from v1 — C compiler (C99), CMake ≥ 3.15, Flex, Bison, Python 3.
- **Test-time Python dependency:** PyYAML, used by `run_iterative.py` only. Install via `pip install pyyaml` or system package. Iterative tests skipped if absent; core build and v1 tests are unaffected.

## Backwards Compatibility

- **Source-program compatibility:** every v1 `.edge` source program compiles under v2 unchanged. Default merge is `passthrough`.
- **Generated Python source:** v1's inline-single-line form is gone. v2 always emits the general skeleton wrapped in `def run(...)` plus a `__main__` block. v1's codegen unit-test golden strings are regenerated.
- **Runtime behavior:** v1 programs produce the same printed output under v2 — same numbers, same shapes. v1 e2e tests, which compare stdout, pass without modification.

## Design Alternatives Recorded

These alternatives were considered and deferred. Future work may revisit them.

- **Surface syntax B (paper-faithful nested parens)**: `map(add)(intersect)` rather than comma-separated. Mirrors the paper's `+(\cap)` notation. Rejected for v2 in favor of the smaller-grammar comma form.
- **Surface syntax C (trailing keyword)**: `map(add) merge(intersect)`. More verbose, unambiguous, easy to parse. Rejected as needlessly verbose.
- **Operator scope A (full 16)**: support all 16 merge operators including the (No,No)→Yes set. Rejected for v2 because the (No,No)→Yes operators introduce compute-on-empty semantics that warrant a focused later increment.
- **Operator scope C (full 16 with whitelist)**: support 16 syntactically but restrict (No,No)→Yes to compute ops with known empty-pair behavior. Rejected as adding semantic-check complexity without clear payoff.
- **Pass-through fast path in codegen**: emit v1's inline single-line form when both merges are pass-through, to preserve byte-identical generated Python for v1 programs. Rejected in favor of one-emitter simplicity; v1 e2e tests still pass (they check stdout) and v1 codegen unit-test goldens are mechanically regenerated.
- **Iterative test harness B (`driver { ... }` directive in `.edge`)**: bundle driver config into the source file. Rejected to keep test-harness config separate from EDGE language semantics.
- **Iterative test harness C (Python helper library)**: each test is a Python file using helper functions. Rejected as still requiring the user to hand-write Python drivers.
- **Termination type `predicate`**: custom Python expression for the termination check. Deferred until a workload needs it.
- **Termination type `epsilon`**: float-tolerance comparison for stability. Deferred until a numerical workload needs it.
- **Constant folding of merge guards in codegen**: collapse `if True:` / `if False:` and elide the corresponding branches. Deferred as an optimization.
