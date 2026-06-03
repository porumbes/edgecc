# Take-Left Compute Operator (C2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `take_left` compute operator (the paper's `←`) that emits its left operand verbatim, giving EDGE a value-preserving mask (`map(take_left, take_left_only)`) that is correct for any value — including `0.0` — unlike today's fragile `map(or, take_left_only)` trick.

**Architecture:** `take_left` is a new `OperatorType` value that rides the existing compute-op plumbing (parse → IR → codegen). No lexer/parser/IR changes. As a *map* operator it emits the left operand; as a *reduce* operator it is a semantic error (map-only). One enum value, three switch cases, one `operator_from_str` entry, one semantic guard.

**Tech Stack:** C99, Flex, Bison, CMake; Python 3 for generated output and e2e/negative harnesses.

**Spec:** `docs/superpowers/specs/2026-06-02-edgecc-take-left-compute-c2-design.md`

**Build/test commands (run from repo root):**
- Build: `cmake --build build -j`
- Unit: `./build/test_semantic` · `./build/test_codegen` (also `test_lexer`, `test_parser`, `test_ir`)
- E2E: `python3 tests/e2e/run_e2e.py ./build/edgecc` (auto-discovers `tests/e2e/<name>/`)
- Negative: `python3 tests/negative/run_negative.py ./build/edgecc` (auto-discovers `tests/negative/<name>/`)
- Everything: `cd build && ctest --output-on-failure`

> **Note for subagent execution:** subagents on this project do not run Bash. The coordinator runs all build/test/commit commands between tasks; implementer prompts must say so and provide exact code.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/ast.h` | `OperatorType` enum | add `OP_TAKE_LEFT` |
| `src/ast.c` | op name ↔ enum | `operator_from_str` + `operator_to_str` entries |
| `src/codegen.c` | Python emission | `compute_op_pyexpr` case: emit left operand |
| `src/semantic.c` | identity + validation | `compute_op_identity` case; reduce-op guard (map-only) |
| `tests/unit/test_codegen.c` | codegen unit test | `map(take_left, …)` emits `map_val = a_op` |
| `tests/unit/test_semantic.c` | semantic unit tests | `take_left` map valid; `reduce(take_left)` errors |
| `tests/e2e/take_left_zero_preserve/` | necessity e2e | masks a `0.0` value, preserved |
| `tests/e2e/push_bfs_takeleft/` | demonstrator e2e | efficient push-BFS via `map(take_left, take_left_only)` |
| `tests/negative/reduce_take_left/` | negative | `reduce(take_left)` → map-only error |
| `examples/push_bfs.edge` | example | efficient push-BFS |
| `CLAUDE.md` | docs | compute-op list + deferred list |

No `-Werror`/`-Wswitch` in this build (verified), but Task 1 still adds every `OperatorType` switch case alongside the new enum value, because `compute_op_pyexpr` would otherwise leave its output buffer uninitialized for `take_left`.

---

## Task 1: `take_left` as a map compute operator

**Files:**
- Modify: `src/ast.h:30-37` (enum), `src/ast.c:186-210` (string maps), `src/codegen.c:289-295` (emit), `src/semantic.c` (`compute_op_identity` switch)
- Test: `tests/unit/test_codegen.c`

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_codegen.c`, add this test (the file already has `static AstNode *parse(...)` and uses `SemanticResult sem; ir_lower(ast, sem.symtab); codegen_python(ir)`). The mask Einsum below uses `take_left` as the map op; the generated body must compute the map value as the bare left operand `a_op`:

```c
TEST(test_codegen_take_left_map) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor Z[R=N] : float, empty = inf;\n"
        "tensor A[R=N] : float, empty = inf;\n"
        "tensor B[R=N] : float, empty = inf;\n"
        "init A = [0.0, 2.0, inf];\n"
        "init B = [inf, 5.0, inf];\n"
        "Z[n] = A[n] . B[n] :: map(take_left, take_left_only) reduce(min);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "map_val = a_op") != NULL);   /* left operand verbatim */
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```

Register it in `main()`: `RUN_TEST(test_codegen_take_left_map);`

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j && ./build/test_codegen`
Expected: FAIL — `operator_from_str("take_left")` returns -1 today, so the map op is invalid and the generated body will not contain `map_val = a_op` (the assertion fails; the program may also report errors). This is the red state.

- [ ] **Step 3: Add the enum value**

In `src/ast.h`, change the `OperatorType` enum (currently lines 30-37) to add `OP_TAKE_LEFT` after `OP_OR`:

```c
typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_MIN,
    OP_MAX,
    OP_AND,
    OP_OR,
    OP_TAKE_LEFT
} OperatorType;
```

- [ ] **Step 4: Add the string mappings**

In `src/ast.c`, in `operator_to_str` add a case (before the closing `}` of the switch):

```c
        case OP_TAKE_LEFT: return "take_left";
```

In `src/ast.c`, in `operator_from_str` add (before `return -1;`):

```c
    if (strcmp(name, "take_left") == 0) return OP_TAKE_LEFT;
```

- [ ] **Step 5: Emit the left operand in codegen**

In `src/codegen.c`, in `compute_op_pyexpr` (the switch at lines 289-295), add a case after `OP_OR`:

```c
        case OP_TAKE_LEFT: snprintf(out, cap, "%s", lhs); break;
```

- [ ] **Step 6: Keep `compute_op_identity` exhaustive**

In `src/semantic.c`, in `compute_op_identity`'s switch, add a case (the value is never consumed — `take_left` is map-only, enforced in Task 2 — but the case documents intent and keeps the switch complete):

```c
        case OP_TAKE_LEFT:
            return 0.0;     /* map-only; identity unused */
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cmake --build build -j && ./build/test_codegen`
Expected: PASS, including `test_codegen_take_left_map`, and all pre-existing codegen tests unchanged.

- [ ] **Step 8: Commit**

```bash
git add src/ast.h src/ast.c src/codegen.c src/semantic.c tests/unit/test_codegen.c
git commit -m "feat: take_left compute operator (emits left operand)"
```

---

## Task 2: `take_left` is map-only (reduce is an error)

**Files:**
- Modify: `src/semantic.c:198-…` (`analyze_einsum` — add reduce-op guard near the top, after `EinsumData *d = &node->data.einsum;`)
- Test: `tests/unit/test_semantic.c`

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_semantic.c`, add two tests and register both in `main()`:

```c
TEST(test_take_left_map_valid) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor Z[R=N] : float, empty = inf;\n"
        "tensor A[R=N] : float, empty = inf;\n"
        "tensor B[R=N] : float, empty = inf;\n"
        "init A = [0.0, 2.0, inf];\n"
        "init B = [inf, 5.0, inf];\n"
        "Z[n] = A[n] . B[n] :: map(take_left, take_left_only) reduce(min);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_take_left_reduce_is_error) {
    AstNode *ast = parse(
        "dim K = 3; dim M = 2;\n"
        "tensor Z[R=M] : float, empty = inf;\n"
        "tensor A[R=K, C=M] : float, empty = inf;\n"
        "tensor B[R=K, C=M] : float, empty = inf;\n"
        "Z[m] = A[k,m] . B[k,m] :: map(mul, intersect) reduce(take_left, union);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}
```

Register:
```c
    RUN_TEST(test_take_left_map_valid);
    RUN_TEST(test_take_left_reduce_is_error);
```

- [ ] **Step 2: Run tests to verify the reduce test fails**

Run: `cmake --build build -j && ./build/test_semantic`
Expected: `test_take_left_map_valid` PASSES already (Task 1 made the map op valid); `test_take_left_reduce_is_error` FAILS (no guard yet — `reduce(take_left)` is currently accepted, so `error_count == 0`).

- [ ] **Step 3: Add the map-only guard**

In `src/semantic.c`, in `analyze_einsum` (starts at line 198), immediately after the line `EinsumData *d = &node->data.einsum;` (line 199), insert:

```c
    if (d->reduce_op == OP_TAKE_LEFT) {
        add_error(res,
            "take_left is a map-only compute operator; "
            "it cannot be used as a reduce operator");
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build -j && ./build/test_semantic`
Expected: PASS for both new tests and all pre-existing semantic tests.

- [ ] **Step 5: Commit**

```bash
git add src/semantic.c tests/unit/test_semantic.c
git commit -m "feat(semantic): take_left is map-only (reduce(take_left) is an error)"
```

---

## Task 3: E2E — necessity (0.0-preserving mask) and efficient push-BFS

**Files:**
- Create: `tests/e2e/take_left_zero_preserve/input.edge`, `tests/e2e/take_left_zero_preserve/expected_output.txt`
- Create: `tests/e2e/push_bfs_takeleft/input.edge`, `tests/e2e/push_bfs_takeleft/expected_output.txt`
- Create: `examples/push_bfs.edge`

- [ ] **Step 1: Create the necessity e2e input**

Create `tests/e2e/take_left_zero_preserve/input.edge`:

```
// take_left preserves a meaningful 0.0 value through a mask, where the older
// map(or, take_left_only) trick would drop it (0.0 or inf == inf). Output empty
// is inf so a stored 0.0 is observable. take_left_only emits where A exists and
// B is empty; reduce(min) passes the map value through (min(inf, v) == v).
dim N = 3;

tensor Z[R=N] : float, empty = inf;
tensor A[R=N] : float, empty = inf;
tensor B[R=N] : float, empty = inf;

init A = [0.0, 2.0, inf];   // A[0] = 0.0 is meaningful; A[2] absent
init B = [inf, 5.0, inf];   // B present only at n=1

Z[n] = A[n] . B[n] :: map(take_left, take_left_only) reduce(min);
```

- [ ] **Step 2: Generate and capture the actual output**

Run:
```bash
./build/edgecc tests/e2e/take_left_zero_preserve/input.edge -o /tmp/tlz.py && python3 /tmp/tlz.py
```
Expected: `[0.0, inf, inf]` — n=0 emits (A exists, B empty) and `take_left` keeps `0.0`; n=1 is suppressed (B present); n=2 absent. If the float formatting of infinity differs, use the exact printed line verbatim in Step 3.

- [ ] **Step 3: Write the expected-output file**

Create `tests/e2e/take_left_zero_preserve/expected_output.txt` with the exact line from Step 2:

```
[0.0, inf, inf]
```

- [ ] **Step 4: Create the efficient push-BFS e2e input**

Create `tests/e2e/push_bfs_takeleft/input.edge`:

```
// Efficient (pruned) push-BFS reachability — the frontier is masked by the
// visited set so only newly-discovered nodes propagate. The mask uses
// map(take_left, take_left_only): keep candidate C where P (visited) is empty,
// preserving C's value robustly. Same graph/result as reachability_bfs.
dim N = 6;

tensor G[R=N, C=N]   : float, empty = 0.0;
tensor C[gen I, R=N] : float, empty = 0.0;   // gathered candidates (intermediate, offset i)
tensor F[gen I, R=N] : float, empty = 0.0;   // frontier (cross-generational)
tensor P[gen I, R=N] : float, empty = 0.0;   // visited  (cross-generational)

init G = [[0.0,1.0,1.0,0.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,1.0,0.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,0.0]];
init F = [1.0,0.0,0.0,0.0,0.0,0.0];
init P = [1.0,0.0,0.0,0.0,0.0,0.0];

C[i, d]   = G[s,d] . F[i,s] :: map(and, intersect) reduce(or, union);
F[i+1, d] = C[i,d] . P[i,d] :: map(take_left, take_left_only) reduce(or, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
stop: stable(P);
```

- [ ] **Step 5: Generate and capture the actual output**

Run:
```bash
./build/edgecc tests/e2e/push_bfs_takeleft/input.edge -o /tmp/pbt.py && python3 /tmp/pbt.py
```
Expected: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` (all nodes reachable from node 0). Use the exact printed line in Step 6.

- [ ] **Step 6: Write the expected-output file**

Create `tests/e2e/push_bfs_takeleft/expected_output.txt`:

```
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

- [ ] **Step 7: Add the example**

Create `examples/push_bfs.edge` with the same content as `tests/e2e/push_bfs_takeleft/input.edge` from Step 4.

- [ ] **Step 8: Run the e2e harness**

Run: `python3 tests/e2e/run_e2e.py ./build/edgecc`
Expected: PASS for `take_left_zero_preserve` and `push_bfs_takeleft`, plus all pre-existing e2e cases.

- [ ] **Step 9: Commit**

```bash
git add tests/e2e/take_left_zero_preserve tests/e2e/push_bfs_takeleft examples/push_bfs.edge
git commit -m "test(e2e): take_left preserves 0.0; efficient push-BFS via take_left"
```

---

## Task 4: Negative test — `reduce(take_left)`

**Files:**
- Create: `tests/negative/reduce_take_left/input.edge`, `tests/negative/reduce_take_left/expected_errors.txt`

- [ ] **Step 1: Create the negative test**

Create `tests/negative/reduce_take_left/input.edge`:

```
dim K = 3; dim M = 2;
tensor Z[R=M] : float, empty = inf;
tensor A[R=K, C=M] : float, empty = inf;
tensor B[R=K, C=M] : float, empty = inf;
Z[m] = A[k,m] . B[k,m] :: map(mul, intersect) reduce(take_left, union);
```

Create `tests/negative/reduce_take_left/expected_errors.txt`:

```
map-only
```

- [ ] **Step 2: Run the negative harness**

Run: `python3 tests/negative/run_negative.py ./build/edgecc`
Expected: PASS `reduce_take_left` (its error output contains "map-only"), plus all pre-existing negative cases.

- [ ] **Step 3: Commit**

```bash
git add tests/negative/reduce_take_left
git commit -m "test(negative): reduce(take_left) is rejected as map-only"
```

---

## Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md` (built-in compute operators list; deferred/limitations list)

- [ ] **Step 1: Add `take_left` to the compute-operator list**

In `CLAUDE.md`, find the sentence listing built-in compute operators (it reads `Built-in compute operators: \`add\`, \`sub\`, \`mul\`, \`min\`, \`max\`, \`and\`, \`or\`.`) and change it to:

```
Rank variables in expressions bind to declared rank names by position. Built-in compute operators: `add`, `sub`, `mul`, `min`, `max`, `and`, `or`, `take_left` (map-only — emits its left operand verbatim; pair with the `take_left_only` or `intersect` merge for a value-preserving mask; using it as a `reduce` operator is an error).
```

- [ ] **Step 2: Update the deferred/limitations list**

In `CLAUDE.md`'s "v2 Scope Limitations" paragraph, the list of remaining BFS gaps currently reads `... the remaining gaps for full 3-step BFS (unary \`¬\`, a take-left compute op, depth-valued masking, multi-output return, \`ANY\` compute, occupancy ...)`. Remove `a take-left compute op, ` from that parenthetical (it is now implemented), leaving the others. The clause becomes:

```
the remaining gaps for full 3-step BFS (unary `¬`, multi-output return, `ANY` compute, occupancy (`‖F‖≡0`) termination)
```

- [ ] **Step 3: Full test run**

Run: `cd build && ctest --output-on-failure`
Expected: 100% pass (8 groups).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document take_left compute operator (C2)"
```

---

## Self-review notes

- **Spec coverage:** new `take_left` compute op (Task 1) · map-only/reduce error (Task 2) · `compute_op_identity` exhaustive case (Task 1 Step 6) · canonical mask `map(take_left, take_left_only)` exercised (Tasks 1/3) · necessity test preserving `0.0` (Task 3) · efficient push-BFS demonstrator + example (Task 3) · negative reduce test (Task 4) · docs (Task 5). All spec sections map to a task.
- **No grammar/IR changes** — consistent with the spec; `take_left` is just another `operator_from_str` name flowing through `IRMap.op`.
- **Verification idioms (confirmed):** codegen entry point `codegen_python(IRProgram*)`; the map skeleton names the left operand `a_op` and emits `map_val = <compute_op_pyexpr(...)>`, so `take_left` yields the literal `map_val = a_op`; `compute_op_pyexpr` has no `default`, so the new case is required for correct output. e2e/negative runners take the binary path argument (`./build/edgecc`) and auto-discover directories.
- **Line numbers** are pre-change and may drift; each step quotes surrounding code to anchor the edit.
- **Necessity-test reduce choice:** `reduce(min)` with `empty = inf` passes the map value through (`min(inf, v) == v`), so a `take_left` map value of `0.0` is stored as `0.0` (since `0.0 != inf`). The `or` trick would instead compute `0.0 or inf == inf` and drop it — that contrast is the point, documented in the fixture comment.
```