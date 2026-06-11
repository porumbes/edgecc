# Unary Negation (`!`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unary complement operator `!` as a per-operand prefix on an Einsum map (`C . !P`), realizing the paper's `¬` so masked BFS reads the way the paper writes it.

**Architecture:** A per-operand `negate` flag threaded `lexer → AST(TensorRef) → parser → IR(IRLoad) → codegen`. The only behavioral change is in the inner-loop emitter (`emit_general_skeleton`): a negated operand flips its existence test (`== empty` instead of `!= empty`) and supplies `τ = 1.0` at complement points. Merge guards and compute operators are untouched. Correctness is pinned by a differential oracle: `A . !B :: map(c, intersect)` must equal `A . B :: map(c, take_left_only)` byte-for-byte.

**Tech Stack:** C99, Flex, Bison (2.3-compatible), CMake/CTest; Python 3 for generated code and e2e/negative harnesses.

**Spec:** `docs/superpowers/specs/2026-06-10-edgecc-unary-negation-design.md`

---

## Background the engineer needs

- **Pipeline:** `.edge → lexer.l → parser.y → AST (ast.h/c) → semantic.c → ir.c → codegen.c → .py`.
- **Build:** `cmake -S . -B build >/dev/null && cmake --build build` (regenerates lexer/parser).
- **All tests:** `ctest --test-dir build --output-on-failure`.
- **One unit binary:** `./build/test_parser` (also `test_lexer`, `test_ir`, `test_codegen`, `test_semantic`).
- **Unit tests** live in `tests/unit/*.c`; each uses a `TEST(name){...}` macro and must be registered with `RUN_TEST(name)` inside that file's `main()`. Lexer tests drive `yylex()` over `yy_scan_string`; parser tests call `yyparse()` and read the global `parse_result`; IR/codegen tests use a local `parse()` helper then `semantic_analyze`/`ir_lower`/`codegen_python`.
- **E2E tests:** a directory `tests/e2e/<name>/` containing `input.edge` and `expected_output.txt`; the harness compiles, runs the generated Python, and diffs stdout. No registration needed — directories are auto-discovered.
- **Negative tests:** a directory `tests/negative/<name>/` containing `input.edge` and `expected_errors.txt`; the harness runs `edgecc input.edge -o /dev/null`, requires a non-zero exit, and checks each line of `expected_errors.txt` appears in stderr.
- **Existence model:** a coordinate "exists" iff `value != empty`. Merge operates only on existence Booleans (`codegen.c` `emit_general_skeleton`). `!X` flips that bit.
- **Commit convention:** do NOT add `Co-Authored-By` trailers.

---

## File Structure

- `src/lexer.l` — add `!` to the single-char token class.
- `src/ast.h` / `src/ast.c` — add `int negate` to `TensorRef`; initialize in `tensor_ref_new`.
- `src/parser.y` — new `map_operand` nonterminal carrying the `!` prefix; output stays `tensor_ref` (so `!` on the LHS is a syntax error).
- `src/ir.h` / `src/ir.c` — add `int negate` to `IRLoad`; populate from the two operands; surface in `--dump-ir`.
- `src/codegen.c` — `emit_general_skeleton`: per-operand existence flip + `τ`.
- `tests/unit/test_lexer.c`, `test_parser.c`, `test_ir.c`, `test_codegen.c` — unit coverage.
- `tests/e2e/neg_vs_take_left_only/`, `tests/e2e/push_bfs_neg/` — e2e coverage.
- `examples/push_bfs_neg.edge` — paper-faithful BFS demonstrator.
- `tests/negative/neg_on_output/` — `!` on LHS rejected.
- `CLAUDE.md`, `docs/roadmap.md` — reconciliation.

---

## Task 1: Lexer — `!` token

**Files:**
- Modify: `src/lexer.l:51`
- Test: `tests/unit/test_lexer.c`

- [ ] **Step 1: Write the failing test**

Add this `TEST` to `tests/unit/test_lexer.c` (next to the other token tests, e.g. after `test_plus_token`):

```c
TEST(test_bang_token) {
    YY_BUFFER_STATE buf = yy_scan_string("!");
    ASSERT_INT_EQ('!', yylex());
    yy_delete_buffer(buf);
}
```

Register it in that file's `main()` by adding next to the other `RUN_TEST` lines:

```c
    RUN_TEST(test_bang_token);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/test_lexer`
Expected: build succeeds, `test_bang_token` FAILS (lexer prints `lexer error: unexpected character '!'` and `yylex` does not return `'!'`).

- [ ] **Step 3: Implement the minimal change**

In `src/lexer.l`, line 51, add `!` to the punctuation character class:

```
[=\.\,\;\:\[\]\(\)+*!]  { return yytext[0]; }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/test_lexer`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lexer.l tests/unit/test_lexer.c
git commit -m "lexer: tokenize '!' for unary negation"
```

---

## Task 2: AST — `negate` field on `TensorRef`

**Files:**
- Modify: `src/ast.h:61-67` (the `TensorRef` struct)
- Modify: `src/ast.c` (`tensor_ref_new`)
- Test: covered by Task 3's parser test (no standalone test — a struct field with no behavior yet)

- [ ] **Step 1: Add the field**

In `src/ast.h`, add `int negate;` to `struct TensorRef`:

```c
struct TensorRef {
    char *tensor_name;
    char **rank_vars;
    int num_vars;
    int gen_axis;    /* index of the coord written as `var + N`, else -1 */
    int gen_offset;  /* the N in `var + N` (0 when gen_axis == -1) */
    int negate;      /* 1 if operand carries a unary `!` (existence complement) */
};
```

- [ ] **Step 2: Initialize it**

In `src/ast.c`, in `tensor_ref_new`, set the default after `ref.gen_offset = 0;`:

```c
    ref.gen_offset = 0;
    ref.negate = 0;
```

- [ ] **Step 3: Verify it still builds**

Run: `cmake --build build`
Expected: builds clean (no behavior change yet).

- [ ] **Step 4: Commit**

```bash
git add src/ast.h src/ast.c
git commit -m "ast: add negate flag to TensorRef"
```

---

## Task 3: Parser — `!` prefix on operands, rejected on output

**Files:**
- Modify: `src/parser.y:110-114` (`%type` decls), `:297-318` (`einsum_stmt`), add `map_operand` rule
- Test: `tests/unit/test_parser.c`

- [ ] **Step 1: Write the failing tests**

Add these two `TEST`s to `tests/unit/test_parser.c` (after `test_parse_einsum`):

```c
TEST(test_parse_einsum_negated_operand) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "Z[d] = A[d] . !B[d] :: map(take_left, intersect) reduce(or, union);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM, es->type);
    ASSERT_INT_EQ(0, es->data.einsum.input1.negate);
    ASSERT_INT_EQ(1, es->data.einsum.input2.negate);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_einsum_negated_left_operand) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "Z[d] = !A[d] . B[d] :: map(and, intersect) reduce(or, union);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(1, es->data.einsum.input1.negate);
    ASSERT_INT_EQ(0, es->data.einsum.input2.negate);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}
```

Register both in `main()`:

```c
    RUN_TEST(test_parse_einsum_negated_operand);
    RUN_TEST(test_parse_einsum_negated_left_operand);
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build && ./build/test_parser`
Expected: build OR tests fail — Bison errors on `!` in operand position (no grammar rule), so `yyparse()` returns non-zero.

- [ ] **Step 3: Implement the grammar change**

In `src/parser.y`:

(a) Add `map_operand` to the pointer `%type` list (line ~111):

```
%type <ptr> rank_pair_list ident_list nested_literal lit_list tensor_ref map_operand
```

(b) Change the two operand positions in `einsum_stmt` (line 298) from `tensor_ref` to `map_operand`. The new first line of the rule:

```
    tensor_ref '=' map_operand '.' map_operand TOK_DOUBLE_COLON
```

(The action body is unchanged: `$3`/`$5` are still `TensorRef *`, and the positions of `$9`,`$10`,`$14`,`$15` do not move because each operand is still one grammar symbol.)

(c) Add the `map_operand` rule immediately after the `einsum_stmt` rule (after line 318):

```
/* A map operand is a tensor ref, optionally prefixed with `!` (unary
   complement). `!` is only valid here, not on the einsum output (LHS). */
map_operand:
    tensor_ref
    {
        $$ = $1;
    }
    | '!' tensor_ref
    {
        TensorRef *r = (TensorRef *)$2;
        r->negate = 1;
        $$ = (void *)r;
    }
    ;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build && ./build/test_parser`
Expected: PASS. (If Bison reports a new conflict, it should not — `!` only begins a `map_operand` and `tensor_ref` begins with `TOK_IDENT`, so the choice is unambiguous on one token of lookahead.)

- [ ] **Step 5: Commit**

```bash
git add src/parser.y tests/unit/test_parser.c
git commit -m "parser: accept '!' prefix on einsum map operands"
```

---

## Task 4: IR — carry `negate` on `IRLoad`

**Files:**
- Modify: `src/ir.h` (the `IRLoad` struct)
- Modify: `src/ir.c:169-184` (populate), `:393-399` (dump)
- Test: `tests/unit/test_ir.c`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_ir.c` (after `test_ir_gemm_structure`):

```c
TEST(test_ir_load_carries_negate) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "tensor A[R=N] : float, empty = 0.0;\n"
        "tensor B[R=N] : float, empty = 0.0;\n"
        "init A = [1.0, 0.0, 1.0];\n"
        "init B = [0.0, 1.0, 0.0];\n"
        "Z[d] = A[d] . !B[d] :: map(take_left, intersect) reduce(or, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    ASSERT_NOT_NULL(ir);
    int saw_neg = 0, saw_plain = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_LOAD) continue;
        if (ir->instrs[i].data.load.negate) saw_neg = 1;
        else saw_plain = 1;
    }
    ASSERT_INT_EQ(1, saw_neg);     /* the !B load */
    ASSERT_INT_EQ(1, saw_plain);   /* the A load */
    ast_free(ast);
}
```

Register it in `main()`:

```c
    RUN_TEST(test_ir_load_carries_negate);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/test_ir`
Expected: build fails — `IRLoad` has no `negate` member.

- [ ] **Step 3: Implement**

(a) In `src/ir.h`, add `int negate;` to the `IRLoad` struct (alongside `gen_read`):

```c
    int negate;    /* 1 if this operand carried a unary `!` (existence complement) */
```

(b) In `src/ir.c`, set it in both LOAD emissions. After the input1 `gen_read` line (~174):

```c
        instr->data.load.negate = e->input1.negate;
```

After the input2 `gen_read` line (~183):

```c
        instr->data.load.negate = e->input2.negate;
```

(c) In `src/ir.c`, surface it in the dump (line ~395). Replace the `IR_LOAD` dump block with:

```c
            case IR_LOAD:
                fprintf(out, "LOAD(r%d, %s%s%s, ", instr->data.load.reg,
                        instr->data.load.tensor_name,
                        instr->data.load.negate ? " [neg]" : "",
                        instr->data.load.gen_read ? " [gen_read]" : "");
                dump_coords(out, instr->data.load.coords, instr->data.load.ncoords);
                fprintf(out, ")\n");
                break;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/test_ir`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ir.h src/ir.c tests/unit/test_ir.c
git commit -m "ir: carry negate flag on IRLoad and show it in --dump-ir"
```

---

## Task 5: Codegen — existence flip + τ in the inner loop

**Files:**
- Modify: `src/codegen.c:341-350` (inside `emit_general_skeleton`)
- Test: `tests/unit/test_codegen.c`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_codegen.c` (after `test_codegen_gemm`):

```c
TEST(test_codegen_negated_operand) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "tensor A[R=N] : float, empty = 0.0;\n"
        "tensor B[R=N] : float, empty = 0.0;\n"
        "init A = [1.0, 0.0, 1.0];\n"
        "init B = [0.0, 1.0, 0.0];\n"
        "Z[d] = A[d] . !B[d] :: map(take_left, intersect) reduce(or, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    /* A is plain (!= empty); B is negated (== empty) and supplies 1.0. */
    ASSERT_TRUE(strstr(code, "a_exists = a_val != 0.0") != NULL);
    ASSERT_TRUE(strstr(code, "b_exists = b_val == 0.0") != NULL);
    ASSERT_TRUE(strstr(code, "b_op = 1.0 if b_exists else 0.0") != NULL);
    free(code);
    ast_free(ast);
}
```

Register it in `main()`:

```c
    RUN_TEST(test_codegen_negated_operand);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ./build/test_codegen`
Expected: FAIL — current codegen always emits `b_exists = b_val != 0.0` and `b_op = b_val ...`.

- [ ] **Step 3: Implement**

In `src/codegen.c`, inside `emit_general_skeleton`, replace the MAP existence block (lines 341-343):

```c
    /* MAP existence */
    cb_indent(cb); cb_appendf(cb, "a_exists = a_val != %s\n", eA);
    cb_indent(cb); cb_appendf(cb, "b_exists = b_val != %s\n", eB);
```

with (a negated operand flips empty<->present — the `!` complement):

```c
    /* MAP existence. A negated operand (`!X`) is present exactly where X is
       empty, so its existence test flips to `== empty`. */
    cb_indent(cb);
    cb_appendf(cb, "a_exists = a_val %s %s\n",
               load1->data.load.negate ? "==" : "!=", eA);
    cb_indent(cb);
    cb_appendf(cb, "b_exists = b_val %s %s\n",
               load2->data.load.negate ? "==" : "!=", eB);
```

Then replace the operand-value lines (lines 349-350):

```c
    cb_indent(cb); cb_appendf(cb, "a_op = a_val if a_exists else %s\n", eA);
    cb_indent(cb); cb_appendf(cb, "b_op = b_val if b_exists else %s\n", eB);
```

with (a negated operand supplies the true marker τ = 1.0 at its complement points; its raw value is never used):

```c
    cb_indent(cb);
    cb_appendf(cb, "a_op = %s if a_exists else %s\n",
               load1->data.load.negate ? "1.0" : "a_val", eA);
    cb_indent(cb);
    cb_appendf(cb, "b_op = %s if b_exists else %s\n",
               load2->data.load.negate ? "1.0" : "b_val", eB);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ./build/test_codegen`
Expected: PASS.

- [ ] **Step 5: Run the full unit suite (no regressions)**

Run: `ctest --test-dir build --output-on-failure`
Expected: all targets pass (non-negated programs are byte-identical — the emitter only branches when `negate` is set).

- [ ] **Step 6: Commit**

```bash
git add src/codegen.c tests/unit/test_codegen.c
git commit -m "codegen: emit existence-flip and tau for negated operands"
```

---

## Task 6: E2E — differential oracle (`!` intersect ≡ take_left_only)

**Files:**
- Create: `tests/e2e/neg_vs_take_left_only/input.edge`
- Create: `tests/e2e/neg_vs_take_left_only/expected_output.txt`

This is the spec's key regression oracle. The program computes the same masked result two ways and prints both; they must be equal and match the hand-derived truth table (`A=[1,2,0,0] B=[3,0,5,0] → [0,2,0,0]`).

- [ ] **Step 1: Write the e2e program**

Create `tests/e2e/neg_vs_take_left_only/input.edge`:

```edge
// Differential oracle: A . !B :: map(_, intersect) must equal
// A . B :: map(_, take_left_only). Both keep A where A is present and B absent.
dim N = 4;
tensor A[R=N]   : float, empty = 0.0;
tensor B[R=N]   : float, empty = 0.0;
tensor Zneg[R=N]: float, empty = 0.0;
tensor Ztlo[R=N]: float, empty = 0.0;
init A = [1.0, 2.0, 0.0, 0.0];
init B = [3.0, 0.0, 5.0, 0.0];
Zneg[d] = A[d] . !B[d] :: map(take_left, intersect) reduce(or, union);
Ztlo[d] = A[d] .  B[d] :: map(take_left, take_left_only) reduce(or, union);
```

Note: edgecc emits one `run()` per program with a single output; this program has two einsums writing two outputs. Confirm the harness behavior in Step 2 — if the generator only prints the last output, split into the form below instead.

- [ ] **Step 2: Compile and inspect actual behavior**

Run:
```bash
cmake --build build
./build/edgecc tests/e2e/neg_vs_take_left_only/input.edge -o /tmp/oracle.py
python3 /tmp/oracle.py
```
Expected: a single printed list. edgecc treats the LAST einsum's output as the program output, so this prints `Ztlo` = `[0.0, 2.0, 0.0, 0.0]`.

Because only one output is printed, make this a **single-output** oracle that exercises the `!` path, and rely on the unit/codegen equality plus the hand-derived expectation:

Replace `tests/e2e/neg_vs_take_left_only/input.edge` with:

```edge
// `!` intersect keeps A where A present and B absent. Hand-derived truth table:
// A=[1,2,0,0] B=[3,0,5,0] -> [0,2,0,0]  (matches take_left_only).
dim N = 4;
tensor A[R=N] : float, empty = 0.0;
tensor B[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0, 0.0, 0.0];
init B = [3.0, 0.0, 5.0, 0.0];
Z[d] = A[d] . !B[d] :: map(take_left, intersect) reduce(or, union);
```

- [ ] **Step 3: Capture expected output**

Run:
```bash
./build/edgecc tests/e2e/neg_vs_take_left_only/input.edge -o /tmp/oracle.py && python3 /tmp/oracle.py
```
Expected printed: `[0.0, 2.0, 0.0, 0.0]`.

Create `tests/e2e/neg_vs_take_left_only/expected_output.txt` with exactly:

```
[0.0, 2.0, 0.0, 0.0]
```

- [ ] **Step 4: Confirm equivalence to take_left_only (manual cross-check)**

Run:
```bash
printf 'dim N = 4;\ntensor A[R=N]:float,empty=0.0;\ntensor B[R=N]:float,empty=0.0;\ntensor Z[R=N]:float,empty=0.0;\ninit A=[1.0,2.0,0.0,0.0];\ninit B=[3.0,0.0,5.0,0.0];\nZ[d]=A[d].B[d]::map(take_left,take_left_only) reduce(or,union);\n' > /tmp/tlo.edge
./build/edgecc /tmp/tlo.edge -o /tmp/tlo.py && python3 /tmp/tlo.py
```
Expected printed: `[0.0, 2.0, 0.0, 0.0]` — identical to the `!` form.

- [ ] **Step 5: Run the e2e suite**

Run: `ctest --test-dir build --output-on-failure -R e2e`
Expected: `neg_vs_take_left_only` passes (and all others stay green).

- [ ] **Step 6: Commit**

```bash
git add tests/e2e/neg_vs_take_left_only/
git commit -m "test(e2e): differential oracle for '!' intersect vs take_left_only"
```

---

## Task 7: E2E — paper-faithful BFS demonstrator

**Files:**
- Create: `examples/push_bfs_neg.edge`
- Create: `tests/e2e/push_bfs_neg/input.edge`
- Create: `tests/e2e/push_bfs_neg/expected_output.txt`

Rewrite `push_bfs.edge`'s mask with `!` and assert it matches `push_bfs`'s result on the same graph.

- [ ] **Step 1: Get the reference (push_bfs) output**

Run:
```bash
cmake --build build
./build/edgecc examples/push_bfs.edge -o /tmp/push_bfs.py && python3 /tmp/push_bfs.py
```
Expected: a length-6 list (the visited/reachable set `P`). Record it verbatim for Step 3.

- [ ] **Step 2: Write the `!` version**

Create `examples/push_bfs_neg.edge` (the mask line is the only change from `push_bfs.edge` — `!P` + `intersect` instead of the `take_left_only` trick):

```edge
// Paper-faithful pruned push-BFS: the frontier is masked by the visited set
// using the unary complement `!P` and intersection -- exactly the paper's
// (G·F)·¬P gather/mask. Same graph and result as push_bfs.edge, which uses the
// equivalent map(take_left, take_left_only) encoding.
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

C[i, d]   = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);
F[i+1, d] = C[i,d] . !P[i,d]  :: map(take_left, intersect) reduce(or, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
stop: stable(P);
```

- [ ] **Step 3: Set up the e2e test**

Copy the example into the test dir and capture its output:

```bash
mkdir -p tests/e2e/push_bfs_neg
cp examples/push_bfs_neg.edge tests/e2e/push_bfs_neg/input.edge
./build/edgecc tests/e2e/push_bfs_neg/input.edge -o /tmp/push_bfs_neg.py && python3 /tmp/push_bfs_neg.py
```

Write the printed list into `tests/e2e/push_bfs_neg/expected_output.txt` (one line, exactly as printed). It MUST equal the Step 1 output from `push_bfs.edge`.

- [ ] **Step 4: Assert the two BFS forms agree**

Run:
```bash
diff <(python3 /tmp/push_bfs.py) <(python3 /tmp/push_bfs_neg.py) && echo "MATCH"
```
Expected: `MATCH` (no diff). If they differ, STOP — the `!` semantics are wrong; do not paper over by editing expected output.

- [ ] **Step 5: Run the e2e suite**

Run: `ctest --test-dir build --output-on-failure -R e2e`
Expected: `push_bfs_neg` passes; all others green.

- [ ] **Step 6: Commit**

```bash
git add examples/push_bfs_neg.edge tests/e2e/push_bfs_neg/
git commit -m "test(e2e): paper-faithful push-BFS using '!P' mask"
```

---

## Task 8: Negative test — `!` on the output is rejected

**Files:**
- Create: `tests/negative/neg_on_output/input.edge`
- Create: `tests/negative/neg_on_output/expected_errors.txt`

- [ ] **Step 1: Write the invalid program**

Create `tests/negative/neg_on_output/input.edge`:

```edge
dim N = 3;
tensor Z[R=N] : float, empty = 0.0;
tensor A[R=N] : float, empty = 0.0;
tensor B[R=N] : float, empty = 0.0;
init A = [1.0, 0.0, 1.0];
init B = [0.0, 1.0, 0.0];
!Z[d] = A[d] . B[d] :: map(or, union) reduce(or, union);
```

- [ ] **Step 2: Observe the actual error text**

Run:
```bash
cmake --build build
./build/edgecc tests/negative/neg_on_output/input.edge -o /dev/null; echo "exit=$?"
```
Expected: non-zero exit; stderr contains a parse/syntax error mentioning `!` (e.g. `syntax error`). Note the exact substring printed.

- [ ] **Step 3: Record the expected error**

Create `tests/negative/neg_on_output/expected_errors.txt` containing a substring that actually appears in the Step 2 stderr (the harness checks containment line-by-line). Use the parser's generic message:

```
syntax error
```

(If Step 2 shows a more specific string, prefer that.)

- [ ] **Step 4: Run the negative suite**

Run: `ctest --test-dir build --output-on-failure -R negative`
Expected: `neg_on_output` passes (edgecc failed as required and stderr matched).

- [ ] **Step 5: Commit**

```bash
git add tests/negative/neg_on_output/
git commit -m "test(negative): reject '!' on einsum output"
```

---

## Task 9: Docs reconciliation

**Files:**
- Modify: `CLAUDE.md` (EDGE Syntax v2 operator list; v2 Scope Limitations)
- Modify: `docs/roadmap.md`

- [ ] **Step 1: Update CLAUDE.md operator description**

In `CLAUDE.md`, in the EDGE Syntax (v2) section, after the merge-operators sentence, add a sentence documenting `!`:

```
Unary complement: prefix a map operand with `!` (the paper's `¬`) to use its
existence complement — `!X` is present exactly where `X` is empty. Paper-faithful
masking is `A . !B :: map(take_left, intersect)`, equivalent to the
`map(_, take_left_only)` encoding. `!` is operand-only (not on the output) and
keys off empty-vs-non-empty, so it works for any declared type.
```

In the `## v2 Scope Limitations` section, remove `unary ¬` from the "Not yet supported" list (it appears in the BFS-gap clause and/or the unary-functions clause). General user-defined unary functions remain unsupported — keep that, but `!`/`¬` complement is now supported.

- [ ] **Step 2: Update the roadmap**

In `docs/roadmap.md`, under **Graph algorithms**, change the paper-faithful BFS row to reflect that unary `¬` is done. In the line:

```
| Paper-faithful 3-step BFS (`¬`, multi-action, `ANY`/parent, occupancy stop) | 🚧 |
```

update the detail in the "Partial 🚧" and "Planned ❌" sections: move unary `¬` out of the remaining-gaps list for BFS (the remaining gaps become multi-action Einsums, multi-output return, `ANY`/parent, occupancy stop). Add a bullet under "Implemented ✅":

```
- **Unary complement `!`** (the paper's `¬`): per-operand existence complement; paper-faithful BFS masking `A . !B :: map(take_left, intersect)`.
```

- [ ] **Step 3: Verify the docs build/scan clean**

Run: `grep -n "unary" CLAUDE.md docs/roadmap.md`
Expected: the new `!`/`¬` supported-language text appears; no remaining claim that unary `¬` is unsupported.

- [ ] **Step 4: Full suite once more**

Run: `ctest --test-dir build --output-on-failure`
Expected: everything green.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/roadmap.md
git commit -m "docs: document unary complement '!' and update BFS roadmap"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Surface syntax `!` (ASCII, operand-only, no `¬` alias) → Tasks 1, 3, 8. ✅
- Existence-flip semantics (`ε = ¬∃`, `ν = τ = 1.0`) → Task 5. ✅
- Per-operand `negate` threaded AST→IR→codegen → Tasks 2, 4, 5. ✅
- Differential oracle vs `take_left_only` → Task 6 (+ codegen unit Task 5). ✅
- Paper-faithful BFS example/e2e → Task 7. ✅
- `--dump-ir` surfacing → Task 4. ✅
- Negative: `!` on output → Task 8. ✅
- Reconciliation (CLAUDE.md, roadmap.md, example beside push_bfs) → Tasks 7, 9. ✅
- Regression: non-`!` programs byte-identical → Task 5 Step 5 (full suite). ✅
- Not restricted to declared-boolean (float BFS tensors) → covered by Task 7 (P is `float, empty=0.0`). ✅

**Placeholder scan:** none — every code/step shows actual content. Task 6 deliberately documents a discover-then-decide step (single-output generator), with the resolved final file given inline.

**Type/name consistency:** `negate` used identically on `TensorRef` (Task 2), `IRLoad` (Task 4), read via `load1/load2->data.load.negate` (Task 5); `map_operand` nonterminal consistent (Task 3). `τ = 1.0` literal consistent between spec and Task 5.
