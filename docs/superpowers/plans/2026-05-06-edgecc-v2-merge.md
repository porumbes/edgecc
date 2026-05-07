# edgecc v2 Implementation Plan — Merge Operators

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add merge operators on Map and Reduce actions to v1's surface language, restructure generated Python as a `def run(...)` callable, and add an iterative test harness driven by `driver.yaml` files (BFS as the canonical workload).

**Architecture:** Extend the v1 pipeline (lexer → parser → AST → semantic → IR → codegen) with a `MergeOp` enum carried on the existing einsum action node and on `IRMap`/`IRReduce`. Codegen is rewritten end-to-end: the v1 single-line LOAD+MAP+REDUCE+STORE inliner is deleted; the new emitter produces a `def run(<inputs>) -> <output>` function whose body is the general merge skeleton, plus a `__main__` block that constructs inputs from `init` declarations and prints the output. The iterative harness is a new Python script under `tests/e2e/iterative/` that compiles a kernel, imports the resulting module, and drives `run()` in a loop, applying element-wise feedback (`replace` / `min` / `max` / `add` / `or` / `and`) until the watched tensor stabilizes.

**Tech Stack:** C99, Flex 2.6+, Bison 2.3+ (macOS-compatible), CMake ≥ 3.15, Python 3, PyYAML (test-time only).

**Reference spec:** `docs/superpowers/specs/2026-05-06-edgecc-v2-merge-design.md`. The plan implements that spec; if anything below contradicts the spec, the spec wins.

---

## File Map

**Compiler source — modify:**
- `src/lexer.l` — add 9 merge keyword tokens.
- `src/ast.h` — add `MergeOp` enum, add `merge_op` and `reduce_merge_op` fields on `EinsumData`, add `ast_einsum_new` arity bump, add `merge_op_from_str()` declaration.
- `src/ast.c` — implement `merge_op_from_str()` and update `ast_einsum_new()` signature; add `merge_op_to_str()` for IR dumping.
- `src/parser.y` — extend `einsum_stmt` grammar rule to accept optional `, IDENT` after the compute op for both `map(...)` and `reduce(...)`. Update token list.
- `src/semantic.h` — add identity-table API.
- `src/semantic.c` — validate merge keywords are recognized, populate identity in symbol table for use during IR lowering, emit `nopass` warning to stderr.
- `src/ir.h` — extend `IRMap` and `IRReduce` with `merge_op`, `empty_a`/`empty_b`/`empty_z` literals, `identity` literal.
- `src/ir.c` — propagate empty values and identity into the new IR fields; extend `ir_dump` to show merge op and identity inline.
- `src/codegen.c` — full rewrite. Delete the v1 single-line inliner. Emit `import math` (when needed), module-level dim constants, `__edgecc_meta__`, `def run(...)` body using the general merge skeleton, and `if __name__ == "__main__"` block.
- `src/codegen.h` — unchanged.
- `src/main.c` — unchanged.

**Compiler tests — modify:**
- `tests/unit/test_lexer.c` — add token tests for the 9 merge keywords.
- `tests/unit/test_parser.c` — add tests for `map(mul)`, `map(mul, intersect)`, `reduce(min, union)` parsing and AST fields.
- `tests/unit/test_semantic.c` — add tests for unknown-merge-op error, `nopass` warning, identity lookup.
- `tests/unit/test_ir.c` — add tests asserting `IRMap`/`IRReduce` carry `merge_op`, identity, empty values; assert truth-table table values.
- `tests/unit/test_codegen.c` — update fragment assertions to match the new `def run` ABI; add fragment assertions for general merge skeleton output.

**E2E tests — create:**
- `tests/e2e/merge_intersect_gemm/{input.edge,expected_output.txt}`
- `tests/e2e/merge_union_min/{input.edge,expected_output.txt}`
- `tests/e2e/merge_take_left/{input.edge,expected_output.txt}`

**Negative tests — create:**
- `tests/negative/bad_merge_op/{input.edge,expected_error.txt}`
- `tests/negative/too_many_args/{input.edge,expected_error.txt}`
- `tests/negative/merge_no_compute/{input.edge,expected_error.txt}`

**Iterative test harness — create:**
- `tests/e2e/iterative/run_iterative.py` — runner. Creating this file also creates the `tests/e2e/iterative/` directory; no placeholder file is needed before the first iterative test is added.
- `tests/e2e/iterative/bfs_simple/{kernel.edge,driver.yaml}`
- `tests/e2e/iterative/bfs_disconnected/{kernel.edge,driver.yaml}`

**Build & docs — modify:**
- `CMakeLists.txt` — add `e2e_iterative` test invocation.
- `CLAUDE.md` — describe v2 syntax, the new generated Python ABI, iterative test framework, PyYAML dependency, removal of the v1 inline codegen path.

---

## Task 1: Lexer — Merge Operator Keywords

**Files:**
- Modify: `src/lexer.l`
- Modify: `src/parser.y` (token list only)
- Modify: `tests/unit/test_lexer.c`

The 9 merge keywords need to be recognized as their own tokens (not generic `TOK_IDENT`) so the parser grammar can clearly distinguish them. We declare a single token kind `TOK_MERGE_OP_NAME` that carries the keyword text, leaving runtime classification to semantic analysis. This keeps the lexer simple and the parser grammar small.

Wait — re-reading the design: the parser only needs to accept an arbitrary identifier in the merge slot, and semantic analysis resolves the name. That means the lexer doesn't need to add new tokens at all; existing `TOK_IDENT` already matches keywords like `intersect`. We do not introduce new tokens for merge ops. Skip lexer changes; the test additions move to the parser test file.

**This task is empty — no lexer modifications needed.** Moved tests for merge-keyword lexing into Task 3 (parser tests verify the names parse through `TOK_IDENT` correctly).

- [ ] **Step 1: Confirm — read `src/lexer.l` and verify there is no special-cased keyword for any merge name (e.g. `intersect`, `union`).**

Run: `grep -nE 'intersect|union|take_left|take_right|xor|passthrough|nopass' src/lexer.l`
Expected: no matches.

If matches exist, the assumption is wrong and the lexer needs additions; halt and reassess. Otherwise proceed.

- [ ] **Step 2: Confirm — no commit needed for this task; move on to Task 2.**

---

## Task 2: AST — MergeOp Enum and EinsumData Fields

**Files:**
- Modify: `src/ast.h`
- Modify: `src/ast.c`
- Modify: `tests/unit/test_parser.c` (uses AST; tests come in Task 3)

- [ ] **Step 1: Edit `src/ast.h` — add `MergeOp` enum after `OperatorType`.**

In `src/ast.h`, after the existing `OperatorType` enum (around line 36), add:

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
```

- [ ] **Step 2: Edit `src/ast.h` — extend `EinsumData` struct with two merge fields.**

Find the existing `EinsumData` struct (around line 87). Change it to:

```c
typedef struct {
    TensorRef output;
    TensorRef input1;
    TensorRef input2;
    OperatorType map_op;
    MergeOp      map_merge_op;       /* defaults to MERGE_PASSTHROUGH */
    OperatorType reduce_op;
    MergeOp      reduce_merge_op;    /* defaults to MERGE_PASSTHROUGH */
} EinsumData;
```

- [ ] **Step 3: Edit `src/ast.h` — update `ast_einsum_new` declaration.**

Find:
```c
AstNode *ast_einsum_new(TensorRef output, TensorRef input1, TensorRef input2,
                         OperatorType map_op, OperatorType reduce_op);
```

Replace with:
```c
AstNode *ast_einsum_new(TensorRef output, TensorRef input1, TensorRef input2,
                         OperatorType map_op, MergeOp map_merge_op,
                         OperatorType reduce_op, MergeOp reduce_merge_op);
```

- [ ] **Step 4: Edit `src/ast.h` — add `merge_op_from_str` and `merge_op_to_str` declarations.**

Add to the utilities section near `operator_from_str`:

```c
MergeOp merge_op_from_str(const char *name);     /* returns -1 on unknown */
const char *merge_op_to_str(MergeOp op);
```

- [ ] **Step 5: Edit `src/ast.c` — update `ast_einsum_new` to accept and store the new fields.**

Find the existing `ast_einsum_new` implementation. Update its signature and body so it stores `map_merge_op` and `reduce_merge_op` into the AST node:

```c
AstNode *ast_einsum_new(TensorRef output, TensorRef input1, TensorRef input2,
                         OperatorType map_op, MergeOp map_merge_op,
                         OperatorType reduce_op, MergeOp reduce_merge_op) {
    AstNode *node = malloc(sizeof(AstNode));
    node->type = NODE_EINSUM;
    node->data.einsum.output = output;
    node->data.einsum.input1 = input1;
    node->data.einsum.input2 = input2;
    node->data.einsum.map_op = map_op;
    node->data.einsum.map_merge_op = map_merge_op;
    node->data.einsum.reduce_op = reduce_op;
    node->data.einsum.reduce_merge_op = reduce_merge_op;
    return node;
}
```

- [ ] **Step 6: Edit `src/ast.c` — implement `merge_op_from_str` and `merge_op_to_str`.**

Add at the bottom of `src/ast.c`:

```c
MergeOp merge_op_from_str(const char *name) {
    if (strcmp(name, "passthrough") == 0)      return MERGE_PASSTHROUGH;
    if (strcmp(name, "nopass") == 0)           return MERGE_NOPASS;
    if (strcmp(name, "intersect") == 0)        return MERGE_INTERSECT;
    if (strcmp(name, "take_left") == 0)        return MERGE_TAKE_LEFT;
    if (strcmp(name, "take_left_only") == 0)   return MERGE_TAKE_LEFT_ONLY;
    if (strcmp(name, "take_right") == 0)       return MERGE_TAKE_RIGHT;
    if (strcmp(name, "take_right_only") == 0)  return MERGE_TAKE_RIGHT_ONLY;
    if (strcmp(name, "xor") == 0)              return MERGE_XOR;
    if (strcmp(name, "union") == 0)            return MERGE_UNION;
    return (MergeOp)-1;
}

const char *merge_op_to_str(MergeOp op) {
    switch (op) {
        case MERGE_PASSTHROUGH:     return "passthrough";
        case MERGE_NOPASS:          return "nopass";
        case MERGE_INTERSECT:       return "intersect";
        case MERGE_TAKE_LEFT:       return "take_left";
        case MERGE_TAKE_LEFT_ONLY:  return "take_left_only";
        case MERGE_TAKE_RIGHT:      return "take_right";
        case MERGE_TAKE_RIGHT_ONLY: return "take_right_only";
        case MERGE_XOR:             return "xor";
        case MERGE_UNION:           return "union";
    }
    return "?";
}
```

Add `#include <string.h>` at the top if not already present.

- [ ] **Step 7: Verify build — `cd build && cmake .. && make` should fail in `parser.y` (its call to `ast_einsum_new` is now wrong-arity).**

Run: `cd build && cmake .. && make 2>&1 | head -30`
Expected: compilation error in `parser.y` about argument count mismatch in `ast_einsum_new`. This is the correct intermediate state — Task 3 fixes the parser.

- [ ] **Step 8: Commit.**

```bash
git add src/ast.h src/ast.c
git commit -m "ast: add MergeOp enum and merge fields on einsum node"
```

---

## Task 3: Parser — Optional Merge Argument

**Files:**
- Modify: `src/parser.y`
- Modify: `tests/unit/test_parser.c`

The action argument lists become `compute_op` or `compute_op , merge_op`, matching the spec's grammar:

```
action_args  : compute_op
             | compute_op ',' merge_op
```

We implement this by extending the existing `einsum_stmt` rule. The merge slot is optional; absent means `MERGE_PASSTHROUGH`.

- [ ] **Step 1: Edit `src/parser.y` — add a `merge_op` enum to the `%union`.**

Currently `%union` has:
```c
%union {
    int int_val;
    double float_val;
    char *str_val;
    void *node;        /* AstNode* */
    void *ptr;         /* generic pointer for helper structs */
    int datatype;      /* DataType enum */
    int op;            /* OperatorType enum */
}
```

Add a comment for `merge_op`; reuse the `int` type since both enums fit:
```c
%union {
    int int_val;
    double float_val;
    char *str_val;
    void *node;        /* AstNode* */
    void *ptr;         /* generic pointer for helper structs */
    int datatype;      /* DataType enum */
    int op;            /* OperatorType enum */
    int mergeop;       /* MergeOp enum */
}
```

- [ ] **Step 2: Edit `src/parser.y` — replace the existing `einsum_stmt` rule.**

Find the rule (around line 224) and replace it with:

```yacc
einsum_stmt:
    tensor_ref '=' tensor_ref '.' tensor_ref TOK_DOUBLE_COLON
        TOK_MAP '(' TOK_IDENT map_merge_arg ')'
        TOK_REDUCE '(' TOK_IDENT reduce_merge_arg ')' ';'
    {
        TensorRef *out = (TensorRef *)$1;
        TensorRef *in1 = (TensorRef *)$3;
        TensorRef *in2 = (TensorRef *)$5;
        OperatorType map_op    = operator_from_str($9);
        MergeOp     map_merge  = (MergeOp)$10;
        OperatorType reduce_op = operator_from_str($13);
        MergeOp     red_merge  = (MergeOp)$14;
        $$ = (void *)ast_einsum_new(*out, *in1, *in2,
                                    map_op, map_merge,
                                    reduce_op, red_merge);
        free($9);
        free($13);
        free(out);
        free(in1);
        free(in2);
    }
    ;

map_merge_arg:
    /* empty */                { $$ = MERGE_PASSTHROUGH; }
    | ',' TOK_IDENT
    {
        MergeOp m = merge_op_from_str($2);
        if ((int)m == -1) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "unknown merge operator '%s' (expected one of: "
                     "passthrough, nopass, intersect, take_left, "
                     "take_left_only, take_right, take_right_only, xor, union)",
                     $2);
            yyerror(buf);
            YYERROR;
        }
        free($2);
        $$ = m;
    }
    ;

reduce_merge_arg:
    /* empty */                { $$ = MERGE_PASSTHROUGH; }
    | ',' TOK_IDENT
    {
        MergeOp m = merge_op_from_str($2);
        if ((int)m == -1) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "unknown merge operator '%s' (expected one of: "
                     "passthrough, nopass, intersect, take_left, "
                     "take_left_only, take_right, take_right_only, xor, union)",
                     $2);
            yyerror(buf);
            YYERROR;
        }
        free($2);
        $$ = m;
    }
    ;
```

- [ ] **Step 3: Edit `src/parser.y` — declare the new `%type` for merge-arg productions.**

In the `%type` declarations region (around line 99-102), add:
```yacc
%type <mergeop> map_merge_arg reduce_merge_arg
```

- [ ] **Step 4: Edit `tests/unit/test_parser.c` — write the failing test for merge parsing.**

Add a test after the existing einsum test:

```c
TEST(test_parse_einsum_with_merge) {
    const char *src =
        "dim D = 3; dim S = 3;\n"
        "tensor F1[R=D] : float, empty = inf;\n"
        "tensor G[R=S, C=D] : float, empty = 0.0;\n"
        "tensor F[R=S]  : float, empty = inf;\n"
        "F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);\n";
    YY_BUFFER_STATE buf = yy_scan_string(src);
    ASSERT_INT_EQ(0, yyparse());
    yy_delete_buffer(buf);
    AstNode *prog = parse_result;
    ASSERT_NOT_NULL(prog);
    /* find the einsum statement (last child) */
    int n = prog->data.program.count;
    AstNode *ein = prog->data.program.statements[n - 1];
    ASSERT_INT_EQ(NODE_EINSUM, ein->type);
    ASSERT_INT_EQ(OP_ADD, ein->data.einsum.map_op);
    ASSERT_INT_EQ(MERGE_INTERSECT, ein->data.einsum.map_merge_op);
    ASSERT_INT_EQ(OP_MIN, ein->data.einsum.reduce_op);
    ASSERT_INT_EQ(MERGE_UNION, ein->data.einsum.reduce_merge_op);
    ast_free(prog);
    parse_result = NULL;
}

TEST(test_parse_einsum_default_merge) {
    const char *src =
        "dim M = 2; dim K = 2; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n";
    YY_BUFFER_STATE buf = yy_scan_string(src);
    ASSERT_INT_EQ(0, yyparse());
    yy_delete_buffer(buf);
    AstNode *prog = parse_result;
    ASSERT_NOT_NULL(prog);
    int n = prog->data.program.count;
    AstNode *ein = prog->data.program.statements[n - 1];
    ASSERT_INT_EQ(MERGE_PASSTHROUGH, ein->data.einsum.map_merge_op);
    ASSERT_INT_EQ(MERGE_PASSTHROUGH, ein->data.einsum.reduce_merge_op);
    ast_free(prog);
    parse_result = NULL;
}
```

Add `RUN_TEST(test_parse_einsum_with_merge);` and `RUN_TEST(test_parse_einsum_default_merge);` to `main()` of `test_parser.c`.

- [ ] **Step 5: Build and run the new tests; expect failures (build fails because grammar isn't compiled until parser.y change is in).**

Run: `cd build && cmake .. && make test_parser 2>&1 | tail -30`
Expected: compilation succeeds (parser.y changes from Step 2 are in).
Run: `./build/test_parser`
Expected: `PASS test_parse_einsum_with_merge` and `PASS test_parse_einsum_default_merge`.

If parsing fails at this stage, debug. The merge identifier resolution happens via `merge_op_from_str` which already exists from Task 2.

- [ ] **Step 6: Run all unit tests to confirm no regressions.**

Run: `cd build && ctest --output-on-failure 2>&1 | tail -20`
Expected: at this stage `test_lexer`, `test_parser` pass. `test_semantic`, `test_ir`, `test_codegen` may still pass or fail depending on Bison's output structure for the EinsumData; if they fail it's because they call `ast_einsum_new` directly with the old arity. That's fixed in steps below.

If `test_semantic` / `test_ir` / `test_codegen` compile-fail because they construct AST manually with old `ast_einsum_new` arity, fix the call sites by adding `MERGE_PASSTHROUGH` for both new arguments. Search:
```bash
grep -n 'ast_einsum_new' tests/unit/*.c
```
Update each call to pass `MERGE_PASSTHROUGH` in both new positions.

- [ ] **Step 7: Commit.**

```bash
git add src/parser.y tests/unit/test_parser.c tests/unit/test_semantic.c tests/unit/test_ir.c tests/unit/test_codegen.c
git commit -m "parser: optional merge operator in map/reduce action args"
```

---

## Task 4: Semantic Analysis — Identity Lookup and `nopass` Warning

**Files:**
- Modify: `src/semantic.h`
- Modify: `src/semantic.c`
- Modify: `tests/unit/test_semantic.c`

Semantic analysis already validates indices and dimensions. v2 adds:
1. An identity-lookup helper used by IR lowering: `compute_op_identity(OperatorType op, DataType dt) -> double`.
2. A `nopass` warning emitted to stderr.

The unknown-merge-op error is caught in the parser (Task 3); semantic analysis doesn't re-validate.

- [ ] **Step 1: Edit `src/semantic.h` — add identity-lookup declaration.**

Add at the bottom, before `#endif`:

```c
double compute_op_identity(OperatorType op, DataType dt);
```

- [ ] **Step 2: Edit `src/semantic.c` — implement `compute_op_identity` and a `nopass` walk.**

Add at the bottom of `src/semantic.c`:

```c
#include <math.h>     /* for INFINITY */
#include <limits.h>   /* for INT_MAX, INT_MIN */

double compute_op_identity(OperatorType op, DataType dt) {
    switch (op) {
        case OP_ADD: case OP_SUB:
            return 0.0;
        case OP_MUL:
            return 1.0;
        case OP_MIN:
            return (dt == TYPE_INTEGER) ? (double)INT_MAX : INFINITY;
        case OP_MAX:
            return (dt == TYPE_INTEGER) ? (double)INT_MIN : -INFINITY;
        case OP_AND:
            return 1.0;     /* true */
        case OP_OR:
            return 0.0;     /* false */
    }
    return 0.0;
}
```

- [ ] **Step 3: Edit `src/semantic.c` — emit `nopass` warning during analysis.**

Find the place where einsum statements are walked in `semantic_analyze`. After the existing index-classification logic for an einsum, add:

```c
if (ein->data.einsum.map_merge_op == MERGE_NOPASS) {
    fprintf(stderr,
        "warning: 'nopass' merge on map always produces empty output; "
        "the action has no effect\n");
}
if (ein->data.einsum.reduce_merge_op == MERGE_NOPASS) {
    fprintf(stderr,
        "warning: 'nopass' merge on reduce always produces empty output; "
        "the action has no effect\n");
}
```

- [ ] **Step 4: Edit `tests/unit/test_semantic.c` — add tests for identity lookup.**

```c
TEST(test_identity_add_float) {
    ASSERT_TRUE(compute_op_identity(OP_ADD, TYPE_FLOAT) == 0.0);
}
TEST(test_identity_mul_float) {
    ASSERT_TRUE(compute_op_identity(OP_MUL, TYPE_FLOAT) == 1.0);
}
TEST(test_identity_min_float) {
    ASSERT_TRUE(compute_op_identity(OP_MIN, TYPE_FLOAT) > 1e308); /* +inf */
}
TEST(test_identity_max_float) {
    ASSERT_TRUE(compute_op_identity(OP_MAX, TYPE_FLOAT) < -1e308); /* -inf */
}
TEST(test_identity_min_int) {
    ASSERT_TRUE(compute_op_identity(OP_MIN, TYPE_INTEGER) == (double)INT_MAX);
}
TEST(test_identity_and_bool) {
    ASSERT_TRUE(compute_op_identity(OP_AND, TYPE_BOOLEAN) == 1.0);
}
TEST(test_identity_or_bool) {
    ASSERT_TRUE(compute_op_identity(OP_OR, TYPE_BOOLEAN) == 0.0);
}
```

Add `#include <limits.h>` at the top of `test_semantic.c` if not already present, and `RUN_TEST(...)` for each new test in `main()`.

- [ ] **Step 5: Build and run `test_semantic`.**

Run: `cd build && make test_semantic && ./test_semantic`
Expected: all new tests pass.

- [ ] **Step 6: Commit.**

```bash
git add src/semantic.h src/semantic.c tests/unit/test_semantic.c
git commit -m "semantic: add compute-op identity lookup and nopass warning"
```

---

## Task 5: IR — Carry Merge, Identity, and Empty Values

**Files:**
- Modify: `src/ir.h`
- Modify: `src/ir.c`
- Modify: `tests/unit/test_ir.c`

`IRMap` and `IRReduce` gain merge_op, identity, and empty-value fields. `ir_lower` populates them from the AST and the symbol table. `ir_dump` prints them inline.

- [ ] **Step 1: Edit `src/ir.h` — extend `IRMap` and `IRReduce`.**

Replace the existing struct definitions with:

```c
typedef struct {
    int dst_reg;
    OperatorType op;
    MergeOp      merge_op;
    int src_reg1;
    int src_reg2;
    double empty_a;        /* empty value of left operand */
    double empty_b;        /* empty value of right operand */
    double empty_z;        /* empty value of output (for maptmp emission check) */
} IRMap;

typedef struct {
    int dst_reg;
    OperatorType op;
    MergeOp      merge_op;
    int src_reg;
    int acc_reg;
    double identity;       /* compute-op identity for reduction state */
    double empty_z;        /* empty value of output */
} IRReduce;
```

- [ ] **Step 2: Edit `src/ir.c` — populate the new fields during lowering.**

Find where the einsum is lowered (the function that produces `IRMap` and `IRReduce` instructions). It currently reads `ein->data.einsum.map_op` and `ein->data.einsum.reduce_op`. Locate the symbol table entries for the input operand tensors and the output tensor by name to read their `empty_val`. Then populate the new fields:

```c
/* Inside ir_lower's einsum handler, after building the IRMap instruction: */
ir_map.merge_op = ein->data.einsum.map_merge_op;
ir_map.empty_a  = symtab_find_tensor(symtab, ein->data.einsum.input1.tensor_name)->empty_val;
ir_map.empty_b  = symtab_find_tensor(symtab, ein->data.einsum.input2.tensor_name)->empty_val;
ir_map.empty_z  = symtab_find_tensor(symtab, ein->data.einsum.output.tensor_name)->empty_val;

/* And for the IRReduce: */
TensorEntry *zent = symtab_find_tensor(symtab, ein->data.einsum.output.tensor_name);
ir_red.merge_op = ein->data.einsum.reduce_merge_op;
ir_red.identity = compute_op_identity(ein->data.einsum.reduce_op, zent->datatype);
ir_red.empty_z  = zent->empty_val;
```

If `symtab_find_tensor` does not exist, add it to `semantic.h`/`semantic.c`:

```c
/* in semantic.h */
TensorEntry *symtab_find_tensor(SymbolTable *st, const char *name);

/* in semantic.c */
TensorEntry *symtab_find_tensor(SymbolTable *st, const char *name) {
    for (int i = 0; i < st->num_tensors; i++) {
        if (strcmp(st->tensors[i].name, name) == 0) return &st->tensors[i];
    }
    return NULL;
}
```

- [ ] **Step 3: Edit `src/ir.c` — extend `ir_dump` to print merge op and identity.**

Find the dump cases for `IR_MAP` and `IR_REDUCE`. Replace with:

```c
case IR_MAP:
    fprintf(out, "  MAP %s %s (eA=%g, eB=%g, eZ=%g) r%d, r%d -> r%d\n",
        operator_to_str(instr->data.map.op),
        merge_op_to_str(instr->data.map.merge_op),
        instr->data.map.empty_a,
        instr->data.map.empty_b,
        instr->data.map.empty_z,
        instr->data.map.src_reg1,
        instr->data.map.src_reg2,
        instr->data.map.dst_reg);
    break;

case IR_REDUCE:
    fprintf(out, "  REDUCE %s %s (id=%g, eZ=%g) r%d -> r%d\n",
        operator_to_str(instr->data.reduce.op),
        merge_op_to_str(instr->data.reduce.merge_op),
        instr->data.reduce.identity,
        instr->data.reduce.empty_z,
        instr->data.reduce.src_reg,
        instr->data.reduce.dst_reg);
    break;
```

- [ ] **Step 4: Edit `tests/unit/test_ir.c` — add tests for the new fields.**

```c
TEST(test_ir_carries_merge_and_identity) {
    AstNode *ast = parse(
        "dim S = 3; dim D = 3;\n"
        "tensor F1[R=D] : float, empty = inf;\n"
        "tensor G[R=S, C=D] : float, empty = 0.0;\n"
        "tensor F[R=S]  : float, empty = inf;\n"
        "F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    int found_map = 0, found_red = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_MAP) {
            ASSERT_INT_EQ(MERGE_INTERSECT, ir->instrs[i].data.map.merge_op);
            ASSERT_TRUE(ir->instrs[i].data.map.empty_a == 0.0);
            ASSERT_TRUE(ir->instrs[i].data.map.empty_b > 1e308);   /* inf */
            ASSERT_TRUE(ir->instrs[i].data.map.empty_z > 1e308);   /* inf */
            found_map = 1;
        }
        if (ir->instrs[i].type == IR_REDUCE) {
            ASSERT_INT_EQ(MERGE_UNION, ir->instrs[i].data.reduce.merge_op);
            ASSERT_TRUE(ir->instrs[i].data.reduce.identity > 1e308); /* +inf for min */
            ASSERT_TRUE(ir->instrs[i].data.reduce.empty_z  > 1e308);
            found_red = 1;
        }
    }
    ASSERT_TRUE(found_map);
    ASSERT_TRUE(found_red);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_ir_default_merge_is_passthrough) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 2; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_MAP) {
            ASSERT_INT_EQ(MERGE_PASSTHROUGH, ir->instrs[i].data.map.merge_op);
        }
        if (ir->instrs[i].type == IR_REDUCE) {
            ASSERT_INT_EQ(MERGE_PASSTHROUGH, ir->instrs[i].data.reduce.merge_op);
        }
    }
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```

`RUN_TEST(...)` both. The `parse` helper exists in `test_codegen.c`; copy it into `test_ir.c` if not already present.

- [ ] **Step 5: Build and run `test_ir`.**

Run: `cd build && make test_ir && ./test_ir`
Expected: all tests pass. The `--dump-ir` change will be exercised manually below.

- [ ] **Step 6: Manually run `--dump-ir` on a merge program to spot-check the output.**

Run:
```bash
echo 'dim S = 2; dim D = 2;
tensor F1[R=D] : float, empty = inf;
tensor G[R=S, C=D] : float, empty = 0.0;
tensor F[R=S]  : float, empty = inf;
init G = [[0,1],[1,0]];
init F = [0, inf];
F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);' > /tmp/iter.edge
./build/edgecc /tmp/iter.edge -o /tmp/iter.py --dump-ir 2>&1 | head -40
```
Expected: `--dump-ir` output contains lines like `MAP add intersect (eA=0, eB=inf, eZ=inf) r1, r2 -> r3` and `REDUCE min union (id=inf, eZ=inf) r3 -> rN`. Compilation will continue but the resulting Python will be wrong (codegen is rewritten in Task 6).

- [ ] **Step 7: Commit.**

```bash
git add src/ir.h src/ir.c src/semantic.h src/semantic.c tests/unit/test_ir.c
git commit -m "ir: extend MAP/REDUCE with merge op, identity, and empty values"
```

---

## Task 6: Codegen — `def run` ABI and General Merge Skeleton

**Files:**
- Rewrite: `src/codegen.c`
- Modify: `tests/unit/test_codegen.c`
- Modify: existing E2E test expected outputs only if needed (check at end)

This is the largest task. The v1 inline codegen is replaced with a general emitter that always produces the same skeleton structure regardless of merge configuration.

- [ ] **Step 1: Add the merge truth-table reference table at the top of `src/codegen.c`.**

After the existing `#include` lines and helper definitions, add:

```c
/* Per-operator Python expression emitted as the merge guard.
 * For Map, substitute "a_exists" / "b_exists".
 * For Reduce, substitute "b_s" / "b_m".
 * Index by MergeOp enum value. */
static const char *MERGE_GUARD_TEMPLATE[] = {
    [MERGE_PASSTHROUGH]      = "True",
    [MERGE_NOPASS]           = "False",
    [MERGE_INTERSECT]        = "%s and %s",
    [MERGE_TAKE_LEFT]        = "%s",
    [MERGE_TAKE_LEFT_ONLY]   = "%s and not %s",
    [MERGE_TAKE_RIGHT]       = "%s",                       /* substitute b_exists/b_m as the only operand below */
    [MERGE_TAKE_RIGHT_ONLY]  = "%s and not %s",            /* substitute b first, a second */
    [MERGE_XOR]              = "%s != %s",
    [MERGE_UNION]            = "%s or %s",
};

/* For "take_right" / "take_right_only", the template uses the right operand
 * first (or alone). The helper below picks the operand-substitution order. */
static int merge_guard_arity(MergeOp m) {
    switch (m) {
        case MERGE_PASSTHROUGH: case MERGE_NOPASS:
            return 0;
        case MERGE_TAKE_LEFT: case MERGE_TAKE_RIGHT:
            return 1;
        default:
            return 2;
    }
}
```

Then add a small helper that writes the merge expression into the buffer with the right operand names. For unary cases (`take_left`, `take_right`), only one operand is referenced; for binary cases, two:

```c
/* Emit the merge guard expression for a given merge op.
 * `lhs_name` is the left-side existence variable name (e.g. "a_exists" or "b_s").
 * `rhs_name` is the right-side existence variable name (e.g. "b_exists" or "b_m"). */
static void emit_merge_guard(CodeBuf *cb, MergeOp m,
                              const char *lhs_name, const char *rhs_name) {
    switch (m) {
        case MERGE_PASSTHROUGH:
            cb_append(cb, "True");
            return;
        case MERGE_NOPASS:
            cb_append(cb, "False");
            return;
        case MERGE_INTERSECT:
            cb_appendf(cb, "%s and %s", lhs_name, rhs_name);
            return;
        case MERGE_TAKE_LEFT:
            cb_append(cb, lhs_name);
            return;
        case MERGE_TAKE_LEFT_ONLY:
            cb_appendf(cb, "%s and not %s", lhs_name, rhs_name);
            return;
        case MERGE_TAKE_RIGHT:
            cb_append(cb, rhs_name);
            return;
        case MERGE_TAKE_RIGHT_ONLY:
            cb_appendf(cb, "%s and not %s", rhs_name, lhs_name);
            return;
        case MERGE_XOR:
            cb_appendf(cb, "%s != %s", lhs_name, rhs_name);
            return;
        case MERGE_UNION:
            cb_appendf(cb, "%s or %s", lhs_name, rhs_name);
            return;
    }
}
```

- [ ] **Step 2: Add a helper `format_double_literal` to render empty values and identities as Python literals.**

```c
/* Format a double for emission as a Python literal.
 * +inf -> "math.inf", -inf -> "-math.inf", integral values for non-bool
 * tensors keep one decimal (e.g. "0.0"), other values use %g.
 * Returns a malloc'd string; caller frees. */
static char *format_double_literal(double v) {
    char buf[64];
    if (v > 1e308) {           /* +inf */
        snprintf(buf, sizeof(buf), "math.inf");
    } else if (v < -1e308) {   /* -inf */
        snprintf(buf, sizeof(buf), "-math.inf");
    } else if (v == (int)v && v >= -1e9 && v <= 1e9) {
        snprintf(buf, sizeof(buf), "%.1f", v);
    } else {
        snprintf(buf, sizeof(buf), "%g", v);
    }
    return strdup(buf);
}
```

The codegen's existing `emit_alloc` should be reviewed: where it currently formats values with `%.1f`, replace that block to use `format_double_literal` so `inf` empty values render as `math.inf`.

- [ ] **Step 3: Add helpers for collecting input/output tensors.**

```c
/* Collect distinct input tensor names (those appearing in IR_LOAD, in load order).
 * Output tensor name is the operand of IR_STORE. Caller-owned. */
typedef struct {
    char *names[8];
    int count;
} TensorNameList;

static void collect_io(IRProgram *ir, TensorNameList *inputs, char **output) {
    inputs->count = 0;
    *output = NULL;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type == IR_LOAD) {
            const char *n = instr->data.load.tensor_name;
            int seen = 0;
            for (int j = 0; j < inputs->count; j++) {
                if (strcmp(inputs->names[j], n) == 0) { seen = 1; break; }
            }
            if (!seen && inputs->count < 8) {
                inputs->names[inputs->count++] = strdup(n);
            }
        } else if (instr->type == IR_STORE) {
            *output = strdup(instr->data.store.tensor_name);
        }
    }
}

static void free_tensor_name_list(TensorNameList *l) {
    for (int i = 0; i < l->count; i++) free(l->names[i]);
    l->count = 0;
}
```

- [ ] **Step 4: Replace `codegen_python` with the new emitter.**

The new function emits in this order:

1. Header comment + `import math`.
2. Module-level dim constants (from `IR_DIM_DEF`).
3. `__edgecc_meta__` dict.
4. `def run(<inputs>):` — body emits output allocation (re-init each call), the loops, and `return <output>`.
5. `if __name__ == "__main__":` — emits input alloc + init, calls `run`, prints output.

Replace the existing function with the structure below. Read the existing function for buffer / indent helpers; reuse them.

```c
char *codegen_python(IRProgram *ir) {
    if (!ir) return NULL;
    CodeBuf cb;
    cb_init(&cb);

    cb_append(&cb, "# Generated by edgecc\n");
    cb_append(&cb, "import math\n\n");

    /* Module-level dim constants */
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_DIM_DEF) {
            cb_appendf(&cb, "%s = %d\n",
                ir->instrs[i].data.dim_def.name,
                ir->instrs[i].data.dim_def.size);
        }
    }
    cb_append(&cb, "\n");

    /* Discover inputs and output */
    TensorNameList inputs;
    char *output = NULL;
    collect_io(ir, &inputs, &output);

    /* __edgecc_meta__ */
    cb_append(&cb, "__edgecc_meta__ = {\n");
    cb_append(&cb, "    \"inputs\": [");
    for (int i = 0; i < inputs.count; i++) {
        if (i > 0) cb_append(&cb, ", ");
        cb_appendf(&cb, "\"%s\"", inputs.names[i]);
    }
    cb_append(&cb, "],\n");
    cb_appendf(&cb, "    \"output\": \"%s\",\n", output ? output : "");
    cb_append(&cb, "}\n\n");

    /* def run(inputs) */
    cb_append(&cb, "def run(");
    for (int i = 0; i < inputs.count; i++) {
        if (i > 0) cb_append(&cb, ", ");
        cb_append(&cb, inputs.names[i]);
    }
    cb_append(&cb, "):\n");
    cb.indent = 1;

    /* Allocate output to identity-equals-empty (use the output's IR_TENSOR_ALLOC) */
    /* Walk the IR and emit:
       - IR_TENSOR_ALLOC for the OUTPUT tensor only (inputs are passed in)
       - All loops with the new merge skeleton emitted at IR_STORE
       - `return <output>` at end of function
    */
    int produced_loops = 0;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        switch (instr->type) {
            case IR_DIM_DEF: break;          /* already emitted at module scope */
            case IR_TENSOR_ALLOC:
                /* Only emit allocation for the OUTPUT tensor. Inputs come in via parameters. */
                if (output && strcmp(instr->data.tensor_alloc.name, output) == 0) {
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s = ", instr->data.tensor_alloc.name);
                    emit_alloc(&cb, &instr->data.tensor_alloc, ir, i);
                    cb_append(&cb, "\n");
                }
                break;
            case IR_TENSOR_INIT:
                /* Init statements lift into __main__; skip inside run(). */
                break;
            case IR_LOOP_BEGIN: {
                if (!produced_loops) {
                    /* leading blank line before first loop */
                    cb_append(&cb, "\n");
                    produced_loops = 1;
                }
                cb_indent(&cb);
                const char *var = instr->data.loop_begin.index_var;
                int range = instr->data.loop_begin.range;
                const char *dim_name = find_dim_name(ir, i, range);
                if (dim_name) {
                    cb_appendf(&cb, "for %s in range(%s):\n", var, dim_name);
                } else {
                    cb_appendf(&cb, "for %s in range(%d):\n", var, range);
                }
                cb.indent++;
                break;
            }
            case IR_LOOP_END:
                cb.indent--;
                break;
            case IR_LOAD: case IR_MAP: case IR_REDUCE:
                break;     /* combined into STORE emission below */
            case IR_STORE:
                if (i >= 4) {
                    IRInstr *load1 = &ir->instrs[i - 4];
                    IRInstr *load2 = &ir->instrs[i - 3];
                    IRInstr *map   = &ir->instrs[i - 2];
                    IRInstr *red   = &ir->instrs[i - 1];
                    if (load1->type == IR_LOAD && load2->type == IR_LOAD &&
                        map->type == IR_MAP && red->type == IR_REDUCE) {
                        emit_general_skeleton(&cb,
                            load1, load2, map, red, instr);
                    }
                }
                break;
        }
    }

    if (output) {
        cb_indent(&cb);
        cb_appendf(&cb, "return %s\n", output);
    }
    cb.indent = 0;
    cb_append(&cb, "\n");

    /* __main__ block: re-emit allocs+inits+call */
    cb_append(&cb, "if __name__ == \"__main__\":\n");
    cb.indent = 1;
    /* Emit allocs for INPUT tensors only (the output gets recomputed by run) */
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type != IR_TENSOR_ALLOC) continue;
        int is_input = 0;
        for (int j = 0; j < inputs.count; j++) {
            if (strcmp(instr->data.tensor_alloc.name, inputs.names[j]) == 0) { is_input = 1; break; }
        }
        if (!is_input) continue;
        cb_indent(&cb);
        cb_appendf(&cb, "%s = ", instr->data.tensor_alloc.name);
        emit_alloc(&cb, &instr->data.tensor_alloc, ir, i);
        cb_append(&cb, "\n");
    }
    /* Emit init statements (inputs only — the output is initialized inside run) */
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type != IR_TENSOR_INIT) continue;
        int is_input = 0;
        for (int j = 0; j < inputs.count; j++) {
            if (strcmp(instr->data.tensor_init.name, inputs.names[j]) == 0) { is_input = 1; break; }
        }
        if (!is_input) continue;
        cb_indent(&cb);
        cb_appendf(&cb, "%s = ", instr->data.tensor_init.name);
        emit_literal(&cb, instr->data.tensor_init.data);
        cb_append(&cb, "\n");
    }

    /* Call run and print */
    if (output) {
        cb_indent(&cb);
        cb_appendf(&cb, "%s = run(", output);
        for (int i = 0; i < inputs.count; i++) {
            if (i > 0) cb_append(&cb, ", ");
            cb_append(&cb, inputs.names[i]);
        }
        cb_append(&cb, ")\n");
        cb_indent(&cb);
        cb_appendf(&cb, "print(%s)\n", output);
    }

    free(output);
    free_tensor_name_list(&inputs);
    return cb.buf;
}
```

- [ ] **Step 5: Implement `emit_general_skeleton`.**

Add this function above `codegen_python` in `src/codegen.c`. It emits the spec's general skeleton (LOAD, MAP existence/merge, REDUCE existence/merge, store-or-empty) into the current loop body:

```c
static const char *compute_op_pyexpr(OperatorType op,
                                     const char *lhs, const char *rhs,
                                     char *out, size_t cap) {
    switch (op) {
        case OP_ADD: snprintf(out, cap, "%s + %s", lhs, rhs); break;
        case OP_SUB: snprintf(out, cap, "%s - %s", lhs, rhs); break;
        case OP_MUL: snprintf(out, cap, "%s * %s", lhs, rhs); break;
        case OP_MIN: snprintf(out, cap, "min(%s, %s)", lhs, rhs); break;
        case OP_MAX: snprintf(out, cap, "max(%s, %s)", lhs, rhs); break;
        case OP_AND: snprintf(out, cap, "%s and %s", lhs, rhs); break;
        case OP_OR:  snprintf(out, cap, "%s or %s",  lhs, rhs); break;
    }
    return out;
}

static void emit_general_skeleton(CodeBuf *cb,
                                   IRInstr *load1, IRInstr *load2,
                                   IRInstr *map_i, IRInstr *red_i,
                                   IRInstr *store_i) {
    char *eA = format_double_literal(map_i->data.map.empty_a);
    char *eB = format_double_literal(map_i->data.map.empty_b);
    char *eZ = format_double_literal(map_i->data.map.empty_z);

    /* LOAD operands */
    cb_indent(cb); cb_append(cb, "a_val = ");
    emit_tensor_access(cb, load1->data.load.tensor_name,
                       load1->data.load.coords, load1->data.load.ncoords);
    cb_append(cb, "\n");
    cb_indent(cb); cb_append(cb, "b_val = ");
    emit_tensor_access(cb, load2->data.load.tensor_name,
                       load2->data.load.coords, load2->data.load.ncoords);
    cb_append(cb, "\n");

    /* MAP existence */
    cb_indent(cb); cb_appendf(cb, "a_exists = a_val != %s\n", eA);
    cb_indent(cb); cb_appendf(cb, "b_exists = b_val != %s\n", eB);

    cb_indent(cb); cb_append(cb, "if ");
    emit_merge_guard(cb, map_i->data.map.merge_op, "a_exists", "b_exists");
    cb_append(cb, ":\n");
    cb->indent++;
    cb_indent(cb); cb_appendf(cb, "a_op = a_val if a_exists else %s\n", eA);
    cb_indent(cb); cb_appendf(cb, "b_op = b_val if b_exists else %s\n", eB);

    char compute_buf[256];
    cb_indent(cb); cb_append(cb, "map_val = ");
    cb_append(cb, compute_op_pyexpr(map_i->data.map.op, "a_op", "b_op",
                                     compute_buf, sizeof(compute_buf)));
    cb_append(cb, "\n");
    cb_indent(cb); cb_appendf(cb, "map_emitted = map_val != %s\n", eZ);
    cb->indent--;
    cb_indent(cb); cb_append(cb, "else:\n");
    cb->indent++;
    cb_indent(cb); cb_append(cb, "map_emitted = False\n");
    cb_indent(cb); cb_appendf(cb, "map_val = %s\n", eZ);
    cb->indent--;

    /* REDUCE existence */
    cb_indent(cb); cb_append(cb, "state = ");
    emit_tensor_access(cb, store_i->data.store.tensor_name,
                       store_i->data.store.coords, store_i->data.store.ncoords);
    cb_append(cb, "\n");
    cb_indent(cb); cb_appendf(cb, "b_s = state != %s\n", eZ);
    cb_indent(cb); cb_append(cb, "b_m = map_emitted\n");

    cb_indent(cb); cb_append(cb, "if ");
    emit_merge_guard(cb, red_i->data.reduce.merge_op, "b_s", "b_m");
    cb_append(cb, ":\n");
    cb->indent++;
    cb_indent(cb); cb_appendf(cb, "val_for_red = map_val if map_emitted else %s\n", eZ);
    cb_indent(cb); cb_append(cb, "new_state = ");
    cb_append(cb, compute_op_pyexpr(red_i->data.reduce.op, "state", "val_for_red",
                                     compute_buf, sizeof(compute_buf)));
    cb_append(cb, "\n");
    cb_indent(cb);
    emit_tensor_access(cb, store_i->data.store.tensor_name,
                       store_i->data.store.coords, store_i->data.store.ncoords);
    cb_appendf(cb, " = new_state if new_state != %s else %s\n", eZ, eZ);
    cb->indent--;
    cb_indent(cb); cb_append(cb, "else:\n");
    cb->indent++;
    cb_indent(cb);
    emit_tensor_access(cb, store_i->data.store.tensor_name,
                       store_i->data.store.coords, store_i->data.store.ncoords);
    cb_appendf(cb, " = %s\n", eZ);
    cb->indent--;

    free(eA); free(eB); free(eZ);
}
```

Forward-declare both helpers near the top of `src/codegen.c` (above `codegen_python`):

```c
static void emit_general_skeleton(CodeBuf *cb,
                                   IRInstr *load1, IRInstr *load2,
                                   IRInstr *map_i, IRInstr *red_i,
                                   IRInstr *store_i);
static char *format_double_literal(double v);
```

- [ ] **Step 6: Update `tests/unit/test_codegen.c` — adjust fragment assertions for the new ABI.**

Replace the existing `test_codegen_gemm` body with assertions that match the v2 form:

```c
TEST(test_codegen_gemm) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 3; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "init A = [[1, 2], [3, 4], [5, 6]];\n"
        "init B = [[7, 8], [9, 10], [11, 12]];\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "import math") != NULL);
    ASSERT_TRUE(strstr(code, "M = 2") != NULL);
    ASSERT_TRUE(strstr(code, "__edgecc_meta__") != NULL);
    ASSERT_TRUE(strstr(code, "\"inputs\": [\"A\", \"B\"]") != NULL);
    ASSERT_TRUE(strstr(code, "\"output\": \"Z\"") != NULL);
    ASSERT_TRUE(strstr(code, "def run(A, B):") != NULL);
    ASSERT_TRUE(strstr(code, "for m in range(") != NULL);
    ASSERT_TRUE(strstr(code, "if True:") != NULL);                 /* passthrough merge */
    ASSERT_TRUE(strstr(code, "a_exists = a_val") != NULL);
    ASSERT_TRUE(strstr(code, "map_val = a_op * b_op") != NULL);
    ASSERT_TRUE(strstr(code, "return Z") != NULL);
    ASSERT_TRUE(strstr(code, "if __name__ == \"__main__\":") != NULL);
    ASSERT_TRUE(strstr(code, "A = [[1, 2], [3, 4], [5, 6]]") != NULL);
    ASSERT_TRUE(strstr(code, "Z = run(A, B)") != NULL);
    ASSERT_TRUE(strstr(code, "print(Z)") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_intersect_union) {
    AstNode *ast = parse(
        "dim S = 2; dim D = 2;\n"
        "tensor F1[R=D] : float, empty = inf;\n"
        "tensor G[R=S, C=D] : float, empty = 0.0;\n"
        "tensor F[R=S]  : float, empty = inf;\n"
        "init G = [[0,1],[1,0]];\n"
        "init F = [0, inf];\n"
        "F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "if a_exists and b_exists:") != NULL);     /* intersect */
    ASSERT_TRUE(strstr(code, "if b_s or b_m:") != NULL);                /* union */
    ASSERT_TRUE(strstr(code, "math.inf") != NULL);
    ASSERT_TRUE(strstr(code, "map_val = a_op + b_op") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```

Add `RUN_TEST(test_codegen_intersect_union);` to `main()`.

- [ ] **Step 7: Build and run all unit tests.**

```bash
cd build && make && ctest --output-on-failure
```
Expected: all unit tests pass, including `test_codegen` with the new fragments.

- [ ] **Step 8: Run E2E tests; verify `gemm`, `min_plus`, `boolean_and_or` still produce expected stdout.**

```bash
cd build && ctest --output-on-failure -R 'e2e_tests'
```
Expected: all 3 v1 E2E tests pass. The generated `.py` files will look very different (verbose skeleton wrapped in `def run`), but the printed output is identical because the algorithm is unchanged.

If any E2E test fails: inspect the generated `.py` (run `./build/edgecc tests/e2e/gemm/input.edge -o /tmp/g.py && python3 /tmp/g.py`). Common failure modes are missing `import math` for non-`inf` programs (acceptable to leave the import even when unused) and incorrect inputs/output classification.

- [ ] **Step 9: Commit.**

```bash
git add src/codegen.c tests/unit/test_codegen.c
git commit -m "codegen: rewrite as def run() with general merge skeleton"
```

---

## Task 7: Single-Shot E2E Tests for Merge

**Files:**
- Create: `tests/e2e/merge_intersect_gemm/input.edge`
- Create: `tests/e2e/merge_intersect_gemm/expected_output.txt`
- Create: `tests/e2e/merge_union_min/input.edge`
- Create: `tests/e2e/merge_union_min/expected_output.txt`
- Create: `tests/e2e/merge_take_left/input.edge`
- Create: `tests/e2e/merge_take_left/expected_output.txt`

The existing `tests/e2e/run_e2e.py` automatically picks up any new subdirectory with `input.edge` + `expected_output.txt`, so no runner changes are needed.

- [ ] **Step 1: Create `tests/e2e/merge_intersect_gemm/input.edge`.**

```
dim M = 2;
dim K = 3;
dim N = 2;

tensor Z[R=M, C=N] : float, empty = 0.0;
tensor A[R=K, C=M] : float, empty = 0.0;
tensor B[R=K, C=N] : float, empty = 0.0;

init A = [[1, 2], [0, 4], [5, 6]];
init B = [[0, 8], [9, 10], [11, 12]];

Z[m,n] = A[k,m] . B[k,n] :: map(mul, intersect) reduce(add, union);
```

- [ ] **Step 2: Compute the expected output by hand and create `tests/e2e/merge_intersect_gemm/expected_output.txt`.**

For `m=0, n=0`: pairs (k=0,1,2) → A[0][0]=1, B[0][0]=0 (skip via intersect); A[1][0]=0, B[1][0]=9 (skip); A[2][0]=5, B[2][0]=11 (compute 55). Sum = 55.
For `m=0, n=1`: A[0][0]=1·B[0][1]=8 = 8; A[1][0]=0 (skip); A[2][0]=5·B[2][1]=12=60. Sum = 68.
For `m=1, n=0`: A[0][1]=2·B[0][0]=0 (skip); A[1][1]=4·B[1][0]=9=36; A[2][1]=6·B[2][0]=11=66. Sum = 102.
For `m=1, n=1`: A[0][1]=2·B[0][1]=8=16; A[1][1]=4·B[1][1]=10=40; A[2][1]=6·B[2][1]=12=72. Sum = 128.

So expected:
```
[[55.0, 68.0], [102.0, 128.0]]
```

- [ ] **Step 3: Create `tests/e2e/merge_union_min/input.edge` (single BFS gather step).**

```
dim S = 4;
dim D = 4;

tensor F1[R=D] : float, empty = inf;
tensor G[R=S, C=D] : float, empty = 0.0;
tensor F[R=S]  : float, empty = inf;

init G = [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]];
init F = [0, inf, inf, inf];

F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);
```

- [ ] **Step 4: Compute and create `tests/e2e/merge_union_min/expected_output.txt`.**

Single gather from F=[0,∞,∞,∞]: F1[0]=∞ (no incoming), F1[1]=1 (s=0 contributes 1+0=1), F1[2]=1 (s=0), F1[3]=∞ (no s with both G[s][3]=1 and F[s] non-empty).

Expected (Python prints `math.inf` as `inf`):
```
[inf, 1.0, 1.0, inf]
```

- [ ] **Step 5: Create `tests/e2e/merge_take_left/input.edge`.**

```
dim N = 3;

tensor Z[R=N] : integer, empty = 0;
tensor A[R=N] : integer, empty = 0;
tensor B[R=N] : integer, empty = 0;

init A = [1, 2, 3];
init B = [10, 0, 30];

Z[n] = A[n] . B[n] :: map(mul, take_left) reduce(add);
```

- [ ] **Step 6: Compute and create `tests/e2e/merge_take_left/expected_output.txt`.**

`take_left` admits coordinates where A is non-empty (regardless of B). For each n, if `A[n] != 0`, run `mul(A[n] if a_exists else 0, B[n] if b_exists else 0)`:
- n=0: A[0]=1, B[0]=10 → 1*10=10. → Z[0]=10.
- n=1: A[1]=2, B[1]=0 (empty), b_exists=False, b_op=0 (substituted). → 2*0=0. map_val=0=empty_Z, so map_emitted=False; reduce(passthrough) still runs but with empty val → state=add(state=0, 0)=0. Z[1]=0.
- n=2: A[2]=3, B[2]=30 → 90. → Z[2]=90.

Expected:
```
[10, 0, 90]
```

- [ ] **Step 7: Run the E2E suite.**

```bash
cd build && ctest --output-on-failure -R 'e2e_tests'
```
Expected: all 6 E2E tests pass (3 v1 + 3 new).

If `merge_take_left` fails because the integer formatting is `10.0` instead of `10`, double-check the `format_double_literal` and `emit_alloc` paths for integer-typed tensors. Integer empty values should remain integer-formatted in the generated Python; if the codegen forces `.0`, the comparison fails.

- [ ] **Step 8: Commit.**

```bash
git add tests/e2e/merge_intersect_gemm tests/e2e/merge_union_min tests/e2e/merge_take_left
git commit -m "tests: add single-shot e2e tests for merge operators"
```

---

## Task 8: Negative Tests — Merge Errors

**Files:**
- Create: `tests/negative/bad_merge_op/input.edge`
- Create: `tests/negative/bad_merge_op/expected_error.txt`
- Create: `tests/negative/too_many_args/input.edge`
- Create: `tests/negative/too_many_args/expected_error.txt`
- Create: `tests/negative/merge_no_compute/input.edge`
- Create: `tests/negative/merge_no_compute/expected_error.txt`

`tests/negative/run_negative.py` reads each subdirectory's `input.edge` and `expected_error.txt`, runs `edgecc`, and asserts the compiler exits non-zero with the expected fragment in stderr.

- [ ] **Step 1: Create `tests/negative/bad_merge_op/input.edge`.**

```
dim M = 2;
dim N = 2;

tensor Z[R=M, C=N] : float, empty = 0.0;
tensor A[R=M, C=N] : float, empty = 0.0;
tensor B[R=M, C=N] : float, empty = 0.0;

Z[m,n] = A[m,n] . B[m,n] :: map(mul, frobnicate) reduce(add);
```

- [ ] **Step 2: Create `tests/negative/bad_merge_op/expected_error.txt`.**

```
unknown merge operator 'frobnicate'
```

(Match a fragment of the parser's error message; the runner checks for substring presence.)

- [ ] **Step 3: Create `tests/negative/too_many_args/input.edge`.**

```
dim M = 2;
dim N = 2;

tensor Z[R=M, C=N] : float, empty = 0.0;
tensor A[R=M, C=N] : float, empty = 0.0;
tensor B[R=M, C=N] : float, empty = 0.0;

Z[m,n] = A[m,n] . B[m,n] :: map(mul, intersect, union) reduce(add);
```

- [ ] **Step 4: Create `tests/negative/too_many_args/expected_error.txt`.**

```
parse error
```

(The Bison parser surfaces this as a generic `syntax error` or `parse error`; the runner just needs *any* error indication.)

- [ ] **Step 5: Create `tests/negative/merge_no_compute/input.edge`.**

```
dim M = 2;
dim N = 2;

tensor Z[R=M, C=N] : float, empty = 0.0;
tensor A[R=M, C=N] : float, empty = 0.0;
tensor B[R=M, C=N] : float, empty = 0.0;

Z[m,n] = A[m,n] . B[m,n] :: map(, intersect) reduce(add);
```

- [ ] **Step 6: Create `tests/negative/merge_no_compute/expected_error.txt`.**

```
parse error
```

- [ ] **Step 7: Run the negative test suite.**

```bash
cd build && ctest --output-on-failure -R 'negative_tests'
```
Expected: all 6 negative tests pass (3 v1 + 3 new).

If any test passes silently when it should fail: confirm `edgecc` is exiting with code != 0. If the parser error message text doesn't match, adjust `expected_error.txt` to match the actual fragment (or fix the parser error text in Task 3 to be friendlier).

- [ ] **Step 8: Commit.**

```bash
git add tests/negative/bad_merge_op tests/negative/too_many_args tests/negative/merge_no_compute
git commit -m "tests: add negative tests for merge syntax errors"
```

---

## Task 9: Iterative Test Harness Scaffold

**Files:**
- Create: `tests/e2e/iterative/run_iterative.py` (creating this file also creates the `tests/e2e/iterative/` directory)
- Modify: `CMakeLists.txt`

The runner is self-contained Python and doesn't depend on the kernel format beyond the `__edgecc_meta__` introspection dict. We add it before any iterative tests so we can exercise it interactively in Task 10.

- [ ] **Step 1: Create `tests/e2e/iterative/run_iterative.py`.**

```python
#!/usr/bin/env python3
"""Iterative end-to-end test driver for edgecc.

Reads each subdirectory under tests/e2e/iterative/ that contains a kernel.edge
and a driver.yaml, compiles the kernel, imports the resulting Python module,
drives mod.run(...) in a loop until the watched tensor stabilizes, and
compares the final state against expected values.
"""
import argparse
import importlib.util
import math
import os
import subprocess
import sys

try:
    import yaml
except ImportError:
    print("SKIP: PyYAML not installed; iterative tests skipped.", file=sys.stderr)
    sys.exit(0)


def deep_equal(a, b):
    """Recursive comparison; handles inf == inf as True (Python's default)."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def elementwise(op, a, b):
    """Apply binary op element-wise to nested lists of equal shape."""
    if isinstance(a, list):
        return [elementwise(op, x, y) for x, y in zip(a, b)]
    if op == "min":     return min(a, b)
    if op == "max":     return max(a, b)
    if op == "add":     return a + b
    if op == "or":      return bool(a) or bool(b)
    if op == "and":     return bool(a) and bool(b)
    raise ValueError(f"unknown combine op: {op}")


def apply_feedback(prev, new, combine):
    if combine == "replace":
        return new
    return elementwise(combine, prev, new)


def import_module_from_file(path):
    spec = importlib.util.spec_from_file_location("kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_test(test_dir, edgecc_bin):
    name = os.path.basename(test_dir)
    kernel_edge = os.path.join(test_dir, "kernel.edge")
    driver_yaml = os.path.join(test_dir, "driver.yaml")
    if not (os.path.exists(kernel_edge) and os.path.exists(driver_yaml)):
        return None

    out_py = os.path.join(test_dir, "_kernel_generated.py")
    try:
        result = subprocess.run(
            [edgecc_bin, kernel_edge, "-o", out_py],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"  FAIL {name}: edgecc failed: {result.stderr.strip()}")
            return False

        mod = import_module_from_file(out_py)
        meta = getattr(mod, "__edgecc_meta__", None)
        if not meta:
            print(f"  FAIL {name}: kernel missing __edgecc_meta__")
            return False

        with open(driver_yaml) as f:
            cfg = yaml.safe_load(f)

        inputs = dict(cfg["inputs"])
        feedback_to = cfg["feedback"]["to"]
        feedback_combine = cfg["feedback"].get("combine", "replace")
        watched = cfg["termination"]["on"]
        max_iter = int(cfg["max_iterations"])

        for it in range(max_iter):
            snapshot = inputs[watched]
            output = mod.run(*[inputs[name] for name in meta["inputs"]])
            inputs[feedback_to] = apply_feedback(
                inputs[feedback_to], output, feedback_combine,
            )
            if deep_equal(inputs[watched], snapshot) and it > 0:
                break
        else:
            print(f"  FAIL {name}: did not stabilize within {max_iter} iterations")
            return False

        for tensor_name, expected in cfg["expected"]["final"].items():
            actual = inputs.get(tensor_name)
            if actual is None and tensor_name == meta["output"]:
                actual = output
            if not deep_equal(actual, expected):
                print(f"  FAIL {name}: {tensor_name} mismatch")
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")
                return False

        print(f"  PASS {name}")
        return True
    finally:
        if os.path.exists(out_py):
            os.unlink(out_py)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--tests-dir", required=True)
    args = parser.parse_args()

    edgecc_bin = os.path.join(args.build_dir, "edgecc")
    passed = 0
    failed = 0
    found_any = False
    for entry in sorted(os.listdir(args.tests_dir)):
        d = os.path.join(args.tests_dir, entry)
        if not os.path.isdir(d):
            continue
        result = run_test(d, edgecc_bin)
        if result is None:
            continue
        found_any = True
        if result:
            passed += 1
        else:
            failed += 1

    if not found_any:
        print("SKIP: no iterative tests found.", file=sys.stderr)
        sys.exit(0)

    print(f"\n{passed}/{passed + failed} iterative e2e tests passed")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
```

Make it executable: `chmod +x tests/e2e/iterative/run_iterative.py`.

- [ ] **Step 2: Edit `CMakeLists.txt` — add an `e2e_iterative` test invocation.**

After the existing `add_test` for `negative_tests`, add:

```cmake
add_test(
    NAME e2e_iterative
    COMMAND ${CMAKE_COMMAND} -E env python3
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/iterative/run_iterative.py
        --build-dir ${CMAKE_CURRENT_BINARY_DIR}
        --tests-dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/iterative
)
```

- [ ] **Step 3: Build and run the test suite (no iterative tests yet, runner should report SKIP).**

```bash
cd build && cmake .. && ctest --output-on-failure -R 'e2e_iterative'
```
Expected: `e2e_iterative` test exits 0 with `SKIP: no iterative tests found.` (or `SKIP: PyYAML not installed.` if PyYAML is missing — install it: `pip install pyyaml`).

- [ ] **Step 4: Commit.**

```bash
git add tests/e2e/iterative/run_iterative.py CMakeLists.txt
git commit -m "tests: add iterative e2e harness with driver.yaml schema"
```

---

## Task 10: BFS Iterative Tests

**Files:**
- Create: `tests/e2e/iterative/bfs_simple/kernel.edge`
- Create: `tests/e2e/iterative/bfs_simple/driver.yaml`
- Create: `tests/e2e/iterative/bfs_disconnected/kernel.edge`
- Create: `tests/e2e/iterative/bfs_disconnected/driver.yaml`

- [ ] **Step 1: Create `tests/e2e/iterative/bfs_simple/kernel.edge`.**

```
dim S = 4;
dim D = 4;

tensor F1[R=D] : float, empty = inf;
tensor G[R=S, C=D] : float, empty = 0.0;
tensor F[R=S]  : float, empty = inf;

init G = [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]];
init F = [0, inf, inf, inf];

F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);
```

- [ ] **Step 2: Create `tests/e2e/iterative/bfs_simple/driver.yaml`.**

```yaml
inputs:
  G: [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
  F: [0, .inf, .inf, .inf]

feedback:
  to: F
  combine: min

termination:
  type: stable
  on: F

max_iterations: 10

expected:
  final:
    F: [0, 1, 1, 2]
```

- [ ] **Step 3: Run the iterative suite to confirm `bfs_simple` passes.**

```bash
cd build && ctest --output-on-failure -R 'e2e_iterative'
```
Expected: `PASS bfs_simple`. Trace if it fails: run `python3 tests/e2e/iterative/run_iterative.py --build-dir build --tests-dir tests/e2e/iterative` directly for verbose output. The harness prints which tensor mismatched.

- [ ] **Step 4: Create `tests/e2e/iterative/bfs_disconnected/kernel.edge`.**

```
dim S = 5;
dim D = 5;

tensor F1[R=D] : float, empty = inf;
tensor G[R=S, C=D] : float, empty = 0.0;
tensor F[R=S]  : float, empty = inf;

init G = [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]];
init F = [0, inf, inf, inf, inf];

F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);
```

This graph has two components: {0,1,2} (reachable from 0) and {3,4} (unreachable).

- [ ] **Step 5: Create `tests/e2e/iterative/bfs_disconnected/driver.yaml`.**

```yaml
inputs:
  G: [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
  F: [0, .inf, .inf, .inf, .inf]

feedback:
  to: F
  combine: min

termination:
  type: stable
  on: F

max_iterations: 10

expected:
  final:
    F: [0, 1, 1, .inf, .inf]
```

- [ ] **Step 6: Run the suite again.**

```bash
cd build && ctest --output-on-failure -R 'e2e_iterative'
```
Expected: `PASS bfs_simple` and `PASS bfs_disconnected`.

- [ ] **Step 7: Run the full test suite to make sure nothing else broke.**

```bash
cd build && ctest --output-on-failure
```
Expected: every test passes — unit tests, both v1 e2e, three new merge e2e, three new merge negative, two iterative tests.

- [ ] **Step 8: Commit.**

```bash
git add tests/e2e/iterative/bfs_simple tests/e2e/iterative/bfs_disconnected
git commit -m "tests: add bfs_simple and bfs_disconnected iterative tests"
```

---

## Task 11: CLAUDE.md and Final Cleanup

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Edit `CLAUDE.md` to document the v2 changes.**

Replace the existing "EDGE Syntax (v1)" and "v1 Scope Limitations" sections; update the "Architecture" section's codegen description. New content:

```markdown
## EDGE Syntax (v2)

```
dim M = 4;
tensor Z[R=M, C=N] : float, empty = 0.0;
init A = [[1, 2], [3, 4]];
Z[m,n] = A[k,m] . B[k,n] :: map(mul, intersect) reduce(add, union);
```

Rank variables in expressions bind to declared rank names by position. Built-in compute operators: `add`, `sub`, `mul`, `min`, `max`, `and`, `or`. Built-in merge operators (optional, default `passthrough`): `passthrough`, `nopass`, `intersect`, `take_left`, `take_left_only`, `take_right`, `take_right_only`, `xor`, `union`.

## Generated Python ABI

```python
import math

S = 4
D = 4

__edgecc_meta__ = {"inputs": ["G", "F"], "output": "F1"}

def run(G, F):
    F1 = ...
    for d in range(D):
        ...
    return F1

if __name__ == "__main__":
    G = ...
    F = ...
    F1 = run(G, F)
    print(F1)
```

The `if __name__ == "__main__"` block runs the kernel once with values from `init` declarations and prints the output. Iterative drivers import the module and call `run()` repeatedly with feedback applied between iterations (see `tests/e2e/iterative/run_iterative.py`).

## Iterative Tests

Tests under `tests/e2e/iterative/<name>/` consist of `kernel.edge` plus `driver.yaml`. The driver YAML specifies initial inputs, a feedback rule (`replace` / `min` / `max` / `add` / `or` / `and`), a termination condition (`stable` on a watched tensor), a max-iteration cap, and the expected final state of each tensor. Run via `ctest`; PyYAML must be installed (`pip install pyyaml`) — without it, iterative tests are skipped.

## v2 Scope Limitations

Not yet supported: multi-operation expressions with compute labels, generational/iterative ranks expressed in EDGE source (use the iterative test harness or a Python driver), anonymous tensors / parenthesized sub-expressions, unary functions, the 7 (No,No)→Yes merge operators (NOR, equivalence, NAND, ¬→, ¬←, ⇐, ⇒), file-based init, sparse tensor storage, optimizations of any kind, user-defined compute operators.
```

Also update the `Architecture` section's `src/codegen.h/c` description:

> - `src/codegen.h/c` — Walks IR sequentially, emits Python wrapped in `def run(<inputs>) -> <output>` plus a `__main__` block. The inner-loop body emits the general merge skeleton: existence checks, merge guard, compute, then store-or-empty. No special-case fast path for pass-through merge — all merge configurations go through the same emitter.

- [ ] **Step 2: Verify everything works from a clean state.**

```bash
cd /Users/porumbes/Dropbox/Home/Serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc
rm -rf build
mkdir build && cd build
cmake .. && make && ctest --output-on-failure
```

Expected: full clean build succeeds, all tests pass — every unit test, every E2E (v1 + 3 merge), every negative (v1 + 3 merge), and both iterative tests.

- [ ] **Step 3: Commit.**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for v2 — merge operators, def run ABI, iterative tests"
```

---

## Self-Review Checklist (for the engineer)

After completing all tasks, sanity-check against the spec (`docs/superpowers/specs/2026-05-06-edgecc-v2-merge-design.md`):

- [ ] All 9 merge operators (`passthrough`, `nopass`, `intersect`, `take_left`, `take_left_only`, `take_right`, `take_right_only`, `xor`, `union`) parse, lower to IR, and codegen to the correct Python expression. Spot-check each in `--dump-ir` and in generated Python.
- [ ] v1 source programs produce identical *runtime* output under v2 — same numbers, same shapes (e.g., `tests/e2e/gemm` still prints `[[89.0, 98.0], [116.0, 128.0]]`).
- [ ] `__edgecc_meta__` is present in every generated `.py`, with `inputs` and `output` correctly populated.
- [ ] `def run(...)` signature lists inputs in declaration order; `__main__` block reproduces v1's print behavior.
- [ ] BFS iterative tests stabilize within `max_iterations` and produce the expected final state.
- [ ] No PyYAML-related test failures when PyYAML is installed; clean SKIP when it isn't.
- [ ] The `nopass` warning fires (run `./build/edgecc /tmp/nopass.edge -o /tmp/x.py 2>&1` on a synthesized `nopass`-using program and check stderr).

If any item fails, fix before declaring v2 complete.
