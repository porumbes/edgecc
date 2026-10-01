# Paper-Faithful Generational Tensors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store every generation of a generational tensor (no `_prev`/`_next` ping-pong), return the stop-watched tensor in full from `run()`, and add the paper's stop predicates (`||X[g]|| == 0`, `X[g] == X[h]`, with `stable(X)` as sugar).

**Architecture:** A generational tensor becomes a Python list indexed by generation. Each iteration appends one empty slice per generational tensor it writes, reads/writes index by the counter `_i` (`F[_i]`, `F[_i + 1]`, `T[_i]`), and the stop predicate is evaluated after the whole cascade (compute-then-check; the satisfying generation is kept). IR loads/stores carry a uniform generation kind (0 none, 1 = `i`, 2 = `i+1`) for cross-generational tensors *and* intermediates; the gen loop carries the resolved predicate.

**Tech Stack:** C99, Flex, Bison 2.3-compatible, CMake/CTest; Python 3 for generated code and test harnesses; scipy/numpy for the reference suite.

**Spec:** `docs/superpowers/specs/2026-09-30-edgecc-paper-faithful-generations-design.md`

---

## Background the engineer needs

- **Work only in the worktree** `/Users/porumbes/Dropbox/Home/Serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc/.worktrees/paper-faithful-generations` (branch `feature/paper-faithful-generations`). Never `cd` to the repo root: another session owns that checkout.
- **Pipeline:** `.edge → src/lexer.l → src/parser.y → AST (src/ast.h/.c) → src/semantic.c → src/ir.c → src/codegen.c → .py`.
- **First build in a fresh worktree:** `cmake -S . -B build >/dev/null && cmake --build build -j8`. Afterwards `cmake --build build -j8` (regenerates lexer/parser). Check build output for `warning` and for Bison `conflicts`.
- **Tests:** `ctest --test-dir build --output-on-failure` (all). One unit binary: `./build/test_lexer`, `test_parser`, `test_semantic`, `test_ir`, `test_codegen`. Negative tests: `python3 tests/negative/run_negative.py build/edgecc tests/negative`. E2E: `python3 tests/e2e/run_e2e.py build/edgecc`.
- **Unit test idiom:** `TEST(name) { ... }` plus `RUN_TEST(name);` in that file's `main()`. Parser tests use `yy_scan_string` + `yyparse()` + global `parse_result`. Semantic/IR/codegen tests use the file-local `parse(src)` helper, then `semantic_analyze`, `ir_lower`, `codegen_python` / `codegen_python_ex(ir, emit_trace, kernel_name)`.
- **Negative test:** `tests/negative/<name>/input.edge` + `expected_errors.txt` (each non-empty line must appear as a substring of stderr; exit code must be non-zero).
- **E2E test:** `tests/e2e/<name>/input.edge` + `expected_output.txt` (stdout compared after `.strip()`); optional `needs_scipy` (empty file), `roundtrip_check` (space-separated floats that `result.mtx`, saved in the test dir, must flatten to).
- **Today's generational codegen** (what we replace): `F_prev = F`, `while True:`, `F_next = [0.0] * N`, reads `F_prev[s]` / writes `F_next[d]`, `if F_next == F_prev: break`, `F_prev = F_next`, `return F_prev`. Loads carry `gen_read` (0/1/2) only for *cross-generational* tensors; intermediates (`T[i, d]`) are plain names reallocated each generation.
- **Commit convention:** do NOT add `Co-Authored-By` trailers.
- Python on this machine is 3.8: no `dict | dict`, no `match`.

---

## Task 1: Lexer — `||` and `==` tokens

**Files:**
- Modify: `src/lexer.l` (after the `"::"` rule)
- Modify: `src/parser.y` (token declarations, near `%token TOK_GEN TOK_STOP TOK_STABLE`)
- Test: `tests/unit/test_lexer.c`

- [x] **Step 1: Declare the tokens and write failing lexer tests**

In `src/parser.y`, change `%token TOK_GEN TOK_STOP TOK_STABLE` to:
```
%token TOK_GEN TOK_STOP TOK_STABLE TOK_OROR TOK_EQEQ
```
In `tests/unit/test_lexer.c`, add after `test_stable_keyword`:
```c
TEST(test_oror_token) {
    YY_BUFFER_STATE buf = yy_scan_string("||");
    ASSERT_INT_EQ(TOK_OROR, yylex());
    yy_delete_buffer(buf);
}

TEST(test_eqeq_token_vs_single_equals) {
    YY_BUFFER_STATE buf = yy_scan_string("== =");
    ASSERT_INT_EQ(TOK_EQEQ, yylex());
    ASSERT_INT_EQ('=', yylex());
    yy_delete_buffer(buf);
}
```
Register both with `RUN_TEST(...)` in `main()`.

- [x] **Step 2: Run to verify failure**

Run: `cmake -S . -B build >/dev/null && cmake --build build -j8 2>&1 | grep -E 'error|warning'; ./build/test_lexer | tail -3`
Expected: `test_oror_token` and `test_eqeq_token_vs_single_equals` FAIL (the lexer returns `'|'`-less garbage / two `'='`).

- [x] **Step 3: Add the lexer rules**

In `src/lexer.l`, directly after the `"::"` rule:
```
"||"                { return TOK_OROR; }
"=="                { return TOK_EQEQ; }
```

- [x] **Step 4: Verify pass**

Run: `cmake --build build -j8 2>&1 | grep -E 'error|warning'; ./build/test_lexer | tail -1`
Expected: all lexer tests pass, no warnings.

- [x] **Step 5: Commit**
```bash
git add src/lexer.l src/parser.y tests/unit/test_lexer.c
git commit -m "lexer: add || and == tokens for stop predicates"
```

---

## Task 2: AST + parser — stop predicate forms

**Files:**
- Modify: `src/ast.h` (`StopData`, constructor prototypes), `src/ast.c` (`ast_stop_new` → three constructors, `ast_free`)
- Modify: `src/parser.y` (`stop_stmt`)
- Modify: `src/semantic.c` (`analyze_stop` — minimal adaptation so the build stays green; fully replaced in Task 3)
- Test: `tests/unit/test_parser.c`; new negative test `tests/negative/stop_unknown_predicate/`

- [x] **Step 1: Write the failing parser tests**

In `tests/unit/test_parser.c`, replace the body assertions of `test_parse_stop_stmt`:
```c
TEST(test_parse_stop_stmt) {
    YY_BUFFER_STATE buf = yy_scan_string("stop: stable(F);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *s = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_STOP, s->type);
    ASSERT_INT_EQ(STOP_STABLE, s->data.stop.kind);
    ASSERT_STR_EQ("F", s->data.stop.stable_tensor);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}
```
and add:
```c
TEST(test_parse_stop_occupancy) {
    YY_BUFFER_STATE buf = yy_scan_string("stop: ||F[i+1]|| == 0;");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *s = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_STOP, s->type);
    ASSERT_INT_EQ(STOP_OCCUPANCY, s->data.stop.kind);
    ASSERT_STR_EQ("F", s->data.stop.lhs.tensor_name);
    ASSERT_INT_EQ(1, s->data.stop.lhs.num_vars);
    ASSERT_STR_EQ("i", s->data.stop.lhs.rank_vars[0]);
    ASSERT_INT_EQ(1, s->data.stop.lhs.gen_offset);
    ASSERT_INT_EQ(0, s->data.stop.occupancy_rhs);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_stop_occupancy_nonzero_parses) {
    /* Parses; rejected later by semantic analysis. */
    YY_BUFFER_STATE buf = yy_scan_string("stop: ||F[i+1]|| == 3;");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *s = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(3, s->data.stop.occupancy_rhs);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_stop_equal) {
    YY_BUFFER_STATE buf = yy_scan_string("stop: P[i+1] == P[i];");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *s = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(STOP_EQUAL, s->data.stop.kind);
    ASSERT_STR_EQ("P", s->data.stop.lhs.tensor_name);
    ASSERT_INT_EQ(1, s->data.stop.lhs.gen_offset);
    ASSERT_STR_EQ("P", s->data.stop.rhs.tensor_name);
    ASSERT_STR_EQ("i", s->data.stop.rhs.rank_vars[0]);
    ASSERT_INT_EQ(0, s->data.stop.rhs.gen_offset);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_stop_unknown_predicate_fails) {
    YY_BUFFER_STATE buf = yy_scan_string("stop: settled(P);");
    ASSERT_TRUE(yyparse() != 0);
    if (parse_result) { ast_free(parse_result); parse_result = NULL; }
    yy_delete_buffer(buf);
}
```
Register the four new tests with `RUN_TEST`.

Create the negative test:
```bash
mkdir -p tests/negative/stop_unknown_predicate
cat > tests/negative/stop_unknown_predicate/input.edge <<'EOF'
// Only stable(X) is a named stop predicate.
dim N = 3;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
tensor P[gen I, R=N] : float, empty = 0.0;
init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];
init F = [1.0, 0.0, 0.0];
init P = [1.0, 0.0, 0.0];
F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
stop: settled(P);
EOF
echo "unknown stop predicate 'settled'" > tests/negative/stop_unknown_predicate/expected_errors.txt
```

- [x] **Step 2: Run to verify failure**

Run: `cmake --build build -j8 2>&1 | grep -E 'error' | head`
Expected: compile errors in `test_parser.c` (`STOP_STABLE`, `stable_tensor`, `lhs` undefined). That is the failing state.

- [x] **Step 3: Redefine `StopData` and constructors**

In `src/ast.h`, replace
```c
typedef struct {
    char *watch_tensor;
} StopData;
```
with
```c
typedef enum {
    STOP_STABLE,      /* stop: stable(X);      sugar for X[i+1] == X[i], resolved in semantic */
    STOP_EQUAL,       /* stop: X[g] == X[h];                                             */
    STOP_OCCUPANCY    /* stop: ||X[g]|| == N;  N must be 0 (checked in semantic)         */
} StopKind;

typedef struct {
    StopKind  kind;
    char     *stable_tensor;  /* STOP_STABLE: the X in stable(X); else NULL */
    TensorRef lhs;            /* STOP_EQUAL / STOP_OCCUPANCY; zeroed otherwise */
    TensorRef rhs;            /* STOP_EQUAL only; zeroed otherwise */
    int       occupancy_rhs;  /* STOP_OCCUPANCY: the integer after '==' */
} StopData;
```
Replace the prototype `AstNode *ast_stop_new(char *watch_tensor);` with:
```c
AstNode *ast_stop_stable_new(char *tensor);
AstNode *ast_stop_equal_new(TensorRef lhs, TensorRef rhs);
AstNode *ast_stop_occupancy_new(TensorRef ref, int rhs_value);
```
In `src/ast.c`, replace `ast_stop_new` with:
```c
AstNode *ast_stop_stable_new(char *tensor) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_STOP;
    node->data.stop.kind = STOP_STABLE;
    node->data.stop.stable_tensor = tensor;
    return node;
}

AstNode *ast_stop_equal_new(TensorRef lhs, TensorRef rhs) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_STOP;
    node->data.stop.kind = STOP_EQUAL;
    node->data.stop.lhs = lhs;
    node->data.stop.rhs = rhs;
    return node;
}

AstNode *ast_stop_occupancy_new(TensorRef ref, int rhs_value) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_STOP;
    node->data.stop.kind = STOP_OCCUPANCY;
    node->data.stop.lhs = ref;
    node->data.stop.occupancy_rhs = rhs_value;
    return node;
}
```
In `ast_free`, replace the `NODE_STOP` case body with (freeing a zeroed `TensorRef` is a no-op):
```c
        case NODE_STOP:
            free(node->data.stop.stable_tensor);
            tensor_ref_free_contents(&node->data.stop.lhs);
            tensor_ref_free_contents(&node->data.stop.rhs);
            break;
```

- [x] **Step 4: Grammar**

In `src/parser.y`, replace the whole `stop_stmt:` rule with:
```
stop_stmt:
    TOK_STOP ':' TOK_STABLE '(' TOK_IDENT ')' ';'
    {
        $$ = (void *)ast_stop_stable_new($5);
    }
    | TOK_STOP ':' TOK_IDENT '(' TOK_IDENT ')' ';'
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "unknown stop predicate '%s' (expected 'stable')", $3);
        yyerror(buf);
        free($3); free($5);
        YYERROR;
    }
    | TOK_STOP ':' TOK_OROR tensor_ref TOK_OROR TOK_EQEQ TOK_INT_LIT ';'
    {
        TensorRef *r = (TensorRef *)$4;
        $$ = (void *)ast_stop_occupancy_new(*r, $7);
        free(r);
    }
    | TOK_STOP ':' tensor_ref TOK_EQEQ tensor_ref ';'
    {
        TensorRef *a = (TensorRef *)$3;
        TensorRef *b = (TensorRef *)$5;
        $$ = (void *)ast_stop_equal_new(*a, *b);
        free(a); free(b);
    }
    ;
```

- [x] **Step 5: Keep semantic compiling (temporary; replaced in Task 3)**

In `src/semantic.c`, in `analyze_stop`, replace `const char *watch = node->data.stop.watch_tensor;` with:
```c
    StopData *sd = &node->data.stop;
    const char *watch = sd->kind == STOP_STABLE ? sd->stable_tensor : sd->lhs.tensor_name;
```

- [x] **Step 6: Verify**

Run: `cmake --build build -j8 2>&1 | grep -iE 'error|warning|conflict'; ./build/test_parser | tail -1; ctest --test-dir build 2>&1 | tail -3; python3 tests/negative/run_negative.py build/edgecc tests/negative | grep -E 'stop_unknown|passed'`
Expected: no warnings or Bison conflicts; parser tests all pass; full ctest passes; `PASS stop_unknown_predicate`.

- [x] **Step 7: Commit**
```bash
git add src/ast.h src/ast.c src/parser.y src/semantic.c tests/unit/test_parser.c tests/negative/stop_unknown_predicate
git commit -m "parser: stop predicates ||X[g]|| == 0 and X[g] == X[h]; stable(X) kept"
```

---

## Task 3: Semantic — predicate resolution, gen-first rule, save rule

**Files:**
- Modify: `src/semantic.h` (`SymbolTable` stop fields)
- Modify: `src/semantic.c` (`analyze_tensor_decl`, remove `analyze_stop`, new `resolve_stop`, save pass, post-pass cleanup)
- Test: `tests/unit/test_semantic.c`; new negative tests (listed below); reconcile `tests/negative/gen_missing_stop/expected_errors.txt`

- [x] **Step 1: Write the failing semantic unit tests**

Add to `tests/unit/test_semantic.c` (before `main`), and `RUN_TEST` each:
```c
#define BFS_DECLS \
    "dim N = 3;\n" \
    "tensor G[R=N, C=N]   : float, empty = 0.0;\n" \
    "tensor F[gen I, R=N] : float, empty = 0.0;\n" \
    "tensor P[gen I, R=N] : float, empty = 0.0;\n" \
    "init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n" \
    "init F = [1.0, 0.0, 0.0];\n" \
    "init P = [1.0, 0.0, 0.0];\n" \
    "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n" \
    "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"

TEST(test_semantic_stop_occupancy) {
    AstNode *ast = parse(BFS_DECLS "stop: ||F[i+1]|| == 0;\n");
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_STR_EQ("F", res.symtab->stop_watch);
    ASSERT_INT_EQ(STOP_OCCUPANCY, res.symtab->stop_kind);
    ASSERT_INT_EQ(1, res.symtab->stop_gen_a);
    semantic_result_free(&res); ast_free(ast);
}

TEST(test_semantic_stop_equal) {
    AstNode *ast = parse(BFS_DECLS "stop: P[i+1] == P[i];\n");
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_STR_EQ("P", res.symtab->stop_watch);
    ASSERT_INT_EQ(STOP_EQUAL, res.symtab->stop_kind);
    ASSERT_INT_EQ(1, res.symtab->stop_gen_a);
    ASSERT_INT_EQ(0, res.symtab->stop_gen_b);
    semantic_result_free(&res); ast_free(ast);
}

TEST(test_semantic_stable_is_equal_sugar) {
    AstNode *ast = parse(BFS_DECLS "stop: stable(P);\n");
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_STR_EQ("P", res.symtab->stop_watch);
    ASSERT_INT_EQ(STOP_EQUAL, res.symtab->stop_kind);
    ASSERT_INT_EQ(1, res.symtab->stop_gen_a);
    ASSERT_INT_EQ(0, res.symtab->stop_gen_b);
    semantic_result_free(&res); ast_free(ast);
}

TEST(test_semantic_stop_on_intermediate_at_i) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor W[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init W = [[0.0,1.0,4.0],[0.0,0.0,2.0],[0.0,0.0,0.0]];\n"
        "init D = [0.0, inf, inf];\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: ||T[i]|| == 0;\n");
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_STR_EQ("T", res.symtab->stop_watch);
    ASSERT_INT_EQ(0, res.symtab->stop_gen_a);
    semantic_result_free(&res); ast_free(ast);
}
```

- [x] **Step 2: Write the failing negative tests**

Create them with this script (run from the worktree root). `BFS` and `BF` are the base programs; each test substitutes one line.
```bash
mk() {  # mk <name> <base: bfs|bf> <stop line> <expected substring> [sed expr on body]
  d=tests/negative/$1; mkdir -p "$d"
  if [ "$2" = bfs ]; then
    body='dim N = 3;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
tensor P[gen I, R=N] : float, empty = 0.0;
init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];
init F = [1.0, 0.0, 0.0];
init P = [1.0, 0.0, 0.0];
F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);'
  else
    body='dim N = 3;
tensor W[R=N, C=N]   : float, empty = 0.0;
tensor T[gen I, R=N] : float, empty = inf;
tensor D[gen I, R=N] : float, empty = inf;
init W = [[0.0,1.0,4.0],[0.0,0.0,2.0],[0.0,0.0,0.0]];
init D = [0.0, inf, inf];
T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);'
  fi
  if [ -n "$5" ]; then body=$(printf '%s\n' "$body" | sed "$5"); fi
  printf '%s\n%s\n' "$body" "$3" > "$d/input.edge"
  printf '%s\n' "$4" > "$d/expected_errors.txt"
}
mk gen_rank_not_first     bfs 'stop: stable(P);'          "generational rank 'I' must be the first rank of 'F'" 's/tensor F\[gen I, R=N\]/tensor F[R=N, gen I]/'
mk stop_occupancy_not_gen bfs 'stop: ||G[i]|| == 0;'      "stop predicate: tensor 'G' is not generational"
mk stop_undeclared        bfs 'stop: ||Q[i+1]|| == 0;'    "stop predicate: undeclared tensor 'Q'"
mk stop_bad_offset        bfs 'stop: ||F[i+2]|| == 0;'    "stop predicate: generation must be 'i' or 'i+1'"
mk stop_intermediate_next bf  'stop: ||T[i+1]|| == 0;'    "intermediate 'T' has only generation 'i' when the stop predicate is checked"
mk stop_stable_intermediate bf 'stop: stable(T);'         "intermediate 'T' has only generation 'i' when the stop predicate is checked"
mk stop_equal_two_tensors bfs 'stop: F[i+1] == P[i];'     "stop predicate: '==' compares two generations of the same tensor"
mk stop_equal_same_gen    bfs 'stop: P[i] == P[i];'       "stop predicate compares a generation with itself"
mk stop_occupancy_nonzero bfs 'stop: ||F[i+1]|| == 3;'    "stop predicate: only '||X[g]|| == 0' is supported"
mk stop_wrong_gen_var     bfs 'stop: ||F[j+1]|| == 0;'    "stop predicate uses 'j', but the generational rank variable is 'i'"
mk stop_not_whole_gen     bfs 'stop: ||F[i+1, d]|| == 0;' "stop predicate: 'F' must name a whole generation, e.g. F[i+1]"
mkdir -p tests/negative/save_generational_2d
cat > tests/negative/save_generational_2d/input.edge <<'EOF'
// Every generation of a 2-D generational tensor is 3-D: not storable as .mtx.
dim N = 3;
tensor G[R=N, C=N]        : float, empty = 0.0;
tensor T[gen I, R=N, C=N] : float, empty = 0.0;
init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];
init T = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];
T[i+1, a, c] = G[a,b] . T[i,b,c] :: map(and, intersect) reduce(or, union);
stop: stable(T);
save: T -> "result.mtx";
EOF
echo "save: generational tensor 'T' has more than one data rank" > tests/negative/save_generational_2d/expected_errors.txt
```
Reconcile the existing fixture whose message changes:
```bash
echo "requires a 'stop:' clause" > tests/negative/gen_missing_stop/expected_errors.txt
```

- [x] **Step 3: Run to verify failure**

Run: `cmake --build build -j8 2>&1 | grep -E 'error' | head; python3 tests/negative/run_negative.py build/edgecc tests/negative | grep -E 'FAIL|passed'`
Expected: `test_semantic.c` fails to compile (`stop_kind`, `STOP_OCCUPANCY` members missing) and the 13 new/reconciled negative tests FAIL.

- [x] **Step 4: Symbol-table fields**

In `src/semantic.h`, replace the `stop_watch` field line with:
```c
    char *stop_watch;    /* tensor named by the stop predicate (borrowed) */
    int   stop_kind;     /* STOP_EQUAL or STOP_OCCUPANCY (stable(X) desugars to STOP_EQUAL) */
    int   stop_gen_a;    /* generation offset of the (left) predicate reference: 0 = i, 1 = i+1 */
    int   stop_gen_b;    /* STOP_EQUAL only: offset of the right reference */
```

- [x] **Step 5: Gen-first rule**

In `src/semantic.c` `analyze_tensor_decl`, inside `if (d->ranks[i].is_generational) {`, directly after the existing "more than one generational rank" check, add:
```c
            if (i != 0)
                add_error(res, "generational rank '%s' must be the first rank of '%s'",
                          d->ranks[i].rank_name, d->name);
```

- [x] **Step 6: Replace `analyze_stop` with post-pass `resolve_stop`**

Delete the whole `analyze_stop` function. In `semantic_analyze`'s statement switch, change the `NODE_STOP` case to:
```c
            case NODE_STOP:
                /* Resolved after all Einsums (needs the generational rank variable). */
                break;
```
Add these functions above `semantic_analyze` (after `validate_cascade_ordering`):
```c
static int name_in(char *const *list, int n, const char *name) {
    for (int i = 0; i < n; i++)
        if (strcmp(list[i], name) == 0) return 1;
    return 0;
}

/* Resolve one predicate reference X[g]. X must be a declared generational
   tensor, the reference must name a whole generation using the program's
   generational rank variable, and g must be i or i+1 (only i for an
   intermediate, which has no generation i+1 when the predicate runs).
   Returns the offset (0 or 1), or -1 after reporting an error. */
static int resolve_stop_ref(SemanticResult *res, const TensorRef *r, TensorEntry **te_out) {
    SymbolTable *st = res->symtab;
    TensorEntry *te = symtab_find_tensor(st, r->tensor_name);
    *te_out = te;
    if (!te) {
        add_error(res, "stop predicate: undeclared tensor '%s'", r->tensor_name);
        return -1;
    }
    if (te->gen_axis < 0) {
        add_error(res, "stop predicate: tensor '%s' is not generational", r->tensor_name);
        return -1;
    }
    if (r->num_vars != 1) {
        add_error(res, "stop predicate: '%s' must name a whole generation, e.g. %s[%s+1]",
                  r->tensor_name, r->tensor_name, st->gen_rank_var ? st->gen_rank_var : "i");
        return -1;
    }
    const char *v = r->rank_vars[0];
    if (st->gen_rank_var && strcmp(v, st->gen_rank_var) != 0) {
        add_error(res, "stop predicate uses '%s', but the generational rank variable is '%s'",
                  v, st->gen_rank_var);
        return -1;
    }
    int off = r->gen_offset;
    if (off != 0 && off != 1) {
        add_error(res, "stop predicate: generation must be '%s' or '%s+1'", v, v);
        return -1;
    }
    if (off == 1 && name_in(st->intermediates, st->num_intermediates, te->name)) {
        add_error(res, "intermediate '%s' has only generation '%s' when the stop predicate is checked",
                  te->name, v);
        return -1;
    }
    return off;
}

/* Find the stop clause (the last one, as before), check it against the
   program's iterativeness, and resolve it into symtab->stop_*. */
static void resolve_stop(SemanticResult *res, AstNode *program) {
    SymbolTable *st = res->symtab;
    ProgramData *pd = &program->data.program;
    AstNode *stop = NULL;
    for (int i = 0; i < pd->count; i++)
        if (pd->statements[i]->type == NODE_STOP) stop = pd->statements[i];

    if (!stop) {
        if (st->is_iterative)
            add_error(res, "generational einsum on '%s' requires a 'stop:' clause",
                      st->gen_tensor ? st->gen_tensor : "?");
        return;
    }
    if (!st->is_iterative) {
        add_error(res, "'stop' clause present but no generational einsum");
        return;
    }

    StopData *sd = &stop->data.stop;
    TensorEntry *ta = NULL, *tb = NULL;
    switch (sd->kind) {
        case STOP_STABLE: {
            /* stable(X) is sugar for X[i+1] == X[i]. */
            ta = symtab_find_tensor(st, sd->stable_tensor);
            if (!ta) {
                add_error(res, "stop predicate: undeclared tensor '%s'", sd->stable_tensor);
                return;
            }
            if (ta->gen_axis < 0) {
                add_error(res, "stop predicate: tensor '%s' is not generational", sd->stable_tensor);
                return;
            }
            if (name_in(st->intermediates, st->num_intermediates, ta->name)) {
                add_error(res, "intermediate '%s' has only generation '%s' when the stop predicate is checked",
                          ta->name, st->gen_rank_var ? st->gen_rank_var : "i");
                return;
            }
            st->stop_kind = STOP_EQUAL;
            st->stop_gen_a = 1;
            st->stop_gen_b = 0;
            break;
        }
        case STOP_OCCUPANCY: {
            int a = resolve_stop_ref(res, &sd->lhs, &ta);
            if (sd->occupancy_rhs != 0) {
                add_error(res, "stop predicate: only '||X[g]|| == 0' is supported");
                return;
            }
            if (a < 0) return;
            st->stop_kind = STOP_OCCUPANCY;
            st->stop_gen_a = a;
            break;
        }
        case STOP_EQUAL: {
            int a = resolve_stop_ref(res, &sd->lhs, &ta);
            int b = resolve_stop_ref(res, &sd->rhs, &tb);
            if (a < 0 || b < 0) return;
            if (ta != tb) {
                add_error(res, "stop predicate: '==' compares two generations of the same tensor "
                               "('%s' vs '%s')", ta->name, tb->name);
                return;
            }
            if (a == b) {
                add_error(res, "stop predicate compares a generation with itself");
                return;
            }
            st->stop_kind = STOP_EQUAL;
            st->stop_gen_a = a;
            st->stop_gen_b = b;
            break;
        }
    }
    st->stop_watch = ta->name;
}
```

- [x] **Step 7: Post-pass cleanup**

In `semantic_analyze`, directly after `validate_cascade_ordering(&res, program);`, add `resolve_stop(&res, program);`. Then delete these now-superseded blocks that follow it:
- `if (st->is_iterative && st->stop_watch == NULL) { ... requires a 'stop: stable(...)' clause ... }`
- `if (st->stop_watch != NULL && !st->is_iterative) { ... 'stop' clause present ... }`
- `if (st->is_iterative && st->stop_watch != NULL && st->num_gen_tensors > 0) { ... stop watches '%s' but it is not a cross-generational tensor ... }`

Keep `SymbolTable *st = res.symtab;` (later code uses it) and the "no cross-generational tensor" check.

- [x] **Step 8: Save rule**

In the save pass, directly after the `if (!is_output) { ... continue; }` block, add:
```c
        TensorEntry *ste = symtab_find_tensor(st, tgt);
        if (ste->gen_axis >= 0 && ste->num_ranks - 1 > 1) {
            add_error(&res, "save: generational tensor '%s' has more than one data rank "
                            "(with every generation it would be 3-D; .mtx stores at most 2-D)", tgt);
            continue;
        }
```

- [x] **Step 9: Verify**

Run: `cmake --build build -j8 2>&1 | grep -E 'error|warning'; ./build/test_semantic | tail -1; python3 tests/negative/run_negative.py build/edgecc tests/negative | grep -E 'FAIL|passed'; ctest --test-dir build 2>&1 | tail -3`
Expected: all semantic tests pass; all negative tests pass (including the 13 new/reconciled); full ctest passes (codegen still uses `stop_watch` and the old loop, so e2e output is unchanged at this point).
Also run `grep -rn "not a cross-generational tensor" tests/` — expected: no hits (no fixture depended on the removed message).

- [x] **Step 10: Commit**
```bash
git add src/semantic.h src/semantic.c tests/unit/test_semantic.c tests/negative
git commit -m "semantic: resolve stop predicates; gen rank must be first; reject 3-D generational save"
```

---

## Task 4: IR — uniform generation kinds and the predicate on the gen loop

**Files:**
- Modify: `src/ir.h` (field comments; `IRGenLoop` stop fields)
- Modify: `src/ir.c` (`gen_read_kind` → `gen_kind`; stores; fused; `ir_lower`; `ir_dump`)
- Test: `tests/unit/test_ir.c`

- [x] **Step 1: Write failing IR tests / update the ones that encode the old encoding**

In `tests/unit/test_ir.c`:
- In the Bellman-Ford cascade test (the one asserting `ASSERT_INT_EQ(1, cross_writes);`), change the counting to distinguish the two kinds:
```c
            case IR_STORE:
                stores++;
                if (ir->instrs[i].data.store.gen_write == 2) cross_writes++;
                if (ir->instrs[i].data.store.gen_write == 1) inter_writes++;
                break;
```
declare `int inter_writes = 0;` with the other counters, and add `ASSERT_INT_EQ(1, inter_writes);` next to `ASSERT_INT_EQ(1, cross_writes);`.
- In `test_ir_two_cross_gen` (the `stores` counter), change `in->data.store.gen_write == 1` to `in->data.store.gen_write == 2`.
- Add:
```c
TEST(test_ir_intermediate_generation_kinds) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor W[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init W = [[0.0,1.0,4.0],[0.0,0.0,2.0],[0.0,0.0,0.0]];\n"
        "init D = [0.0, inf, inf];\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: stable(D);\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    int t_read_i = 0, t_write_i = 0, d_write_next = 0, saw_loop = 0;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *in = &ir->instrs[i];
        if (in->type == IR_LOAD && strcmp(in->data.load.tensor_name, "T") == 0 &&
            in->data.load.gen_read == 1) t_read_i = 1;
        if (in->type == IR_STORE && strcmp(in->data.store.tensor_name, "T") == 0 &&
            in->data.store.gen_write == 1) t_write_i = 1;
        if (in->type == IR_STORE && strcmp(in->data.store.tensor_name, "D") == 0 &&
            in->data.store.gen_write == 2) d_write_next = 1;
        if (in->type == IR_GEN_LOOP_BEGIN) {
            saw_loop = 1;
            ASSERT_STR_EQ("D", in->data.gen_loop.watch);
            ASSERT_INT_EQ(STOP_EQUAL, in->data.gen_loop.stop_kind);
            ASSERT_INT_EQ(1, in->data.gen_loop.stop_gen_a);
            ASSERT_INT_EQ(0, in->data.gen_loop.stop_gen_b);
        }
    }
    ASSERT_INT_EQ(1, t_read_i);
    ASSERT_INT_EQ(1, t_write_i);
    ASSERT_INT_EQ(1, d_write_next);
    ASSERT_INT_EQ(1, saw_loop);
    ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

TEST(test_ir_occupancy_stop) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor F[gen I, R=N] : float, empty = inf;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n"
        "init F = [0.0, inf, inf];\n"
        "init P = [1.0, 0.0, 0.0];\n"
        "T[i, d]   = G[s,d] . F[i,s]   :: map(add, intersect) reduce(min, union);\n"
        "F[i+1, d] = T[i,d] . !P[i,d]  :: map(take_left, intersect) reduce(min, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: ||F[i+1]|| == 0;\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    int saw = 0;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *in = &ir->instrs[i];
        if (in->type != IR_GEN_LOOP_BEGIN) continue;
        saw = 1;
        ASSERT_STR_EQ("F", in->data.gen_loop.watch);
        ASSERT_INT_EQ(STOP_OCCUPANCY, in->data.gen_loop.stop_kind);
        ASSERT_INT_EQ(1, in->data.gen_loop.stop_gen_a);
        ASSERT_TRUE(isinf(in->data.gen_loop.watch_empty));
    }
    ASSERT_INT_EQ(1, saw);
    ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}
```
Add `#include <math.h>` at the top of `test_ir.c` if not present, and `RUN_TEST` both new tests.

- [x] **Step 2: Run to verify failure**

Run: `cmake --build build -j8 2>&1 | grep -E 'error' | head`
Expected: compile errors (`stop_kind`, `stop_gen_a`, `watch_empty` not members of `IRGenLoop`).

- [x] **Step 3: IR header**

In `src/ir.h`:
- `IRLoad.gen_read` comment → `/* 0: not generational; 1: generation i (X[_i]); 2: generation i+1 (X[_i + 1]) */`
- `IRStore.gen_write` comment → `/* same encoding as IRLoad.gen_read: 1 = intermediate (i), 2 = cross-gen (i+1) */`
- In `IRFused`, add a comment after `int gen_write;` in the out line: change the line to
  `char *out_name; char **out_coords; int out_ncoords; double empty_z; int gen_write; /* gen kind, as IRStore */`
- Replace the `IRGenLoop` struct with:
```c
typedef struct {
    IRGenAlloc cross_gens[8];         /* cross-generational tensors (written at i+1) */
    int num_cross_gens;
    char *watch;                      /* tensor named by the stop predicate (owned) */
    int stop_kind;                    /* STOP_EQUAL or STOP_OCCUPANCY */
    int stop_gen_a;                   /* offset of the (left) predicate reference: 0 = i, 1 = i+1 */
    int stop_gen_b;                   /* STOP_EQUAL only */
    double watch_empty;               /* empty value of the watched tensor (occupancy) */
    IRGenAlloc intermediates[8];      /* intra-generation intermediates (written at i) */
    int num_intermediates;
    int max_gen;                      /* safety iteration cap */
} IRGenLoop;
```

- [x] **Step 4: Lowering**

In `src/ir.c`, replace `gen_read_kind` with:
```c
static int is_intermediate_name(SymbolTable *st, const char *name) {
    for (int m = 0; m < st->num_intermediates; m++)
        if (strcmp(st->intermediates[m], name) == 0) return 1;
    return 0;
}

/* Generation kind of a reference: 0 not generational; 1 generation i;
   2 generation i+1. Applies to cross-generational tensors and intermediates
   alike -- every generational tensor stores all of its generations. */
static int gen_kind(SymbolTable *st, TensorRef *ref, TensorEntry *te) {
    if (!te || te->gen_axis < 0) return 0;
    if (!is_cross_gen_name(st, te->name) && !is_intermediate_name(st, te->name)) return 0;
    int eff = (ref->gen_axis == te->gen_axis) ? ref->gen_offset : 0;
    return (eff == 1) ? 2 : 1;
}
```
Then replace every call `gen_read_kind(` with `gen_kind(` (three sites: `fill_fused_operand`, and the two LOADs in `emit_einsum`), and replace the two writes:
- in `emit_einsum_fused`: `fu->gen_write = is_cross_gen_name(st, f->output.tensor_name);` → `fu->gen_write = gen_kind(st, &f->output, out_te);`
- in `emit_einsum` STORE: `instr->data.store.gen_write = is_cross_gen_name(symtab, e->output.tensor_name);` → `instr->data.store.gen_write = gen_kind(symtab, &e->output, out_te);`

In `ir_lower`, after `gl->data.gen_loop.watch = ...;` add:
```c
        gl->data.gen_loop.stop_kind  = symtab->stop_kind;
        gl->data.gen_loop.stop_gen_a = symtab->stop_gen_a;
        gl->data.gen_loop.stop_gen_b = symtab->stop_gen_b;
        TensorEntry *wte = symtab->stop_watch ? symtab_find_tensor(symtab, symtab->stop_watch) : NULL;
        gl->data.gen_loop.watch_empty = wte ? wte->empty_val : 0.0;
```

- [x] **Step 5: Dump**

In `ir_dump`: LOAD flag `instr->data.load.gen_read ? " [gen_read]" : ""` →
`instr->data.load.gen_read == 2 ? " [gen i+1]" : instr->data.load.gen_read == 1 ? " [gen i]" : ""`;
STORE flag likewise with `gen_write`. In `IR_GEN_LOOP_BEGIN`, replace the final `fprintf(out, "], max_gen=%d)\n", ...)` with:
```c
                {
                    IRGenLoop *g = &instr->data.gen_loop;
                    const char *ga = g->stop_gen_a ? "i+1" : "i";
                    if (g->stop_kind == STOP_OCCUPANCY)
                        fprintf(out, "], stop=||%s[%s]||==0, max_gen=%d)\n",
                                g->watch ? g->watch : "?", ga, g->max_gen);
                    else
                        fprintf(out, "], stop=%s[%s]==%s[%s], max_gen=%d)\n",
                                g->watch ? g->watch : "?", ga,
                                g->watch ? g->watch : "?", g->stop_gen_b ? "i+1" : "i",
                                g->max_gen);
                }
```

- [x] **Step 6: Verify**

Run: `cmake --build build -j8 2>&1 | grep -E 'error|warning'; ./build/test_ir | tail -1; ./build/edgecc examples/bellman_ford.edge -o /dev/null --dump-ir | grep -E 'GEN_LOOP|gen i'`
Expected: all IR tests pass; the dump shows `stop=D[i+1]==D[i]` and `[gen i]` / `[gen i+1]` markers.
Note: e2e tests for generational programs that use intermediates now FAIL (codegen still emits `_prev`/`_next` but intermediates now carry kind 1 and the old codegen maps kind 1 to `_prev`). That is expected; Task 5 fixes codegen. Do not commit a red suite for long: go straight to Task 5 after committing.

- [x] **Step 7: Commit**
```bash
git add src/ir.h src/ir.c tests/unit/test_ir.c
git commit -m "ir: uniform generation kinds for all generational tensors; stop predicate on the gen loop"
```

---

## Task 5: Codegen — store every generation

**Files:**
- Modify: `src/codegen.c`
- Test: `tests/unit/test_codegen.c`

- [x] **Step 1: Rewrite the codegen tests that encode the old loop, add new ones**

In `tests/unit/test_codegen.c` replace the assertion blocks as follows (keep each test's program text).

`test_codegen_generational_2d_alloc_uses_dim_names`:
```c
    ASSERT_TRUE(strstr(code, "T.append([[0.0] * M for _ in range(N)])") != NULL);
    ASSERT_TRUE(strstr(code, "U.append([[0.0] * M for _ in range(N)])") != NULL);
```
`test_codegen_generational_2d_alloc_inferred_dim`:
```c
    ASSERT_TRUE(strstr(code, "T.append([[0.0] * N for _ in range(N)])") != NULL);
```
`test_codegen_generational`:
```c
    ASSERT_TRUE(strstr(code, "__EDGECC_MAX_GEN__ = 1000") != NULL);
    ASSERT_TRUE(strstr(code, "def run(G, F):") != NULL);
    ASSERT_TRUE(strstr(code, "F = [F]") != NULL);
    ASSERT_TRUE(strstr(code, "_i = 0") != NULL);
    ASSERT_TRUE(strstr(code, "while True:") != NULL);
    ASSERT_TRUE(strstr(code, "F.append([0.0] * N)") != NULL);
    ASSERT_TRUE(strstr(code, "b_val = F[_i][s]") != NULL);
    ASSERT_TRUE(strstr(code, "state = F[_i + 1][d]") != NULL);
    ASSERT_TRUE(strstr(code, "if F[_i + 1] == F[_i]:") != NULL);
    ASSERT_TRUE(strstr(code, "_i += 1") != NULL);
    ASSERT_TRUE(strstr(code, "raise RuntimeError(\"generational loop did not stop\")") != NULL);
    ASSERT_TRUE(strstr(code, "return F\n") != NULL);
    ASSERT_TRUE(strstr(code, "_prev") == NULL);
    ASSERT_TRUE(strstr(code, "_next") == NULL);
    ASSERT_TRUE(strstr(code, "_edgecc_occupancy") == NULL);   /* only when used */
```
`test_codegen_cascade_bellman_ford`:
```c
    ASSERT_TRUE(strstr(code, "def run(W, D):") != NULL);
    ASSERT_TRUE(strstr(code, "\"output\": \"D\"") != NULL);
    ASSERT_TRUE(strstr(code, "D = [D]") != NULL);
    ASSERT_TRUE(strstr(code, "T = []") != NULL);
    ASSERT_TRUE(strstr(code, "D.append([math.inf] * N)") != NULL);
    ASSERT_TRUE(strstr(code, "T.append([math.inf] * N)") != NULL);
    ASSERT_TRUE(strstr(code, "state = T[_i][d]") != NULL);     /* T written at i */
    ASSERT_TRUE(strstr(code, "b_val = T[_i][d]") != NULL);     /* T read at i */
    ASSERT_TRUE(strstr(code, "a_val = D[_i][d]") != NULL);
    ASSERT_TRUE(strstr(code, "if D[_i + 1] == D[_i]:") != NULL);
    ASSERT_TRUE(strstr(code, "return D\n") != NULL);
```
`test_codegen_two_cross_gen`:
```c
    ASSERT_TRUE(strstr(code, "def run(G, F, P):") != NULL);
    ASSERT_TRUE(strstr(code, "F = [F]") != NULL);
    ASSERT_TRUE(strstr(code, "P = [P]") != NULL);
    ASSERT_TRUE(strstr(code, "F.append(") != NULL);
    ASSERT_TRUE(strstr(code, "P.append(") != NULL);
    ASSERT_TRUE(strstr(code, "b_val = F[_i + 1][d]") != NULL);  /* P einsum reads F at i+1 */
    ASSERT_TRUE(strstr(code, "if P[_i + 1] == P[_i]:") != NULL);
    ASSERT_TRUE(strstr(code, "return P\n") != NULL);
```
`test_codegen_independent_cross_gen` (update its comment to "both are stored in full; the loop stops on the single watched tensor (A)"):
```c
    ASSERT_TRUE(strstr(code, "def run(G, A, B):") != NULL);
    ASSERT_TRUE(strstr(code, "A.append(") != NULL);
    ASSERT_TRUE(strstr(code, "B.append(") != NULL);
    ASSERT_TRUE(strstr(code, "if A[_i + 1] == A[_i]:") != NULL);   /* gate on watched A */
    ASSERT_TRUE(strstr(code, "if B[") == NULL);                    /* B is NOT the gate */
    ASSERT_TRUE(strstr(code, "return A\n") != NULL);
```
`test_codegen_trace_scaffolding`:
```c
    ASSERT_TRUE(strstr(code, "_trace=None") != NULL);
    ASSERT_TRUE(strstr(code, "\"kind\": \"generation\", \"gen\": 0") != NULL);
    ASSERT_TRUE(strstr(code, "for _g in range(1, _i + 2):") != NULL);
    ASSERT_TRUE(strstr(code, "\"P\": P[_g]") != NULL);
    ASSERT_TRUE(strstr(code, "\"schema\": \"edgecc-trace/1\"") != NULL);
    ASSERT_TRUE(strstr(code, "\"dims\": {\"N\": len(P[0])}") != NULL);
    ASSERT_TRUE(strstr(code, ".trace.json") != NULL);
    ASSERT_TRUE(strstr(code, "_edgecc_snapshot") == NULL);   /* history needs no copies */
```
Add new tests (and `RUN_TEST` them):
```c
TEST(test_codegen_occupancy_stop) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor F[gen I, R=N] : float, empty = inf;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n"
        "init F = [0.0, inf, inf];\n"
        "init P = [1.0, 0.0, 0.0];\n"
        "T[i, d]   = G[s,d] . F[i,s]   :: map(add, intersect) reduce(min, union);\n"
        "F[i+1, d] = T[i,d] . !P[i,d]  :: map(take_left, intersect) reduce(min, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: ||F[i+1]|| == 0;\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "def _edgecc_occupancy(x, empty):") != NULL);
    ASSERT_TRUE(strstr(code, "if _edgecc_occupancy(F[_i + 1], math.inf) == 0:") != NULL);
    ASSERT_TRUE(strstr(code, "\"output\": \"F\"") != NULL);
    ASSERT_TRUE(strstr(code, "return F\n") != NULL);
    free(code); ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

TEST(test_codegen_fused_generational_indexing) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n"
        "init F = [1.0, 0.0, 0.0];\n"
        "init P = [1.0, 0.0, 0.0];\n"
        "F[i+1, d] = (G[s,d] .1 F[i,s]) .2 !P[i,d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or,union) reduce(or,union);\n"
        "stop: stable(P);\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "F[_i][s]") != NULL);        /* inner operand at i */
    ASSERT_TRUE(strstr(code, "P[_i][d]") != NULL);        /* outer operand at i */
    ASSERT_TRUE(strstr(code, "F[_i + 1][d]") != NULL);    /* fused output at i+1 */
    ASSERT_TRUE(strstr(code, "_prev") == NULL);
    ASSERT_TRUE(strstr(code, "_next") == NULL);
    free(code); ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}
```

- [x] **Step 2: Run to verify failure**

Run: `cmake --build build -j8 2>&1 | grep -E 'error|warning'; ./build/test_codegen | grep -cE 'FAIL'`
Expected: many FAILs (old loop still emitted).

- [x] **Step 3: Slice naming helper; use it everywhere `_prev`/`_next` were formed**

In `src/codegen.c`, add near `emit_tensor_access`:
```c
/* A generational tensor stores every generation (the paper's rank I), so a
   reference names one slice: kind 1 -> X[_i], kind 2 -> X[_i + 1]; kind 0 is
   a non-generational tensor (plain name). */
static void gen_slice_name(char *out, size_t cap, const char *name, int kind) {
    if (kind == 2)      snprintf(out, cap, "%s[_i + 1]", name);
    else if (kind == 1) snprintf(out, cap, "%s[_i]", name);
    else                snprintf(out, cap, "%s", name);
}
```
In `emit_general_skeleton`, replace the comment + the three `if/else` naming blocks (from `/* Generational tensors are double-buffered` through the `z_name` else) with:
```c
    /* Generational tensors store every generation; index the slice. */
    char a_name[128], b_name[128], z_name[128];
    gen_slice_name(a_name, sizeof(a_name), load1->data.load.tensor_name, load1->data.load.gen_read);
    gen_slice_name(b_name, sizeof(b_name), load2->data.load.tensor_name, load2->data.load.gen_read);
    gen_slice_name(z_name, sizeof(z_name), store_i->data.store.tensor_name, store_i->data.store.gen_write);
```
Replace the body of `fused_operand_name` with `gen_slice_name(out, cap, op->name, op->gen_read);` and update its comment to "Resolve a fused operand's generation slice into `out`." In `emit_fused_skeleton`, replace the `z_name` `if/else` with:
```c
    char z_name[160];
    gen_slice_name(z_name, sizeof(z_name), fu->out_name, fu->gen_write);
```

- [x] **Step 4: Capture the predicate; occupancy helper; drop the snapshot helper**

In `codegen_python_ex`, in the gen-loop metadata scan, add `int stop_kind = STOP_EQUAL, stop_gen_a = 1, stop_gen_b = 0; double watch_empty = 0.0;` to the declarations above the loop, and inside the `IR_GEN_LOOP_BEGIN` branch after `watch = gl->watch;`:
```c
            stop_kind   = gl->stop_kind;
            stop_gen_a  = gl->stop_gen_a;
            stop_gen_b  = gl->stop_gen_b;
            watch_empty = gl->watch_empty;
```
Replace the `_edgecc_snapshot` helper block (`if (emit_trace) { cb_append(&cb, "def _edgecc_snapshot(x):\n" ...); }`) with:
```c
    /* Occupancy helper for `stop: ||X[g]|| == 0`; emitted only when used. */
    if (iterative && stop_kind == STOP_OCCUPANCY) {
        cb_append(&cb,
            "def _edgecc_occupancy(x, empty):\n"
            "    if isinstance(x, list):\n"
            "        return sum(_edgecc_occupancy(v, empty) for v in x)\n"
            "    return 0 if x == empty else 1\n"
            "\n");
    }
```

- [x] **Step 5: Allocation exclusion for intermediates**

In the `IR_TENSOR_ALLOC` case inside `run()`, change the condition's exclusion from
`!(iterative && name_in_list(instr->data.tensor_alloc.name, cross_gens, num_cross_gens))`
to
```c
                    !(iterative &&
                      (name_in_list(instr->data.tensor_alloc.name, cross_gens, num_cross_gens) ||
                       name_in_list(instr->data.tensor_alloc.name, intermediates, num_intermediates)))
```
and update the comment to: "a generational tensor is built by the gen loop (from the caller's generation 0, or empty for an intermediate)."

- [x] **Step 6: Replace the gen-loop emission**

Add this helper above `codegen_python_ex`:
```c
/* Build the per-generation trace from the stored history, after the loop.
   Step g holds each cross-generational tensor at generation g and, for g >= 1,
   each intermediate at generation g-1 (written during iteration g-1) -- the
   same steps edgecc-trace/1 always recorded. Slices are never written after
   their iteration, so no copies are needed. */
static void emit_trace_from_history(CodeBuf *cb,
                                    const char **cross, int nc,
                                    const char **inter, int ni) {
    cb_indent(cb); cb_append(cb, "if _trace is not None:\n");
    cb->indent++;
    cb_indent(cb);
    cb_append(cb, "_trace.append({\"kind\": \"generation\", \"gen\": 0, \"tensors\": {");
    for (int c = 0; c < nc; c++) {
        if (c) cb_append(cb, ", ");
        cb_appendf(cb, "\"%s\": %s[0]", cross[c], cross[c]);
    }
    cb_append(cb, "}})\n");
    cb_indent(cb); cb_append(cb, "for _g in range(1, _i + 2):\n");
    cb->indent++;
    cb_indent(cb);
    cb_append(cb, "_trace.append({\"kind\": \"generation\", \"gen\": _g, \"tensors\": {");
    int first = 1;
    for (int c = 0; c < nc; c++) {
        if (!first) cb_append(cb, ", ");
        cb_appendf(cb, "\"%s\": %s[_g]", cross[c], cross[c]);
        first = 0;
    }
    for (int m = 0; m < ni; m++) {
        if (!first) cb_append(cb, ", ");
        cb_appendf(cb, "\"%s\": %s[_g - 1]", inter[m], inter[m]);
        first = 0;
    }
    cb_append(cb, "}})\n");
    cb->indent -= 2;
}
```
Replace the entire `case IR_GEN_LOOP_BEGIN: { ... break; }` block with:
```c
            case IR_GEN_LOOP_BEGIN: {
                IRGenLoop *gloop = &instr->data.gen_loop;
                /* Every generational tensor is a list indexed by generation.
                   The caller supplies generation 0 of each cross-generational
                   tensor; an intermediate starts with no generations. */
                for (int c = 0; c < gloop->num_cross_gens; c++) {
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s = [%s]\n",
                               gloop->cross_gens[c].name, gloop->cross_gens[c].name);
                }
                for (int m = 0; m < gloop->num_intermediates; m++) {
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s = []\n", gloop->intermediates[m].name);
                }
                cb_indent(&cb); cb_append(&cb, "_i = 0\n");
                cb_indent(&cb); cb_append(&cb, "while True:\n");
                cb.indent++;
                /* Each iteration first appends one empty slice per tensor it
                   writes: cross-gens at i+1, intermediates at i. */
                IRGenAlloc *allocs[16]; int na = 0;
                for (int c = 0; c < gloop->num_cross_gens; c++) allocs[na++] = &gloop->cross_gens[c];
                for (int m = 0; m < gloop->num_intermediates; m++) allocs[na++] = &gloop->intermediates[m];
                for (int k = 0; k < na; k++) {
                    IRTensorAlloc tmp; memset(&tmp, 0, sizeof(tmp));
                    tmp.ndims = allocs[k]->spatial_ndims;
                    tmp.empty_val = allocs[k]->empty_val;
                    for (int j = 0; j < tmp.ndims; j++) {
                        tmp.dim_sizes[j] = allocs[k]->spatial_dims[j];
                        tmp.dim_names[j] = allocs[k]->spatial_dim_names[j];
                    }
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s.append(", allocs[k]->name);
                    emit_alloc(&cb, &tmp, ir, i);
                    cb_append(&cb, ")\n");
                }
                break;
            }
```
Replace the entire `case IR_GEN_LOOP_END: { ... break; }` block with:
```c
            case IR_GEN_LOOP_END: {
                /* Compute-then-check: the generation that satisfies the stop
                   predicate is kept, then the loop ends. */
                char pa[160], pb[160];
                gen_slice_name(pa, sizeof(pa), watch, stop_gen_a + 1);
                cb_indent(&cb);
                if (stop_kind == STOP_OCCUPANCY) {
                    char *ev = format_double_literal(watch_empty);
                    cb_appendf(&cb, "if _edgecc_occupancy(%s, %s) == 0:\n", pa, ev);
                    free(ev);
                } else {
                    gen_slice_name(pb, sizeof(pb), watch, stop_gen_b + 1);
                    cb_appendf(&cb, "if %s == %s:\n", pa, pb);
                }
                cb.indent++;
                cb_indent(&cb); cb_append(&cb, "break\n");
                cb.indent--;
                cb_indent(&cb); cb_append(&cb, "_i += 1\n");
                cb_indent(&cb); cb_append(&cb, "if _i >= __EDGECC_MAX_GEN__:\n");
                cb.indent++;
                cb_indent(&cb);
                cb_append(&cb, "raise RuntimeError(\"generational loop did not stop\")\n");
                cb.indent--;
                cb.indent--;   /* leave the while loop */
                if (emit_trace)
                    emit_trace_from_history(&cb, cross_gens, num_cross_gens,
                                            intermediates, num_intermediates);
                break;
            }
```

- [x] **Step 7: Return the full tensor; fix the trace's spatial size**

Replace
```c
        if (iterative && watch) {
            cb_appendf(&cb, "return %s_prev\n", watch);
        } else {
```
with
```c
        if (iterative && watch) {
            cb_appendf(&cb, "return %s\n", watch);   /* every generation */
        } else {
```
In the trace `json.dump` line, change `"\"dims\": {\"N\": len(%s)}, \"gen_tensors\": ["` to `"\"dims\": {\"N\": len(%s[0])}, \"gen_tensors\": ["` (the returned tensor is a list of generations; `N` is a slice's length).

- [x] **Step 8: Verify**

Run:
```bash
cmake --build build -j8 2>&1 | grep -E 'error|warning'
for t in lexer parser semantic ir codegen; do ./build/test_$t | tail -1; done
python3 tests/trace/run_trace_tests.py build/edgecc | tail -2
python3 tests/viz/run_viz_tests.py | tail -2
./build/edgecc examples/push_bfs_fused.edge -o /tmp/new.py && python3 /tmp/new.py > /tmp/new.out
../premerge-gate/build/edgecc examples/push_bfs_fused.edge -o /tmp/old.py && python3 /tmp/old.py > /tmp/old.out
python3 -c "import math; e={'inf': math.inf}; n=eval(open('/tmp/new.out').read(), e); o=eval(open('/tmp/old.out').read(), e); print(len(n), 'generations; last == main:', n[-1] == o)"
```
Expected: all unit binaries pass; trace and viz tests pass unchanged; the last command prints `... generations; last == main: True` (`../premerge-gate/build/edgecc` is the `main` build).
The 14 generational e2e tests and the reference suite still FAIL on output shape: fixed in Task 6.

- [x] **Step 9: Commit**
```bash
git add src/codegen.c tests/unit/test_codegen.c
git commit -m "codegen: store every generation (no ping-pong); stop predicates; return the full tensor"
```

---

## Task 6: Migrate expected outputs; reference suite compares the last generation

**Files:**
- Modify: 14 `tests/e2e/*/expected_output.txt` (via script)
- Modify: `tests/reference/run_reference.py`
- Modify: example header comments in `examples/bellman_ford.edge`, `examples/bellman_ford_2stage.edge`, `examples/connected_components.edge`, `examples/reachability.edge`, `examples/transitive_closure.edge`

- [x] **Step 1: Write the guarded migration script (not committed)**

Create `build/migrate_expected.py` (`build/` is gitignored):
```python
#!/usr/bin/env python3
"""One-off: migrate generational e2e expected outputs from the final slice to
the full history. Refuses to write unless the new output's LAST generation
equals the old expected output exactly (the change must be representational)."""
import glob, math, os, subprocess, sys, tempfile

REPO = os.getcwd()
EDGECC = os.path.join(REPO, "build", "edgecc")
ENV = {"inf": math.inf, "nan": math.nan}
refused = 0
for inp in sorted(glob.glob("tests/e2e/*/input.edge")):
    if "stop:" not in open(inp).read():
        continue
    d = os.path.dirname(inp)
    exp_path = os.path.join(d, "expected_output.txt")
    old = open(exp_path).read().strip()
    args = []
    if os.path.exists(os.path.join(d, "args")):
        args = [t.replace("{DATA}", os.path.join(REPO, "data"))
                for t in open(os.path.join(d, "args")).read().split()]
    with tempfile.TemporaryDirectory() as tmp:
        py = os.path.join(tmp, "k.py")
        subprocess.run([EDGECC, inp, "-o", py], check=True, capture_output=True)
        r = subprocess.run([sys.executable, py, *args], capture_output=True,
                           text=True, cwd=d, check=True)
    new = r.stdout.strip()
    new_val, old_val = eval(new, ENV), eval(old, ENV)
    if new_val[-1] != old_val:
        print(f"REFUSE {d}: last generation {new_val[-1]} != old {old_val}")
        refused += 1
        continue
    with open(exp_path, "w") as f:
        f.write(new + "\n")
    print(f"migrated {d}: {len(new_val)} generations")
sys.exit(1 if refused else 0)
```

- [x] **Step 2: Run it**

Run: `python3 build/migrate_expected.py; echo "exit=$?"`
Expected: 14 `migrated ...` lines, no `REFUSE`, `exit=0`. If anything is refused, STOP and investigate: it means the change altered a result, not just its shape.

- [x] **Step 3: Hand-verify one history**

Run: `cat tests/e2e/reachability_bfs/expected_output.txt; cat tests/e2e/reachability_bfs/input.edge | grep -E 'init|stop'`
Check by hand that generation 0 equals the `init` of the watched tensor, each generation adds exactly the vertices reachable in one more step, and the last two generations are equal (stable stop). Record the check in the commit message.

- [x] **Step 4: Reference harness compares the last generation**

In `tests/reference/run_reference.py`:
- Add `gen=True` to every `CASES` entry except `diag_g2`.
- In `main()`, replace
```python
                exp = case["ref"](G)
```
with
```python
                if case.get("gen"):
                    out = out[-1]   # run() returns every generation; compare the final state
                exp = case["ref"](G)
```

- [x] **Step 5: Example header comments**

In each of the five examples listed above, change the `// Output:` label to `// Output (last generation):` (the value shown is still correct: it is the final generation; `run()` now returns every generation). Verify each still matches:
```bash
for f in bellman_ford bellman_ford_2stage connected_components reachability transitive_closure; do
  ./build/edgecc examples/$f.edge -o /tmp/ex.py >/dev/null && python3 /tmp/ex.py > /tmp/ex.out
  echo "$f last generation:"; python3 -c "import math; print(eval(open('/tmp/ex.out').read(), {'inf': math.inf})[-1])"
  grep -n -A2 'Output (last generation)' examples/$f.edge
done
```
Expected: each printed `last=` value matches the comment.

- [x] **Step 6: Full suite**

Run: `ctest --test-dir build --output-on-failure 2>&1 | tail -4`
Expected: 100% pass.

- [x] **Step 7: Commit**
```bash
git add tests/e2e tests/reference/run_reference.py examples
git commit -m "test: generational e2e outputs are full histories (guarded: last generation unchanged)

Migrated 14 expected outputs with a script that refused to write unless
the new output's last generation equalled the old output exactly.
Hand-checked tests/e2e/reachability_bfs. Reference suite compares
out[-1] for generational kernels."
```

---

## Task 7: Retire the `<T>_prev` / `<T>_next` name reservation

**Files:**
- Modify: `src/semantic.c` (remove `is_gen_buffer_name` and its use)
- Delete: `tests/negative/reserved_gen_suffix/`, `tests/negative/reserved_gen_suffix_dim_var/`
- Modify: `tests/reserved/run_reserved_names.py` (drop the derived-name allowance)
- Modify: `CLAUDE.md` (Names paragraph)
- Test: `tests/unit/test_semantic.c`

- [x] **Step 1: Write the failing positive test**

Add to `tests/unit/test_semantic.c` (and `RUN_TEST` it):
```c
/* Generated code no longer uses <T>_prev/<T>_next, so those names are free. */
TEST(test_semantic_gen_buffer_names_are_free) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor F_next[R=N]   : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n"
        "init F = [1.0, 0.0, 0.0];\n"
        "init F_next = [0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s] :: map(and, intersect) reduce(or, union);\n"
        "stop: stable(F);\n");
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    semantic_result_free(&res); ast_free(ast);
}
```

- [x] **Step 2: Run to verify failure**

Run: `cmake --build build -j8 2>&1 | grep error; ./build/test_semantic | grep -A1 gen_buffer_names`
Expected: FAIL (`tensor 'F_next' is reserved ... (buffer of a generational tensor)`).

- [x] **Step 3: Remove the reservation**

In `src/semantic.c`: delete the `is_gen_buffer_name` function and its comment; in `check_reserved`, replace the `why` computation with:
```c
    const char *why = is_py_reserved(name) ? "keyword, builtin, or compiler temporary" : NULL;
```
(`check_reserved` no longer needs `program`; drop that parameter from its signature and from all its call sites.)
Delete the two fixture directories:
```bash
git rm -r -q tests/negative/reserved_gen_suffix tests/negative/reserved_gen_suffix_dim_var
```
In `tests/reserved/run_reserved_names.py`, make `user_names` return only `names` (delete the `gen` and `derived` computations) and change the call site to `users = user_names(src)` and the filter to `c.names - users`, so any `_prev`/`_next` emission would now fail the sync test. Update the module docstring: remove "a generational buffer (<T>_prev / <T>_next),".
In `CLAUDE.md`, in the **Names** paragraph, delete `, and a
tensor may not be named `<T>_prev`/`<T>_next` for a generational `T`` (keep the sentence ending with "…may not reuse a dim or tensor name.").

- [x] **Step 4: Verify**

Run: `cmake --build build -j8 2>&1 | grep -E 'error|warning'; ./build/test_semantic | tail -1; python3 tests/reserved/run_reserved_names.py build/edgecc; ctest --test-dir build 2>&1 | tail -3`
Expected: semantic tests pass; `reserved-name check: ... 0 unlisted names`; full ctest passes.

- [x] **Step 5: Commit**
```bash
git add -A src/semantic.c tests/unit/test_semantic.c tests/negative tests/reserved CLAUDE.md
git commit -m "semantic: retire the <T>_prev/<T>_next name reservation (codegen no longer emits them)"
```

---

## Task 8: Paper BFS demonstrator and generational save

**Files:**
- Create: `examples/bfs_paper.edge`
- Create: `tests/e2e/bfs_paper/{input.edge,expected_output.txt}`
- Create: `tests/e2e/save_generational/{input.edge,expected_output.txt,roundtrip_check,needs_scipy}`

- [x] **Step 1: Write the demonstrator and its hand-verified expected history**

`examples/bfs_paper.edge`:
```
// The paper's BFS (tex/bfs-example/07-iteration.tex, equation bfsi), as written:
//   T[i,d]   = G[s,d] . F[i,s]   :: map(add, intersect) reduce(min, union)   advance
//   F[i+1,d] = T[i,d] . !P[i,d]  :: map(take_left, intersect)                 filter
//   P[i+1,d] = P[i,d] . F[i+1,d] :: map(or, union)                            update
//   stop: ||F[i+1]|| == 0                                                     frontier empty
// The paper's full specification reduces with ANY; edgecc has no ANY yet, and
// min gives the same depths. run() returns F with every generation: F[g] holds
// the vertices discovered at depth g (valued g), and the last generation is empty.
// Graph: 0->1, 0->2, 1->3, 2->3, 2->4, 3->5, 4->5; source 0.
// Output: [[0.0, inf, inf, inf, inf, inf], [inf, 1.0, 1.0, inf, inf, inf],
//          [inf, inf, inf, 2.0, 2.0, inf], [inf, inf, inf, inf, inf, 3.0],
//          [inf, inf, inf, inf, inf, inf]]
dim N = 6;

tensor G[R=N, C=N]   : float, empty = 0.0;   // adjacency, 1.0 = edge
tensor T[gen I, R=N] : float, empty = inf;   // advanced candidates (intermediate)
tensor F[gen I, R=N] : float, empty = inf;   // frontier, valued by depth
tensor P[gen I, R=N] : float, empty = 0.0;   // visited

init G = [[0.0,1.0,1.0,0.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,1.0,0.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,0.0]];
init F = [0.0, inf, inf, inf, inf, inf];
init P = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

T[i, d]   = G[s,d] . F[i,s]   :: map(add, intersect) reduce(min, union);
F[i+1, d] = T[i,d] . !P[i,d]  :: map(take_left, intersect) reduce(min, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
stop: ||F[i+1]|| == 0;
```
Then:
```bash
mkdir -p tests/e2e/bfs_paper
cp examples/bfs_paper.edge tests/e2e/bfs_paper/input.edge
echo "[[0.0, inf, inf, inf, inf, inf], [inf, 1.0, 1.0, inf, inf, inf], [inf, inf, inf, 2.0, 2.0, inf], [inf, inf, inf, inf, inf, 3.0], [inf, inf, inf, inf, inf, inf]]" > tests/e2e/bfs_paper/expected_output.txt
```
(Hand derivation: depths 0:0, 1:1, 2:1, 3:2, 4:2, 5:3; F[g] = vertices at depth g valued g; 5 has no out-edges, so F[4] is empty and the occupancy predicate stops the loop, keeping F[4].)

`tests/e2e/save_generational/input.edge`:
```
// Saving a generational tensor writes every generation: P is 1-D per
// generation, so result.mtx is a (generations x N) matrix.
dim N = 3;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
tensor P[gen I, R=N] : float, empty = 0.0;
init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];
init F = [1.0, 0.0, 0.0];
init P = [1.0, 0.0, 0.0];
F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
stop: stable(P);
save: P -> "result.mtx";
```
```bash
cd tests/e2e/save_generational
echo "[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]" > expected_output.txt
echo "1.0 0.0 0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0" > roundtrip_check
: > needs_scipy
cd -
```

- [x] **Step 2: Run**

Run: `python3 tests/e2e/run_e2e.py build/edgecc | grep -E 'bfs_paper|save_generational|passed'`
Expected: `PASS bfs_paper`, `PASS save_generational`. If `bfs_paper` fails, compare the printed history against the hand derivation above generation by generation before changing anything.

- [x] **Step 3: Commit**
```bash
git add examples/bfs_paper.edge tests/e2e/bfs_paper tests/e2e/save_generational
git commit -m "examples: the paper's BFS with occupancy stop; e2e for saving a generational tensor"
```

---

## Task 9: Reference suite — independent history checks

**Files:**
- Create: `tests/reference/kernels/bfs_paper.edge`
- Modify: `tests/reference/run_reference.py`

- [x] **Step 1: Kernel**

`tests/reference/kernels/bfs_paper.edge`:
```
// The paper's BFS with occupancy stop, on any loaded graph (run on the 0/1 pattern).
dim N = *;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor T[gen I, R=N] : float, empty = inf;
tensor F[gen I, R=N] : float, empty = inf;
tensor P[gen I, R=N] : float, empty = 0.0;
init G = load("graph.mtx");
init F = seed(source=0, value=0.0);
init P = seed(source=0, value=1.0);
T[i, d]   = G[s,d] . F[i,s]   :: map(add, intersect) reduce(min, union);
F[i+1, d] = T[i,d] . !P[i,d]  :: map(take_left, intersect) reduce(min, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
stop: ||F[i+1]|| == 0;
```

- [x] **Step 2: History references and comparator**

In `tests/reference/run_reference.py`, add after `ref_closure`:
```python
def hop_dist(G):
    return list(shortest_path(csr(G), directed=True, unweighted=True, indices=0))


def eccentricity(dist):
    return int(max(d for d in dist if d < INF))


def ref_reach_history(G):
    """P[g] support = vertices within hop distance g. A stable-stop BFS keeps
    the first repeated generation, so it has eccentricity + 2 generations."""
    dist = hop_dist(G)
    return [[1.0 if d <= g else 0.0 for d in dist] for g in range(eccentricity(dist) + 2)]


def ref_depth_history(G):
    """D[g][v] = hop distance if it is <= g, else inf (eccentricity + 2 generations)."""
    dist = hop_dist(G)
    return [[d if d <= g else INF for d in dist] for g in range(eccentricity(dist) + 2)]


def ref_frontier_history(G):
    """Paper BFS: F[g][v] = g for vertices at hop distance exactly g; the
    occupancy stop keeps the first empty generation (eccentricity + 2 in all)."""
    dist = hop_dist(G)
    return [[float(g) if d == g else INF for d in dist] for g in range(eccentricity(dist) + 2)]
```
after `close`:
```python
def history(cmp):
    """Compare every generation (and the number of generations)."""
    def check(out, exp):
        return len(out) == len(exp) and all(cmp(o, e) for o, e in zip(out, exp))
    return check
```
after `widest_inputs`:
```python
def paper_bfs_inputs(G):
    n = len(G)
    return {"G": G, "F": seed(n, 0.0, INF), "P": seed(n, 1.0, 0.0)}
```

- [x] **Step 3: Cases**

In `CASES`, for the five `bfs_*` reachability kernels (`bfs_push_fused`, `bfs_pull_fused`, `bfs_push_fused_rhs`, `bfs_neg_cascade`, `bfs_naive`) replace `ref=ref_reach, cmp=same_support` with `ref=ref_reach_history, cmp=history(same_support), history=True` (and drop their `gen=True`). For `bfs_depth` use `ref=ref_depth_history, cmp=history(close), history=True`. Add:
```python
    dict(kernel="bfs_paper", graph="pattern", inputs=paper_bfs_inputs,
         ref=ref_frontier_history, cmp=history(close), history=True),
```
In `main()`, the Task 6 `out[-1]` line stays as is (`history` cases have no `gen=True`, so they compare the full list).
Remove `ref_reach` if it is now unused.

- [x] **Step 4: Verify, then prove the history oracle can fail**

Run: `python3 tests/reference/run_reference.py build/edgecc | tail -3`
Expected: all checks pass (13 kernels + `bfs_paper`, 15 graphs).
Sabotage check: temporarily change `bfs_paper.edge`'s stop to `stop: stable(P);` and rerun.
Expected: `bfs_paper` FAILs on every graph (the kernel now returns the watched tensor `P` instead of `F`, so the frontier-history comparison cannot match). For a second, subtler sabotage, change `map(take_left, intersect)` to `map(take_left, union)` in the `F` Einsum (visited vertices are no longer filtered out of the frontier) and confirm failures again. Restore the file and confirm it passes.

- [x] **Step 5: Commit**
```bash
git add tests/reference
git commit -m "test(reference): check whole generational histories against scipy hop distances"
```

---

## Task 10: Docs

**Files:**
- Modify: `CLAUDE.md`, `docs/roadmap.md`

- [x] **Step 1: CLAUDE.md — generational section**

In `CLAUDE.md` "### Generational (iterative) ranks":
- Replace the first paragraph ("A single Einsum may iterate …") and its following paragraph ("Generated `run()` loops internally with a **fresh (empty) output slice each generation** … See `…generational-ranks-a-design.md`.") with:
```
A generational rank, declared with `gen` (no dimension) as a tensor's **first**
rank, makes the tensor hold **every generation**, as in the paper: `F[gen I, R=N]`
is a list indexed by generation (`F[i][d]`). `init` supplies generation 0. The
cascade runs once per generation in source order; cross-generational tensors are
written at `i+1`, intermediates at `i`, and every new slice starts empty (the
paper's "all output tensors are initially empty"). After each generation the
**stop predicate** is checked (compute-then-check: the generation that satisfies
it is kept):

    stop: ||F[i+1]|| == 0;     // occupancy: generation i+1 of F is all-empty (paper BFS)
    stop: P[i+1] == P[i];      // two generations of the same tensor are equal
    stop: stable(P);           // sugar for P[i+1] == P[i]

References name whole generations (`F[i+1]`, not `F[i+1, d]`) at `i` or `i+1`
(`i` only for an intermediate). `run()` returns the tensor named in the predicate
**with every generation**; a fixed safety cap (`__EDGECC_MAX_GEN__ = 1000`) raises
`RuntimeError`. Storing all generations is the paper's semantics; reclaiming
memory (e.g. ping-ponging two buffers) is a future liveness optimization. See
`docs/superpowers/specs/2026-09-30-edgecc-paper-faithful-generations-design.md`.
```
- In the "Cascades" paragraph, replace "(it carries the `init`, is double-buffered, and is returned)" with "(it carries the `init`)", and "recomputed fresh each generation" with "one new generation per iteration".
- In the C1 paragraph, replace "`run()` returns the single tensor named by `stop: stable(...)`; the others are computed and double-buffered each generation but not returned." with "`run()` returns the single tensor named by the stop predicate (in full); the others are computed and stored but not returned."
- In "## Generated Python ABI", append: "For a generational program, `run()` returns the watched tensor as a list of generations (`F[0]` is the caller's generation-0 input), and `__main__` prints it."
- In "## v2 Scope Limitations", replace "occupancy (`‖F‖≡0`) termination; " with "" (occupancy termination is now supported as `stop: ||F[i+1]|| == 0;`).
- In "### Visualization", after "records every generational tensor's value per generation," add "(built after the loop from the stored generations)".

- [x] **Step 2: roadmap.md**

- Update `**Last updated:**` to today's date.
- Change the row `| Generational ranks (\`gen\`, \`stop: stable\`) | ✅ |` to `| Generational ranks (\`gen\`): every generation stored, as in the paper | ✅ |` and add after it `| Stop predicates: occupancy \`\|\|F[i+1]\|\| == 0\`, equality, \`stable(X)\` sugar | ✅ |`.
- In the BFS row, change `(\`ANY\`/parent, occupancy stop; unary \`¬\`=\`!\` ✅, multi-action fusion ✅)` to `(\`ANY\`/parent; unary \`¬\`=\`!\` ✅, multi-action fusion ✅, occupancy stop ✅)`.
- In "### Implemented ✅", change the line "- **Generational ranks** (`gen`) with `stop: stable(T)`; internal double-buffered loop in `run()`." to "- **Generational ranks** (`gen`) store every generation (paper semantics; `run()` returns the full watched tensor); stop predicates `||F[i+1]|| == 0`, `X[i+1] == X[i]`, `stable(X)`. Demonstrator: `examples/bfs_paper.edge`."
- In "### Partial 🚧" BFS line, remove "and occupancy (`‖F‖≡0`) termination".
- In "### Planned ❌ (by area)" → Tooling, add "liveness-based reclamation of generational storage (ping-pong)".

- [x] **Step 3: Verify and commit**

Run: `ctest --test-dir build 2>&1 | tail -2; grep -n '_prev\|double-buffer' CLAUDE.md docs/roadmap.md`
Expected: suite passes; grep finds nothing (or only historical spec references).
```bash
git add CLAUDE.md docs/roadmap.md
git commit -m "docs: paper-faithful generational tensors and stop predicates"
```

---

## Task 11: Finish

- [x] **Step 1: Gate**

Run: `tools/premerge.sh`
Expected: `GATE PASSED`. Fix anything it finds (failing test first), then rerun.

- [x] **Step 2: Reconcile the spec**

Append a "## Deviations and findings (recorded at Finish)" section to the spec listing anything that differed from it during implementation (at minimum: the trace test's `dims.N` now uses a slice's length; any pre-existing bug the gate found). Tick every box in this plan (`sed -i '' 's/^- \[ \] /- [x] /' docs/superpowers/plans/2026-09-30-edgecc-paper-faithful-generations.md`). Commit:
```bash
git add docs/superpowers
git commit -m "docs: record deviations for paper-faithful generations; tick plan"
```

- [x] **Step 3: Merge without touching the repo-root checkout**

`main` is checked out in `.worktrees/premerge-gate` (the repo root belongs to the exploration session and is on another branch). From this worktree:
```bash
git -C ../premerge-gate merge --no-ff feature/paper-faithful-generations \
  -m "Merge feature/paper-faithful-generations: store every generation (paper semantics) and stop predicates"
git -C ../premerge-gate log --oneline -1
```
Then build and run ctest in that worktree:
```bash
cmake --build ../premerge-gate/build -j8 2>&1 | grep -E 'error|warning'; ctest --test-dir ../premerge-gate/build 2>&1 | tail -2
```
Expected: merge commit created; suite passes on `main`.

- [x] **Step 4: Memory and handoff**

Update the BFS-milestone memory (occupancy stop done; remaining: multi-output `run()`, `ANY`/parent pointers, `<<`, >1 generational rank) and note that generational tensors now store every generation. Tell the user `main` moved (the exploration session should merge `main` into its branch and rebuild when convenient, since every generational program's output is now a full history).
