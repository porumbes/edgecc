# Generational Ranks (Sub-project A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-Einsum **generational (iterative) rank** to the EDGE language so the compiler emits the fixed-point loop itself, replacing the external YAML harness for this class of program.

**Architecture:** A generational rank is declared per-tensor with `gen` (no dimension). One Einsum reads generation `i` and writes generation `i+1`; `stop: stable(F);` ends the loop. Codegen emits a double-buffered `while` loop inside `run()` with a **fresh** (empty) output slice each generation, returning the final state. The non-iterative pipeline is untouched; generational support is an additive branch keyed off a SymbolTable flag surfaced through new IR instructions.

**Tech Stack:** C99, Flex, Bison (2.3-compatible), CMake/CTest, generated Python 3. Design spec: `docs/superpowers/specs/2026-06-01-edgecc-generational-ranks-a-design.md`.

**Conventions for this repo:**
- Build: `mkdir -p build && cd build && cmake .. && make`
- All tests: `cd build && ctest --output-on-failure`
- One unit test binary: `./build/test_lexer` (etc.)
- Unit tests use `tests/unit/test_helpers.h` macros (`TEST`, `RUN_TEST`, `ASSERT_INT_EQ`, `ASSERT_STR_EQ`, `ASSERT_TRUE`, `ASSERT_NOT_NULL`, `test_report`). New tests are added to the existing `test_*.c` files and registered with `RUN_TEST(...)` in their `main()`.
- e2e and negative tests are **data-only** directories auto-discovered by the Python harnesses — no CMake changes needed to add a case.
- **Do not** add `Co-Authored-By` trailers to commits.

---

## File Structure

All changes are edits to existing files plus two new test-data directories. No new source files, no CMake changes.

- `src/lexer.l` — add `gen`/`stop`/`stable` keywords and the `+` token.
- `src/ast.h` / `src/ast.c` — `RankPair.is_generational`; `TensorRef.gen_axis`/`gen_offset`; new `NODE_STOP` node + constructor/destructor.
- `src/parser.y` — generational rank production, `i+1` coordinate, `stop` statement.
- `src/semantic.h` / `src/semantic.c` — `TensorEntry.gen_axis`; SymbolTable iterative fields; generational validation + classification; init-shape over spatial ranks.
- `src/ir.h` / `src/ir.c` — `IR_GEN_LOOP_BEGIN`/`IR_GEN_LOOP_END`; `IRLoad.gen_read`/`IRStore.gen_write`; gen-aware lowering; `ir_dump` cases.
- `src/codegen.c` — iterative branch: double-buffered loop, fresh slice, `_prev`/`_next` substitution, stable check, cap, `return F_prev`.
- `tests/unit/test_lexer.c`, `test_parser.c`, `test_semantic.c`, `test_ir.c`, `test_codegen.c` — new unit tests.
- `tests/e2e/reachability/` — new e2e case (`input.edge`, `expected_output.txt`).
- `tests/negative/gen_*` — new negative cases.
- `CLAUDE.md` — documentation.

---

## Task 1: Lexer tokens (`gen`, `stop`, `stable`, `+`)

**Files:**
- Modify: `src/lexer.l`
- Modify: `src/parser.y` (declare the three new tokens so `parser.tab.h` defines them)
- Test: `tests/unit/test_lexer.c`

- [ ] **Step 1: Declare the new tokens in the parser so the header generates**

In `src/parser.y`, find the line:
```
%token TOK_DIM TOK_TENSOR TOK_INIT TOK_MAP TOK_REDUCE TOK_EMPTY
```
Add a new `%token` line directly beneath it:
```
%token TOK_GEN TOK_STOP TOK_STABLE
```

- [ ] **Step 2: Write the failing lexer tests**

In `tests/unit/test_lexer.c`, add these tests after `test_double_colon` (before `test_full_token_sequence`):
```c
TEST(test_gen_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("gen");
    ASSERT_INT_EQ(TOK_GEN, yylex());
    yy_delete_buffer(buf);
}

TEST(test_stop_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("stop");
    ASSERT_INT_EQ(TOK_STOP, yylex());
    yy_delete_buffer(buf);
}

TEST(test_stable_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("stable");
    ASSERT_INT_EQ(TOK_STABLE, yylex());
    yy_delete_buffer(buf);
}

TEST(test_plus_token) {
    YY_BUFFER_STATE buf = yy_scan_string("+");
    ASSERT_INT_EQ('+', yylex());
    yy_delete_buffer(buf);
}
```
Register them in `main()` alongside the other `RUN_TEST(...)` calls:
```c
    RUN_TEST(test_gen_keyword);
    RUN_TEST(test_stop_keyword);
    RUN_TEST(test_stable_keyword);
    RUN_TEST(test_plus_token);
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd build && cmake .. && make test_lexer && ./test_lexer`
Expected: build fails (`TOK_GEN` undeclared) or, once it builds, FAILs — either way the tokens are not yet recognized.

- [ ] **Step 4: Add the keywords and `+` to the lexer**

In `src/lexer.l`, add three keyword rules immediately after the `"reduce"` rule (line ~20), so they are matched before the identifier rule:
```
"gen"               { return TOK_GEN; }
"stop"              { return TOK_STOP; }
"stable"            { return TOK_STABLE; }
```
Then change the single-character token rule (currently `[=\.\,\;\:\[\]\(\)]`) to include `+`:
```
[=\.\,\;\:\[\]\(\)+]  { return yytext[0]; }
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd build && make test_lexer && ./test_lexer`
Expected: PASS for all four new tests.

- [ ] **Step 6: Commit**

```bash
git add src/lexer.l src/parser.y tests/unit/test_lexer.c
git commit -m "feat(lexer): add gen/stop/stable keywords and + token"
```

---

## Task 2: AST nodes + parser grammar

This task adds the AST fields/node and the grammar that builds them: generational rank declaration (`gen I`), the `i+1` coordinate offset, and the `stop: stable(F);` statement.

**Files:**
- Modify: `src/ast.h`, `src/ast.c`
- Modify: `src/parser.y`
- Test: `tests/unit/test_parser.c`

- [ ] **Step 1: Add the AST struct fields and the NODE_STOP node**

In `src/ast.h`:

Add `NODE_STOP` to the `NodeType` enum:
```c
typedef enum {
    NODE_PROGRAM,
    NODE_DIM_DECL,
    NODE_TENSOR_DECL,
    NODE_INIT,
    NODE_EINSUM,
    NODE_STOP
} NodeType;
```

Add `is_generational` to `RankPair`:
```c
struct RankPair {
    char *rank_name;
    char *dim_name;        /* NULL when is_generational */
    int   is_generational; /* 1 if declared `gen NAME` (unbounded, no dim) */
};
```

Add offset fields to `TensorRef`:
```c
struct TensorRef {
    char *tensor_name;
    char **rank_vars;
    int num_vars;
    int gen_axis;    /* index of the coord written as `var + N`, else -1 */
    int gen_offset;  /* the N in `var + N` (0 when gen_axis == -1) */
};
```

Add a `StopData` struct (place it near the other `*Data` structs, e.g. after `InitData`):
```c
typedef struct {
    char *watch_tensor;
} StopData;
```

Add `StopData stop;` to the `AstNode` union:
```c
    union {
        ProgramData program;
        DimDeclData dim_decl;
        TensorDeclData tensor_decl;
        InitData init;
        EinsumData einsum;
        StopData stop;
    } data;
```

Declare the constructor with the other constructors:
```c
AstNode *ast_stop_new(char *watch_tensor);
```

- [ ] **Step 2: Implement the AST constructor, defaults, and destructor**

In `src/ast.c`:

Add the constructor (after `ast_init_new`):
```c
AstNode *ast_stop_new(char *watch_tensor) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_STOP;
    node->data.stop.watch_tensor = watch_tensor;
    return node;
}
```

Set the new `TensorRef` defaults inside `tensor_ref_new`:
```c
TensorRef tensor_ref_new(char *name, char **rank_vars, int num_vars) {
    TensorRef ref;
    ref.tensor_name = name;
    ref.rank_vars = rank_vars;
    ref.num_vars = num_vars;
    ref.gen_axis = -1;
    ref.gen_offset = 0;
    return ref;
}
```

In `ast_free`, add a `NODE_STOP` case and make the tensor-decl rank cleanup safe for NULL `dim_name` (already safe — `free(NULL)` is a no-op — so only the new case is required):
```c
        case NODE_STOP:
            free(node->data.stop.watch_tensor);
            break;
```

- [ ] **Step 3: Write the failing parser tests**

In `tests/unit/test_parser.c`, add these tests before `main()`:
```c
TEST(test_parse_generational_rank) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "tensor F[gen I, R=N] : float, empty = 0.0;");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *td = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_TENSOR_DECL, td->type);
    ASSERT_INT_EQ(2, td->data.tensor_decl.num_ranks);
    ASSERT_INT_EQ(1, td->data.tensor_decl.ranks[0].is_generational);
    ASSERT_STR_EQ("I", td->data.tensor_decl.ranks[0].rank_name);
    ASSERT_INT_EQ(0, td->data.tensor_decl.ranks[1].is_generational);
    ASSERT_STR_EQ("R", td->data.tensor_decl.ranks[1].rank_name);
    ASSERT_STR_EQ("N", td->data.tensor_decl.ranks[1].dim_name);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_offset_coord) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *e = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM, e->type);
    /* output F[i+1, d]: offset on axis 0 */
    ASSERT_INT_EQ(0, e->data.einsum.output.gen_axis);
    ASSERT_INT_EQ(1, e->data.einsum.output.gen_offset);
    ASSERT_STR_EQ("i", e->data.einsum.output.rank_vars[0]);
    ASSERT_STR_EQ("d", e->data.einsum.output.rank_vars[1]);
    /* input2 F[i, s]: no explicit offset */
    ASSERT_INT_EQ(-1, e->data.einsum.input2.gen_axis);
    ASSERT_INT_EQ(0, e->data.einsum.input2.gen_offset);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_stop_stmt) {
    YY_BUFFER_STATE buf = yy_scan_string("stop: stable(F);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *s = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_STOP, s->type);
    ASSERT_STR_EQ("F", s->data.stop.watch_tensor);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}
```
Register them in `main()`:
```c
    RUN_TEST(test_parse_generational_rank);
    RUN_TEST(test_parse_offset_coord);
    RUN_TEST(test_parse_stop_stmt);
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cd build && make test_parser && ./test_parser`
Expected: build error (new fields/grammar absent) or FAIL.

- [ ] **Step 5: Implement the grammar**

In `src/parser.y`:

(a) Add the `StrList` offset tracking. Replace the `StrList` struct and its helpers (the `typedef struct { char **items; ... } StrList;` block and `strlist_new`/`strlist_add`) with:
```c
typedef struct {
    char **items;
    int   *offsets;
    int count;
    int capacity;
} StrList;

static StrList *strlist_new(void) {
    StrList *sl = malloc(sizeof(StrList));
    sl->capacity = 4;
    sl->items = malloc(sl->capacity * sizeof(char *));
    sl->offsets = malloc(sl->capacity * sizeof(int));
    sl->count = 0;
    return sl;
}

static void strlist_add(StrList *sl, char *s, int offset) {
    if (sl->count >= sl->capacity) {
        sl->capacity *= 2;
        sl->items = realloc(sl->items, sl->capacity * sizeof(char *));
        sl->offsets = realloc(sl->offsets, sl->capacity * sizeof(int));
    }
    sl->items[sl->count] = s;
    sl->offsets[sl->count] = offset;
    sl->count++;
}
```

(b) Add a generational alternative to `rank_pair_list`. The existing two alternatives must also set `is_generational = 0`. Replace the whole `rank_pair_list:` rule with:
```c
rank_pair_list:
    TOK_IDENT '=' TOK_IDENT
    {
        RankPairList *rpl = rankpairlist_new();
        RankPair rp;
        rp.rank_name = $1;
        rp.dim_name = $3;
        rp.is_generational = 0;
        rankpairlist_add(rpl, rp);
        $$ = (void *)rpl;
    }
    | TOK_GEN TOK_IDENT
    {
        RankPairList *rpl = rankpairlist_new();
        RankPair rp;
        rp.rank_name = $2;
        rp.dim_name = NULL;
        rp.is_generational = 1;
        rankpairlist_add(rpl, rp);
        $$ = (void *)rpl;
    }
    | rank_pair_list ',' TOK_IDENT '=' TOK_IDENT
    {
        RankPairList *rpl = (RankPairList *)$1;
        RankPair rp;
        rp.rank_name = $3;
        rp.dim_name = $5;
        rp.is_generational = 0;
        rankpairlist_add(rpl, rp);
        $$ = (void *)rpl;
    }
    | rank_pair_list ',' TOK_GEN TOK_IDENT
    {
        RankPairList *rpl = (RankPairList *)$1;
        RankPair rp;
        rp.rank_name = $4;
        rp.dim_name = NULL;
        rp.is_generational = 1;
        rankpairlist_add(rpl, rp);
        $$ = (void *)rpl;
    }
    ;
```

(c) Add the `stop_stmt` to the `stmt` rule and define it. Change:
```c
stmt:
    dim_decl
    | tensor_decl
    | init_stmt
    | einsum_stmt
    ;
```
to:
```c
stmt:
    dim_decl
    | tensor_decl
    | init_stmt
    | einsum_stmt
    | stop_stmt
    ;

stop_stmt:
    TOK_STOP ':' TOK_STABLE '(' TOK_IDENT ')' ';'
    {
        $$ = (void *)ast_stop_new($5);
    }
    ;
```
Add `stop_stmt` to the `%type <node>` line:
```c
%type <node> program stmt_list stmt dim_decl tensor_decl init_stmt einsum_stmt stop_stmt
```

(d) Replace the `ident_list` rule with offset-aware coordinates:
```c
ident_list:
    TOK_IDENT
    {
        StrList *sl = strlist_new();
        strlist_add(sl, $1, 0);
        $$ = (void *)sl;
    }
    | TOK_IDENT '+' TOK_INT_LIT
    {
        StrList *sl = strlist_new();
        strlist_add(sl, $1, $3);
        $$ = (void *)sl;
    }
    | ident_list ',' TOK_IDENT
    {
        StrList *sl = (StrList *)$1;
        strlist_add(sl, $3, 0);
        $$ = (void *)sl;
    }
    | ident_list ',' TOK_IDENT '+' TOK_INT_LIT
    {
        StrList *sl = (StrList *)$1;
        strlist_add(sl, $3, $5);
        $$ = (void *)sl;
    }
    ;
```

(e) Update the `tensor_ref` action to record the offset axis and free the new `offsets` array:
```c
tensor_ref:
    TOK_IDENT '[' ident_list ']'
    {
        StrList *sl = (StrList *)$3;
        TensorRef *ref = malloc(sizeof(TensorRef));
        *ref = tensor_ref_new($1, sl->items, sl->count);
        for (int i = 0; i < sl->count; i++) {
            if (sl->offsets[i] != 0) {
                ref->gen_axis = i;
                ref->gen_offset = sl->offsets[i];
            }
        }
        free(sl->offsets);
        free(sl);  /* free list struct but not items */
        $$ = (void *)ref;
    }
    ;
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd build && make test_parser && ./test_parser`
Expected: PASS for the three new tests. (Bison may report shift/reduce conflicts; the grammar above is unambiguous for these inputs. If `make` fails on the bison step, re-run `cmake ..` first.)

- [ ] **Step 7: Run the full unit suite to confirm no regressions**

Run: `cd build && make && ctest --output-on-failure -R "test_lexer|test_parser"`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/ast.h src/ast.c src/parser.y tests/unit/test_parser.c
git commit -m "feat(parser): generational rank decl, i+1 coordinate, stop statement"
```

---

## Task 3: Semantic analysis

Detect the generational rank, validate offsets and the stop clause, exclude the generational rank from index classification, and validate `init` against spatial ranks only.

**Files:**
- Modify: `src/semantic.h`, `src/semantic.c`
- Test: `tests/unit/test_semantic.c`

- [ ] **Step 1: Add SymbolTable / TensorEntry fields**

In `src/semantic.h`:

Add `gen_axis` to `TensorEntry`:
```c
typedef struct {
    char *name;
    RankPair *ranks;
    int num_ranks;
    DataType datatype;
    double empty_val;
    int dim_sizes[8];
    int gen_axis;   /* index of generational rank, or -1 */
} TensorEntry;
```

Add iterative fields to `SymbolTable` (after `num_reduction_indices`):
```c
    /* Generational (iterative) program metadata. is_iterative == 0 for
       ordinary single-pass programs. */
    int is_iterative;
    char *gen_tensor;    /* name of the generational tensor (borrowed) */
    char *gen_rank_var;  /* the index var bound to the gen rank, e.g. "i" (borrowed) */
    char *stop_watch;    /* tensor named in `stop: stable(...)` (borrowed) */
```

- [ ] **Step 2: Write the failing semantic tests**

In `tests/unit/test_semantic.c`, add a shared valid-program string via tests. Add before `main()`:
```c
TEST(test_generational_valid) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_INT_EQ(1, res.symtab->is_iterative);
    ASSERT_STR_EQ("F", res.symtab->gen_tensor);
    ASSERT_STR_EQ("i", res.symtab->gen_rank_var);
    ASSERT_STR_EQ("F", res.symtab->stop_watch);
    /* gen rank var "i" is excluded from classification */
    ASSERT_INT_EQ(1, res.symtab->num_free_indices);       /* d */
    ASSERT_INT_EQ(1, res.symtab->num_reduction_indices);  /* s */
    ASSERT_STR_EQ("d", res.symtab->free_indices[0].name);
    ASSERT_STR_EQ("s", res.symtab->reduction_indices[0].name);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_generational_stop_wrong_tensor) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(G);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_generational_missing_stop) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_generational_bad_offset) {
    /* write offset must be +1; +2 is rejected */
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "F[i+2, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_generational_init_spatial_shape) {
    /* init provides the spatial slice only (1-D of size N), not the gen axis */
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    semantic_result_free(&res);
    ast_free(ast);
}
```
Register all five in `main()`:
```c
    RUN_TEST(test_generational_valid);
    RUN_TEST(test_generational_stop_wrong_tensor);
    RUN_TEST(test_generational_missing_stop);
    RUN_TEST(test_generational_bad_offset);
    RUN_TEST(test_generational_init_spatial_shape);
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd build && make test_semantic && ./test_semantic`
Expected: build error (new symtab fields referenced) or FAIL.

- [ ] **Step 4: Handle generational ranks in tensor-decl analysis**

In `src/semantic.c`, replace the rank-resolution loop inside `analyze_tensor_decl` (the `for (int i = 0; i < d->num_ranks; i++) { ... }` block plus the surrounding `te->...` assignments) so it records `gen_axis` and skips dim lookup for generational ranks:
```c
    TensorEntry *te = &st->tensors[st->num_tensors];
    te->name = d->name;
    te->ranks = d->ranks;
    te->num_ranks = d->num_ranks;
    te->datatype = d->datatype;
    te->empty_val = d->empty_val;
    te->gen_axis = -1;

    for (int i = 0; i < d->num_ranks; i++) {
        if (d->ranks[i].is_generational) {
            if (te->gen_axis != -1) {
                add_error(res, "tensor '%s' has more than one generational rank",
                          d->name);
            }
            te->gen_axis = i;
            te->dim_sizes[i] = -1;   /* unbounded sentinel */
            continue;
        }
        int sz = find_dim(st, d->ranks[i].dim_name);
        if (sz == -1) {
            add_error(res, "tensor '%s' references undeclared dim '%s'",
                      d->name, d->ranks[i].dim_name);
            te->dim_sizes[i] = 0;
        } else {
            te->dim_sizes[i] = sz;
        }
    }
    st->num_tensors++;
```
(Delete the old `int ok = 1; ... (void)ok; st->num_tensors++;` block this replaces.)

- [ ] **Step 5: Validate init against spatial ranks only**

In `src/semantic.c`, replace the body of `analyze_init` with a version that strips the generational axis before the shape check:
```c
static void analyze_init(SemanticResult *res, AstNode *node) {
    InitData *d = &node->data.init;
    SymbolTable *st = res->symtab;

    TensorEntry *te = symtab_find_tensor(st, d->tensor_name);
    if (te == NULL) {
        add_error(res, "init references undeclared tensor '%s'", d->tensor_name);
        return;
    }

    /* Build spatial dim sizes (excluding the generational axis, if any). */
    int spatial_sizes[8];
    int spatial_ranks = 0;
    for (int i = 0; i < te->num_ranks; i++) {
        if (i == te->gen_axis) continue;
        spatial_sizes[spatial_ranks++] = te->dim_sizes[i];
    }

    check_init_shape(res, d->data, 0, spatial_ranks, spatial_sizes,
                     d->tensor_name);
}
```

- [ ] **Step 6: Detect the generational tensor and validate offsets in the einsum**

In `src/semantic.c`, inside `analyze_einsum`, after the existing rank-count validation block (the three `if (d->...num_vars != ...num_ranks)` checks) and before the `IndexInfo indices[...]` declaration, insert generational detection + validation:
```c
    /* ---- Generational rank handling ---- */
    /* The generational tensor is whichever operand's TensorEntry has gen_axis >= 0.
       For A there is exactly one such tensor; it must be the output and one input. */
    const char *gen_var = NULL;   /* the index var bound to the gen rank */
    {
        TensorEntry *gen_te = NULL;
        if (out_te->gen_axis >= 0) gen_te = out_te;
        else if (in1_te->gen_axis >= 0) gen_te = in1_te;
        else if (in2_te->gen_axis >= 0) gen_te = in2_te;

        if (gen_te != NULL) {
            st->is_iterative = 1;
            st->gen_tensor = gen_te->name;

            /* Output must be the generational tensor, written at offset +1. */
            if (strcmp(d->output.tensor_name, gen_te->name) != 0) {
                add_error(res,
                    "generational tensor '%s' must be the output of the einsum",
                    gen_te->name);
            } else {
                if (d->output.gen_axis != out_te->gen_axis ||
                    d->output.gen_offset != 1) {
                    add_error(res,
                        "generational output '%s' must be written at offset +1 "
                        "on its generational rank (e.g. %s[i+1, ...])",
                        gen_te->name, gen_te->name);
                }
                if (d->output.gen_axis == out_te->gen_axis)
                    gen_var = d->output.rank_vars[out_te->gen_axis];
            }

            /* The matching read of the generational tensor must use offset 0
               (no `+N`) on its generational rank. */
            TensorRef *in_gen = NULL;
            TensorEntry *in_gen_te = NULL;
            if (strcmp(d->input1.tensor_name, gen_te->name) == 0) {
                in_gen = &d->input1; in_gen_te = in1_te;
            } else if (strcmp(d->input2.tensor_name, gen_te->name) == 0) {
                in_gen = &d->input2; in_gen_te = in2_te;
            }
            if (in_gen == NULL) {
                add_error(res,
                    "generational tensor '%s' must also appear as an input "
                    "read at generation i", gen_te->name);
            } else if (in_gen->gen_offset != 0) {
                add_error(res,
                    "generational input '%s' must be read at offset 0 "
                    "(e.g. %s[i, ...]), not with a +N offset",
                    gen_te->name, gen_te->name);
            } else if (gen_var != NULL &&
                       in_gen_te->gen_axis >= 0 &&
                       in_gen_te->gen_axis < in_gen->num_vars &&
                       strcmp(in_gen->rank_vars[in_gen_te->gen_axis], gen_var) != 0) {
                add_error(res,
                    "generational rank variable mismatch: output uses '%s' but "
                    "input uses '%s'",
                    gen_var, in_gen->rank_vars[in_gen_te->gen_axis]);
            }
        }
    }
```

- [ ] **Step 7: Exclude the generational rank from index classification**

Still in `analyze_einsum`, the three `REGISTER_INDEX` loops must skip the generational axis of each ref. Replace the three loops:
```c
    for (int i = 0; i < d->output.num_vars && i < out_te->num_ranks; i++)
        REGISTER_INDEX(d->output.rank_vars[i], out_te->dim_sizes[i]);
    for (int i = 0; i < d->input1.num_vars && i < in1_te->num_ranks; i++)
        REGISTER_INDEX(d->input1.rank_vars[i], in1_te->dim_sizes[i]);
    for (int i = 0; i < d->input2.num_vars && i < in2_te->num_ranks; i++)
        REGISTER_INDEX(d->input2.rank_vars[i], in2_te->dim_sizes[i]);
```
with:
```c
    for (int i = 0; i < d->output.num_vars && i < out_te->num_ranks; i++)
        if (i != out_te->gen_axis)
            REGISTER_INDEX(d->output.rank_vars[i], out_te->dim_sizes[i]);
    for (int i = 0; i < d->input1.num_vars && i < in1_te->num_ranks; i++)
        if (i != in1_te->gen_axis)
            REGISTER_INDEX(d->input1.rank_vars[i], in1_te->dim_sizes[i]);
    for (int i = 0; i < d->input2.num_vars && i < in2_te->num_ranks; i++)
        if (i != in2_te->gen_axis)
            REGISTER_INDEX(d->input2.rank_vars[i], in2_te->dim_sizes[i]);
```
Then record the gen rank var into the symbol table. Immediately after the `#undef REGISTER_INDEX` line, add:
```c
    if (gen_var != NULL) st->gen_rank_var = gen_var;
```

- [ ] **Step 8: Analyze the stop statement and finalize iterative checks**

In `src/semantic.c`, add a stop analyzer (place it after `analyze_einsum`):
```c
static void analyze_stop(SemanticResult *res, AstNode *node) {
    SymbolTable *st = res->symtab;
    const char *watch = node->data.stop.watch_tensor;
    TensorEntry *te = symtab_find_tensor(st, watch);
    if (te == NULL) {
        add_error(res, "stop: undeclared tensor '%s'", watch);
        return;
    }
    if (te->gen_axis < 0) {
        add_error(res, "stop: tensor '%s' is not generational", watch);
    }
    st->stop_watch = te->name;
}
```

Add the `NODE_STOP` case to the statement-dispatch switch in `semantic_analyze`:
```c
            case NODE_STOP:
                analyze_stop(&res, stmt);
                break;
```

After the statement loop in `semantic_analyze` (just before `return res;`), add the cross-statement consistency checks:
```c
    /* Iterative consistency: a generational einsum needs a stop clause that
       watches the generational tensor, and vice versa. */
    SymbolTable *st = res.symtab;
    if (st->is_iterative && st->stop_watch == NULL) {
        add_error(&res, "generational einsum on '%s' requires a 'stop: stable(...)' clause",
                  st->gen_tensor ? st->gen_tensor : "?");
    }
    if (st->stop_watch != NULL && !st->is_iterative) {
        add_error(&res, "'stop' clause present but no generational einsum");
    }
    if (st->is_iterative && st->stop_watch != NULL && st->gen_tensor != NULL &&
        strcmp(st->stop_watch, st->gen_tensor) != 0) {
        add_error(&res, "stop watches '%s' but the generational tensor is '%s'",
                  st->stop_watch, st->gen_tensor);
    }
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `cd build && make test_semantic && ./test_semantic`
Expected: PASS for all five new tests, and existing semantic tests still PASS.

- [ ] **Step 10: Commit**

```bash
git add src/semantic.h src/semantic.c tests/unit/test_semantic.c
git commit -m "feat(semantic): generational rank detection, offset/stop validation, spatial init"
```

---

## Task 4: IR lowering

Emit `IR_GEN_LOOP_BEGIN`/`IR_GEN_LOOP_END` around the spatial loops, strip the generational coordinate from the gen-tensor load/store, and tag them `gen_read`/`gen_write`.

**Files:**
- Modify: `src/ir.h`, `src/ir.c`
- Test: `tests/unit/test_ir.c`

- [ ] **Step 1: Add IR types, the gen-loop struct, and load/store tags**

In `src/ir.h`:

Add the two new types to the `IRType` enum:
```c
typedef enum {
    IR_DIM_DEF,
    IR_TENSOR_ALLOC,
    IR_TENSOR_INIT,
    IR_LOOP_BEGIN,
    IR_LOOP_END,
    IR_LOAD,
    IR_MAP,
    IR_REDUCE,
    IR_STORE,
    IR_GEN_LOOP_BEGIN,
    IR_GEN_LOOP_END
} IRType;
```

Add `gen_read` to `IRLoad` and `gen_write` to `IRStore`:
```c
typedef struct {
    int reg;
    char *tensor_name;
    char **coords;
    int ncoords;
    int gen_read;   /* 1 if this loads the generational tensor's prev slice */
} IRLoad;
```
```c
typedef struct {
    char *tensor_name;
    char **coords;
    int ncoords;
    int src_reg;
    int gen_write;  /* 1 if this stores into the generational tensor's next slice */
} IRStore;
```

Add a gen-loop struct (after `IRStore`):
```c
typedef struct {
    char *tensor_name;       /* the generational tensor */
    int spatial_dims[8];     /* shape excluding the generational axis */
    int spatial_ndims;
    double empty_val;        /* for the fresh slice */
    int max_gen;             /* safety iteration cap */
} IRGenLoop;
```

Add it to the instruction union:
```c
    union {
        IRDimDef dim_def;
        IRTensorAlloc tensor_alloc;
        IRTensorInit tensor_init;
        IRLoopBegin loop_begin;
        IRLoad load;
        IRMap map;
        IRReduce reduce;
        IRStore store;
        IRGenLoop gen_loop;
    } data;
```

- [ ] **Step 2: Write the failing IR test**

In `tests/unit/test_ir.c`, add this test before `main()`:
```c
TEST(test_ir_generational_structure) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    ASSERT_NOT_NULL(ir);

    int gen_begin = 0, gen_end = 0, gen_reads = 0, gen_writes = 0;
    int loop_begins = 0;
    for (int i = 0; i < ir->count; i++) {
        switch (ir->instrs[i].type) {
            case IR_GEN_LOOP_BEGIN:
                gen_begin++;
                ASSERT_STR_EQ("F", ir->instrs[i].data.gen_loop.tensor_name);
                ASSERT_INT_EQ(1, ir->instrs[i].data.gen_loop.spatial_ndims);
                ASSERT_INT_EQ(4, ir->instrs[i].data.gen_loop.spatial_dims[0]);
                break;
            case IR_GEN_LOOP_END: gen_end++; break;
            case IR_LOOP_BEGIN:   loop_begins++; break;
            case IR_LOAD:
                if (ir->instrs[i].data.load.gen_read) {
                    gen_reads++;
                    /* gen coordinate stripped: F[i,s] -> coords [s] */
                    ASSERT_INT_EQ(1, ir->instrs[i].data.load.ncoords);
                    ASSERT_STR_EQ("s", ir->instrs[i].data.load.coords[0]);
                }
                break;
            case IR_STORE:
                if (ir->instrs[i].data.store.gen_write) {
                    gen_writes++;
                    ASSERT_INT_EQ(1, ir->instrs[i].data.store.ncoords);
                    ASSERT_STR_EQ("d", ir->instrs[i].data.store.coords[0]);
                }
                break;
            default: break;
        }
    }
    ASSERT_INT_EQ(1, gen_begin);
    ASSERT_INT_EQ(1, gen_end);
    ASSERT_INT_EQ(1, gen_reads);
    ASSERT_INT_EQ(1, gen_writes);
    ASSERT_INT_EQ(2, loop_begins);   /* spatial: d (free) + s (reduction) */

    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```
Register it in `main()`:
```c
    RUN_TEST(test_ir_generational_structure);
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd build && make test_ir && ./test_ir`
Expected: build error (new fields/types) or FAIL.

- [ ] **Step 4: Implement gen-aware lowering**

In `src/ir.c`:

Add a coordinate-stripping helper near `dup_str_array` (after it):
```c
/* Duplicate coords, omitting index `strip_axis` (pass -1 to keep all). */
static char **dup_coords_strip(char **src, int count, int strip_axis, int *out_count) {
    int n = 0;
    for (int i = 0; i < count; i++) if (i != strip_axis) n++;
    *out_count = n;
    if (n == 0) return NULL;
    char **dst = malloc(n * sizeof(char *));
    int j = 0;
    for (int i = 0; i < count; i++) {
        if (i == strip_axis) continue;
        dst[j++] = dup_str(src[i]);
    }
    return dst;
}
```

Rewrite `emit_einsum` to handle the iterative case. Replace the entire function body with:
```c
static void emit_einsum(IRProgram *ir, AstNode *einsum_node, SymbolTable *symtab) {
    EinsumData *e = &einsum_node->data.einsum;

    /* Resolve the generational tensor's axis per operand (-1 if not the gen tensor). */
    int out_gen_axis = -1, in1_gen_axis = -1, in2_gen_axis = -1;
    if (symtab->is_iterative && symtab->gen_tensor) {
        TensorEntry *gte = symtab_find_tensor(symtab, symtab->gen_tensor);
        int gax = gte ? gte->gen_axis : -1;
        if (strcmp(e->output.tensor_name, symtab->gen_tensor) == 0) out_gen_axis = gax;
        if (strcmp(e->input1.tensor_name, symtab->gen_tensor) == 0) in1_gen_axis = gax;
        if (strcmp(e->input2.tensor_name, symtab->gen_tensor) == 0) in2_gen_axis = gax;

        IRInstr *gl = ir_emit(ir, IR_GEN_LOOP_BEGIN);
        gl->data.gen_loop.tensor_name = dup_str(symtab->gen_tensor);
        gl->data.gen_loop.empty_val = gte ? gte->empty_val : 0.0;
        gl->data.gen_loop.max_gen = 1000;
        int sn = 0;
        for (int i = 0; gte && i < gte->num_ranks; i++) {
            if (i == gte->gen_axis) continue;
            gl->data.gen_loop.spatial_dims[sn++] = gte->dim_sizes[i];
        }
        gl->data.gen_loop.spatial_ndims = sn;
    }

    /* Spatial loops: free indices, then reduction indices (gen rank excluded
       by semantic analysis). */
    for (int i = 0; i < symtab->num_free_indices; i++) {
        IRInstr *instr = ir_emit(ir, IR_LOOP_BEGIN);
        instr->data.loop_begin.index_var = dup_str(symtab->free_indices[i].name);
        instr->data.loop_begin.range = symtab->free_indices[i].size;
    }
    for (int i = 0; i < symtab->num_reduction_indices; i++) {
        IRInstr *instr = ir_emit(ir, IR_LOOP_BEGIN);
        instr->data.loop_begin.index_var = dup_str(symtab->reduction_indices[i].name);
        instr->data.loop_begin.range = symtab->reduction_indices[i].size;
    }

    /* LOAD input1 -> reg 0 */
    {
        IRInstr *instr = ir_emit(ir, IR_LOAD);
        instr->data.load.reg = 0;
        instr->data.load.tensor_name = dup_str(e->input1.tensor_name);
        instr->data.load.coords = dup_coords_strip(e->input1.rank_vars,
            e->input1.num_vars, in1_gen_axis, &instr->data.load.ncoords);
        instr->data.load.gen_read = (in1_gen_axis >= 0);
    }

    /* LOAD input2 -> reg 1 */
    {
        IRInstr *instr = ir_emit(ir, IR_LOAD);
        instr->data.load.reg = 1;
        instr->data.load.tensor_name = dup_str(e->input2.tensor_name);
        instr->data.load.coords = dup_coords_strip(e->input2.rank_vars,
            e->input2.num_vars, in2_gen_axis, &instr->data.load.ncoords);
        instr->data.load.gen_read = (in2_gen_axis >= 0);
    }

    /* MAP: reg 2 = map_op(reg 0, reg 1) */
    {
        IRInstr *instr = ir_emit(ir, IR_MAP);
        instr->data.map.dst_reg = 2;
        instr->data.map.op = e->map_op;
        instr->data.map.merge_op = e->map_merge_op;
        instr->data.map.src_reg1 = 0;
        instr->data.map.src_reg2 = 1;
        instr->data.map.empty_a = symtab_find_tensor(symtab, e->input1.tensor_name)->empty_val;
        instr->data.map.empty_b = symtab_find_tensor(symtab, e->input2.tensor_name)->empty_val;
        instr->data.map.empty_z = symtab_find_tensor(symtab, e->output.tensor_name)->empty_val;
    }

    /* REDUCE: reg 3 = reduce_op(reg 2, reg 3) */
    {
        TensorEntry *zent = symtab_find_tensor(symtab, e->output.tensor_name);
        IRInstr *instr = ir_emit(ir, IR_REDUCE);
        instr->data.reduce.dst_reg = 3;
        instr->data.reduce.op = e->reduce_op;
        instr->data.reduce.merge_op = e->reduce_merge_op;
        instr->data.reduce.src_reg = 2;
        instr->data.reduce.acc_reg = 3;
        instr->data.reduce.identity = compute_op_identity(e->reduce_op, zent->datatype);
        instr->data.reduce.empty_z  = zent->empty_val;
    }

    /* STORE: output[coords] = reg 3 */
    {
        IRInstr *instr = ir_emit(ir, IR_STORE);
        instr->data.store.tensor_name = dup_str(e->output.tensor_name);
        instr->data.store.coords = dup_coords_strip(e->output.rank_vars,
            e->output.num_vars, out_gen_axis, &instr->data.store.ncoords);
        instr->data.store.src_reg = 3;
        instr->data.store.gen_write = (out_gen_axis >= 0);
    }

    /* LOOP_END in reverse order: reduction first, then free */
    for (int i = symtab->num_reduction_indices - 1; i >= 0; i--) {
        ir_emit(ir, IR_LOOP_END);
    }
    for (int i = symtab->num_free_indices - 1; i >= 0; i--) {
        ir_emit(ir, IR_LOOP_END);
    }

    if (symtab->is_iterative) {
        ir_emit(ir, IR_GEN_LOOP_END);
    }
}
```

- [ ] **Step 5: Add free/dump cases for the new IR types**

In `src/ir.c`, in `ir_free`'s switch, add cases for the gen-loop instructions (place near `IR_LOOP_END`):
```c
            case IR_GEN_LOOP_BEGIN:
                free(instr->data.gen_loop.tensor_name);
                break;
            case IR_GEN_LOOP_END:
                break;
```

In `ir_dump`'s switch, add:
```c
            case IR_GEN_LOOP_BEGIN:
                fprintf(out, "GEN_LOOP_BEGIN(%s, spatial=[",
                        instr->data.gen_loop.tensor_name);
                for (int j = 0; j < instr->data.gen_loop.spatial_ndims; j++) {
                    if (j > 0) fprintf(out, ",");
                    fprintf(out, "%d", instr->data.gen_loop.spatial_dims[j]);
                }
                fprintf(out, "], empty=%g, max_gen=%d)\n",
                        instr->data.gen_loop.empty_val,
                        instr->data.gen_loop.max_gen);
                break;
            case IR_GEN_LOOP_END:
                fprintf(out, "GEN_LOOP_END\n");
                break;
```

Also update the `IR_LOAD` and `IR_STORE` dump cases to surface the tags (optional but useful for `--dump-ir`). Change the `IR_LOAD` dump to:
```c
            case IR_LOAD:
                fprintf(out, "LOAD(r%d, %s%s, ", instr->data.load.reg,
                        instr->data.load.tensor_name,
                        instr->data.load.gen_read ? " [gen_read]" : "");
                dump_coords(out, instr->data.load.coords, instr->data.load.ncoords);
                fprintf(out, ")\n");
                break;
```
and `IR_STORE` to:
```c
            case IR_STORE:
                fprintf(out, "STORE(%s%s, ", instr->data.store.tensor_name,
                        instr->data.store.gen_write ? " [gen_write]" : "");
                dump_coords(out, instr->data.store.coords, instr->data.store.ncoords);
                fprintf(out, ", r%d)\n", instr->data.store.src_reg);
                break;
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd build && make test_ir && ./test_ir`
Expected: PASS for `test_ir_generational_structure`, existing IR tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ir.h src/ir.c tests/unit/test_ir.c
git commit -m "feat(ir): gen-loop instructions, coordinate stripping, prev/next tags"
```

---

## Task 5: Codegen — double-buffered fixed-point loop

Emit the iterative `run()` with a fresh empty slice per generation, `_prev`/`_next` substitution, a stable check, and the safety cap. The non-iterative path is unchanged.

**Files:**
- Modify: `src/codegen.c`
- Test: `tests/unit/test_codegen.c`

- [ ] **Step 1: Write the failing codegen test**

In `tests/unit/test_codegen.c`, add this test before `main()`:
```c
TEST(test_codegen_generational) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[1.0,1.0,0.0,0.0],[0.0,1.0,1.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]];\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "__EDGECC_MAX_GEN__ = 1000") != NULL);
    ASSERT_TRUE(strstr(code, "def run(G, F):") != NULL);
    ASSERT_TRUE(strstr(code, "F_prev = F") != NULL);
    ASSERT_TRUE(strstr(code, "while True:") != NULL);
    ASSERT_TRUE(strstr(code, "F_next = [0.0] * N") != NULL);
    ASSERT_TRUE(strstr(code, "b_val = F_prev[s]") != NULL);   /* gen read */
    ASSERT_TRUE(strstr(code, "state = F_next[d]") != NULL);   /* gen write slot */
    ASSERT_TRUE(strstr(code, "if F_next == F_prev:") != NULL);
    ASSERT_TRUE(strstr(code, "break") != NULL);
    ASSERT_TRUE(strstr(code, "F_prev = F_next") != NULL);
    ASSERT_TRUE(strstr(code, "raise RuntimeError") != NULL);
    ASSERT_TRUE(strstr(code, "return F_prev") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```
Register it in `main()`:
```c
    RUN_TEST(test_codegen_generational);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd build && make test_codegen && ./test_codegen`
Expected: FAIL — current codegen has no iterative branch.

- [ ] **Step 3: Make the skeleton substitute `_prev`/`_next` names**

In `src/codegen.c`, change `emit_general_skeleton` so gen-tagged loads/stores use the buffer names. Replace the three tensor-access emissions inside `emit_general_skeleton` (the `a_val`, `b_val`, and the `state`/output stores) so they compute an effective name first. At the top of the function (after the `eA`/`eB`/`eZ` lines), add:
```c
    /* Generational tensors are double-buffered: reads use <name>_prev,
       writes use <name>_next. Non-generational tensors keep their name. */
    char a_name[128], b_name[128], z_name[128];
    if (load1->data.load.gen_read)
        snprintf(a_name, sizeof(a_name), "%s_prev", load1->data.load.tensor_name);
    else
        snprintf(a_name, sizeof(a_name), "%s", load1->data.load.tensor_name);
    if (load2->data.load.gen_read)
        snprintf(b_name, sizeof(b_name), "%s_prev", load2->data.load.tensor_name);
    else
        snprintf(b_name, sizeof(b_name), "%s", load2->data.load.tensor_name);
    if (store_i->data.store.gen_write)
        snprintf(z_name, sizeof(z_name), "%s_next", store_i->data.store.tensor_name);
    else
        snprintf(z_name, sizeof(z_name), "%s", store_i->data.store.tensor_name);
```
Then change the `a_val`/`b_val` loads to use these names. Replace:
```c
    cb_indent(cb); cb_append(cb, "a_val = ");
    emit_tensor_access(cb, load1->data.load.tensor_name,
                       load1->data.load.coords, load1->data.load.ncoords);
    cb_append(cb, "\n");
    cb_indent(cb); cb_append(cb, "b_val = ");
    emit_tensor_access(cb, load2->data.load.tensor_name,
                       load2->data.load.coords, load2->data.load.ncoords);
    cb_append(cb, "\n");
```
with:
```c
    cb_indent(cb); cb_append(cb, "a_val = ");
    emit_tensor_access(cb, a_name,
                       load1->data.load.coords, load1->data.load.ncoords);
    cb_append(cb, "\n");
    cb_indent(cb); cb_append(cb, "b_val = ");
    emit_tensor_access(cb, b_name,
                       load2->data.load.coords, load2->data.load.ncoords);
    cb_append(cb, "\n");
```
Then replace **all three** remaining `emit_tensor_access(cb, store_i->data.store.tensor_name, ...)` calls in the function (the `state = ...` read and the two output writes) so they pass `z_name` instead of `store_i->data.store.tensor_name`. For example the `state` read becomes:
```c
    cb_indent(cb); cb_append(cb, "state = ");
    emit_tensor_access(cb, z_name,
                       store_i->data.store.coords, store_i->data.store.ncoords);
    cb_append(cb, "\n");
```
and likewise the two `emit_tensor_access(...store_i...)` calls that emit the assignment target both become `emit_tensor_access(cb, z_name, store_i->data.store.coords, store_i->data.store.ncoords);`.

- [ ] **Step 4: Detect iterative mode and emit the gen-loop scaffolding**

In `src/codegen.c`, inside `codegen_python`, after `collect_io(ir, &inputs, &output);`, scan for the gen loop:
```c
    /* Detect generational mode and capture the gen-loop metadata. */
    int iterative = 0;
    const char *gen_tensor = NULL;
    IRInstr *gen_begin = NULL;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_GEN_LOOP_BEGIN) {
            iterative = 1;
            gen_tensor = ir->instrs[i].data.gen_loop.tensor_name;
            gen_begin = &ir->instrs[i];
            break;
        }
    }
```

Emit the cap constant. Immediately after the `import math\n\n` append at the top of the function, add:
```c
    if (iterative) {
        cb_append(&cb, "__EDGECC_MAX_GEN__ = 1000\n\n");
    }
```

In the `run()` body emission loop, change the `IR_TENSOR_ALLOC` case so the generational tensor is **not** allocated inside `run()` (it arrives as a parameter; `F_next` is allocated by the gen loop). Replace the `case IR_TENSOR_ALLOC:` block with:
```c
            case IR_TENSOR_ALLOC:
                /* Output tensor is allocated inside run(); but a generational
                   tensor arrives as a parameter and is buffered by the gen loop. */
                if (output && strcmp(instr->data.tensor_alloc.name, output) == 0 &&
                    !(iterative && gen_tensor &&
                      strcmp(instr->data.tensor_alloc.name, gen_tensor) == 0)) {
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s = ", instr->data.tensor_alloc.name);
                    emit_alloc(&cb, &instr->data.tensor_alloc, ir, i);
                    cb_append(&cb, "\n");
                }
                break;
```

Add `IR_GEN_LOOP_BEGIN` / `IR_GEN_LOOP_END` cases to the same switch (place them after `IR_LOOP_END`):
```c
            case IR_GEN_LOOP_BEGIN: {
                cb_indent(&cb);
                cb_appendf(&cb, "%s_prev = %s\n",
                           instr->data.gen_loop.tensor_name,
                           instr->data.gen_loop.tensor_name);
                cb_indent(&cb);
                cb_append(&cb, "_it = 0\n");
                cb_indent(&cb);
                cb_append(&cb, "while True:\n");
                cb.indent++;
                /* fresh empty slice for this generation */
                IRTensorAlloc tmp;
                memset(&tmp, 0, sizeof(tmp));
                tmp.ndims = instr->data.gen_loop.spatial_ndims;
                tmp.empty_val = instr->data.gen_loop.empty_val;
                for (int k = 0; k < tmp.ndims; k++)
                    tmp.dim_sizes[k] = instr->data.gen_loop.spatial_dims[k];
                cb_indent(&cb);
                cb_appendf(&cb, "%s_next = ", instr->data.gen_loop.tensor_name);
                emit_alloc(&cb, &tmp, ir, i);
                cb_append(&cb, "\n");
                break;
            }
            case IR_GEN_LOOP_END: {
                const char *t = gen_tensor;
                cb_indent(&cb);
                cb_appendf(&cb, "if %s_next == %s_prev:\n", t, t);
                cb.indent++;
                cb_indent(&cb); cb_append(&cb, "break\n");
                cb.indent--;
                cb_indent(&cb);
                cb_appendf(&cb, "%s_prev = %s_next\n", t, t);
                cb_indent(&cb); cb_append(&cb, "_it += 1\n");
                cb_indent(&cb); cb_append(&cb, "if _it >= __EDGECC_MAX_GEN__:\n");
                cb.indent++;
                cb_indent(&cb);
                cb_append(&cb, "raise RuntimeError(\"generational loop did not stabilize\")\n");
                cb.indent--;
                cb.indent--;   /* close the while body */
                break;
            }
```

Change the final return so iterative kernels return the final buffer. Replace:
```c
    if (output) {
        cb_indent(&cb);
        cb_appendf(&cb, "return %s\n", output);
    }
```
with:
```c
    if (output) {
        cb_indent(&cb);
        if (iterative && gen_tensor) {
            cb_appendf(&cb, "return %s_prev\n", gen_tensor);
        } else {
            cb_appendf(&cb, "return %s\n", output);
        }
    }
```

- [ ] **Step 5: Skip the gen tensor's dead alloc in `__main__`**

Still in `codegen_python`, in the `__main__` input-alloc loop, skip the generational tensor (its `init` fully defines generation 0; its declared shape includes the unbounded axis). Change the loop body that emits input allocs so it also skips the gen tensor. Replace:
```c
        if (!is_input) continue;
        cb_indent(&cb);
        cb_appendf(&cb, "%s = ", instr->data.tensor_alloc.name);
        emit_alloc(&cb, &instr->data.tensor_alloc, ir, i);
        cb_append(&cb, "\n");
```
with:
```c
        if (!is_input) continue;
        if (iterative && gen_tensor &&
            strcmp(instr->data.tensor_alloc.name, gen_tensor) == 0) continue;
        cb_indent(&cb);
        cb_appendf(&cb, "%s = ", instr->data.tensor_alloc.name);
        emit_alloc(&cb, &instr->data.tensor_alloc, ir, i);
        cb_append(&cb, "\n");
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd build && make test_codegen && ./test_codegen`
Expected: PASS for `test_codegen_generational`, existing codegen tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add src/codegen.c tests/unit/test_codegen.c
git commit -m "feat(codegen): double-buffered generational loop with fresh-slice populate"
```

---

## Task 6: End-to-end reachability test

Compile and run a complete generational program through the plain e2e harness (the loop is internal now).

**Files:**
- Create: `tests/e2e/reachability/input.edge`
- Create: `tests/e2e/reachability/expected_output.txt`

- [ ] **Step 1: Create the reachability program**

Create `tests/e2e/reachability/input.edge`:
```
// Boolean reachability / transitive closure over a reflexive adjacency matrix.
// G is reflexive (1.0 on the diagonal) so each generation re-derives the prior
// frontier and adds one hop; F grows monotonically and goes stable at the
// reachable set. Edges: 0->1, 1->2 (node 3 is unreachable from node 0).
dim N = 4;

tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;

init G = [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
init F = [1.0, 0.0, 0.0, 0.0];

F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);
stop: stable(F);
```

- [ ] **Step 2: Create the expected output**

Create `tests/e2e/reachability/expected_output.txt` with exactly:
```
[1.0, 1.0, 1.0, 0.0]
```

- [ ] **Step 3: Manually verify the generated program runs and prints the expected value**

Run:
```bash
cd build && make edgecc && ./edgecc ../tests/e2e/reachability/input.edge -o /tmp/reach.py && python3 /tmp/reach.py
```
Expected stdout: `[1.0, 1.0, 1.0, 0.0]`

If it differs, inspect with `./edgecc ../tests/e2e/reachability/input.edge -o /tmp/reach.py --dump-ir` and read `/tmp/reach.py`.

- [ ] **Step 4: Run the e2e suite (auto-discovers the new directory)**

Run: `cd build && ctest --output-on-failure -R e2e_tests`
Expected: includes `PASS reachability`, overall PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/reachability/input.edge tests/e2e/reachability/expected_output.txt
git commit -m "test(e2e): generational reachability / transitive-closure kernel"
```

---

## Task 7: Negative tests

Lock in the semantic guardrails with data-only negative cases (auto-discovered by `tests/negative/run_negative.py`; each needs `input.edge` + `expected_errors.txt` whose lines are substrings expected in stderr).

**Files:**
- Create: `tests/negative/gen_stop_wrong_tensor/{input.edge,expected_errors.txt}`
- Create: `tests/negative/gen_missing_stop/{input.edge,expected_errors.txt}`
- Create: `tests/negative/gen_bad_offset/{input.edge,expected_errors.txt}`

- [ ] **Step 1: Create the "stop watches wrong tensor" case**

`tests/negative/gen_stop_wrong_tensor/input.edge`:
```
dim N = 4;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);
stop: stable(G);
```
`tests/negative/gen_stop_wrong_tensor/expected_errors.txt`:
```
not generational
```

- [ ] **Step 2: Create the "missing stop" case**

`tests/negative/gen_missing_stop/input.edge`:
```
dim N = 4;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);
```
`tests/negative/gen_missing_stop/expected_errors.txt`:
```
requires a 'stop: stable(...)' clause
```

- [ ] **Step 3: Create the "bad offset" case**

`tests/negative/gen_bad_offset/input.edge`:
```
dim N = 4;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
F[i+2, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);
stop: stable(F);
```
`tests/negative/gen_bad_offset/expected_errors.txt`:
```
offset +1
```

- [ ] **Step 4: Run the negative suite**

Run: `cd build && ctest --output-on-failure -R negative_tests`
Expected: includes PASS for the three new cases, overall PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/negative/gen_stop_wrong_tensor tests/negative/gen_missing_stop tests/negative/gen_bad_offset
git commit -m "test(negative): generational stop/offset guardrails"
```

---

## Task 8: Documentation + full suite

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document the generational syntax in CLAUDE.md**

In `CLAUDE.md`, in the "EDGE Syntax (v2)" section, after the existing code block, add:
```markdown
### Generational (iterative) ranks

A single Einsum may iterate via a **generational rank** declared with `gen` (no
dimension). The Einsum reads generation `i` and writes generation `i+1`; the
loop runs until the watched tensor is stable:

```
dim N = 4;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
init F = [1.0, 0.0, 0.0, 0.0];           // generation 0
F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);
stop: stable(F);
```

Generated `run()` loops internally with a **fresh (empty) output slice each
generation** (the paper's default populate; no cross-generation accumulation),
double-buffered as `F_prev`/`F_next`, and returns the final state. A fixed
safety cap (`__EDGECC_MAX_GEN__ = 1000`) raises `RuntimeError` if the watched
tensor never stabilizes. Limitations: one generational tensor / one
generational rank, offsets `i` and `i+1` only, `stable` stop only, single
Einsum, no `<<` accumulation. See `docs/superpowers/specs/2026-06-01-edgecc-generational-ranks-a-design.md`.
```

- [ ] **Step 2: Update the deferred-scope list**

In `CLAUDE.md`, in the "v2 Scope Limitations" section, change the deferred entry:
```
generational/iterative ranks expressed in EDGE source (use the iterative test harness or a Python driver instead)
```
to:
```
multi-Einsum generational cascades (the `<<` update/accumulation populate, and chained Einsums sharing a generational rank — single-Einsum generational ranks are supported; see "Generational (iterative) ranks")
```

- [ ] **Step 3: Run the complete test suite**

Run: `cd build && cmake .. && make && ctest --output-on-failure`
Expected: all suites PASS — `test_lexer`, `test_parser`, `test_semantic`, `test_ir`, `test_codegen`, `e2e_tests` (incl. `reachability`), `negative_tests` (incl. the three `gen_*`), `e2e_iterative` (unchanged; skipped if PyYAML missing).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document single-Einsum generational ranks"
```

---

## Self-Review Notes

- **Spec coverage:** surface syntax (Tasks 1–2), fresh-slice populate + cap (Task 5), spatial init (Task 3), reachability demonstrator (Task 6), all five negative validations split across Task 3 unit tests + Task 7 data cases, docs incl. scope-list move (Task 8). `--dump-ir` for the new instructions (Task 4 Step 5).
- **Type consistency:** `gen_axis`/`gen_offset` (TensorRef), `is_generational` (RankPair), `gen_axis` (TensorEntry), `is_iterative`/`gen_tensor`/`gen_rank_var`/`stop_watch` (SymbolTable), `gen_read`/`gen_write` (IRLoad/IRStore), `IRGenLoop.{tensor_name,spatial_dims,spatial_ndims,empty_val,max_gen}` are used identically across tasks.
- **Buffer naming:** `<name>_prev` / `<name>_next` / `_it` / `__EDGECC_MAX_GEN__` are consistent between codegen (Task 5) and the codegen test assertions.
- **No CMake changes:** unit tests append to existing binaries; e2e/negative cases are auto-discovered data dirs.
