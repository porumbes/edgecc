# File Input/Output (densify-on-load .mtx) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let EDGE programs load `.mtx` graphs from disk (`init G = load("g.mtx")`), infer tensor dimensions from the loaded file (`dim N = *;`), and write the result tensor to disk (`save: Z -> "out.mtx";`), with input paths overridable via argv.

**Architecture:** The compiler core carries only an **opaque path string** through lexer → parser → AST → semantic → IR → codegen. All Matrix Market parsing lives in two emitted Python runtime helpers (`_edgecc_load`/`_edgecc_save`) that wrap scipy and **densify** the graph into edgecc's existing dense nested-list representation — the `run()` compute is unchanged. Inferred dims become runtime values derived from input shapes at the top of `run()`; this requires threading dim *names* (not just sizes) into the IR tensor-alloc.

**Tech Stack:** C99, Flex, Bison 2.3, CMake; generated Python 3 + scipy (lazy-imported); ctest.

**Spec:** `docs/superpowers/specs/2026-06-07-edgecc-file-io-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/lexer.l` | Tokenize string literals, `load`, `save`, `*`, `->` | Modify |
| `src/parser.y` | Grammar: `load(STR)` init, `dim = *`, `save:` directive | Modify |
| `src/ast.h` / `src/ast.c` | Init source variant, inferred-dim flag, `NODE_SAVE` | Modify |
| `src/semantic.h` / `src/semantic.c` | Record inferred dims / loaded tensors / save target; validate | Modify |
| `src/ir.h` / `src/ir.c` | `is_inferred` dim, `is_load`+path init, `IR_SAVE`, dim *names* in alloc | Modify |
| `src/codegen.c` | Emit runtime helpers, inferred-dim derivation in `run()`, load/save in `__main__`, argv override | Modify |
| `tests/unit/test_lexer.c` ... `test_codegen.c` | Per-stage unit tests | Modify |
| `tests/e2e/load_mtx/` | E2E: load → run → assert | Create |
| `tests/e2e/load_save_roundtrip/` | E2E: load → run → save → re-read | Create |
| `tests/negative/{save_unknown_tensor,inferred_dim_unpinned}/` | Negative tests | Create |
| `tests/e2e/run_e2e.py` | Skip-if-scipy-missing for `.mtx` tests | Modify |
| `CMakeLists.txt` | (no new targets; existing e2e/negative pick up new dirs) | — |
| `CLAUDE.md` | Document `load`/`save`/`dim = *` + scipy runtime dep | Modify |

**Conventions discovered (follow these):**
- Bison 2.3: `%union` uses `void *` for node/ptr; semantic actions cast.
- Lexer single-char tokens come through the class on `lexer.l:39`; multi-char tokens get named `TOK_*`.
- Tests use the tiny harness in `tests/unit/test_helpers.h` (`TEST`, `RUN_TEST`, `ASSERT_INT_EQ`, `ASSERT_STR_EQ`, `ASSERT_TRUE`, `test_report`). Every new unit test must be added to **both** the `TEST(...)` body and the `RUN_TEST(...)` list in `main()`.
- `IRTensorInit.data` is a **borrowed** pointer into the AST (`ir.c:91`); do not free it in `ir_free`.

---

## Task 1: Lexer — string literals, `load`, `save`, `*`, `->`

**Files:**
- Modify: `src/lexer.l`
- Test: `tests/unit/test_lexer.c`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_lexer.c` (after `test_map_reduce_keywords`, before `int main`):

```c
TEST(test_load_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("load");
    ASSERT_INT_EQ(TOK_LOAD, yylex());
    yy_delete_buffer(buf);
}

TEST(test_save_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("save");
    ASSERT_INT_EQ(TOK_SAVE, yylex());
    yy_delete_buffer(buf);
}

TEST(test_star_token) {
    YY_BUFFER_STATE buf = yy_scan_string("*");
    ASSERT_INT_EQ('*', yylex());
    yy_delete_buffer(buf);
}

TEST(test_arrow_token) {
    YY_BUFFER_STATE buf = yy_scan_string("->");
    ASSERT_INT_EQ(TOK_ARROW, yylex());
    yy_delete_buffer(buf);
}

TEST(test_string_literal) {
    YY_BUFFER_STATE buf = yy_scan_string("\"graph.mtx\"");
    ASSERT_INT_EQ(TOK_STRING_LIT, yylex());
    ASSERT_STR_EQ("graph.mtx", yylval.str_val);
    free(yylval.str_val);
    yy_delete_buffer(buf);
}
```

Register them in `main()` with matching `RUN_TEST(...)` lines.

- [ ] **Step 2: Run to verify failure**

Run: `cd build && cmake .. >/dev/null && make test_lexer 2>&1 | tail -20`
Expected: FAIL — compile error, `TOK_LOAD`/`TOK_SAVE`/`TOK_ARROW`/`TOK_STRING_LIT` undeclared (they are added to `parser.tab.h` in Task 3; for now the lexer tokens don't exist).

> Note: tokens are declared in the Bison grammar (Task 3), which generates `parser.tab.h`. To keep Task 1 self-contained and green on its own, add the token rules to the lexer now AND add the `%token` declarations to `parser.y` in this task's Step 3 (grammar rules using them come in Task 3). This is the one place two files move together.

- [ ] **Step 3: Implement lexer rules + token declarations**

In `src/lexer.l`, add keyword rules after line 24 (`"empty"`):

```
"load"              { return TOK_LOAD; }
"save"              { return TOK_SAVE; }
```

Add the arrow rule before the `"::"` rule (line 32) so `->` is matched as one token:

```
"->"                { return TOK_ARROW; }
```

Add a string-literal rule after the float/int literal rules (after line 35). Strip the surrounding quotes:

```
\"[^\"\n]*\"        {
                      int n = (int)strlen(yytext);
                      yylval.str_val = malloc(n - 1);     /* drop 2 quotes, add NUL */
                      memcpy(yylval.str_val, yytext + 1, n - 2);
                      yylval.str_val[n - 2] = '\0';
                      return TOK_STRING_LIT;
                    }
```

Add `*` to the single-char token class on line 39:

```
[=\.\,\;\:\[\]\(\)+*]  { return yytext[0]; }
```

In `src/parser.y`, add the token declarations after line 106 (`%token <str_val> TOK_IDENT`):

```
%token TOK_LOAD TOK_SAVE TOK_ARROW
%token <str_val> TOK_STRING_LIT
```

- [ ] **Step 4: Run to verify pass**

Run: `cd build && make test_lexer 2>&1 | tail -5 && ./test_lexer`
Expected: PASS — all lexer tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lexer.l src/parser.y tests/unit/test_lexer.c
git commit -m "feat(lexer): string literals, load/save/-> tokens, * for inferred dims"
```

---

## Task 2: AST — init source variant, inferred dim, save node

**Files:**
- Modify: `src/ast.h`, `src/ast.c`

No standalone AST test harness exists; this task makes the data model compile and keeps existing tests green. The new constructors are exercised by parser tests in Task 3.

- [ ] **Step 1: Extend AST structs and declarations**

In `src/ast.h`:

Add `NODE_SAVE` to `NodeType` (after `NODE_STOP`):

```c
typedef enum {
    NODE_PROGRAM,
    NODE_DIM_DECL,
    NODE_TENSOR_DECL,
    NODE_INIT,
    NODE_EINSUM,
    NODE_STOP,
    NODE_SAVE
} NodeType;
```

Add an `is_inferred` flag to `DimDeclData` (size is ignored when inferred):

```c
typedef struct {
    char *name;
    int size;
    int is_inferred;   /* 1 if declared `dim N = *;` (size from a loaded file) */
} DimDeclData;
```

Make `InitData` a tagged variant (literal vs file-load). `data` is NULL when `is_load`; `path` is NULL otherwise:

```c
typedef struct {
    char *tensor_name;
    int is_load;          /* 1 if `init X = load("path");` */
    LiteralData *data;    /* literal init (NULL when is_load) */
    char *path;           /* file path (NULL when literal) */
} InitData;
```

Add a `SaveData` struct (after `StopData`) and a union member + constructor/declaration:

```c
typedef struct {
    char *tensor_name;
    char *path;
} SaveData;
```

Add to the `AstNode` union (after `stop`):

```c
        StopData stop;
        SaveData save;
```

Add constructor declarations (after `ast_stop_new`):

```c
AstNode *ast_init_load_new(char *tensor_name, char *path);
AstNode *ast_dim_decl_inferred_new(char *name);
AstNode *ast_save_new(char *tensor_name, char *path);
```

- [ ] **Step 2: Implement constructors and destructor cases**

In `src/ast.c`:

Set the new `InitData` fields in the existing literal constructor (`ast_init_new`, line 45) so `is_load`/`path` are well-defined:

```c
AstNode *ast_init_new(char *tensor_name, LiteralData *data) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_INIT;
    node->data.init.tensor_name = tensor_name;
    node->data.init.is_load = 0;
    node->data.init.data = data;
    node->data.init.path = NULL;
    return node;
}
```

Add the new constructors (after `ast_init_new`):

```c
AstNode *ast_init_load_new(char *tensor_name, char *path) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_INIT;
    node->data.init.tensor_name = tensor_name;
    node->data.init.is_load = 1;
    node->data.init.data = NULL;
    node->data.init.path = path;
    return node;
}

AstNode *ast_dim_decl_inferred_new(char *name) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_DIM_DECL;
    node->data.dim_decl.name = name;
    node->data.dim_decl.size = 0;
    node->data.dim_decl.is_inferred = 1;
    return node;
}

AstNode *ast_save_new(char *tensor_name, char *path) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_SAVE;
    node->data.save.tensor_name = tensor_name;
    node->data.save.path = path;
    return node;
}
```

Update `ast_free` (line 159) `NODE_INIT` case to free the path, and add a `NODE_SAVE` case:

```c
        case NODE_INIT:
            free(node->data.init.tensor_name);
            literal_data_free(node->data.init.data);   /* NULL-safe */
            free(node->data.init.path);                 /* NULL-safe */
            break;
        case NODE_EINSUM:
            tensor_ref_free_contents(&node->data.einsum.output);
            tensor_ref_free_contents(&node->data.einsum.input1);
            tensor_ref_free_contents(&node->data.einsum.input2);
            break;
        case NODE_STOP:
            free(node->data.stop.watch_tensor);
            break;
        case NODE_SAVE:
            free(node->data.save.tensor_name);
            free(node->data.save.path);
            break;
```

- [ ] **Step 3: Build to verify it compiles and existing tests pass**

Run: `cd build && make 2>&1 | tail -5 && ctest --output-on-failure 2>&1 | tail -15`
Expected: builds clean; all existing tests still PASS (no behavior change yet).

- [ ] **Step 4: Commit**

```bash
git add src/ast.h src/ast.c
git commit -m "feat(ast): init-from-load variant, inferred dim flag, save node"
```

---

## Task 3: Parser — grammar for load / inferred dim / save

**Files:**
- Modify: `src/parser.y`
- Test: `tests/unit/test_parser.c`

- [ ] **Step 1: Read the existing parser-test pattern**

Run: `sed -n '1,60p' tests/unit/test_parser.c`
This shows how tests parse a string (via `yy_scan_string` + `yyparse`) and inspect `parse_result`. Mirror that pattern exactly for the new tests.

- [ ] **Step 2: Write failing parser tests**

Add to `tests/unit/test_parser.c` (and register each in `main()` with `RUN_TEST`). These assert the AST shape produced by the new grammar:

```c
TEST(test_parse_init_load) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "init G = load(\"graph.mtx\");\n");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *prog = parse_result;
    /* statement 2 is the init */
    AstNode *init = prog->data.program.statements[2];
    ASSERT_INT_EQ(NODE_INIT, init->type);
    ASSERT_INT_EQ(1, init->data.init.is_load);
    ASSERT_STR_EQ("graph.mtx", init->data.init.path);
    ast_free(prog);
    yy_delete_buffer(buf);
}

TEST(test_parse_inferred_dim) {
    YY_BUFFER_STATE buf = yy_scan_string("dim N = *;\n");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *dim = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_DIM_DECL, dim->type);
    ASSERT_INT_EQ(1, dim->data.dim_decl.is_inferred);
    ast_free(parse_result);
    yy_delete_buffer(buf);
}

TEST(test_parse_save_directive) {
    YY_BUFFER_STATE buf = yy_scan_string("save: Z -> \"out.mtx\";\n");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *save = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_SAVE, save->type);
    ASSERT_STR_EQ("Z", save->data.save.tensor_name);
    ASSERT_STR_EQ("out.mtx", save->data.save.path);
    ast_free(parse_result);
    yy_delete_buffer(buf);
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cd build && make test_parser 2>&1 | tail -20`
Expected: FAIL — grammar rejects `load(...)`, `= *`, and `save:` (parse errors / `yyparse` returns nonzero).

- [ ] **Step 4: Implement grammar rules**

In `src/parser.y`:

Add `save_stmt` to the `%type <node>` line (line 108):

```
%type <node> program stmt_list stmt dim_decl tensor_decl init_stmt einsum_stmt stop_stmt save_stmt
```

Add `save_stmt` to the `stmt` alternatives (line 135):

```
stmt:
    dim_decl
    | tensor_decl
    | init_stmt
    | einsum_stmt
    | stop_stmt
    | save_stmt
    ;
```

Add the inferred-dim alternative to `dim_decl` (line 150):

```
dim_decl:
    TOK_DIM TOK_IDENT '=' TOK_INT_LIT ';'
    {
        $$ = (void *)ast_dim_decl_new($2, $4);
    }
    | TOK_DIM TOK_IDENT '=' '*' ';'
    {
        $$ = (void *)ast_dim_decl_inferred_new($2);
    }
    ;
```

Add the load alternative to `init_stmt` (line 221):

```
init_stmt:
    TOK_INIT TOK_IDENT '=' nested_literal ';'
    {
        $$ = (void *)ast_init_new($2, (LiteralData *)$4);
    }
    | TOK_INIT TOK_IDENT '=' TOK_LOAD '(' TOK_STRING_LIT ')' ';'
    {
        $$ = (void *)ast_init_load_new($2, $6);
    }
    ;
```

Add a new `save_stmt` rule (after `stop_stmt`, line 148):

```
save_stmt:
    TOK_SAVE ':' TOK_IDENT TOK_ARROW TOK_STRING_LIT ';'
    {
        $$ = (void *)ast_save_new($3, $5);
    }
    ;
```

- [ ] **Step 5: Run to verify pass**

Run: `cd build && make test_parser 2>&1 | tail -5 && ./test_parser`
Expected: PASS. If Bison reports shift/reduce conflicts, they are pre-existing-style; verify the three new tests pass and existing parser tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/parser.y tests/unit/test_parser.c
git commit -m "feat(parser): load(\"path\") init, dim = *, save: directive"
```

---

## Task 4: Semantic — record + validate inferred dims, loaded tensors, save target

**Files:**
- Modify: `src/semantic.h`, `src/semantic.c`
- Test: `tests/unit/test_semantic.c`, and negative tests in Task 9

- [ ] **Step 1: Extend the symbol table**

In `src/semantic.h`:

Add `is_inferred` to `DimEntry`:

```c
typedef struct {
    char *name;
    int size;
    int is_inferred;   /* `dim N = *;` — size resolved at runtime from a load */
} DimEntry;
```

Add load tracking to `TensorEntry` (after `has_init`):

```c
    int has_init;            /* 1 if an `init` statement provides this tensor's values */
    int loaded_from_file;    /* 1 if init is `load("...")` */
```

Add save-directive fields to `SymbolTable` (after `num_intermediates`):

```c
    char *save_tensor;   /* tensor named in `save: X -> "..."` (borrowed), or NULL */
    char *save_path;     /* path in the save directive (borrowed), or NULL */
```

- [ ] **Step 2: Write failing semantic tests**

Add to `tests/unit/test_semantic.c` (register in `main()`). Use the file's existing helper for analyzing a source string (read the top of the file first: `sed -n '1,40p' tests/unit/test_semantic.c` to see whether it exposes an `analyze_source(...)` helper or builds via `yy_scan_string`+`yyparse`+`semantic_analyze`; mirror it).

```c
TEST(test_semantic_records_inferred_dim) {
    SemanticResult r = analyze_source(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n");
    ASSERT_INT_EQ(0, r.error_count);
    /* dim N is inferred and is pinned by loaded tensor G -> no error */
    semantic_result_free(&r);
}

TEST(test_semantic_unpinned_inferred_dim_errors) {
    SemanticResult r = analyze_source(
        "dim N = *;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0];\n"
        "Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);\n");
    ASSERT_TRUE(r.error_count > 0);   /* N inferred but no loaded tensor pins it */
    semantic_result_free(&r);
}

TEST(test_semantic_save_unknown_tensor_errors) {
    SemanticResult r = analyze_source(
        "dim N = 2;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0];\n"
        "Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);\n"
        "save: Q -> \"out.mtx\";\n");   /* Q is not a declared tensor */
    ASSERT_TRUE(r.error_count > 0);
    semantic_result_free(&r);
}
```

> If `analyze_source` does not exist in the test file, add a small static helper at the top mirroring the parse+analyze sequence already used by the other tests there.

- [ ] **Step 3: Run to verify failure**

Run: `cd build && make test_semantic 2>&1 | tail -20`
Expected: FAIL — inferred dim and save validations not implemented; `loaded_from_file` not set.

- [ ] **Step 4: Implement recording + validation**

In `src/semantic.c`:

1. **Inferred dims:** where dim declarations are processed (search `NODE_DIM_DECL`), copy `is_inferred` into the `DimEntry`. For inferred dims, leave `size` as 0 (resolved at runtime).

2. **Loaded tensors:** where `NODE_INIT` sets `has_init` (search `has_init`), also set `loaded_from_file = stmt->data.init.is_load` on the matching `TensorEntry`.

3. **Save directive:** add a pass over statements for `NODE_SAVE`; set `symtab->save_tensor`/`symtab->save_path` (borrow the AST strings). Error if the named tensor is not found via `symtab_find_tensor`:

```c
/* Save directive: target must be a declared tensor. */
for (int i = 0; i < program->data.program.count; i++) {
    AstNode *s = program->data.program.statements[i];
    if (s->type != NODE_SAVE) continue;
    if (!symtab_find_tensor(st, s->data.save.tensor_name)) {
        add_error(res, "save: unknown tensor '%s'", s->data.save.tensor_name);
        continue;
    }
    st->save_tensor = s->data.save.tensor_name;  /* borrowed */
    st->save_path   = s->data.save.path;          /* borrowed */
}
```

4. **Unpinned inferred dim:** after tensors and dims are recorded, every inferred dim must be a `dim_name` of at least one tensor whose `loaded_from_file` is set:

```c
/* Every inferred dim must be pinned by a loaded tensor. */
for (int d = 0; d < st->num_dims; d++) {
    if (!st->dims[d].is_inferred) continue;
    int pinned = 0;
    for (int t = 0; t < st->num_tensors && !pinned; t++) {
        if (!st->tensors[t].loaded_from_file) continue;
        for (int k = 0; k < st->tensors[t].num_ranks; k++) {
            const char *dn = st->tensors[t].ranks[k].dim_name;
            if (dn && strcmp(dn, st->dims[d].name) == 0) { pinned = 1; break; }
        }
    }
    if (!pinned)
        add_error(res, "inferred dim '%s' is not pinned by any load(...) tensor",
                  st->dims[d].name);
}
```

> Use the file's existing error-append function. Search for how other errors are added (e.g. `grep -n "error_count\|errors\[" src/semantic.c`) and match that call (shown here as `add_error`). If existing dim-size resolution asserts a positive size, guard it with `if (!dims[d].is_inferred)`.

- [ ] **Step 5: Run to verify pass**

Run: `cd build && make test_semantic 2>&1 | tail -5 && ./test_semantic`
Expected: PASS. Run full suite to confirm no regressions: `ctest --output-on-failure 2>&1 | tail -15`.

- [ ] **Step 6: Commit**

```bash
git add src/semantic.h src/semantic.c tests/unit/test_semantic.c
git commit -m "feat(semantic): record inferred dims/loaded tensors/save; validate pinning + save target"
```

---

## Task 5: IR — inferred dim, load-path init, SAVE instr, dim names in alloc

**Files:**
- Modify: `src/ir.h`, `src/ir.c`
- Test: `tests/unit/test_ir.c`

This task also threads dim **names** into `IRTensorAlloc` (needed so codegen can allocate inferred-dim tensors by name instead of by size).

- [ ] **Step 1: Extend IR structs**

In `src/ir.h`:

Add `is_inferred` to `IRDimDef`:

```c
typedef struct {
    char *name;
    int size;
    int is_inferred;   /* size unknown at compile time; derived in run() */
} IRDimDef;
```

Add dim names to `IRTensorAlloc`:

```c
typedef struct {
    char *name;
    int dim_sizes[8];
    char *dim_names[8];   /* per-axis dim name (NULL if anonymous); owned */
    int ndims;
    double empty_val;
} IRTensorAlloc;
```

Add load fields to `IRTensorInit`:

```c
typedef struct {
    char *name;
    LiteralData *data;  /* borrowed pointer, not owned (NULL when is_load) */
    int is_load;        /* 1 if `load("path")` */
    char *path;         /* owned; NULL when literal */
} IRTensorInit;
```

Add an `IR_SAVE` type to the `IRType` enum (after `IR_GEN_LOOP_END`) and a struct + union member:

```c
typedef struct {
    char *tensor_name;   /* owned */
    char *path;          /* owned */
} IRSave;
```

Add to the `IRInstr` union:

```c
        IRGenLoop gen_loop;
        IRSave save;
```

- [ ] **Step 2: Write failing IR tests**

Add to `tests/unit/test_ir.c` (mirror its existing lowering-and-inspect pattern; read `sed -n '1,50p' tests/unit/test_ir.c` first). Register in `main()`.

```c
TEST(test_ir_load_init_carries_path) {
    IRProgram *ir = lower_source(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n");
    int found = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_TENSOR_INIT &&
            strcmp(ir->instrs[i].data.tensor_init.name, "G") == 0) {
            ASSERT_INT_EQ(1, ir->instrs[i].data.tensor_init.is_load);
            ASSERT_STR_EQ("g.mtx", ir->instrs[i].data.tensor_init.path);
            found = 1;
        }
    }
    ASSERT_TRUE(found);
    ir_free(ir);
}

TEST(test_ir_inferred_dim_flag) {
    IRProgram *ir = lower_source(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n");
    int seen = 0;
    for (int i = 0; i < ir->count; i++)
        if (ir->instrs[i].type == IR_DIM_DEF &&
            strcmp(ir->instrs[i].data.dim_def.name, "N") == 0) {
            ASSERT_INT_EQ(1, ir->instrs[i].data.dim_def.is_inferred);
            seen = 1;
        }
    ASSERT_TRUE(seen);
    ir_free(ir);
}

TEST(test_ir_save_instruction) {
    IRProgram *ir = lower_source(
        "dim N = 2;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0];\n"
        "Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);\n"
        "save: Z -> \"out.mtx\";\n");
    int found = 0;
    for (int i = 0; i < ir->count; i++)
        if (ir->instrs[i].type == IR_SAVE) {
            ASSERT_STR_EQ("Z", ir->instrs[i].data.save.tensor_name);
            ASSERT_STR_EQ("out.mtx", ir->instrs[i].data.save.path);
            found = 1;
        }
    ASSERT_TRUE(found);
    ir_free(ir);
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cd build && make test_ir 2>&1 | tail -20`
Expected: FAIL — fields/instr not populated.

- [ ] **Step 4: Implement lowering**

In `src/ir.c`:

1. `emit_dim_defs` (line 48): copy `is_inferred`:

```c
instr->data.dim_def.name = dup_str(symtab->dims[i].name);
instr->data.dim_def.size = symtab->dims[i].size;
instr->data.dim_def.is_inferred = symtab->dims[i].is_inferred;
```

2. Tensor alloc lowering (around line 72): populate `dim_names` per axis from the tensor's ranks. Find the `TensorEntry` for the tensor and copy `ranks[d].dim_name` (dup), leaving generational axis as NULL:

```c
for (int d = 0; d < instr->data.tensor_alloc.ndims; d++) {
    const char *dn = te->ranks[d].dim_name;   /* te = matching TensorEntry */
    instr->data.tensor_alloc.dim_names[d] = dn ? dup_str(dn) : NULL;
}
```

> Use whatever local already holds the tensor's `TensorEntry` in that function; if none, look it up with `symtab_find_tensor(symtab, name)`.

3. `emit_tensor_inits` (line 84): carry the load flag/path:

```c
IRInstr *instr = ir_emit(ir, IR_TENSOR_INIT);
instr->data.tensor_init.name = dup_str(stmt->data.init.tensor_name);
instr->data.tensor_init.data = stmt->data.init.data;   /* borrowed (NULL when load) */
instr->data.tensor_init.is_load = stmt->data.init.is_load;
instr->data.tensor_init.path = stmt->data.init.is_load
                             ? dup_str(stmt->data.init.path) : NULL;
```

4. Emit a `IR_SAVE` instruction from the symtab save fields (near the end of `ir_lower`, after the gen-loop / main lowering):

```c
if (symtab->save_tensor) {
    IRInstr *s = ir_emit(ir, IR_SAVE);
    s->data.save.tensor_name = dup_str(symtab->save_tensor);
    s->data.save.path = dup_str(symtab->save_path);
}
```

5. `ir_free` (line 249): free new owned strings:
   - `IR_TENSOR_ALLOC`: free each non-NULL `dim_names[d]`.
   - `IR_TENSOR_INIT`: free `path` (NULL-safe).
   - add `IR_SAVE` case freeing `tensor_name` and `path`.

6. `ir_dump`: add an `IR_SAVE` line and show `is_load`/path on `TENSOR_INIT`:

```c
case IR_SAVE:
    fprintf(out, "SAVE(%s -> %s)\n",
            instr->data.save.tensor_name, instr->data.save.path);
    break;
```

In the `IR_TENSOR_INIT` dump case (line 329), distinguish load:

```c
case IR_TENSOR_INIT:
    if (instr->data.tensor_init.is_load)
        fprintf(out, "TENSOR_INIT(%s, load=%s)\n",
                instr->data.tensor_init.name, instr->data.tensor_init.path);
    else
        fprintf(out, "TENSOR_INIT(%s, ...)\n", instr->data.tensor_init.name);
    break;
```

- [ ] **Step 5: Run to verify pass**

Run: `cd build && make test_ir 2>&1 | tail -5 && ./test_ir`
Expected: PASS. Full suite: `ctest --output-on-failure 2>&1 | tail -15` — existing tests still pass (dim_names is additive; codegen still uses size matching until Task 6).

- [ ] **Step 6: Commit**

```bash
git add src/ir.h src/ir.c tests/unit/test_ir.c
git commit -m "feat(ir): inferred-dim flag, load-path init, SAVE instr, dim names in alloc"
```

---

## Task 6: Codegen — runtime helpers, inferred-dim derivation, load/save in __main__, argv

**Files:**
- Modify: `src/codegen.c`
- Test: `tests/unit/test_codegen.c`

This is the largest task. Sub-steps build it incrementally.

- [ ] **Step 1: Write failing codegen tests**

Add to `tests/unit/test_codegen.c` (mirror its existing "generate and substring-assert" pattern; read `sed -n '1,50p' tests/unit/test_codegen.c`). Register in `main()`.

```c
TEST(test_codegen_emits_load_helper) {
    char *py = generate_source(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n");
    ASSERT_TRUE(strstr(py, "def _edgecc_load(") != NULL);
    ASSERT_TRUE(strstr(py, "_edgecc_load(\"g.mtx\")") != NULL);
    free(py);
}

TEST(test_codegen_inferred_dim_from_input_shape) {
    char *py = generate_source(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n");
    /* N derived inside run() from an input's shape */
    ASSERT_TRUE(strstr(py, "N = len(") != NULL);
    /* inferred dim is NOT a module constant `N = <int>` */
    ASSERT_TRUE(strstr(py, "\nN = 0\n") == NULL);
    free(py);
}

TEST(test_codegen_emits_save) {
    char *py = generate_source(
        "dim N = 2;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0];\n"
        "Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);\n"
        "save: Z -> \"out.mtx\";\n");
    ASSERT_TRUE(strstr(py, "def _edgecc_save(") != NULL);
    ASSERT_TRUE(strstr(py, "_edgecc_save(Z, \"out.mtx\")") != NULL);
    free(py);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd build && make test_codegen 2>&1 | tail -20`
Expected: FAIL — helpers/derivation/save not emitted.

- [ ] **Step 3: Detect file-I/O use and emit runtime helpers**

In `codegen_python` (after the `import math` line, `codegen.c:409`), scan the IR for any `IR_TENSOR_INIT` with `is_load` or any `IR_SAVE`. If present, emit the helper block:

```c
int uses_fileio = 0;
for (int i = 0; i < ir->count; i++) {
    if ((ir->instrs[i].type == IR_TENSOR_INIT && ir->instrs[i].data.tensor_init.is_load) ||
        ir->instrs[i].type == IR_SAVE) { uses_fileio = 1; break; }
}
if (uses_fileio) {
    cb_append(&cb,
        "def _edgecc_load(path):\n"
        "    if path.endswith(\".mtx\"):\n"
        "        try:\n"
        "            from scipy.io import mmread\n"
        "        except ImportError:\n"
        "            raise ImportError(\"edgecc load/save requires scipy: pip install scipy\")\n"
        "        return mmread(path).toarray().tolist()\n"
        "    raise ValueError(\"edgecc load: unsupported file extension for %r (supported: .mtx)\" % path)\n"
        "\n"
        "def _edgecc_save(tensor, path):\n"
        "    if path.endswith(\".mtx\"):\n"
        "        try:\n"
        "            from scipy.io import mmwrite\n"
        "            import numpy as np\n"
        "        except ImportError:\n"
        "            raise ImportError(\"edgecc load/save requires scipy: pip install scipy\")\n"
        "        mmwrite(path, np.array(tensor))\n"
        "        return\n"
        "    raise ValueError(\"edgecc save: unsupported file extension for %r (supported: .mtx)\" % path)\n"
        "\n");
}
```

- [ ] **Step 4: Skip module constants for inferred dims**

In the module-level dim-constant loop (`codegen.c:412`), skip inferred dims:

```c
for (int i = 0; i < ir->count; i++) {
    if (ir->instrs[i].type == IR_DIM_DEF &&
        !ir->instrs[i].data.dim_def.is_inferred) {
        cb_appendf(&cb, "%s = %d\n",
            ir->instrs[i].data.dim_def.name,
            ir->instrs[i].data.dim_def.size);
    }
}
```

- [ ] **Step 5: Derive inferred dims at the top of run()**

Immediately after emitting `def run(...):` and setting `cb.indent = 1` (`codegen.c:487`), emit a `len()` derivation for each inferred dim, found from the first input tensor that uses it. Add a helper above `codegen_python`:

```c
/* For an inferred dim `dimname`, find the first input tensor alloc that has it
   on some axis and emit `dimname = len(input)` (axis 0) or `len(input[0])`. */
static void emit_inferred_dim_derivation(CodeBuf *cb, IRProgram *ir,
                                         const char *dimname,
                                         TensorNameList *inputs) {
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_TENSOR_ALLOC) continue;
        IRTensorAlloc *a = &ir->instrs[i].data.tensor_alloc;
        int is_input = 0;
        for (int j = 0; j < inputs->count; j++)
            if (strcmp(a->name, inputs->names[j]) == 0) { is_input = 1; break; }
        if (!is_input) continue;
        for (int d = 0; d < a->ndims; d++) {
            if (a->dim_names[d] && strcmp(a->dim_names[d], dimname) == 0) {
                cb_indent(cb);
                if (d == 0) cb_appendf(cb, "%s = len(%s)\n", dimname, a->name);
                else        cb_appendf(cb, "%s = len(%s%s)\n", dimname, a->name,
                                       "[0]");   /* d==1: one [0]; deeper axes unused in v2 */
                return;
            }
        }
    }
}
```

Then, after `cb.indent = 1;` at line 487, call it for each inferred dim:

```c
for (int i = 0; i < ir->count; i++) {
    if (ir->instrs[i].type == IR_DIM_DEF &&
        ir->instrs[i].data.dim_def.is_inferred) {
        emit_inferred_dim_derivation(&cb, ir, ir->instrs[i].data.dim_def.name, &inputs);
    }
}
```

> v2 tensors are at most 2D, so axis 0 → `len(x)` and axis 1 → `len(x[0])` cover all cases. `emit_inferred_dim_derivation` must be declared/defined after `TensorNameList` (line 250) and after `collect_io`, but before `codegen_python` — place it just before `codegen_python` and forward-declare `TensorNameList` is unnecessary since it's defined earlier at line 250.

- [ ] **Step 6: Make emit_alloc prefer dim names**

In `emit_alloc` (`codegen.c:125`), before the size-matching block, prefer `alloc->dim_names[d]` when non-NULL. Replace the per-axis name resolution so that a present `dim_names[d]` wins:

```c
for (int d = 0; d < alloc->ndims; d++) {
    dim_names[d] = alloc->dim_names[d];   /* prefer the threaded name */
    if (dim_names[d]) continue;           /* fall back to size matching only if absent */
    for (int j = 0; j < ndims_found; j++) {
        if (!all_dims[j].used && all_dims[j].size == alloc->dim_sizes[d]) {
            dim_names[d] = all_dims[j].name;
            all_dims[j].used = 1;
            break;
        }
    }
}
```

This makes inferred-dim allocations emit `[[0.0] * N for _ in range(N)]` (names) instead of relying on a numeric size that does not exist.

- [ ] **Step 7: Emit load instead of literal alloc in __main__; emit save; argv override**

In the `__main__` block (`codegen.c:633`):

(a) For **input allocs**, if the input is loaded-from-file, emit a `_edgecc_load(...)` call instead of `emit_alloc`. First build a quick lookup of load paths:

Add before the `__main__` input-alloc loop:

```c
/* Map input name -> load path (NULL if literal). */
const char *load_path_of[8] = {0};
for (int i = 0; i < ir->count; i++) {
    if (ir->instrs[i].type == IR_TENSOR_INIT &&
        ir->instrs[i].data.tensor_init.is_load) {
        for (int j = 0; j < inputs.count; j++)
            if (strcmp(ir->instrs[i].data.tensor_init.name, inputs.names[j]) == 0)
                load_path_of[j] = ir->instrs[i].data.tensor_init.path;
    }
}
/* Ordered list of file-backed input names for argv override + meta. */
```

In the input-alloc loop (line 637), skip the dense alloc for loaded inputs (they are assigned by the load loop below). Then after that loop, before the init loop, emit argv handling + loads:

```c
/* Positional argv overrides file-backed inputs, in inputs order. */
cb_indent(&cb); cb_append(&cb, "import sys\n");
cb_indent(&cb); cb_append(&cb, "_argv = sys.argv[1:]\n");
cb_indent(&cb); cb_append(&cb, "if _argv and _argv[0] in (\"-h\", \"--help\"):\n");
cb.indent++;
cb_indent(&cb);
cb_append(&cb, "print(\"usage: <prog> [");
/* list file-backed inputs in order */
{
    int first = 1;
    for (int j = 0; j < inputs.count; j++) {
        if (!load_path_of[j]) continue;
        if (!first) cb_append(&cb, " ");
        cb_appendf(&cb, "%s.mtx", inputs.names[j]);
        first = 0;
    }
}
cb_append(&cb, "]\"); sys.exit(0)\n");
cb.indent--;
/* For each file-backed input k (0-based among file inputs), use argv[k] or default. */
{
    int k = 0;
    for (int j = 0; j < inputs.count; j++) {
        if (!load_path_of[j]) continue;
        cb_indent(&cb);
        cb_appendf(&cb,
            "%s = _edgecc_load(_argv[%d]) if len(_argv) > %d else _edgecc_load(\"%s\")\n",
            inputs.names[j], k, k, load_path_of[j]);
        k++;
    }
    if (k > 0) {
        /* Validate count: too many args is an error. */
        cb_indent(&cb);
        cb_appendf(&cb,
            "if len(_argv) > %d:\n", k);
        cb.indent++;
        cb_indent(&cb);
        cb_appendf(&cb,
            "raise SystemExit(\"error: expected at most %d input path(s), got %%d\" %% len(_argv))\n", k);
        cb.indent--;
    }
}
```

(b) After the `run(...)`+`print` block (line 678), emit the save call if an `IR_SAVE` exists:

```c
for (int i = 0; i < ir->count; i++) {
    if (ir->instrs[i].type != IR_SAVE) continue;
    cb_indent(&cb);
    cb_appendf(&cb, "_edgecc_save(%s, \"%s\")\n",
               ir->instrs[i].data.save.tensor_name,
               ir->instrs[i].data.save.path);
}
```

(c) Extend `__edgecc_meta__` with the file-backed inputs list (after the `"output"` line, `codegen.c:477`):

```c
cb_append(&cb, "    \"file_inputs\": [");
{
    int first = 1;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_TENSOR_INIT &&
            ir->instrs[i].data.tensor_init.is_load) {
            if (!first) cb_append(&cb, ", ");
            cb_appendf(&cb, "\"%s\"", ir->instrs[i].data.tensor_init.name);
            first = 0;
        }
    }
}
cb_append(&cb, "],\n");
```

> Note on indices: the `load_path_of[8]` / `inputs.count` arrays assume ≤ 8 inputs, consistent with the rest of codegen's fixed `[8]` buffers.

- [ ] **Step 8: Run to verify pass**

Run: `cd build && make test_codegen 2>&1 | tail -5 && ./test_codegen`
Expected: PASS. Full suite: `ctest --output-on-failure 2>&1 | tail -20` — all existing tests pass (non-file programs are unaffected: `uses_fileio` is 0, no inferred dims, `emit_alloc` still resolves names/sizes as before).

- [ ] **Step 9: Commit**

```bash
git add src/codegen.c tests/unit/test_codegen.c
git commit -m "feat(codegen): file-io helpers, inferred-dim derivation, load/save + argv in __main__"
```

---

## Task 7: E2E — load → run → assert

**Files:**
- Create: `tests/e2e/load_mtx/input.edge`, `tests/e2e/load_mtx/graph.mtx`, `tests/e2e/load_mtx/expected_output.txt`
- Modify: `tests/e2e/run_e2e.py` (skip if scipy missing; run from the test dir so relative paths resolve)

- [ ] **Step 1: Add scipy-skip + working-directory to the e2e runner**

In `tests/e2e/run_e2e.py`, at the top of `run_test` (after computing `name`), skip `.mtx`-based tests when scipy is absent, and run the generated program with `cwd=test_dir` so `load("graph.mtx")` resolves relative to the test directory.

Add near the imports:

```python
try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
```

Detect file-I/O tests by a marker file and skip:

```python
    needs_scipy = os.path.exists(os.path.join(test_dir, "needs_scipy"))
    if needs_scipy and not HAVE_SCIPY:
        print(f"  SKIP {name}: scipy not installed")
        return None
```

Change the Python-run subprocess (line 28) to set the working directory:

```python
        result = subprocess.run(
            [sys.executable, tmp_py],
            capture_output=True, text=True, timeout=10, cwd=test_dir
        )
```

- [ ] **Step 2: Create the fixture (failing — directory not yet present)**

`tests/e2e/load_mtx/needs_scipy` (empty marker file):

```
```

`tests/e2e/load_mtx/graph.mtx` — a 4×4 adjacency matrix in Matrix Market coordinate form (1→2, 1→3, 3→4; 1-indexed):

```
%%MatrixMarket matrix coordinate real general
4 4 3
1 2 1.0
1 3 1.0
3 4 1.0
```

`tests/e2e/load_mtx/input.edge` — one push step of reachability from node 1 (dense), output is the neighbor set:

```
dim N = *;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor F[R=N]      : float, empty = 0.0;
init G = load("graph.mtx");
init F = [1.0, 0.0, 0.0, 0.0];
Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);
```

`tests/e2e/load_mtx/expected_output.txt` — neighbors of node 1 are nodes 2 and 3:

```
[0.0, 1.0, 1.0, 0.0]
```

> Verify the expected vector by reasoning: `F=[1,0,0,0]` selects row `s=0` (node 1); `G[0]` has 1.0 at columns 1 and 2 (nodes 2,3). `Z[d]=sum_s G[s][d]*F[s]` → `Z=[0,1,1,0]`. If the generated print formats differently (e.g. integer `0` vs `0.0`), adjust `expected_output.txt` to match the actual single run once (it is deterministic).

- [ ] **Step 3: Run the e2e suite**

Run: `cd build && ctest -R e2e_tests --output-on-failure 2>&1 | tail -20`
Expected: PASS `load_mtx` (or SKIP if scipy missing). If the printed format differs, update `expected_output.txt` to the observed deterministic output and re-run.

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/run_e2e.py tests/e2e/load_mtx
git commit -m "test(e2e): load a .mtx graph and run a dense push step"
```

---

## Task 8: E2E — load → run → save → re-read roundtrip

**Files:**
- Create: `tests/e2e/load_save_roundtrip/{input.edge,graph.mtx,needs_scipy,expected_output.txt}`

This test saves the output and re-reads it inside the EDGE program's own run by printing a verification line. Since the e2e harness only compares stdout, we verify the saved file by having the generated program print it back: the simplest robust check is to assert the normal stdout (the printed output tensor) AND rely on `_edgecc_save` not raising. A dedicated round-trip read is added as a tiny post-check script.

- [ ] **Step 1: Create the fixture**

`tests/e2e/load_save_roundtrip/needs_scipy` (empty marker).

`tests/e2e/load_save_roundtrip/graph.mtx`:

```
%%MatrixMarket matrix coordinate real general
4 4 3
1 2 1.0
1 3 1.0
3 4 1.0
```

`tests/e2e/load_save_roundtrip/input.edge`:

```
dim N = *;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor F[R=N]      : float, empty = 0.0;
init G = load("graph.mtx");
init F = [1.0, 0.0, 0.0, 0.0];
Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);
save: Z -> "result.mtx";
```

`tests/e2e/load_save_roundtrip/expected_output.txt`:

```
[0.0, 1.0, 1.0, 0.0]
```

- [ ] **Step 2: Run the e2e suite (verifies save does not raise + output matches)**

Run: `cd build && ctest -R e2e_tests --output-on-failure 2>&1 | tail -20`
Expected: PASS (the program prints the tensor and writes `result.mtx` without error). The generated program runs with `cwd=test_dir`, so `result.mtx` is written there.

- [ ] **Step 3: Add a round-trip read assertion**

Append to `tests/e2e/run_e2e.py` `run_test`, after the stdout comparison passes, a check that re-reads any saved `.mtx` for tests carrying a `roundtrip_check` marker:

```python
    if os.path.exists(os.path.join(test_dir, "roundtrip_check")) and HAVE_SCIPY:
        from scipy.io import mmread
        saved = os.path.join(test_dir, "result.mtx")
        if not os.path.exists(saved):
            print(f"  FAIL {name}: expected saved result.mtx not found")
            return False
        vals = mmread(saved).toarray().flatten().tolist()
        if vals != [0.0, 1.0, 1.0, 0.0]:
            print(f"  FAIL {name}: round-tripped values {vals} != expected")
            return False
        os.unlink(saved)
```

Create the marker `tests/e2e/load_save_roundtrip/roundtrip_check` (empty).

> Keep the hard-coded expected vector here; this test is specifically the round-trip fixture. Other tests without the marker skip this block.

- [ ] **Step 4: Run + commit**

Run: `cd build && ctest -R e2e_tests --output-on-failure 2>&1 | tail -20`
Expected: PASS.

```bash
git add tests/e2e/run_e2e.py tests/e2e/load_save_roundtrip
git commit -m "test(e2e): load -> run -> save -> re-read roundtrip"
```

---

## Task 9: Negative tests — unpinned inferred dim, save unknown tensor

**Files:**
- Create: `tests/negative/inferred_dim_unpinned/{input.edge,expected_errors.txt}`
- Create: `tests/negative/save_unknown_tensor/{input.edge,expected_errors.txt}`

- [ ] **Step 1: Create the fixtures**

`tests/negative/inferred_dim_unpinned/input.edge`:

```
dim N = *;
tensor F[R=N] : float, empty = 0.0;
init F = [1.0, 0.0];
Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);
```

`tests/negative/inferred_dim_unpinned/expected_errors.txt`:

```
inferred dim 'N' is not pinned by any load
```

`tests/negative/save_unknown_tensor/input.edge`:

```
dim N = 2;
tensor F[R=N] : float, empty = 0.0;
init F = [1.0, 0.0];
Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);
save: Q -> "out.mtx";
```

`tests/negative/save_unknown_tensor/expected_errors.txt`:

```
save: unknown tensor 'Q'
```

- [ ] **Step 2: Run the negative suite**

Run: `cd build && ctest -R negative_tests --output-on-failure 2>&1 | tail -20`
Expected: PASS — both new fixtures produce the expected error substrings. (The error strings must match the messages emitted in Task 4; adjust the `expected_errors.txt` text to be a substring of the actual stderr if wording differs.)

- [ ] **Step 3: Commit**

```bash
git add tests/negative/inferred_dim_unpinned tests/negative/save_unknown_tensor
git commit -m "test(negative): unpinned inferred dim and save of unknown tensor"
```

---

## Task 10: Docs + example

**Files:**
- Modify: `CLAUDE.md`
- Create: `examples/load_push.edge`, `examples/load_push.mtx`

- [ ] **Step 1: Add an example**

`examples/load_push.mtx`:

```
%%MatrixMarket matrix coordinate real general
4 4 3
1 2 1.0
1 3 1.0
3 4 1.0
```

`examples/load_push.edge`:

```
// One dense push step over a graph loaded from Matrix Market.
dim N = *;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor F[R=N]      : float, empty = 0.0;
init G = load("load_push.mtx");
init F = [1.0, 0.0, 0.0, 0.0];
Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);
save: Z -> "load_push_out.mtx";
```

- [ ] **Step 2: Verify the example end-to-end**

Run:
```bash
cd build && make edgecc && cd ../examples && \
  ../build/edgecc load_push.edge -o /tmp/load_push.py && \
  python3 /tmp/load_push.py && ls load_push_out.mtx && rm -f load_push_out.mtx
```
Expected: prints `[0.0, 1.0, 1.0, 0.0]`, writes and lists `load_push_out.mtx`. (Requires scipy.)

- [ ] **Step 3: Document in CLAUDE.md**

In the **EDGE Syntax (v2)** section, add after the built-in operators paragraph:

```markdown
### File input/output

A tensor may be initialized from a file instead of inline literals, and the
output may be written to a file:

```
dim N = *;                                  // dimension inferred from the loaded file
tensor G[R=N, C=N] : float, empty = 0.0;
init G = load("graph.mtx");                 // dense-materialized from Matrix Market
Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);
save: Z -> "result.mtx";                    // written after run()
```

`load("path")` reads a Matrix Market (`.mtx`) file and **densifies** it into the
normal nested-list representation (the compute stays dense). `dim N = *;` infers
the size from a loaded tensor at runtime; an inferred dim must be pinned by at
least one `load(...)` tensor. `save: T -> "path";` writes tensor `T` after
`run()`. Generated programs accept positional argv overrides for the
file-backed inputs (in `__edgecc_meta__["file_inputs"]` order); run with `-h`
for usage.

**Runtime dependency:** `load`/`save` require `scipy` (`pip install scipy`),
lazily imported only when used. The e2e harness skips `.mtx` tests when scipy is
absent.
```

In the **Generated Python ABI** section, note that `__edgecc_meta__` may include
a `"file_inputs"` list and that `_edgecc_load`/`_edgecc_save` helpers are emitted
when file I/O is used.

In **v2 Scope Limitations**, append to the sparse-storage item the note: *sparse
end-to-end compute (CSR/CSC/COO iteration) is deferred; file loading currently
densifies on load — see `docs/superpowers/specs/2026-06-07-edgecc-file-io-design.md`.*

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md examples/load_push.edge examples/load_push.mtx
git commit -m "docs: document load/save file I/O; add load_push example"
```

---

## Task 11: Full verification

- [ ] **Step 1: Clean build + full suite**

Run:
```bash
rm -rf build && mkdir build && cd build && cmake .. >/dev/null && make 2>&1 | tail -5 && ctest --output-on-failure 2>&1 | tail -30
```
Expected: everything builds; all unit, e2e, negative, and iterative tests pass (file-I/O e2e tests SKIP if scipy is not installed).

- [ ] **Step 2: Confirm non-file programs are byte-identical**

Run (spot-check a pre-existing example is unchanged by the refactor):
```bash
cd /Users/porumbes/.../edgecc && ./build/edgecc examples/gemm.edge -o /tmp/gemm_new.py && diff <(git show HEAD~10:examples/gemm.py 2>/dev/null || true) /tmp/gemm_new.py; echo "compare gemm output manually if needed"
```
Expected: dense, non-file programs emit the same compute (dim names already matched names before; the `dim_names` threading is additive). Any diff should be limited to additive lines, never changed compute.

- [ ] **Step 3: Final commit (if any doc/cleanup remains)**

```bash
git status   # should be clean
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- `load("path")` builtin → Tasks 1–3, 5, 6. ✓
- Generic, extension-dispatch loader → Task 6 Step 3 (`_edgecc_load` dispatches on `.endswith(".mtx")`). ✓
- `load`/`save` naming → throughout. ✓
- Inferred dims (`dim N = *`) → Tasks 1–6 (derive in `run()`, Task 6 Step 5). ✓
- Shared-dim disagreement = runtime error → emergent from `len()`-derived dims + dense indexing raising `IndexError`; **note:** the spec wanted a *clear* `ValueError`. See Gap 1 below.
- `save:` directive → Tasks 1–6. ✓
- argv override of file-backed inputs, `-h`, count validation → Task 6 Step 7. ✓
- Errors (compile vs runtime) → Task 4 (compile), Task 6 helpers (runtime ImportError/ValueError). ✓
- scipy lazy import + ctest skip + docs → Tasks 6, 7, 10. ✓
- Sparse-compute deferral note in CLAUDE.md → Task 10 Step 3. ✓
- VS Code / build-system dep management / non-mtx formats → deferred (spec §Deferred); intentionally no tasks.

**Gap 1 (surface to user):** The spec's "two loads disagree on a shared dim → clear `ValueError` naming both tensors" is only partially met — inferred dims are each derived from the *first* input that uses them, so a second input of the wrong size would surface as a Python `IndexError` deep in the loop, not a friendly message. A clean fix is a runtime shape-assert emitted at the top of `run()` for every input sharing an inferred dim. This is a small added sub-step in Task 6 if desired; flagged for the execution decision rather than silently dropped.

**Placeholder scan:** No TBD/TODO; all code blocks concrete. Test fixtures include exact expected outputs with the reasoning to verify them. ✓

**Type/name consistency:** `is_inferred`, `is_load`, `path`, `dim_names`, `IR_SAVE`/`IRSave`, `_edgecc_load`/`_edgecc_save`, `load_path_of`, `file_inputs` used consistently across AST→IR→codegen→tests. ✓
