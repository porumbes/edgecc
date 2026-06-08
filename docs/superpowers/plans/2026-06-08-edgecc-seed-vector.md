# Seed-Vector Construct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `seed(source=I, value=V)` init form that builds a length-N (inferred) source vector filled with the tensor's `empty` value, unblocking BFS/SSSP on loaded graphs.

**Architecture:** A new init kind threaded through lex/parse/AST → semantic → IR → codegen `__main__`. It is materialized as Python in the `__main__` block (fill with the tensor's empty, set the source index), sized from the loaded pinning tensor. The einsum/loop lowering and `run()` body are untouched. Added as an additive `is_seed` flag (a literal init is `is_load==0 && is_seed==0`) to keep the diff small.

**Tech Stack:** C (C99, Flex/Bison 2.3-compatible), Python 3 (generated code + tests), scipy (test oracle), CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-06-08-edgecc-seed-vector-design.md`

---

## File Structure

**Modify:**
- `src/lexer.l` — `seed` keyword → `TOK_SEED`
- `src/parser.y` — `TOK_SEED` token + `init = seed(source=INT, value=number)` production
- `src/ast.h` — `InitData` gains `is_seed`, `seed_source`, `seed_value`
- `src/ast.c` — `ast_init_seed_new` constructor (destructor needs no change)
- `src/semantic.c` — validate seed tensor is rank-1, `source >= 0`
- `src/ir.h` — `IRTensorInit` gains `is_seed`, `seed_source`, `seed_value`
- `src/ir.c` — `emit_tensor_inits` copies the seed fields
- `src/codegen.c` — `emit_seed_init` helper + dispatch in the `__main__` init loop
- `tests/unit/test_lexer.c`, `tests/unit/test_parser.c`, `tests/unit/test_semantic.c`, `tests/unit/test_codegen.c` — unit tests
- `CLAUDE.md` — document `seed(...)`

**Create:**
- `tests/e2e/seed_sssp_karate/` — `input.edge`, `args`, `expected_output.txt`, `needs_scipy`
- `tests/negative/seed_2d/` — `input.edge`, `expected_errors.txt`
- `tests/negative/seed_unpinned/` — `input.edge`, `expected_errors.txt`

---

## Task 1: Surface → AST (`seed` keyword, grammar, AST node)

**Files:**
- Modify: `src/lexer.l` (after the `"load"` rule, ~line 25)
- Modify: `src/parser.y` (token decl ~line 107; `init_stmt` production ~lines 235-244)
- Modify: `src/ast.h` (`InitData`, ~lines 101-106)
- Modify: `src/ast.c` (constructors, ~lines 45-63)
- Test: `tests/unit/test_lexer.c`, `tests/unit/test_parser.c`

- [ ] **Step 1: Write the failing lexer + parser tests**

In `tests/unit/test_lexer.c`, add (place beside the other token tests; register it in the test runner the same way existing `TEST(...)` cases are registered at the bottom of the file):

```c
TEST(test_lex_seed_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("seed");
    ASSERT_INT_EQ(TOK_SEED, yylex());
    yy_delete_buffer(buf);
}
```

In `tests/unit/test_parser.c`, add (and register it like the existing cases):

```c
TEST(test_parse_init_seed) {
    YY_BUFFER_STATE buf = yy_scan_string("init D = seed(source=2, value=0.0);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *init = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_INIT, init->type);
    ASSERT_STR_EQ("D", init->data.init.tensor_name);
    ASSERT_INT_EQ(0, init->data.init.is_load);
    ASSERT_INT_EQ(1, init->data.init.is_seed);
    ASSERT_INT_EQ(2, init->data.init.seed_source);
    ASSERT_TRUE(init->data.init.seed_value == 0.0);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd build && cmake .. >/dev/null && make test_lexer test_parser 2>&1 | tail -20`
Expected: compile failure — `TOK_SEED` undeclared and `is_seed`/`seed_source`/`seed_value` not members of `InitData`.

- [ ] **Step 3: Add the `seed` lexer rule**

In `src/lexer.l`, immediately after the `"load"` rule (line 25, `"load"  { return TOK_LOAD; }`), add:

```c
"seed"              { return TOK_SEED; }
```

- [ ] **Step 4: Extend the AST `InitData` and add the constructor**

In `src/ast.h`, replace the `InitData` struct (lines 101-106):

```c
typedef struct {
    char *tensor_name;
    int is_load;          /* 1 if `init X = load("path");` */
    int is_seed;          /* 1 if `init X = seed(source=I, value=V);` */
    LiteralData *data;    /* literal init (NULL when is_load/is_seed) */
    char *path;           /* file path (NULL unless is_load) */
    int seed_source;      /* source index (valid when is_seed) */
    double seed_value;    /* value at source (valid when is_seed) */
} InitData;
```

Add the constructor prototype near the other init constructors in `src/ast.h` (beside `ast_init_load_new`):

```c
AstNode *ast_init_seed_new(char *tensor_name, int source, double value);
```

In `src/ast.c`, after `ast_init_load_new` (line 63), add:

```c
AstNode *ast_init_seed_new(char *tensor_name, int source, double value) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_INIT;
    node->data.init.tensor_name = tensor_name;
    node->data.init.is_load = 0;
    node->data.init.is_seed = 1;
    node->data.init.data = NULL;
    node->data.init.path = NULL;
    node->data.init.seed_source = source;
    node->data.init.seed_value = value;
    return node;
}
```

(The destructor at `src/ast.c:188-192` needs no change — seed adds no owned heap pointers; `calloc` zeroes `is_seed` for the literal/load constructors automatically.)

- [ ] **Step 5: Add the parser token and grammar production**

In `src/parser.y`, add `TOK_SEED` to a `%token` line (next to `TOK_LOAD`, ~line 107):

```c
%token TOK_LOAD TOK_SAVE TOK_ARROW TOK_SEED
```

Add a `seed_value` helper nonterminal so `value=` accepts an int or float. Near the other `%type` declarations, add:

```c
%type <float_val> seed_value
```

In the `init_stmt` production (lines 235-244), add a third alternative before the closing `;`:

```c
init_stmt:
    TOK_INIT TOK_IDENT '=' nested_literal ';'
    {
        $$ = (void *)ast_init_new($2, (LiteralData *)$4);
    }
    | TOK_INIT TOK_IDENT '=' TOK_LOAD '(' TOK_STRING_LIT ')' ';'
    {
        $$ = (void *)ast_init_load_new($2, $6);
    }
    | TOK_INIT TOK_IDENT '=' TOK_SEED '(' TOK_IDENT '=' TOK_INT_LIT ',' TOK_IDENT '=' seed_value ')' ';'
    {
        if (strcmp($6, "source") != 0 || strcmp($10, "value") != 0) {
            yyerror("seed expects seed(source=<int>, value=<number>)");
            free($6); free($10); free($2);
            YYERROR;
        }
        free($6); free($10);
        $$ = (void *)ast_init_seed_new($2, $8, $12);
    }
    ;

seed_value:
      TOK_FLOAT_LIT { $$ = $1; }
    | TOK_INT_LIT   { $$ = (double)$1; }
    ;
```

(`$6`/`$10` are the `source`/`value` keyword idents — `strdup`'d by the lexer, not stored, so free them. `$2` is the tensor name, consumed by the constructor.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd build && make test_lexer test_parser 2>&1 | tail -5 && ./test_lexer && ./test_parser`
Expected: both binaries build (Bison may warn but not error) and all tests print PASS, including `test_lex_seed_keyword` and `test_parse_init_seed`.

- [ ] **Step 7: Commit**

```bash
git add src/lexer.l src/parser.y src/ast.h src/ast.c tests/unit/test_lexer.c tests/unit/test_parser.c
git commit -m "feat(seed): lex/parse seed(source=,value=) into INIT_SEED AST node"
```

---

## Task 2: Semantic validation (rank-1, non-negative source)

**Files:**
- Modify: `src/semantic.c` (`analyze_init`, ~lines 165-204)
- Test: `tests/unit/test_semantic.c`

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_semantic.c`, add (register like existing cases). Match the file's existing helper for running analysis and reading errors — the pattern below mirrors how other negative-path semantic tests in that file assert on `result.num_errors` / error strings; adjust the accessor names to the ones already used in the file if they differ:

```c
TEST(test_semantic_seed_rank1_ok) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor D[R=N]      : float, empty = inf;\n"
        "init G = load(\"g.mtx\");\n"
        "init D = seed(source=0, value=0.0);\n"
        "D[d] = G[s,d] . D[s] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.num_errors);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_semantic_seed_2d_rejected) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor D[R=N, C=N] : float, empty = inf;\n"
        "init G = load(\"g.mtx\");\n"
        "init D = seed(source=0, value=0.0);\n"
        "D[a,b] = G[a,b] . G[a,b] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_TRUE(sem.num_errors > 0);
    semantic_result_free(&sem);
    ast_free(ast);
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd build && make test_semantic 2>&1 | tail -5 && ./test_semantic`
Expected: `test_semantic_seed_2d_rejected` FAILS (no validation yet → `num_errors == 0`). (`test_semantic_seed_rank1_ok` may already pass.)

- [ ] **Step 3: Add the seed validation branch**

In `src/semantic.c`, in `analyze_init` (after the `te->loaded_from_file = d->is_load;` line and the existing `if (d->is_load) return;`, ~line 178), add:

```c
    if (d->is_seed) {
        if (te->num_ranks != 1)
            add_error(res, "seed: tensor '%s' must be 1-D", d->tensor_name);
        if (d->seed_source < 0)
            add_error(res, "seed: source index must be non-negative");
        return;
    }
```

(Place it after `if (d->is_load) return;` so seed inits skip the literal shape-checking below, exactly as load inits do. The existing inferred-dim pin check at `src/semantic.c:655-670` already errors if the seed's dim is unpinned — no change needed there.)

- [ ] **Step 4: Run to verify they pass**

Run: `cd build && make test_semantic 2>&1 | tail -5 && ./test_semantic`
Expected: all PASS, including both new tests.

- [ ] **Step 5: Commit**

```bash
git add src/semantic.c tests/unit/test_semantic.c
git commit -m "feat(seed): semantic validation — rank-1 seed tensor, non-negative source"
```

---

## Task 3: IR plumbing + codegen `__main__` materialization

**Files:**
- Modify: `src/ir.h` (`IRTensorInit`, ~lines 37-42)
- Modify: `src/ir.c` (`emit_tensor_inits`, ~lines 87-97)
- Modify: `src/codegen.c` (new `emit_seed_init` helper + dispatch in `__main__` init loop ~lines 905-920)
- Test: `tests/unit/test_codegen.c`

- [ ] **Step 1: Write the failing codegen test**

In `tests/unit/test_codegen.c`, add (register like existing cases):

```c
TEST(test_codegen_seed_init) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor D[R=N]      : float, empty = inf;\n"
        "init G = load(\"g.mtx\");\n"
        "init D = seed(source=2, value=0.0);\n"
        "D[d] = G[s,d] . D[s] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "D = [math.inf] * len(G)") != NULL);
    ASSERT_TRUE(strstr(code, "if not (0 <= 2 < len(D)):") != NULL);
    ASSERT_TRUE(strstr(code, "D[2] = 0.0") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd build && make test_codegen 2>&1 | tail -5 && ./test_codegen`
Expected: `test_codegen_seed_init` FAILS — the emitted code does not contain the seed fill/guard/assignment (the seed currently falls through to `emit_literal(NULL)` producing `D = ` with no RHS).

- [ ] **Step 3: Extend `IRTensorInit` and copy the seed fields**

In `src/ir.h`, replace `IRTensorInit` (lines 37-42):

```c
typedef struct {
    char *name;
    LiteralData *data;  /* borrowed pointer, not owned (NULL when is_load/is_seed) */
    int is_load;        /* 1 if `load("path")` */
    int is_seed;        /* 1 if `seed(source=, value=)` */
    char *path;         /* owned (NULL unless is_load) */
    int seed_source;
    double seed_value;
} IRTensorInit;
```

In `src/ir.c`, in `emit_tensor_inits` (after line 97, where `path` is copied), add:

```c
        instr->data.tensor_init.is_seed = stmt->data.init.is_seed;
        instr->data.tensor_init.seed_source = stmt->data.init.seed_source;
        instr->data.tensor_init.seed_value = stmt->data.init.seed_value;
```

- [ ] **Step 4: Add the `emit_seed_init` codegen helper**

In `src/codegen.c`, immediately after `emit_inferred_dim_derivation` (ends ~line 425), add:

```c
/* Materialize a seed(...) init in __main__: fill a length-N vector with the
   tensor's empty value, then set the source index to the seed value. N comes
   from the loaded tensor that pins the seed tensor's (single) dim, or from a
   fixed dim's constant size. */
static void emit_seed_init(CodeBuf *cb, IRProgram *ir, const char *name,
                           int source, double value) {
    double empty_val = 0.0;
    const char *dimname = NULL;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_TENSOR_ALLOC) continue;
        if (strcmp(ir->instrs[i].data.tensor_alloc.name, name) != 0) continue;
        empty_val = ir->instrs[i].data.tensor_alloc.empty_val;
        dimname = ir->instrs[i].data.tensor_alloc.dim_names[0];
        break;
    }

    int inferred = 0, dimsize = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_DIM_DEF) continue;
        if (dimname && strcmp(ir->instrs[i].data.dim_def.name, dimname) == 0) {
            inferred = ir->instrs[i].data.dim_def.is_inferred;
            dimsize = ir->instrs[i].data.dim_def.size;
            break;
        }
    }

    char lenexpr[128];
    snprintf(lenexpr, sizeof(lenexpr), "0");   /* defensive default */
    if (!inferred) {
        snprintf(lenexpr, sizeof(lenexpr), "%d", dimsize);
    } else {
        int found = 0;
        for (int i = 0; i < ir->count && !found; i++) {
            if (ir->instrs[i].type != IR_TENSOR_INIT) continue;
            if (!ir->instrs[i].data.tensor_init.is_load) continue;
            const char *pin = ir->instrs[i].data.tensor_init.name;
            for (int j = 0; j < ir->count && !found; j++) {
                if (ir->instrs[j].type != IR_TENSOR_ALLOC) continue;
                if (strcmp(ir->instrs[j].data.tensor_alloc.name, pin) != 0) continue;
                IRTensorAlloc *pa = &ir->instrs[j].data.tensor_alloc;
                for (int d = 0; d < pa->ndims; d++) {
                    if (pa->dim_names[d] && dimname &&
                        strcmp(pa->dim_names[d], dimname) == 0) {
                        if (d == 0) snprintf(lenexpr, sizeof(lenexpr), "len(%s)", pin);
                        else        snprintf(lenexpr, sizeof(lenexpr), "len(%s[0])", pin);
                        found = 1;
                        break;
                    }
                }
            }
        }
    }

    char *emptys = format_double_literal(empty_val);
    char *vals = format_double_literal(value);
    cb_indent(cb);
    cb_appendf(cb, "%s = [%s] * %s\n", name, emptys, lenexpr);
    cb_indent(cb);
    cb_appendf(cb, "if not (0 <= %d < len(%s)):\n", source, name);
    cb->indent++;
    cb_indent(cb);
    cb_appendf(cb,
        "raise ValueError(\"edgecc: seed source %d out of range [0, %%d) for '%s'\" %% len(%s))\n",
        source, name, name);
    cb->indent--;
    cb_indent(cb);
    cb_appendf(cb, "%s[%d] = %s\n", name, source, vals);
    free(emptys);
    free(vals);
}
```

- [ ] **Step 5: Dispatch seed inits in the `__main__` init loop**

In `src/codegen.c`, in the `__main__` init emission loop (~lines 905-920), find:

```c
        if (instr->data.tensor_init.is_load) continue;
        int is_input = 0;
        for (int j = 0; j < inputs.count; j++) {
            if (strcmp(instr->data.tensor_init.name, inputs.names[j]) == 0) { is_input = 1; break; }
        }
        if (!is_input) continue;
        cb_indent(&cb);
        cb_appendf(&cb, "%s = ", instr->data.tensor_init.name);
        emit_literal(&cb, instr->data.tensor_init.data);
        cb_append(&cb, "\n");
```

and insert the seed branch right after the `if (!is_input) continue;` line:

```c
        if (instr->data.tensor_init.is_seed) {
            emit_seed_init(&cb, ir, instr->data.tensor_init.name,
                           instr->data.tensor_init.seed_source,
                           instr->data.tensor_init.seed_value);
            continue;
        }
```

- [ ] **Step 6: Run to verify it passes**

Run: `cd build && make test_codegen 2>&1 | tail -5 && ./test_codegen`
Expected: all PASS, including `test_codegen_seed_init`.

- [ ] **Step 7: Commit**

```bash
git add src/ir.h src/ir.c src/codegen.c tests/unit/test_codegen.c
git commit -m "feat(seed): IR fields + __main__ materialization (fill-empty, range guard, source set)"
```

---

## Task 4: End-to-end SSSP-on-loaded-karate test

**Files:**
- Create: `tests/e2e/seed_sssp_karate/input.edge`
- Create: `tests/e2e/seed_sssp_karate/args`
- Create: `tests/e2e/seed_sssp_karate/expected_output.txt`
- Create: `tests/e2e/seed_sssp_karate/needs_scipy`

- [ ] **Step 1: Create the kernel, args, and marker**

`tests/e2e/seed_sssp_karate/input.edge`:

```
// Single-source shortest paths (Bellman-Ford) over a LOADED graph with a
// seed(...) source vector. karate is a pattern matrix (weights 1.0), so this
// computes BFS hop-distance from node 0 to every node. Validates seed-vector
// sizing to a loaded graph end-to-end.
//
// W keeps empty=0.0 (load densifies absent->0.0 so the intersect map excludes
// non-edges). T/D use empty=inf (the min reduce identity). D is seeded.
dim N = *;
tensor W[R=N, C=N]   : float, empty = 0.0;
tensor T[gen I, R=N] : float, empty = inf;
tensor D[gen I, R=N] : float, empty = inf;
init W = load("graph.mtx");
init D = seed(source=0, value=0.0);
T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);
stop: stable(D);
```

`tests/e2e/seed_sssp_karate/args`:

```
{DATA}/suitesparse/karate.mtx
```

```bash
touch tests/e2e/seed_sssp_karate/needs_scipy
```

- [ ] **Step 2: Build, generate expected output**

Run:
```bash
cd build && make >/dev/null && cd ..
./build/edgecc tests/e2e/seed_sssp_karate/input.edge -o /tmp/seed_karate.py
python3 /tmp/seed_karate.py data/suitesparse/karate.mtx > tests/e2e/seed_sssp_karate/expected_output.txt
cat tests/e2e/seed_sssp_karate/expected_output.txt
```
Expected: a 34-element list of floats (distances from node 0), e.g. `[0.0, 1.0, 1.0, 1.0, 1.0, ...]`. Node 0 has degree 16, so most entries are 1.0 or 2.0; no `inf` (karate is connected).

- [ ] **Step 3: Independently verify against a scipy BFS oracle**

```bash
python3 - <<'PY'
from scipy.io import mmread
from scipy.sparse.csgraph import shortest_path
import ast
m = mmread("data/suitesparse/karate.mtx")
a = (m.toarray() if hasattr(m, "toarray") else m)
# unweighted shortest path (BFS hop distance) from node 0
dist = shortest_path(a != 0, method="D", unweighted=True, indices=0)
want = [float(x) for x in dist]
got = ast.literal_eval(open("tests/e2e/seed_sssp_karate/expected_output.txt").read())
assert got == want, ("MISMATCH", [(j, got[j], want[j]) for j in range(len(got)) if got[j] != want[j]][:10])
print("oracle OK; SSSP/BFS distances from node 0 match; max hop =", max(want))
PY
```
Expected: `oracle OK; SSSP/BFS distances from node 0 match; max hop = 3.0` (karate's eccentricity from node 0). If this mismatches, do NOT overwrite — investigate the seed materialization or kernel before proceeding.

- [ ] **Step 4: Run through the harness**

Run: `python3 tests/e2e/run_e2e.py ./build/edgecc 2>&1 | grep -E "seed_sssp|passed"`
Expected: `PASS seed_sssp_karate` and the e2e count increased by 1.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/seed_sssp_karate
git commit -m "test(e2e): SSSP/BFS on loaded karate via seed(...) source vector"
```

---

## Task 5: Negative tests

**Files:**
- Create: `tests/negative/seed_2d/input.edge`, `tests/negative/seed_2d/expected_errors.txt`
- Create: `tests/negative/seed_unpinned/input.edge`, `tests/negative/seed_unpinned/expected_errors.txt`

- [ ] **Step 1: Create the rank-violation negative test**

`tests/negative/seed_2d/input.edge`:

```
dim N = *;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor D[R=N, C=N] : float, empty = inf;
init G = load("g.mtx");
init D = seed(source=0, value=0.0);
D[a,b] = G[a,b] . G[a,b] :: map(add, intersect) reduce(min, union);
```

`tests/negative/seed_2d/expected_errors.txt`:

```
seed: tensor 'D' must be 1-D
```

- [ ] **Step 2: Create the unpinned-dim negative test**

`tests/negative/seed_unpinned/input.edge`:

```
dim N = *;
tensor D[R=N] : float, empty = inf;
init D = seed(source=0, value=0.0);
D[d] = D[d] . D[d] :: map(add, union) reduce(add, union);
```

`tests/negative/seed_unpinned/expected_errors.txt`:

```
inferred dim 'N' is not pinned by any load
```

- [ ] **Step 3: Run the negative suite**

Run: `python3 tests/negative/run_negative.py ./build/edgecc 2>&1 | grep -E "seed_|passed"`
Expected: `PASS seed_2d` and `PASS seed_unpinned`.

- [ ] **Step 4: Commit**

```bash
git add tests/negative/seed_2d tests/negative/seed_unpinned
git commit -m "test(negative): seed on 2-D tensor and on unpinned inferred dim"
```

---

## Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md` (EDGE Syntax section — near the built-in operators / init description)

- [ ] **Step 1: Document `seed(...)` in CLAUDE.md**

In `CLAUDE.md`, in the EDGE Syntax (v2) area, after the "File input/output" subsection's "Real test graphs" paragraph (added by the SuiteSparse work) and before "### Visualization", add:

```markdown
**Seed/source vectors.** A 1-D tensor may be initialized with
`init D = seed(source=I, value=V);` — it builds a length-N vector (N inferred
from the loaded graph that pins the dim) filled with the tensor's own declared
`empty` value, with 0-based index `I` set to `V`. Filling with `empty` makes one
construct serve both families: SSSP uses `empty = inf` + `seed(source=0,
value=0.0)`; BFS uses `empty = 0.0` + `seed(source=0, value=1.0)`. This unblocks
BFS/SSSP on arbitrarily-sized loaded graphs (the source vector auto-sizes to the
graph instead of being a fixed-length literal). The seed is materialized in the
generated `__main__` and passed to `run()`; a runtime guard rejects an out-of-range
source index. Single-source only for now (the keyword syntax is reserved for a
future multi-source form); the seed tensor must be 1-D and its dim pinned by a
`load(...)`. See `tests/e2e/seed_sssp_karate/`.
```

- [ ] **Step 2: Run the full suite**

Run: `cd build && cmake .. >/dev/null && make >/dev/null && ctest --output-on-failure 2>&1 | tail -6`
Expected: 100% pass — all unit tests, `e2e_tests` (now including `seed_sssp_karate`), `negative_tests` (now including `seed_2d`, `seed_unpinned`), and all pre-existing targets.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document seed(...) source-vector construct in CLAUDE.md"
```

---

## Self-Review Notes

- **Spec coverage:** construct semantics → Task 1 (parse) + Task 3 (materialization); fill-with-empty → Task 3 (`emit_seed_init` uses the tensor's `empty_val`); keyword syntax / both-args-required → Task 1 grammar; rank-1 + non-negative source → Task 2; reuse of pin check → Task 2 (no change) + Task 5 (`seed_unpinned`); runtime range guard → Task 3 + Task 5; IR additive `is_seed` flag → Task 3; primary SSSP/BFS e2e on karate → Task 4; unit tests per stage → Tasks 1-3; negative tests → Task 5; docs → Task 6. All covered.
- **Additive flag, not enum:** `is_seed` added alongside `is_load`; literal init = both zero. No existing `is_load` read site changes behavior.
- **Type/name consistency:** `is_seed`/`seed_source`/`seed_value` named identically in `InitData` (ast.h), `IRTensorInit` (ir.h), and all read sites; constructor `ast_init_seed_new(char*, int, double)` matches the parser action call; `emit_seed_init(CodeBuf*, IRProgram*, const char*, int, double)` matches its dispatch call.
- **Non-circular verification:** the e2e expected output is cross-checked against `scipy.sparse.csgraph.shortest_path` before commit.
- **Bison 2.3:** new production uses the existing `(void *)` cast convention and `YYERROR` for the keyword-mismatch path; `seed_value` nonterminal avoids int/float ambiguity.
