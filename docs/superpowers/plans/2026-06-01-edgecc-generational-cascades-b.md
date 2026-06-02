# Generational Cascades (Sub-project B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend in-language generational iteration from a single Einsum (Sub-project A) to a **cascade** — a sequence of Einsums that run in order each generation, sharing one generational rank, with intermediate tensors flowing between them — so Bellman-Ford is expressible entirely in EDGE.

**Architecture:** The generation loop wraps the whole cascade. One **cross-generational** tensor (written at `i+1`, has `init`, double-buffered `prev`/`next`, returned) plus any number of **intra-generation intermediates** (written/read at `i`, no `init`, fresh each generation). Index classification becomes **per-Einsum**. No lexer/parser/AST changes — multiple Einsum statements and all tokens already parse.

**Tech Stack:** C99, Flex, Bison (2.3-compatible), CMake/CTest, generated Python 3. Spec: `docs/superpowers/specs/2026-06-01-edgecc-generational-cascades-b-design.md`.

**Conventions for this repo:**
- Build: `mkdir -p build && cd build && cmake .. && make`. All tests: `cd build && ctest --output-on-failure`. One unit binary: `./build/test_ir`.
- Unit tests use `tests/unit/test_helpers.h` macros; register new tests with `RUN_TEST(...)` in `main()`. Each `test_*.c` already has a `static AstNode *parse(const char *src)` helper — reuse it.
- e2e and negative tests are data-only directories auto-discovered by the Python harnesses (no CMake change to add a case).
- **Do not** add `Co-Authored-By` trailers to commits.
- **Build-green rule:** the global `free_indices`/`reduction_indices`/`is_iterative`/`gen_tensor`/`gen_rank_var`/`stop_watch` fields on `SymbolTable` stay; B adds per-Einsum classification *alongside* them. This keeps A's behavior and tests intact while the cascade machinery is layered on.

---

## File Structure

All changes are edits to existing files plus new test-data directories and one example. No new source files; no CMake changes.

- `src/semantic.h` / `src/semantic.c` — per-Einsum classification (`EinsumClass`), cascade role detection (cross-gen tensor, intermediates), `TensorEntry.has_init`, cascade validations.
- `src/ir.h` / `src/ir.c` — `IRGenLoop` carries the cross-gen tensor + intermediate fresh-alloc list; one gen loop wraps all Einsums; per-Einsum classification consumed; coordinate stripping for all gen-rank tensors, `prev`/`next` tags only for the cross-gen tensor.
- `src/codegen.c` — cascade I/O (params = inputs + cross-gen, return cross-gen, intermediates excluded); gen-loop scaffolding allocates cross-gen `next` + each intermediate fresh per generation.
- `tests/unit/test_semantic.c`, `test_ir.c`, `test_codegen.c` — new cascade unit tests; two A white-box assertions updated for the renamed IR struct fields.
- `tests/e2e/bellman_ford/`, `examples/bellman_ford.edge`, `tests/negative/cascade_*` — new data.
- `CLAUDE.md` — docs.

---

## Task 1: Semantic — per-Einsum classification + cascade roles

**Files:**
- Modify: `src/semantic.h`, `src/semantic.c`
- Test: `tests/unit/test_semantic.c`

- [ ] **Step 1: Add semantic.h fields**

In `src/semantic.h`, add a max constant near the other `#define`s (after `#define MAX_INDICES 16`):
```c
#define MAX_EINSUMS 16
```

Add `has_init` to `TensorEntry` (after `int gen_axis;`):
```c
    int has_init;   /* 1 if an `init` statement provides this tensor's values */
```

Add an `EinsumClass` struct immediately before `typedef struct { ... } SymbolTable;`:
```c
typedef struct {
    IndexEntry free_indices[MAX_INDICES];
    int num_free_indices;
    IndexEntry reduction_indices[MAX_INDICES];
    int num_reduction_indices;
} EinsumClass;
```

Add these fields to `SymbolTable` (after the existing `char *stop_watch;` line):
```c
    /* Per-Einsum index classification, in source order (cascade support). */
    EinsumClass einsum_class[MAX_EINSUMS];
    int num_einsums;
    /* Intra-generation intermediate tensors (gen-rank, written at offset i). */
    char *intermediates[MAX_TENSORS];   /* borrowed names */
    int num_intermediates;
```

- [ ] **Step 2: Write the failing semantic tests**

In `tests/unit/test_semantic.c`, add before `main()`:
```c
TEST(test_cascade_valid_bellman_ford) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init D = [0.0, inf, inf, inf];\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_INT_EQ(1, res.symtab->is_iterative);
    ASSERT_STR_EQ("D", res.symtab->gen_tensor);          /* cross-generational */
    ASSERT_STR_EQ("i", res.symtab->gen_rank_var);
    ASSERT_STR_EQ("D", res.symtab->stop_watch);
    ASSERT_INT_EQ(1, res.symtab->num_intermediates);     /* T */
    ASSERT_STR_EQ("T", res.symtab->intermediates[0]);
    ASSERT_INT_EQ(2, res.symtab->num_einsums);
    /* einsum 0 (relax): free = d, reduction = s */
    ASSERT_INT_EQ(1, res.symtab->einsum_class[0].num_free_indices);
    ASSERT_STR_EQ("d", res.symtab->einsum_class[0].free_indices[0].name);
    ASSERT_INT_EQ(1, res.symtab->einsum_class[0].num_reduction_indices);
    ASSERT_STR_EQ("s", res.symtab->einsum_class[0].reduction_indices[0].name);
    /* einsum 1 (combine): free = d, no reduction */
    ASSERT_INT_EQ(1, res.symtab->einsum_class[1].num_free_indices);
    ASSERT_STR_EQ("d", res.symtab->einsum_class[1].free_indices[0].name);
    ASSERT_INT_EQ(0, res.symtab->einsum_class[1].num_reduction_indices);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_two_cross_gen_tensors) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "tensor E[gen I, R=N] : float, empty = inf;\n"
        "init D = [0.0, inf, inf, inf];\n"
        "init E = [0.0, inf, inf, inf];\n"
        "D[i+1, d] = D[i, d] . E[i, d] :: map(min, union) reduce(min, union);\n"
        "E[i+1, d] = E[i, d] . D[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_intermediate_with_init) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init D = [0.0, inf, inf, inf];\n"
        "init T = [0.0, inf, inf, inf];\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_intermediate_read_before_write) {
    /* combine reads T before relax writes it (Einsums in wrong order) */
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init D = [0.0, inf, inf, inf];\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}
```
Register all four in `main()`:
```c
    RUN_TEST(test_cascade_valid_bellman_ford);
    RUN_TEST(test_cascade_two_cross_gen_tensors);
    RUN_TEST(test_cascade_intermediate_with_init);
    RUN_TEST(test_cascade_intermediate_read_before_write);
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd build && cmake .. && make test_semantic && ./test_semantic`
Expected: build error (new fields) or FAIL (cascade roles not yet populated).

- [ ] **Step 4: Mark tensors that have an init**

In `src/semantic.c`, in `analyze_init`, after the `TensorEntry *te = symtab_find_tensor(...)` null check and before building the spatial shape, mark the tensor:
```c
    te->has_init = 1;
```
(Place it right after the `if (te == NULL) { ... return; }` block.)

- [ ] **Step 5: Rework the generational block in `analyze_einsum`**

In `src/semantic.c`, change the signature of `analyze_einsum` to take an Einsum index:
```c
static void analyze_einsum(SemanticResult *res, AstNode *node, int eidx) {
```

Replace the entire `/* ---- Generational rank handling ---- */` block (the `char *gen_var = NULL;` declaration and the `{ ... }` block that follows it, through its closing brace just before `/* collect all indices and resolve sizes */`) with this cascade-aware version:
```c
    /* ---- Generational rank handling (cascade-aware) ---- */
    char *gen_var = NULL;   /* the index var bound to the gen rank, this einsum */
    {
        TensorEntry *tes[3]  = { out_te, in1_te, in2_te };
        TensorRef   *refs[3] = { &d->output, &d->input1, &d->input2 };

        /* Detect the shared generational rank variable. */
        for (int k = 0; k < 3; k++) {
            if (tes[k]->gen_axis >= 0 && tes[k]->gen_axis < refs[k]->num_vars) {
                st->is_iterative = 1;
                char *v = refs[k]->rank_vars[tes[k]->gen_axis];
                if (gen_var == NULL) gen_var = v;
                if (st->gen_rank_var == NULL) {
                    st->gen_rank_var = v;
                } else if (strcmp(st->gen_rank_var, v) != 0) {
                    add_error(res,
                        "generational rank variable mismatch: '%s' vs '%s'",
                        st->gen_rank_var, v);
                }
            }
        }

        /* Classify the OUTPUT's generational role by its effective write offset
           on the declared generational axis. A `+N` offset on the ref, if
           present, must be on that axis. (TensorRef.gen_axis is the coordinate
           where a `+N` literally appeared, -1 if none — not the declared axis.) */
        if (out_te->gen_axis >= 0) {
            if (d->output.gen_axis >= 0 && d->output.gen_axis != out_te->gen_axis) {
                add_error(res,
                    "generational tensor '%s' written with an offset on the wrong axis",
                    out_te->name);
            } else {
                int eff_off = (d->output.gen_axis == out_te->gen_axis)
                              ? d->output.gen_offset : 0;
                if (eff_off == 1) {
                    /* cross-generational write: D[i+1, ...] */
                    if (st->gen_tensor == NULL) {
                        st->gen_tensor = out_te->name;
                    } else if (strcmp(st->gen_tensor, out_te->name) != 0) {
                        add_error(res,
                            "more than one cross-generational tensor ('%s' and '%s'); "
                            "only one is supported", st->gen_tensor, out_te->name);
                    }
                } else if (eff_off == 0) {
                    /* intra-generation intermediate write: T[i, ...] */
                    int seen = 0;
                    for (int m = 0; m < st->num_intermediates; m++)
                        if (strcmp(st->intermediates[m], out_te->name) == 0) seen = 1;
                    if (!seen && st->num_intermediates < MAX_TENSORS)
                        st->intermediates[st->num_intermediates++] = out_te->name;
                } else {
                    add_error(res,
                        "generational tensor '%s' written at unsupported offset +%d "
                        "(only i and i+1 are allowed)", out_te->name, eff_off);
                }
            }
        }

        /* Generational inputs must be read at offset 0 (no +N). */
        for (int k = 1; k < 3; k++) {
            if (tes[k]->gen_axis >= 0 && refs[k]->gen_offset != 0) {
                add_error(res,
                    "generational tensor '%s' must be read at offset 0 "
                    "(e.g. %s[i, ...]), not with a +N offset",
                    tes[k]->name, tes[k]->name);
            }
        }
    }
```

- [ ] **Step 6: Store the per-Einsum classification**

Still in `analyze_einsum`, at the very end of the function (after the index-classification loop that fills `st->free_indices`/`st->reduction_indices`, just before the closing `}`), append:
```c
    /* Copy this Einsum's classification into the per-Einsum slot (cascade support).
       The global free/reduction fields remain populated for the single-Einsum path. */
    if (eidx >= 0 && eidx < MAX_EINSUMS) {
        EinsumClass *ec = &st->einsum_class[eidx];
        ec->num_free_indices = st->num_free_indices;
        for (int i = 0; i < st->num_free_indices; i++)
            ec->free_indices[i] = st->free_indices[i];
        ec->num_reduction_indices = st->num_reduction_indices;
        for (int i = 0; i < st->num_reduction_indices; i++)
            ec->reduction_indices[i] = st->reduction_indices[i];
        if (eidx + 1 > st->num_einsums) st->num_einsums = eidx + 1;
    }
```

- [ ] **Step 7: Add the cascade-ordering validator**

In `src/semantic.c`, add this function immediately after `analyze_stop`:
```c
/* Validate SSA ordering: an intra-generation intermediate must be written by an
   earlier Einsum before a later Einsum reads it. */
static void validate_cascade_ordering(SemanticResult *res, AstNode *program) {
    SymbolTable *st = res->symtab;
    if (!st->is_iterative) return;

    char *written[MAX_TENSORS];
    int nwritten = 0;
    ProgramData *pd = &program->data.program;
    for (int i = 0; i < pd->count; i++) {
        AstNode *stmt = pd->statements[i];
        if (stmt->type != NODE_EINSUM) continue;
        EinsumData *e = &stmt->data.einsum;

        TensorRef *ins[2] = { &e->input1, &e->input2 };
        for (int k = 0; k < 2; k++) {
            const char *nm = ins[k]->tensor_name;
            int is_inter = 0;
            for (int m = 0; m < st->num_intermediates; m++)
                if (strcmp(st->intermediates[m], nm) == 0) is_inter = 1;
            if (!is_inter) continue;
            int found = 0;
            for (int w = 0; w < nwritten; w++)
                if (strcmp(written[w], nm) == 0) found = 1;
            if (!found)
                add_error(res, "intermediate '%s' is read before it is written", nm);
        }

        const char *onm = e->output.tensor_name;
        for (int m = 0; m < st->num_intermediates; m++) {
            if (strcmp(st->intermediates[m], onm) == 0) {
                written[nwritten++] = st->intermediates[m];
                break;
            }
        }
    }
}
```

- [ ] **Step 8: Wire the index + new passes into `semantic_analyze`**

In `src/semantic.c`, in `semantic_analyze`, the statement-dispatch loop calls `analyze_einsum(&res, stmt)`. Add a counter and pass it. Replace:
```c
    ProgramData *pd = &program->data.program;
    for (int i = 0; i < pd->count; i++) {
        AstNode *stmt = pd->statements[i];
        switch (stmt->type) {
            case NODE_DIM_DECL:
                analyze_dim_decl(&res, stmt);
                break;
            case NODE_TENSOR_DECL:
                analyze_tensor_decl(&res, stmt);
                break;
            case NODE_INIT:
                analyze_init(&res, stmt);
                break;
            case NODE_EINSUM:
                analyze_einsum(&res, stmt);
                break;
            case NODE_STOP:
                analyze_stop(&res, stmt);
                break;
            default:
                add_error(&res, "unknown statement type %d", stmt->type);
                break;
        }
    }
```
with:
```c
    ProgramData *pd = &program->data.program;
    int einsum_index = 0;
    for (int i = 0; i < pd->count; i++) {
        AstNode *stmt = pd->statements[i];
        switch (stmt->type) {
            case NODE_DIM_DECL:
                analyze_dim_decl(&res, stmt);
                break;
            case NODE_TENSOR_DECL:
                analyze_tensor_decl(&res, stmt);
                break;
            case NODE_INIT:
                analyze_init(&res, stmt);
                break;
            case NODE_EINSUM:
                analyze_einsum(&res, stmt, einsum_index++);
                break;
            case NODE_STOP:
                analyze_stop(&res, stmt);
                break;
            default:
                add_error(&res, "unknown statement type %d", stmt->type);
                break;
        }
    }

    validate_cascade_ordering(&res, program);
```

- [ ] **Step 9: Add role-consistency finalize checks**

Still in `semantic_analyze`, the existing finalize block starts with `SymbolTable *st = res.symtab;` and the three iterative-consistency `if` checks. Immediately AFTER those three existing `if` checks (and before `return res;`), add the init-role checks:
```c
    /* The cross-generational tensor needs a generation-0 init; intermediates must not. */
    if (st->is_iterative && st->gen_tensor != NULL) {
        TensorEntry *cg = symtab_find_tensor(st, st->gen_tensor);
        if (cg != NULL && !cg->has_init) {
            add_error(&res,
                "cross-generational tensor '%s' requires an init (generation 0)",
                st->gen_tensor);
        }
    }
    for (int m = 0; m < st->num_intermediates; m++) {
        TensorEntry *te = symtab_find_tensor(st, st->intermediates[m]);
        if (te != NULL && te->has_init) {
            add_error(&res,
                "intermediate tensor '%s' must not have an init "
                "(it is recomputed each generation)", st->intermediates[m]);
        }
    }
```

- [ ] **Step 10: Run the tests to verify they pass**

Run: `cd build && make test_semantic && ./test_semantic`
Expected: the four new cascade tests PASS, and all existing semantic tests (including A's `test_generational_valid`) still PASS.

- [ ] **Step 11: Commit**

```bash
git add src/semantic.h src/semantic.c tests/unit/test_semantic.c
git commit -m "feat(semantic): cascade roles + per-Einsum index classification"
```

---

## Task 2: IR — one generational loop wrapping the cascade

**Files:**
- Modify: `src/ir.h`, `src/ir.c`
- Test: `tests/unit/test_ir.c` (and update one A white-box assertion)

- [ ] **Step 1: Restructure `IRGenLoop` to carry the cross-gen tensor + intermediates**

In `src/ir.h`, replace the existing `IRGenLoop` struct:
```c
typedef struct {
    char *tensor_name;       /* the generational tensor */
    int spatial_dims[8];     /* shape excluding the generational axis */
    int spatial_ndims;
    double empty_val;        /* for the fresh slice */
    int max_gen;             /* safety iteration cap */
} IRGenLoop;
```
with:
```c
typedef struct {
    char *name;
    int spatial_dims[8];     /* shape excluding the generational axis */
    int spatial_ndims;
    double empty_val;
} IRGenAlloc;

typedef struct {
    IRGenAlloc cross_gen;            /* the cross-generational tensor (prev/next) */
    IRGenAlloc intermediates[8];     /* intra-generation intermediates (fresh each gen) */
    int num_intermediates;
    int max_gen;                     /* safety iteration cap */
} IRGenLoop;
```

- [ ] **Step 2: Write the failing IR test**

In `tests/unit/test_ir.c`, add before `main()`:
```c
TEST(test_ir_cascade_structure) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init D = [0.0, inf, inf, inf];\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    ASSERT_NOT_NULL(ir);

    int gen_begin = 0, gen_end = 0, stores = 0, loop_begins = 0;
    int cross_writes = 0;   /* gen_write set (cross-gen) */
    for (int i = 0; i < ir->count; i++) {
        switch (ir->instrs[i].type) {
            case IR_GEN_LOOP_BEGIN:
                gen_begin++;
                ASSERT_STR_EQ("D", ir->instrs[i].data.gen_loop.cross_gen.name);
                ASSERT_INT_EQ(1, ir->instrs[i].data.gen_loop.cross_gen.spatial_ndims);
                ASSERT_INT_EQ(4, ir->instrs[i].data.gen_loop.cross_gen.spatial_dims[0]);
                ASSERT_INT_EQ(1, ir->instrs[i].data.gen_loop.num_intermediates);
                ASSERT_STR_EQ("T", ir->instrs[i].data.gen_loop.intermediates[0].name);
                break;
            case IR_GEN_LOOP_END:  gen_end++; break;
            case IR_LOOP_BEGIN:    loop_begins++; break;
            case IR_STORE:
                stores++;
                if (ir->instrs[i].data.store.gen_write) cross_writes++;
                break;
            default: break;
        }
    }
    ASSERT_INT_EQ(1, gen_begin);    /* ONE gen loop wraps BOTH einsums */
    ASSERT_INT_EQ(1, gen_end);
    ASSERT_INT_EQ(2, stores);       /* T and D */
    ASSERT_INT_EQ(1, cross_writes); /* only D's store is gen_write */
    ASSERT_INT_EQ(3, loop_begins);  /* relax: d,s ; combine: d */

    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```
Register it in `main()`:
```c
    RUN_TEST(test_ir_cascade_structure);
```

- [ ] **Step 3: Update A's IR white-box assertions for the renamed struct**

In `tests/unit/test_ir.c`, in the existing `test_ir_generational_structure`, the `IR_GEN_LOOP_BEGIN` case currently asserts `gen_loop.tensor_name`, `gen_loop.spatial_ndims`, `gen_loop.spatial_dims[0]`. Update those three lines to the new field path:
```c
            case IR_GEN_LOOP_BEGIN:
                gen_begin++;
                ASSERT_STR_EQ("F", ir->instrs[i].data.gen_loop.cross_gen.name);
                ASSERT_INT_EQ(1, ir->instrs[i].data.gen_loop.cross_gen.spatial_ndims);
                ASSERT_INT_EQ(4, ir->instrs[i].data.gen_loop.cross_gen.spatial_dims[0]);
                break;
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cd build && cmake .. && make test_ir && ./test_ir`
Expected: build error (renamed struct) or FAIL (two gen loops / wrong intermediate handling).

- [ ] **Step 5: Rewrite `emit_einsum` to be cascade-aware and not emit the gen loop**

In `src/ir.c`, replace the entire `emit_einsum` function with this version. It (a) takes the Einsum index and uses `symtab->einsum_class[eidx]`, (b) strips the gen coordinate for **any** gen-rank operand while tagging `gen_read`/`gen_write` only for the cross-gen tensor, and (c) no longer emits `IR_GEN_LOOP_BEGIN/END` (that moves to `ir_lower`):
```c
static void emit_einsum(IRProgram *ir, AstNode *einsum_node, SymbolTable *symtab, int eidx) {
    EinsumData *e = &einsum_node->data.einsum;
    EinsumClass *cls = &symtab->einsum_class[eidx];

    /* Per-operand: strip axis = the tensor's gen axis (any gen-rank tensor);
       cross-gen tag = operand IS the cross-generational tensor. */
    TensorEntry *out_te = symtab_find_tensor(symtab, e->output.tensor_name);
    TensorEntry *in1_te = symtab_find_tensor(symtab, e->input1.tensor_name);
    TensorEntry *in2_te = symtab_find_tensor(symtab, e->input2.tensor_name);
    const char *cg = symtab->gen_tensor;
    int out_cross = cg && strcmp(e->output.tensor_name, cg) == 0;
    int in1_cross = cg && strcmp(e->input1.tensor_name, cg) == 0;
    int in2_cross = cg && strcmp(e->input2.tensor_name, cg) == 0;

    /* Spatial loops: this Einsum's free indices, then its reduction indices. */
    for (int i = 0; i < cls->num_free_indices; i++) {
        IRInstr *instr = ir_emit(ir, IR_LOOP_BEGIN);
        instr->data.loop_begin.index_var = dup_str(cls->free_indices[i].name);
        instr->data.loop_begin.range = cls->free_indices[i].size;
    }
    for (int i = 0; i < cls->num_reduction_indices; i++) {
        IRInstr *instr = ir_emit(ir, IR_LOOP_BEGIN);
        instr->data.loop_begin.index_var = dup_str(cls->reduction_indices[i].name);
        instr->data.loop_begin.range = cls->reduction_indices[i].size;
    }

    /* LOAD input1 -> reg 0 */
    {
        IRInstr *instr = ir_emit(ir, IR_LOAD);
        instr->data.load.reg = 0;
        instr->data.load.tensor_name = dup_str(e->input1.tensor_name);
        instr->data.load.coords = dup_coords_strip(e->input1.rank_vars,
            e->input1.num_vars, in1_te->gen_axis, &instr->data.load.ncoords);
        instr->data.load.gen_read = in1_cross;
    }
    /* LOAD input2 -> reg 1 */
    {
        IRInstr *instr = ir_emit(ir, IR_LOAD);
        instr->data.load.reg = 1;
        instr->data.load.tensor_name = dup_str(e->input2.tensor_name);
        instr->data.load.coords = dup_coords_strip(e->input2.rank_vars,
            e->input2.num_vars, in2_te->gen_axis, &instr->data.load.ncoords);
        instr->data.load.gen_read = in2_cross;
    }

    /* MAP: reg 2 = map_op(reg 0, reg 1) */
    {
        IRInstr *instr = ir_emit(ir, IR_MAP);
        instr->data.map.dst_reg = 2;
        instr->data.map.op = e->map_op;
        instr->data.map.merge_op = e->map_merge_op;
        instr->data.map.src_reg1 = 0;
        instr->data.map.src_reg2 = 1;
        instr->data.map.empty_a = in1_te->empty_val;
        instr->data.map.empty_b = in2_te->empty_val;
        instr->data.map.empty_z = out_te->empty_val;
    }

    /* REDUCE: reg 3 = reduce_op(reg 2, reg 3) */
    {
        IRInstr *instr = ir_emit(ir, IR_REDUCE);
        instr->data.reduce.dst_reg = 3;
        instr->data.reduce.op = e->reduce_op;
        instr->data.reduce.merge_op = e->reduce_merge_op;
        instr->data.reduce.src_reg = 2;
        instr->data.reduce.acc_reg = 3;
        instr->data.reduce.identity = compute_op_identity(e->reduce_op, out_te->datatype);
        instr->data.reduce.empty_z  = out_te->empty_val;
    }

    /* STORE: output[coords] = reg 3 */
    {
        IRInstr *instr = ir_emit(ir, IR_STORE);
        instr->data.store.tensor_name = dup_str(e->output.tensor_name);
        instr->data.store.coords = dup_coords_strip(e->output.rank_vars,
            e->output.num_vars, out_te->gen_axis, &instr->data.store.ncoords);
        instr->data.store.src_reg = 3;
        instr->data.store.gen_write = out_cross;
    }

    /* LOOP_END in reverse order: reduction first, then free. */
    for (int i = cls->num_reduction_indices - 1; i >= 0; i--) ir_emit(ir, IR_LOOP_END);
    for (int i = cls->num_free_indices - 1; i >= 0; i--) ir_emit(ir, IR_LOOP_END);
}
```

- [ ] **Step 6: Add a gen-loop emitter and rewrite `ir_lower`**

In `src/ir.c`, add this helper immediately before `ir_lower`:
```c
/* Fill an IRGenAlloc with a tensor's spatial shape (dims excluding gen axis). */
static void fill_gen_alloc(IRGenAlloc *a, TensorEntry *te) {
    a->name = dup_str(te->name);
    a->empty_val = te->empty_val;
    int sn = 0;
    for (int i = 0; i < te->num_ranks; i++) {
        if (i == te->gen_axis) continue;
        a->spatial_dims[sn++] = te->dim_sizes[i];
    }
    a->spatial_ndims = sn;
}
```

Then replace the einsum-emitting loop at the end of `ir_lower`. The current code is:
```c
    for (int i = 0; i < program->data.program.count; i++) {
        AstNode *stmt = program->data.program.statements[i];
        if (stmt->type == NODE_EINSUM) {
            emit_einsum(ir, stmt, symtab);
        }
    }

    return ir;
```
Replace with:
```c
    if (symtab->is_iterative) {
        /* One generational loop wraps the whole cascade. */
        IRInstr *gl = ir_emit(ir, IR_GEN_LOOP_BEGIN);
        gl->data.gen_loop.max_gen = 1000;
        TensorEntry *cgte = symtab_find_tensor(symtab, symtab->gen_tensor);
        if (cgte) fill_gen_alloc(&gl->data.gen_loop.cross_gen, cgte);
        int ni = 0;
        for (int m = 0; m < symtab->num_intermediates && ni < 8; m++) {
            TensorEntry *te = symtab_find_tensor(symtab, symtab->intermediates[m]);
            if (te) fill_gen_alloc(&gl->data.gen_loop.intermediates[ni++], te);
        }
        gl->data.gen_loop.num_intermediates = ni;

        int eidx = 0;
        for (int i = 0; i < program->data.program.count; i++) {
            AstNode *stmt = program->data.program.statements[i];
            if (stmt->type == NODE_EINSUM)
                emit_einsum(ir, stmt, symtab, eidx++);
        }

        ir_emit(ir, IR_GEN_LOOP_END);
    } else {
        int eidx = 0;
        for (int i = 0; i < program->data.program.count; i++) {
            AstNode *stmt = program->data.program.statements[i];
            if (stmt->type == NODE_EINSUM)
                emit_einsum(ir, stmt, symtab, eidx++);
        }
    }

    return ir;
```

- [ ] **Step 7: Update `ir_free` and `ir_dump` for the new struct**

In `src/ir.c`, in `ir_free`, the `IR_GEN_LOOP_BEGIN` case frees `instr->data.gen_loop.tensor_name`. Replace it with:
```c
            case IR_GEN_LOOP_BEGIN:
                free(instr->data.gen_loop.cross_gen.name);
                for (int j = 0; j < instr->data.gen_loop.num_intermediates; j++)
                    free(instr->data.gen_loop.intermediates[j].name);
                break;
```

In `ir_dump`, the `IR_GEN_LOOP_BEGIN` case references `instr->data.gen_loop.tensor_name`/`spatial_*`. Replace that case with:
```c
            case IR_GEN_LOOP_BEGIN:
                fprintf(out, "GEN_LOOP_BEGIN(cross=%s, spatial=[",
                        instr->data.gen_loop.cross_gen.name);
                for (int j = 0; j < instr->data.gen_loop.cross_gen.spatial_ndims; j++) {
                    if (j > 0) fprintf(out, ",");
                    fprintf(out, "%d", instr->data.gen_loop.cross_gen.spatial_dims[j]);
                }
                fprintf(out, "], intermediates=[");
                for (int j = 0; j < instr->data.gen_loop.num_intermediates; j++) {
                    if (j > 0) fprintf(out, ",");
                    fprintf(out, "%s", instr->data.gen_loop.intermediates[j].name);
                }
                fprintf(out, "], max_gen=%d)\n", instr->data.gen_loop.max_gen);
                break;
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `cd build && make test_ir && ./test_ir`
Expected: `test_ir_cascade_structure` and the updated `test_ir_generational_structure` PASS, plus existing IR tests.

- [ ] **Step 9: Commit**

```bash
git add src/ir.h src/ir.c tests/unit/test_ir.c
git commit -m "feat(ir): single gen loop wrapping a cascade; intermediate fresh-alloc list"
```

---

## Task 3: Codegen — cascade I/O and multi-buffer scaffolding

**Files:**
- Modify: `src/codegen.c`
- Test: `tests/unit/test_codegen.c`

- [ ] **Step 1: Write the failing codegen test**

In `tests/unit/test_codegen.c`, add before `main()`:
```c
TEST(test_codegen_cascade_bellman_ford) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init W = [[inf,1.0,4.0,inf],[inf,inf,2.0,6.0],[inf,inf,inf,3.0],[inf,inf,inf,inf]];\n"
        "init D = [0.0, inf, inf, inf];\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "def run(W, D):") != NULL);   /* T excluded (intermediate) */
    ASSERT_TRUE(strstr(code, "\"output\": \"D\"") != NULL);
    ASSERT_TRUE(strstr(code, "D_prev = D") != NULL);
    ASSERT_TRUE(strstr(code, "while True:") != NULL);
    ASSERT_TRUE(strstr(code, "D_next = [math.inf] * N") != NULL);  /* cross-gen fresh */
    ASSERT_TRUE(strstr(code, "T = [math.inf] * N") != NULL);       /* intermediate fresh */
    ASSERT_TRUE(strstr(code, "b_val = T[d]") != NULL);             /* combine reads T plainly */
    ASSERT_TRUE(strstr(code, "a_val = D_prev[d]") != NULL);        /* combine reads D_prev */
    ASSERT_TRUE(strstr(code, "if D_next == D_prev:") != NULL);
    ASSERT_TRUE(strstr(code, "return D_prev") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```
Register it in `main()`:
```c
    RUN_TEST(test_codegen_cascade_bellman_ford);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd build && cmake .. && make test_codegen && ./test_codegen`
Expected: build error (renamed `gen_loop` fields used by codegen) or FAIL.

- [ ] **Step 3: Update iterative detection + intermediate set**

In `src/codegen.c`, in `codegen_python`, the iterative-detection block currently sets `gen_tensor = ir->instrs[i].data.gen_loop.tensor_name`. Replace that block with one that also records the intermediate names:
```c
    /* Detect generational mode and capture the gen-loop metadata. */
    int iterative = 0;
    const char *gen_tensor = NULL;
    const char *intermediates[8];
    int num_intermediates = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_GEN_LOOP_BEGIN) {
            iterative = 1;
            gen_tensor = ir->instrs[i].data.gen_loop.cross_gen.name;
            num_intermediates = ir->instrs[i].data.gen_loop.num_intermediates;
            for (int j = 0; j < num_intermediates && j < 8; j++)
                intermediates[j] = ir->instrs[i].data.gen_loop.intermediates[j].name;
            break;
        }
    }
```

- [ ] **Step 4: Exclude intermediates from `run()` inputs and force the cross-gen output**

In `src/codegen.c`, after `collect_io(ir, &inputs, &output);` and the iterative-detection block, add a filter that drops intermediate tensors from the discovered inputs and pins the output to the cross-generational tensor:
```c
    /* Cascade I/O: intermediates are neither parameters nor the return value;
       the returned tensor is the cross-generational one. */
    if (iterative) {
        TensorNameList filtered;
        filtered.count = 0;
        for (int i = 0; i < inputs.count; i++) {
            int is_inter = 0;
            for (int j = 0; j < num_intermediates; j++)
                if (strcmp(inputs.names[i], intermediates[j]) == 0) is_inter = 1;
            if (is_inter) { free(inputs.names[i]); continue; }
            filtered.names[filtered.count++] = inputs.names[i];
        }
        inputs = filtered;
        if (gen_tensor) {
            if (output) free(output);
            output = strdup(gen_tensor);
        }
    }
```
(This relies on the existing `TensorNameList` type and `inputs`/`output` locals; `collect_io` allocates each `inputs.names[i]` and `output` with `strdup`, so freeing the dropped ones and re-`strdup`ing `output` keeps ownership consistent.)

- [ ] **Step 5: Emit fresh allocations for cross-gen `next` and every intermediate**

In `src/codegen.c`, in the run-body switch, replace the `IR_GEN_LOOP_BEGIN` case (which currently emits `<name>_prev`, `_it = 0`, `while True:`, and a single `<name>_next` fresh slice) with one that allocates the cross-gen `next` slice **and** each intermediate fresh:
```c
            case IR_GEN_LOOP_BEGIN: {
                IRGenLoop *gloop = &instr->data.gen_loop;
                cb_indent(&cb);
                cb_appendf(&cb, "%s_prev = %s\n", gloop->cross_gen.name, gloop->cross_gen.name);
                cb_indent(&cb);
                cb_append(&cb, "_it = 0\n");
                cb_indent(&cb);
                cb_append(&cb, "while True:\n");
                cb.indent++;
                /* cross-generational next slice (fresh each generation) */
                {
                    IRTensorAlloc tmp;
                    memset(&tmp, 0, sizeof(tmp));
                    tmp.ndims = gloop->cross_gen.spatial_ndims;
                    tmp.empty_val = gloop->cross_gen.empty_val;
                    for (int k = 0; k < tmp.ndims; k++)
                        tmp.dim_sizes[k] = gloop->cross_gen.spatial_dims[k];
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s_next = ", gloop->cross_gen.name);
                    emit_alloc(&cb, &tmp, ir, i);
                    cb_append(&cb, "\n");
                }
                /* each intra-generation intermediate (fresh each generation) */
                for (int m = 0; m < gloop->num_intermediates; m++) {
                    IRTensorAlloc tmp;
                    memset(&tmp, 0, sizeof(tmp));
                    tmp.ndims = gloop->intermediates[m].spatial_ndims;
                    tmp.empty_val = gloop->intermediates[m].empty_val;
                    for (int k = 0; k < tmp.ndims; k++)
                        tmp.dim_sizes[k] = gloop->intermediates[m].spatial_dims[k];
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s = ", gloop->intermediates[m].name);
                    emit_alloc(&cb, &tmp, ir, i);
                    cb_append(&cb, "\n");
                }
                break;
            }
```

- [ ] **Step 6: Verify the `IR_GEN_LOOP_END` case still compiles against the struct**

In `src/codegen.c`, the `IR_GEN_LOOP_END` case uses the `gen_tensor` local (captured in Step 3) for the `if <t>_next == <t>_prev` / swap / cap. It does **not** reference the renamed struct fields, so it needs no change. Confirm by reading it; leave as-is.

- [ ] **Step 7: Run the test to verify it passes**

Run: `cd build && make test_codegen && ./test_codegen`
Expected: `test_codegen_cascade_bellman_ford` PASS; existing codegen tests (GEMM, A's generational) still PASS.

- [ ] **Step 8: Full build + manual end-to-end check**

Run:
```bash
cd build && cmake .. && make 2>&1 | grep -ic warning   # expect 0
cat > /tmp/bf.edge <<'EOF'
dim N = 4;
tensor W[R=N, C=N]   : float, empty = inf;
tensor T[gen I, R=N] : float, empty = inf;
tensor D[gen I, R=N] : float, empty = inf;
init W = [[inf,1.0,4.0,inf],[inf,inf,2.0,6.0],[inf,inf,inf,3.0],[inf,inf,inf,inf]];
init D = [0.0, inf, inf, inf];
T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);
stop: stable(D);
EOF
./edgecc /tmp/bf.edge -o /tmp/bf.py && python3 /tmp/bf.py
```
Expected stdout: `[0.0, 1.0, 3.0, 6.0]`. (If it differs, inspect with `--dump-ir` and read `/tmp/bf.py`.) Then `rm -f /tmp/bf.edge /tmp/bf.py`.

- [ ] **Step 9: Commit**

```bash
git add src/codegen.c tests/unit/test_codegen.c
git commit -m "feat(codegen): cascade I/O and per-generation fresh buffers for intermediates"
```

---

## Task 4: End-to-end Bellman-Ford + example

**Files:**
- Create: `tests/e2e/bellman_ford/input.edge`, `tests/e2e/bellman_ford/expected_output.txt`
- Create: `examples/bellman_ford.edge`

- [ ] **Step 1: Create the e2e program**

Create `tests/e2e/bellman_ford/input.edge`:
```
// Bellman-Ford shortest paths as a 2-Einsum generational cascade.
// D is cross-generational (distances, written at i+1); T is an intra-generation
// intermediate (relaxed candidates, written at i). Each generation: relax all
// edges into T, then keep the smaller of the old and new distance into D.
// Graph edges (weight): 0->1(1), 0->2(4), 1->2(2), 1->3(6), 2->3(3); source 0.
dim N = 4;

tensor W[R=N, C=N]   : float, empty = inf;
tensor T[gen I, R=N] : float, empty = inf;
tensor D[gen I, R=N] : float, empty = inf;

init W = [[inf, 1.0, 4.0, inf], [inf, inf, 2.0, 6.0], [inf, inf, inf, 3.0], [inf, inf, inf, inf]];
init D = [0.0, inf, inf, inf];

T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);   // relax
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);      // combine
stop: stable(D);
```

- [ ] **Step 2: Create the expected output**

Create `tests/e2e/bellman_ford/expected_output.txt` with exactly:
```
[0.0, 1.0, 3.0, 6.0]
```

- [ ] **Step 3: Run the e2e suite**

Run: `cd build && python3 ../tests/e2e/run_e2e.py ./edgecc 2>&1 | grep -iE "bellman|passed"`
Expected: `PASS bellman_ford`, and the overall count includes all prior e2e cases.

- [ ] **Step 4: Add the example**

Create `examples/bellman_ford.edge` (same content as the e2e `input.edge` above, plus the run hint header). Use:
```
// Bellman-Ford shortest paths — a multi-Einsum generational cascade.
//
// D is the cross-generational tensor (distances, written at i+1); T is an
// intra-generation intermediate (relaxed candidates, written at i, fresh each
// generation). Each generation relaxes all edges into T, then keeps the smaller
// of the old and new distance into D, until D is stable.
//
// Run:  ./build/edgecc examples/bellman_ford.edge -o out.py && python3 out.py
// Output: [0.0, 1.0, 3.0, 6.0]
dim N = 4;

tensor W[R=N, C=N]   : float, empty = inf;
tensor T[gen I, R=N] : float, empty = inf;
tensor D[gen I, R=N] : float, empty = inf;

init W = [[inf, 1.0, 4.0, inf], [inf, inf, 2.0, 6.0], [inf, inf, inf, 3.0], [inf, inf, inf, inf]];
init D = [0.0, inf, inf, inf];

T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);
stop: stable(D);
```
Verify it: `./build/edgecc examples/bellman_ford.edge -o /tmp/ex.py && python3 /tmp/ex.py` → `[0.0, 1.0, 3.0, 6.0]`, then `rm -f /tmp/ex.py`.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/bellman_ford examples/bellman_ford.edge
git commit -m "test(e2e): Bellman-Ford cascade kernel; add example"
```

---

## Task 5: Negative tests

**Files:**
- Create: `tests/negative/cascade_two_cross_gen/{input.edge,expected_errors.txt}`
- Create: `tests/negative/cascade_intermediate_init/{input.edge,expected_errors.txt}`
- Create: `tests/negative/cascade_read_before_write/{input.edge,expected_errors.txt}`

- [ ] **Step 1: Two cross-generational tensors**

`tests/negative/cascade_two_cross_gen/input.edge`:
```
dim N = 4;
tensor D[gen I, R=N] : float, empty = inf;
tensor E[gen I, R=N] : float, empty = inf;
init D = [0.0, inf, inf, inf];
init E = [0.0, inf, inf, inf];
D[i+1, d] = D[i, d] . E[i, d] :: map(min, union) reduce(min, union);
E[i+1, d] = E[i, d] . D[i, d] :: map(min, union) reduce(min, union);
stop: stable(D);
```
`tests/negative/cascade_two_cross_gen/expected_errors.txt`:
```
more than one cross-generational tensor
```

- [ ] **Step 2: Intermediate given an init**

`tests/negative/cascade_intermediate_init/input.edge`:
```
dim N = 4;
tensor W[R=N, C=N]   : float, empty = inf;
tensor T[gen I, R=N] : float, empty = inf;
tensor D[gen I, R=N] : float, empty = inf;
init D = [0.0, inf, inf, inf];
init T = [0.0, inf, inf, inf];
T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);
stop: stable(D);
```
`tests/negative/cascade_intermediate_init/expected_errors.txt`:
```
must not have an init
```

- [ ] **Step 3: Intermediate read before written**

`tests/negative/cascade_read_before_write/input.edge`:
```
dim N = 4;
tensor W[R=N, C=N]   : float, empty = inf;
tensor T[gen I, R=N] : float, empty = inf;
tensor D[gen I, R=N] : float, empty = inf;
init D = [0.0, inf, inf, inf];
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);
T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);
stop: stable(D);
```
`tests/negative/cascade_read_before_write/expected_errors.txt`:
```
read before it is written
```

- [ ] **Step 4: Run the negative suite**

Run: `cd build && python3 ../tests/negative/run_negative.py ./edgecc 2>&1 | grep -iE "cascade_|passed"`
Expected: `PASS cascade_two_cross_gen`, `PASS cascade_intermediate_init`, `PASS cascade_read_before_write`, overall pass.

- [ ] **Step 5: Commit**

```bash
git add tests/negative/cascade_two_cross_gen tests/negative/cascade_intermediate_init tests/negative/cascade_read_before_write
git commit -m "test(negative): cascade role/ordering guardrails"
```

---

## Task 6: Documentation + full suite + regression check

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document cascades in CLAUDE.md**

In `CLAUDE.md`, in the "Generational (iterative) ranks" subsection (added in A), append after its existing prose:
```markdown
**Cascades (multiple Einsums per generation).** A generation may run a *sequence*
of Einsums in source order, sharing one generational rank. Exactly one
**cross-generational** tensor is written at `i+1` (it carries the `init`, is
double-buffered, and is returned); any number of **intra-generation
intermediates** are written and read at `i` (no `init`, recomputed fresh each
generation). Bellman-Ford shortest paths:

```
dim N = 4;
tensor W[R=N, C=N]   : float, empty = inf;
tensor T[gen I, R=N] : float, empty = inf;   // intermediate (relaxed candidates)
tensor D[gen I, R=N] : float, empty = inf;   // cross-generational (distances)
init W = [[inf,1.0,4.0,inf],[inf,inf,2.0,6.0],[inf,inf,inf,3.0],[inf,inf,inf,inf]];
init D = [0.0, inf, inf, inf];
T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);   // relax
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);      // combine
stop: stable(D);
```

The cross-generation combine is just another Einsum (`map(min, union)`), so no
special accumulation operator is needed. An Einsum with no reduction rank (the
combine) still writes `reduce(...)`; with zero reduction indices it passes the
map value through. See `docs/superpowers/specs/2026-06-01-edgecc-generational-cascades-b-design.md`.
```

- [ ] **Step 2: Update the deferred-scope list**

In `CLAUDE.md`, in the "v2 Scope Limitations" section, the deferred entry currently reads (from A):
```
multi-Einsum generational cascades (the `<<` update/accumulation populate, and chained Einsums sharing a generational rank — single-Einsum generational ranks are supported; see "Generational (iterative) ranks")
```
Replace that substring with:
```
the `<<` update compute operator, multiple cross-generational tensors / full 3-step BFS (needs unary `¬` and a take-left compute op), `ANY` compute, occupancy (`‖F‖≡0`) termination, and nested cascades / multiple generational ranks — single-Einsum generational ranks and single-cross-generational-tensor cascades (e.g. Bellman-Ford) are supported; see "Generational (iterative) ranks"
```

- [ ] **Step 3: Run the complete test suite**

Run: `cd build && cmake .. && make && ctest --output-on-failure`
Expected: all 8 suites PASS — unit (`test_lexer`/`test_parser`/`test_semantic`/`test_ir`/`test_codegen`), `e2e_tests` (incl. `bellman_ford`, A's `reachability`/`transitive_closure`/`connected_components`, GEMM, etc.), `negative_tests` (incl. the 3 new `cascade_*` and A's `gen_*`), and `e2e_iterative`.

- [ ] **Step 4: Regression — confirm A + non-iterative output unchanged**

Run:
```bash
cd "/Users/porumbes/Dropbox/Home/Serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc"
for ex in gemm reachability transitive_closure connected_components; do
  expected=$(grep -m1 "^// Output:" examples/$ex.edge | sed 's|^// Output: ||')
  ./build/edgecc examples/$ex.edge -o /tmp/$ex.py 2>/dev/null && actual=$(python3 /tmp/$ex.py)
  if [ -n "$expected" ]; then echo "$ex: expected $expected | actual $actual"; else echo "$ex: $actual"; fi
  rm -f /tmp/$ex.py
done
```
Expected: `reachability` → `[1.0, 1.0, 1.0, 0.0]`, `transitive_closure` → the closure matrix, `connected_components` → `[3.0, 3.0, 3.0, 5.0, 5.0]`. (`gemm.edge` has no `// Output:` header; just confirm it runs and prints a matrix.)

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document generational cascades (Bellman-Ford)"
```

---

## Self-Review Notes

- **Spec coverage:** cascade-as-sequence (Task 2 `ir_lower`); one cross-gen tensor + intermediates with auto-classification (Task 1 roles, Task 2 `IRGenLoop`); per-Einsum classification (Task 1 `EinsumClass`, Task 2 `emit_einsum`); trivial-reduce for the combine (verified in Task 3 Step 8 / Task 4); SSA ordering (Task 1 `validate_cascade_ordering`); cascade I/O (Task 3); Bellman-Ford demonstrator (Task 4); all four spec negative cases (Task 1 unit + Task 5 data); A/non-iterative regression (Task 6 Step 4); docs (Task 6).
- **Build-green:** global classification fields retained; Task 1 keeps existing tests passing; Task 2 renames `IRGenLoop` fields and updates the one A white-box assertion in the same task; Task 3 consumes the renamed fields.
- **Type consistency:** `EinsumClass`, `SymbolTable.{einsum_class,num_einsums,intermediates,num_intermediates}`, `TensorEntry.has_init`, `IRGenAlloc`, `IRGenLoop.{cross_gen,intermediates,num_intermediates,max_gen}`, and the codegen locals (`gen_tensor`, `intermediates`, `num_intermediates`, `D_prev`/`D_next`/`_it`) are used identically across tasks.
- **No CMake changes:** unit tests append to existing binaries; e2e/negative cases are auto-discovered data dirs.
