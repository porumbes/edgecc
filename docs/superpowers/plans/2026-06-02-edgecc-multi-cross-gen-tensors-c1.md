# Multiple Cross-Generational Tensors (C1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the edgecc generational cascade from exactly one cross-generational tensor to N of them, including the *next-reads-next* dependency (`P[i+1] = P[i] . F[i+1]`), and ship a naive reachability-BFS demonstrator.

**Architecture:** Pipeline change only (semantic → IR → codegen); no lexer/parser/AST change. The single `gen_tensor` scalar becomes an ordered `gen_tensors[]` set. `IRLoad.gen_read` becomes tri-state (0 none / 1 read-at-`i`→`_prev` / 2 read-at-`i+1`→`_next`). `run()` returns the single `stop`-watched tensor. The N==1 path is kept byte-for-byte identical so Bellman-Ford / A / GEMM output is unchanged.

**Tech Stack:** C99, Flex, Bison, CMake; Python 3 for generated output and e2e/negative harnesses.

**Spec:** `docs/superpowers/specs/2026-06-02-edgecc-multi-cross-gen-tensors-c1-design.md`

**Build/test commands (run from repo root):**
- Build: `cmake --build build -j`
- Unit: `./build/test_semantic` · `./build/test_ir` · `./build/test_codegen` (also `test_lexer`, `test_parser`)
- E2E: `python3 tests/e2e/run_e2e.py` (auto-discovers `tests/e2e/<name>/`)
- Negative: `python3 tests/negative/run_negative.py` (auto-discovers `tests/negative/<name>/`)
- Everything: `cd build && ctest --output-on-failure`

> **Note for subagent execution:** subagents on this project do not run Bash. The coordinator runs all build/test/commit commands between tasks. Implementer prompts must include the exact code to write; the coordinator validates by building and running tests.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/semantic.h` | SymbolTable fields | Add `gen_tensors[]` + `num_gen_tensors`; keep `gen_tensor` as alias = `gen_tensors[0]` |
| `src/semantic.c` | role classification, offset & SSA validation | Collect N cross-gen tensors; allow `i+1` reads; SSA-check `i+1` reads; generalize init / read-at-`i` / stop-watch checks |
| `src/ir.h` | IR structs | `IRGenLoop.cross_gens[]` + `num_cross_gens` + `watch`; `IRLoad.gen_read` doc → tri-state |
| `src/ir.c` | lowering | Tri-state `gen_read`, per-tensor `gen_write`, populate `cross_gens[]` + `watch`, update free/dump |
| `src/codegen.c` | Python emission | Capture N cross-gens + watch; tri-state prev/next naming; fresh `_next` per tensor; N==1 vs N>1 epilogue; return watch |
| `tests/unit/test_semantic.c` | semantic unit tests | Flip the two-cross-gen test to valid; add next-reads-next + i+1-before-write |
| `tests/unit/test_ir.c` | IR unit tests | Add 2-cross-gen IR shape test |
| `tests/unit/test_codegen.c` | codegen unit tests | Add 2-cross-gen codegen test; existing gen tests must still pass unchanged |
| `tests/e2e/reachability_bfs/` | e2e demonstrator | New `input.edge` + `expected_output.txt` |
| `tests/negative/cascade_two_cross_gen/` | obsolete negative | **Delete** |
| `tests/negative/cascade_i1_read_before_write/` | new negative | New `input.edge` + `expected_errors.txt` |
| `examples/reachability_bfs.edge` | example | New |
| `CLAUDE.md` | docs | Cascades subsection + deferred list |

---

## Task 1: Semantic analysis accepts N cross-generational tensors

**Files:**
- Modify: `src/semantic.h:50-62` (SymbolTable fields)
- Modify: `src/semantic.c:284-316` (role + offset rules), `src/semantic.c:423-459` (SSA), `src/semantic.c:512-555` (consistency/init/read-at-i checks)
- Test: `tests/unit/test_semantic.c`

- [ ] **Step 1: Write/modify failing tests**

In `tests/unit/test_semantic.c`, **replace** the existing `TEST(test_cascade_two_cross_gen_tensors)` body (it currently asserts an error) with the new valid-program assertion, and **add** two new tests. Place near the other cascade tests:

```c
TEST(test_cascade_two_cross_gen_tensors) {
    /* Reachability BFS: F and P are both cross-generational (written at i+1).
       Under C1 this is valid, and P[i+1] reads the just-written F[i+1]. */
    AstNode *ast = parse(
        "dim N = 6;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0,0.0],"
        "[0.0,0.0,0.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0],"
        "[0.0,0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0]];\n"
        "init F = [1.0,0.0,0.0,0.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0,0.0,0.0,0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_INT_EQ(2, res.symtab->num_gen_tensors);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_i1_read_before_write) {
    /* P reads F[i+1] but F is written by a LATER Einsum -> SSA error. */
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0,0.0];\n"
        "init P = [1.0,0.0];\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "stop: stable(P);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_cross_gen_missing_init) {
    /* Two cross-gen tensors but P has no init -> error. */
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0,0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}
```

Register the two new tests in `main()` (the existing `RUN_TEST(test_cascade_two_cross_gen_tensors)` line stays):

```c
    RUN_TEST(test_cascade_i1_read_before_write);
    RUN_TEST(test_cascade_cross_gen_missing_init);
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build -j && ./build/test_semantic`
Expected: FAIL — `test_cascade_two_cross_gen_tensors` reports `expected 0, got 1` (old code still errors on two cross-gen); `num_gen_tensors` field does not exist yet (compile error until Step 3's header change). If it fails to compile, that counts as the failing state — proceed to Step 3.

- [ ] **Step 3: Add the SymbolTable fields**

In `src/semantic.h`, change the generational metadata block (currently lines 50-61) to:

```c
    /* Generational (iterative) program metadata. is_iterative == 0 for
       ordinary single-pass programs. */
    int is_iterative;
    /* Cross-generational tensors: gen-rank tensors written at offset i+1.
       C1 allows more than one (e.g. BFS frontier F and visited set P). */
    char *gen_tensors[MAX_TENSORS];   /* borrowed names, source order */
    int num_gen_tensors;
    char *gen_tensor;    /* alias = gen_tensors[0]; kept for diagnostics (borrowed) */
    char *gen_rank_var;  /* the index var bound to the gen rank, e.g. "i" (borrowed) */
    char *stop_watch;    /* tensor named in `stop: stable(...)` (borrowed) */
    /* Per-Einsum index classification, in source order (cascade support). */
    EinsumClass einsum_class[MAX_EINSUMS];
    int num_einsums;
    /* Intra-generation intermediate tensors (gen-rank, written at offset i). */
    char *intermediates[MAX_TENSORS];   /* borrowed names */
    int num_intermediates;
```

- [ ] **Step 4: Collect N cross-gen tensors instead of erroring**

In `src/semantic.c`, replace the cross-generational write branch (currently lines 284-292, the `if (eff_off == 1) { ... only one is supported ... }`) with:

```c
                if (eff_off == 1) {
                    /* cross-generational write: D[i+1, ...] — collect, allow many. */
                    int seen = 0;
                    for (int m = 0; m < st->num_gen_tensors; m++)
                        if (strcmp(st->gen_tensors[m], out_te->name) == 0) seen = 1;
                    if (!seen && st->num_gen_tensors < MAX_TENSORS) {
                        st->gen_tensors[st->num_gen_tensors++] = out_te->name;
                        if (st->gen_tensor == NULL) st->gen_tensor = out_te->name;
                    }
                }
```

- [ ] **Step 5: Allow `i+1` reads of generational inputs (offset > 1 still errors)**

In `src/semantic.c`, replace the offset-0-input loop (currently lines 308-316) with:

```c
        /* Generational inputs may be read at offset 0 (i) or +1 (i+1, the
           just-written cross-gen value from an earlier Einsum). Offsets >1 are
           unsupported. The "written before read at i+1" SSA rule is enforced in
           validate_cascade_ordering. */
        for (int k = 1; k < 3; k++) {
            if (tes[k]->gen_axis >= 0 && refs[k]->gen_offset > 1) {
                add_error(res,
                    "generational tensor '%s' read at unsupported offset +%d "
                    "(only i and i+1 are allowed)", tes[k]->name, refs[k]->gen_offset);
            }
        }
```

- [ ] **Step 6: SSA-check `i+1` reads in validate_cascade_ordering**

In `src/semantic.c`, replace the whole `validate_cascade_ordering` function (currently lines 423-459) with the version below. It keeps the existing intermediate check and adds a cross-gen `i+1`-read-before-write check:

```c
/* Validate SSA ordering within a generation:
   (1) an intra-generation intermediate must be written by an earlier Einsum
       before a later Einsum reads it;
   (2) a cross-generational tensor read at i+1 must already be written at i+1 by
       an earlier Einsum (next-reads-next). */
static void validate_cascade_ordering(SemanticResult *res, AstNode *program) {
    SymbolTable *st = res->symtab;
    if (!st->is_iterative) return;

    char *written[MAX_TENSORS];      int nwritten = 0;     /* intermediates written so far */
    char *cg_written[MAX_TENSORS];   int ncg_written = 0;  /* cross-gen written at i+1 so far */
    ProgramData *pd = &program->data.program;
    for (int i = 0; i < pd->count; i++) {
        AstNode *stmt = pd->statements[i];
        if (stmt->type != NODE_EINSUM) continue;
        EinsumData *e = &stmt->data.einsum;

        TensorRef *ins[2] = { &e->input1, &e->input2 };
        for (int k = 0; k < 2; k++) {
            const char *nm = ins[k]->tensor_name;
            /* (1) intermediate read-before-write */
            int is_inter = 0;
            for (int m = 0; m < st->num_intermediates; m++)
                if (strcmp(st->intermediates[m], nm) == 0) is_inter = 1;
            if (is_inter) {
                int found = 0;
                for (int w = 0; w < nwritten; w++)
                    if (strcmp(written[w], nm) == 0) found = 1;
                if (!found)
                    add_error(res, "intermediate '%s' is read before it is written", nm);
            }
            /* (2) cross-gen i+1 read-before-write */
            TensorEntry *te = symtab_find_tensor(st, nm);
            int eff = (te && ins[k]->gen_axis == te->gen_axis) ? ins[k]->gen_offset : 0;
            if (eff == 1) {
                int found = 0;
                for (int w = 0; w < ncg_written; w++)
                    if (strcmp(cg_written[w], nm) == 0) found = 1;
                if (!found)
                    add_error(res,
                        "cross-generational tensor '%s' is read at i+1 before it is written", nm);
            }
        }

        /* record this Einsum's output */
        const char *onm = e->output.tensor_name;
        for (int m = 0; m < st->num_intermediates; m++) {
            if (strcmp(st->intermediates[m], onm) == 0) {
                written[nwritten++] = st->intermediates[m]; break;
            }
        }
        for (int m = 0; m < st->num_gen_tensors; m++) {
            if (strcmp(st->gen_tensors[m], onm) == 0) {
                cg_written[ncg_written++] = st->gen_tensors[m]; break;
            }
        }
    }
}
```

- [ ] **Step 7: Generalize the stop-watch, init, and read-at-`i` checks to all cross-gen tensors**

In `src/semantic.c`, replace the stop-watch consistency check (currently lines 512-516) with:

```c
    if (st->is_iterative && st->stop_watch != NULL && st->num_gen_tensors > 0) {
        int ok = 0;
        for (int m = 0; m < st->num_gen_tensors; m++)
            if (strcmp(st->stop_watch, st->gen_tensors[m]) == 0) ok = 1;
        if (!ok)
            add_error(&res,
                "stop watches '%s' but it is not a cross-generational tensor",
                st->stop_watch);
    }
```

Replace the single-tensor init check (currently lines 518-526) with a loop over every cross-gen tensor:

```c
    /* Every cross-generational tensor needs a generation-0 init. */
    for (int m = 0; m < st->num_gen_tensors; m++) {
        TensorEntry *cg = symtab_find_tensor(st, st->gen_tensors[m]);
        if (cg != NULL && !cg->has_init) {
            add_error(&res,
                "cross-generational tensor '%s' requires an init (generation 0)",
                st->gen_tensors[m]);
        }
    }
```

Replace the single-tensor read-at-`i` check (currently lines 536-555) with a per-tensor loop:

```c
    /* Every cross-generational tensor must be read at generation i somewhere
       (so its generation-0 value is wired into run() as a parameter). */
    for (int m = 0; st->is_iterative && m < st->num_gen_tensors; m++) {
        const char *gt = st->gen_tensors[m];
        int read_at_i = 0;
        for (int i = 0; i < pd->count; i++) {
            AstNode *s = pd->statements[i];
            if (s->type != NODE_EINSUM) continue;
            EinsumData *e = &s->data.einsum;
            TensorEntry *t1 = symtab_find_tensor(st, e->input1.tensor_name);
            TensorEntry *t2 = symtab_find_tensor(st, e->input2.tensor_name);
            int eff1 = (t1 && e->input1.gen_axis == t1->gen_axis) ? e->input1.gen_offset : 0;
            int eff2 = (t2 && e->input2.gen_axis == t2->gen_axis) ? e->input2.gen_offset : 0;
            if ((strcmp(e->input1.tensor_name, gt) == 0 && eff1 == 0) ||
                (strcmp(e->input2.tensor_name, gt) == 0 && eff2 == 0)) {
                read_at_i = 1; break;
            }
        }
        if (!read_at_i) {
            add_error(&res,
                "cross-generational tensor '%s' must be read at generation i "
                "somewhere in the cascade", gt);
        }
    }
```

> Note: the missing-stop diagnostic at line 507 still reads `st->gen_tensor` — leave it; `gen_tensor` is now `gen_tensors[0]`, a fine fallback name.

- [ ] **Step 8: Run tests to verify they pass**

Run: `cmake --build build -j && ./build/test_semantic`
Expected: PASS for all, including the rewritten `test_cascade_two_cross_gen_tensors` (0 errors, `num_gen_tensors == 2`), `test_cascade_i1_read_before_write`, `test_cascade_cross_gen_missing_init`, and every pre-existing test (Bellman-Ford `test_cascade_valid_bellman_ford` still 0 errors; `gen_two_tensors` rank-mismatch test still errors).

- [ ] **Step 9: Commit**

```bash
git add src/semantic.h src/semantic.c tests/unit/test_semantic.c
git commit -m "feat(semantic): accept N cross-generational tensors + i+1 next-reads-next SSA"
```

---

## Task 2: IR carries N cross-gen allocs, tri-state gen_read, per-tensor gen_write

**Files:**
- Modify: `src/ir.h:44-50` (IRLoad doc), `src/ir.h:88-93` (IRGenLoop)
- Modify: `src/ir.c:95-176` (emit_einsum), `src/ir.c:201-212` (ir_lower gen loop), `src/ir.c:256` (ir_free), `src/ir.c:357-360` (dump-ir)
- Test: `tests/unit/test_ir.c`

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_ir.c`, add (matching the file's existing parse→analyze→lower idiom; copy the `parse()` helper pattern already used there):

```c
TEST(test_ir_two_cross_gen) {
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0,0.0];\n"
        "init P = [1.0,0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    IRProgram *ir = ir_lower(ast, res.symtab);
    ASSERT_NOT_NULL(ir);

    int found_begin = 0, f_next_read = 0, p_prev_read = 0, stores = 0;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *in = &ir->instrs[i];
        if (in->type == IR_GEN_LOOP_BEGIN) {
            found_begin = 1;
            ASSERT_INT_EQ(2, in->data.gen_loop.num_cross_gens);
            ASSERT_NOT_NULL(in->data.gen_loop.watch);
            ASSERT_STR_EQ("P", in->data.gen_loop.watch);
        }
        if (in->type == IR_LOAD) {
            if (strcmp(in->data.load.tensor_name, "F") == 0 && in->data.load.gen_read == 2)
                f_next_read = 1;            /* F read at i+1 -> next */
            if (strcmp(in->data.load.tensor_name, "P") == 0 && in->data.load.gen_read == 1)
                p_prev_read = 1;            /* P read at i -> prev */
        }
        if (in->type == IR_STORE && in->data.store.gen_write == 1) stores++;
    }
    ASSERT_INT_EQ(1, found_begin);
    ASSERT_INT_EQ(1, f_next_read);
    ASSERT_INT_EQ(1, p_prev_read);
    ASSERT_INT_EQ(2, stores);              /* both F and P stores are cross-gen writes */

    ir_free(ir);
    semantic_result_free(&res);
    ast_free(ast);
}
```

Register it in `main()`: `RUN_TEST(test_ir_two_cross_gen);`

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j && ./build/test_ir`
Expected: FAIL to compile — `num_cross_gens` / `watch` are not members of `IRGenLoop` yet.

- [ ] **Step 3: Extend the IR structs**

In `src/ir.h`, update the `IRLoad.gen_read` comment (line 49) to:

```c
    int gen_read;   /* 0: not cross-gen; 1: read at i (-> _prev); 2: read at i+1 (-> _next) */
```

Replace the `IRGenLoop` struct (currently lines 88-93) with:

```c
typedef struct {
    IRGenAlloc cross_gens[8];         /* cross-generational tensors (each prev/next) */
    int num_cross_gens;
    char *watch;                      /* the stop-watched cross-gen tensor name (owned) */
    IRGenAlloc intermediates[8];      /* intra-generation intermediates (fresh each gen) */
    int num_intermediates;
    int max_gen;                      /* safety iteration cap */
} IRGenLoop;
```

- [ ] **Step 4: Tri-state gen_read and per-tensor gen_write in emit_einsum**

In `src/ir.c`, add two helpers just above `emit_einsum` (before line 95):

```c
static int is_cross_gen_name(SymbolTable *st, const char *name) {
    for (int m = 0; m < st->num_gen_tensors; m++)
        if (strcmp(st->gen_tensors[m], name) == 0) return 1;
    return 0;
}

/* 0: not cross-gen; 1: cross-gen read at i (prev); 2: cross-gen read at i+1 (next). */
static int gen_read_kind(SymbolTable *st, TensorRef *ref, TensorEntry *te) {
    if (!te || te->gen_axis < 0) return 0;
    if (!is_cross_gen_name(st, te->name)) return 0;
    int eff = (ref->gen_axis == te->gen_axis) ? ref->gen_offset : 0;
    return (eff == 1) ? 2 : 1;
}
```

In `emit_einsum`, delete the old cross-gen booleans (currently lines 102-105: `const char *cg = ...; int out_cross ...; int in1_cross ...; int in2_cross ...;`). Then change the three flag assignments:

- Line 126 `instr->data.load.gen_read = in1_cross;` →
  ```c
  instr->data.load.gen_read = gen_read_kind(symtab, &e->input1, in1_te);
  ```
- Line 135 `instr->data.load.gen_read = in2_cross;` →
  ```c
  instr->data.load.gen_read = gen_read_kind(symtab, &e->input2, in2_te);
  ```
- Line 170 `instr->data.store.gen_write = out_cross;` →
  ```c
  instr->data.store.gen_write = is_cross_gen_name(symtab, e->output.tensor_name);
  ```

- [ ] **Step 5: Populate cross_gens[] and watch in ir_lower**

In `src/ir.c`, replace the gen-loop population block (currently lines 203-212) with:

```c
        IRInstr *gl = ir_emit(ir, IR_GEN_LOOP_BEGIN);
        gl->data.gen_loop.max_gen = 1000;
        int nc = 0;
        for (int m = 0; m < symtab->num_gen_tensors && nc < 8; m++) {
            TensorEntry *te = symtab_find_tensor(symtab, symtab->gen_tensors[m]);
            if (te) fill_gen_alloc(&gl->data.gen_loop.cross_gens[nc++], te);
        }
        gl->data.gen_loop.num_cross_gens = nc;
        gl->data.gen_loop.watch = symtab->stop_watch ? dup_str(symtab->stop_watch) : NULL;
        int ni = 0;
        for (int m = 0; m < symtab->num_intermediates && ni < 8; m++) {
            TensorEntry *te = symtab_find_tensor(symtab, symtab->intermediates[m]);
            if (te) fill_gen_alloc(&gl->data.gen_loop.intermediates[ni++], te);
        }
        gl->data.gen_loop.num_intermediates = ni;
```

- [ ] **Step 6: Update ir_free and dump-ir for the array**

In `src/ir.c`, in `ir_free`, replace the gen-loop free (currently line 256, `free(instr->data.gen_loop.cross_gen.name);`) with:

```c
                for (int c = 0; c < instr->data.gen_loop.num_cross_gens; c++)
                    free(instr->data.gen_loop.cross_gens[c].name);
                free(instr->data.gen_loop.watch);
                for (int m = 0; m < instr->data.gen_loop.num_intermediates; m++)
                    free(instr->data.gen_loop.intermediates[m].name);
```

> Check: confirm the intermediates were already freed elsewhere in `ir_free`. If the existing code already frees `intermediates[*].name`, do NOT duplicate it — only add the `cross_gens`/`watch` frees. Inspect lines around 256 before editing.

In `src/ir.c`, in the `--dump-ir` printer (currently lines 357-360 referencing `cross_gen.name` / `cross_gen.spatial_*`), replace with a loop:

```c
                for (int c = 0; c < instr->data.gen_loop.num_cross_gens; c++) {
                    IRGenAlloc *cg = &instr->data.gen_loop.cross_gens[c];
                    fprintf(out, "  GEN_LOOP cross_gen %s [", cg->name);
                    for (int j = 0; j < cg->spatial_ndims; j++) {
                        if (j) fprintf(out, ",");
                        fprintf(out, "%d", cg->spatial_dims[j]);
                    }
                    fprintf(out, "]\n");
                }
```

> Adapt the exact `fprintf` format to match the surrounding dump style at that site (read lines 352-365 first; keep whatever prefix/format the file already uses, just iterate the array).

- [ ] **Step 7: Run tests to verify they pass**

Run: `cmake --build build -j && ./build/test_ir && ./build/test_semantic`
Expected: PASS, including `test_ir_two_cross_gen` and all pre-existing IR tests (Bellman-Ford lowers with `num_cross_gens == 1`, its `D` loads have `gen_read == 1`).

- [ ] **Step 8: Commit**

```bash
git add src/ir.h src/ir.c tests/unit/test_ir.c
git commit -m "feat(ir): N cross-gen allocs, tri-state gen_read, watch tensor"
```

---

## Task 3: Codegen emits N-cross-gen run()/loop/epilogue (N==1 byte-for-byte)

**Files:**
- Modify: `src/codegen.c:300-323` (prev/next naming), `src/codegen.c:414-447` (capture + I/O), `src/codegen.c:480-491` (run alloc skip), `src/codegen.c:515-569` (gen-loop scaffolding + epilogue), `src/codegen.c:589-596` (return), `src/codegen.c:612-613` (`__main__` alloc skip)
- Test: `tests/unit/test_codegen.c`

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_codegen.c`, add (matching the file's existing parse→analyze→lower→codegen idiom; reuse the helper that returns the generated string — inspect an existing gen test in the file and copy its setup):

```c
TEST(test_codegen_two_cross_gen) {
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0,0.0];\n"
        "init P = [1.0,0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, res.symtab);
    char *code = codegen_python(ir);   /* entry point used by the other tests in this file */
    ASSERT_NOT_NULL(code);

    ASSERT_TRUE(strstr(code, "def run(G, F, P):") != NULL);
    ASSERT_TRUE(strstr(code, "F_prev = F") != NULL);
    ASSERT_TRUE(strstr(code, "P_prev = P") != NULL);
    ASSERT_TRUE(strstr(code, "F_next = ") != NULL);
    ASSERT_TRUE(strstr(code, "P_next = ") != NULL);
    ASSERT_TRUE(strstr(code, "F_next") != NULL);          /* P einsum reads F_next */
    ASSERT_TRUE(strstr(code, "_done = P_next == P_prev") != NULL);
    ASSERT_TRUE(strstr(code, "return P_prev") != NULL);

    free(code);
    ir_free(ir);
    semantic_result_free(&res);
    ast_free(ast);
}
```

Register it: `RUN_TEST(test_codegen_two_cross_gen);`

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j && ./build/test_codegen`
Expected: FAIL — current codegen emits `def run(G, F, P)` but returns `F_prev`/`P_prev` per the single `gen_tensor` (cross_gen.name = first), has no `_done`, and only one `_next`. Several `strstr` asserts fail.

- [ ] **Step 3: Add a list-membership helper**

In `src/codegen.c`, add near the top (file scope, before `generate`/`codegen_generate`):

```c
static int name_in_list(const char *name, const char **list, int n) {
    for (int i = 0; i < n; i++)
        if (strcmp(name, list[i]) == 0) return 1;
    return 0;
}
```

- [ ] **Step 4: Tri-state prev/next operand naming**

In `src/codegen.c` `emit_general_skeleton`, replace the `a_name`/`b_name` assignment block (currently lines 312-319) with:

```c
    if (load1->data.load.gen_read == 2)
        snprintf(a_name, sizeof(a_name), "%s_next", load1->data.load.tensor_name);
    else if (load1->data.load.gen_read == 1)
        snprintf(a_name, sizeof(a_name), "%s_prev", load1->data.load.tensor_name);
    else
        snprintf(a_name, sizeof(a_name), "%s", load1->data.load.tensor_name);
    if (load2->data.load.gen_read == 2)
        snprintf(b_name, sizeof(b_name), "%s_next", load2->data.load.tensor_name);
    else if (load2->data.load.gen_read == 1)
        snprintf(b_name, sizeof(b_name), "%s_prev", load2->data.load.tensor_name);
    else
        snprintf(b_name, sizeof(b_name), "%s", load2->data.load.tensor_name);
```

(The `z_name` block, lines 320-323, is unchanged — `gen_write` stays a bool → `_next`.)

- [ ] **Step 5: Capture N cross-gens + watch; set output = watch**

In `src/codegen.c`, replace the gen-mode capture block (currently lines 415-428) with:

```c
    int iterative = 0;
    const char *cross_gens[8]; int num_cross_gens = 0;
    const char *watch = NULL;
    const char *intermediates[8];
    int num_intermediates = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_GEN_LOOP_BEGIN) {
            IRGenLoop *gl = &ir->instrs[i].data.gen_loop;
            iterative = 1;
            num_cross_gens = gl->num_cross_gens;
            for (int j = 0; j < num_cross_gens && j < 8; j++)
                cross_gens[j] = gl->cross_gens[j].name;
            watch = gl->watch;
            num_intermediates = gl->num_intermediates;
            for (int j = 0; j < num_intermediates && j < 8; j++)
                intermediates[j] = gl->intermediates[j].name;
            break;
        }
    }
```

Replace the iterative I/O override (currently lines 432-447) with — note `output` becomes the watched tensor:

```c
    if (iterative) {
        TensorNameList filtered;
        filtered.count = 0;
        for (int i = 0; i < inputs.count; i++) {
            if (name_in_list(inputs.names[i], intermediates, num_intermediates)) {
                free(inputs.names[i]); continue;
            }
            filtered.names[filtered.count++] = inputs.names[i];
        }
        inputs = filtered;
        if (watch) {
            if (output) free(output);
            output = strdup(watch);
        }
    }
```

- [ ] **Step 6: Skip run()-internal alloc for every cross-gen tensor**

In `src/codegen.c`, replace the `IR_TENSOR_ALLOC` guard inside `run()` (currently lines 483-485) with:

```c
                if (output && strcmp(instr->data.tensor_alloc.name, output) == 0 &&
                    !(iterative &&
                      name_in_list(instr->data.tensor_alloc.name, cross_gens, num_cross_gens))) {
```

- [ ] **Step 7: Emit per-tensor prev init + fresh _next**

In `src/codegen.c`, replace the `IR_GEN_LOOP_BEGIN` case body (currently lines 515-550) with:

```c
            case IR_GEN_LOOP_BEGIN: {
                IRGenLoop *gloop = &instr->data.gen_loop;
                /* <name>_prev = <name>, once per cross-gen tensor, before the loop */
                for (int c = 0; c < gloop->num_cross_gens; c++) {
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s_prev = %s\n",
                               gloop->cross_gens[c].name, gloop->cross_gens[c].name);
                }
                cb_indent(&cb); cb_append(&cb, "_it = 0\n");
                cb_indent(&cb); cb_append(&cb, "while True:\n");
                cb.indent++;
                /* fresh <name>_next for every cross-gen tensor */
                for (int c = 0; c < gloop->num_cross_gens; c++) {
                    IRTensorAlloc tmp; memset(&tmp, 0, sizeof(tmp));
                    tmp.ndims = gloop->cross_gens[c].spatial_ndims;
                    tmp.empty_val = gloop->cross_gens[c].empty_val;
                    for (int k = 0; k < tmp.ndims; k++)
                        tmp.dim_sizes[k] = gloop->cross_gens[c].spatial_dims[k];
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s_next = ", gloop->cross_gens[c].name);
                    emit_alloc(&cb, &tmp, ir, i);
                    cb_append(&cb, "\n");
                }
                /* fresh intra-generation intermediates (unchanged) */
                for (int m = 0; m < gloop->num_intermediates; m++) {
                    IRTensorAlloc tmp; memset(&tmp, 0, sizeof(tmp));
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

> For `num_cross_gens == 1` this emits exactly the previous text: one `X_prev = X`, one `X_next = <alloc>`.

- [ ] **Step 8: Epilogue — N==1 verbatim, N>1 swap-all-gate-watched**

In `src/codegen.c`, replace the `IR_GEN_LOOP_END` case (currently lines 552-569) with:

```c
            case IR_GEN_LOOP_END: {
                if (num_cross_gens <= 1) {
                    const char *t = watch;   /* == cross_gens[0] for the single case */
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
                    cb.indent--;
                } else {
                    cb_indent(&cb);
                    cb_appendf(&cb, "_done = %s_next == %s_prev\n", watch, watch);
                    for (int c = 0; c < num_cross_gens; c++) {
                        cb_indent(&cb);
                        cb_appendf(&cb, "%s_prev = %s_next\n", cross_gens[c], cross_gens[c]);
                    }
                    cb_indent(&cb); cb_append(&cb, "if _done:\n");
                    cb.indent++;
                    cb_indent(&cb); cb_append(&cb, "break\n");
                    cb.indent--;
                    cb_indent(&cb); cb_append(&cb, "_it += 1\n");
                    cb_indent(&cb); cb_append(&cb, "if _it >= __EDGECC_MAX_GEN__:\n");
                    cb.indent++;
                    cb_indent(&cb);
                    cb_append(&cb, "raise RuntimeError(\"generational loop did not stabilize\")\n");
                    cb.indent--;
                    cb.indent--;
                }
                break;
            }
```

- [ ] **Step 9: Return the watched tensor; skip all cross-gen allocs in `__main__`**

In `src/codegen.c`, replace the return block (currently lines 589-596) with:

```c
    if (output) {
        cb_indent(&cb);
        if (iterative && watch) {
            cb_appendf(&cb, "return %s_prev\n", watch);
        } else {
            cb_appendf(&cb, "return %s\n", output);
        }
    }
```

Replace the `__main__` cross-gen alloc-skip (currently lines 612-613) with:

```c
        if (iterative &&
            name_in_list(instr->data.tensor_alloc.name, cross_gens, num_cross_gens)) continue;
```

- [ ] **Step 10: Run tests to verify they pass (and N==1 is byte-for-byte)**

Run: `cmake --build build -j && ./build/test_codegen && ./build/test_ir && ./build/test_semantic`
Expected: PASS, including `test_codegen_two_cross_gen` AND every pre-existing codegen golden test (Bellman-Ford, A's reachability/transitive-closure, GEMM) — the N==1 path text is unchanged, so no golden string should differ.

- [ ] **Step 11: Verify existing e2e still byte-for-byte**

Run: `python3 tests/e2e/run_e2e.py`
Expected: all existing e2e cases PASS (Bellman-Ford prints `[0.0, 1.0, 3.0, 6.0]`, etc.).

- [ ] **Step 12: Commit**

```bash
git add src/codegen.c tests/unit/test_codegen.c
git commit -m "feat(codegen): N cross-gen run()/loop/epilogue; return watched tensor"
```

---

## Task 4: End-to-end reachability BFS + example

**Files:**
- Create: `tests/e2e/reachability_bfs/input.edge`
- Create: `tests/e2e/reachability_bfs/expected_output.txt`
- Create: `examples/reachability_bfs.edge`

- [ ] **Step 1: Create the e2e input**

Create `tests/e2e/reachability_bfs/input.edge`:

```
dim N = 6;

tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
tensor P[gen I, R=N] : float, empty = 0.0;

init G = [[0.0,1.0,1.0,0.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,1.0,0.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,0.0]];
init F = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
init P = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
stop: stable(P);
```

- [ ] **Step 2: Generate and capture actual output**

Run:
```bash
./build/edgecc tests/e2e/reachability_bfs/input.edge -o /tmp/rbfs.py && python3 /tmp/rbfs.py
```
Expected: it prints the visited set with all six nodes reachable from node 0, i.e. `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`. Confirm this matches the hand-computed reachability (0→{1,2}→{3,4}→{5}); if the float formatting differs, use the exact printed line verbatim in Step 3.

- [ ] **Step 3: Write the expected-output file**

Create `tests/e2e/reachability_bfs/expected_output.txt` with the exact line printed in Step 2:

```
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

- [ ] **Step 4: Run the e2e harness**

Run: `python3 tests/e2e/run_e2e.py`
Expected: PASS `reachability_bfs` plus all pre-existing e2e cases.

- [ ] **Step 5: Add the example**

Create `examples/reachability_bfs.edge` with the same content as the e2e `input.edge` from Step 1.

- [ ] **Step 6: Full test run**

Run: `cd build && ctest --output-on-failure`
Expected: 100% pass (8 test groups).

- [ ] **Step 7: Commit**

```bash
git add tests/e2e/reachability_bfs examples/reachability_bfs.edge
git commit -m "test(e2e): naive reachability BFS (two cross-gen tensors)"
```

---

## Task 5: Negative-test reconciliation

**Rationale:** C1 removes B's "exactly one cross-generational tensor" limitation, so the negative test that defended it is now wrong. The rule that *survives* (`gen_two_tensors` = two distinct gen ranks → "rank variable mismatch") is unaffected. There is no existing "input read at offset 0" negative test to tighten, so reconciliation is **delete one, add one**.

**Files:**
- Delete: `tests/negative/cascade_two_cross_gen/` (whole directory)
- Create: `tests/negative/cascade_i1_read_before_write/input.edge`
- Create: `tests/negative/cascade_i1_read_before_write/expected_errors.txt`

- [ ] **Step 1: Delete the obsolete negative test**

```bash
git rm -r tests/negative/cascade_two_cross_gen
```

- [ ] **Step 2: Add the new negative test (i+1 read before write)**

Create `tests/negative/cascade_i1_read_before_write/input.edge`:

```
dim N = 2;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
tensor P[gen I, R=N] : float, empty = 0.0;
init F = [1.0, 0.0];
init P = [1.0, 0.0];
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);
stop: stable(P);
```

Create `tests/negative/cascade_i1_read_before_write/expected_errors.txt`:

```
read at i+1 before it is written
```

- [ ] **Step 3: Run the negative harness**

Run: `python3 tests/negative/run_negative.py`
Expected: PASS — `cascade_i1_read_before_write` matches its expected error; `cascade_two_cross_gen` no longer present; `gen_two_tensors`, `cascade_read_before_write`, `gen_bad_offset` and the rest still pass.

- [ ] **Step 4: Commit**

```bash
git add tests/negative/cascade_i1_read_before_write
git commit -m "test(negative): retire two-cross-gen error; add i+1-read-before-write"
```

---

## Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md` (Cascades subsection + v2 Scope Limitations deferred list)

- [ ] **Step 1: Update the Cascades subsection**

In `CLAUDE.md`, under the "Cascades (multiple Einsums per generation)" paragraph, change the "Exactly one cross-generational tensor" wording to reflect N tensors, and add the reachability-BFS example. Add a sentence:

> A cascade may write **more than one** cross-generational tensor (each declared with `gen`, written at `i+1`, carrying an `init`). A later Einsum may read an earlier one's just-written `i+1` value (*next-reads-next*), e.g. `P[i+1,d] = P[i,d] . F[i+1,d]`. `run()` returns the single tensor named by `stop: stable(...)`. Reachability BFS:
>
> ```
> F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);
> P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
> stop: stable(P);
> ```

- [ ] **Step 2: Update the deferred list**

In the "v2 Scope Limitations" section, remove "multiple cross-generational tensors / full 3-step BFS" from the *unsupported* list as the multi-cross-gen part, and reword so the remaining BFS gaps stay deferred. Change the relevant clause to:

> multiple cross-generational tensors are supported (see Cascades); the remaining gaps for full 3-step BFS are unary `¬`, a take-left compute op, depth-valued masking, multi-output return, `ANY` compute, and occupancy (`‖F‖≡0`) termination

- [ ] **Step 3: Full test run**

Run: `cd build && ctest --output-on-failure`
Expected: 100% pass.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: cascades may write N cross-generational tensors (C1)"
```

---

## Self-review notes

- **Spec coverage:** N cross-gen set (Task 1/2/3) · next-reads-next read + SSA (Task 1 Step 6, Task 2 Step 4, Task 3 Step 4) · single watched return (Task 3 Step 5/9) · N==1 byte-for-byte (Task 3 Steps 7/8/10/11) · reachability-BFS demonstrator (Task 4) · negative reconciliation (Task 5) · docs (Task 6). All spec sections map to a task.
- **Type/name consistency:** `gen_tensors[]`/`num_gen_tensors` (semantic), `cross_gens[]`/`num_cross_gens`/`watch` (IR + codegen), tri-state `gen_read` (0/1/2), `name_in_list`, `is_cross_gen_name`/`gen_read_kind` used consistently across tasks.
- **Line numbers** are from the pre-change source and will drift as edits land; each step quotes the surrounding code to anchor the edit. Re-read the cited region before applying if line numbers no longer match.
- **Verification idioms (confirmed):** codegen entry point is `codegen_python(IRProgram*)` → malloc'd string (`src/codegen.h`); both `test_ir.c` and `test_codegen.c` already define `static AstNode *parse(const char *)` and use `SemanticResult sem; ir_lower(ast, sem.symtab)`. New test snippets compile against these.
```