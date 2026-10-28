# Intermediates out of the `run()` ABI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Every tensor an Einsum writes starts empty and is allocated inside `run()`; only tensors that are read but never written are `run()` parameters (DFT gaps G5/G6).

**Architecture:** Semantic analysis gains one whole-program pass (`validate_written_tensors`) with three new compile errors (E1 init on a written tensor, E2 read before write, E3 non-generational write in a generational program). Codegen collects every stored tensor from the IR in `collect_io`, removes those from `inputs` for non-generational programs, and allocates every written tensor at the top of `run()` through the existing output-allocation path.

**Tech Stack:** C99, Flex/Bison (Bison 2.3 compatible), CMake/ctest, Python 3.8 for generated code and test harnesses.

**Spec:** `docs/superpowers/specs/2026-10-06-edgecc-intermediates-out-of-abi-design.md`

## Global Constraints

- Work only in the worktree `.worktrees/intermediates-abi` (branch `feature/intermediates-abi`). Never `git checkout` in the repo root; it belongs to the DFT exploration session (branch `dft-exploration`).
- Build: `cmake -S . -B build && make -C build`. Tests: `ctest --test-dir build --output-on-failure`.
- Generated Python and harnesses must run on Python 3.8 (no `dict |`, no `match`, no `list[int]` annotations).
- No `Co-Authored-By` trailer in commit messages.
- Exact error messages (tests match them as substrings):
  - E1: `tensor '<T>' is written by an Einsum and must not have an init (output tensors start empty)`
  - E2: `tensor '<T>' is read before the Einsum that writes it`
  - E3: `in a generational program every Einsum must write a generational tensor; '<T>' is not (compute it in a separate program)`
- E1/E2 apply to non-generational tensors only. Generational intermediates already have their own init rule, cross-generational tensors still *require* an init (generation 0), and generational read-before-write is already checked by `validate_cascade_ordering`.
- What `run()` returns is unchanged (last Einsum's output, or the stop-watched tensor).
- Generational programs' ABI is unchanged: a cross-generational tensor is both read and written and stays a `run()` parameter (its generation 0).
- No new named slots → no unknown-value negative test needed. No new semantics with a graph-algorithm meaning → no `tests/reference/` case needed (the spec's acceptance oracle is the DFT checker against numpy).

## Review Focus

- **Generational programs keep their parameters.** `F[i+1,d] = G[s,d] . F[i,s]` must still generate `def run(G, F)`; the read-minus-written rule must not apply to them. Pinned by the existing `test_codegen` asserts `def run(G, F):`, `def run(W, D):`, `def run(G, F, P):`, `def run(G, A, B):`, which Task 2 must keep green.
- **Fused Einsums write too.** An intermediate produced by a fused Einsum (`T[d] = (G[s,d] .1 F[s]) .2 P[d]`) must leave the signature like any other. Task 2 adds `test_codegen_fused_intermediate_not_a_param`.
- **Reading between two writes is legal.** `B = …; C = B…; B = …; Z = C.B` must not trigger E2, and keeps today's accumulate-into-one-buffer result. Task 2 adds the `intermediate_double_write` e2e (`[4.0, 14.0]`).
- **Inferred dims with an intermediate** (`dim N = *`, the G6 crash) must run through the generated program's own `__main__`. Task 2 adds the `intermediate_inferred_dim` e2e.
- **The self-updating non-generational kernel** (`D[d] = G[s,d] . D[s]` with `init D`) silently discarded its init on `main` (prints all-empty). It must now be a compile error (E1 + E2). Task 1 adds the `self_update_nongen` negative test.

---

### Task 1: Semantic rules E1/E2/E3

**Files:**
- Modify: `src/semantic.c` (new `validate_written_tensors`, called from `semantic_analyze`)
- Modify (reconciliation): `tests/unit/test_semantic.c`, `tests/unit/test_codegen.c`, `tests/unit/test_ir.c`, `examples/dotproduct.edge`
- Create: `tests/negative/{intermediate_with_init,output_with_init,read_before_write,self_update_nongen,gen_nongen_write}/{input.edge,expected_errors.txt}`

**Interfaces:**
- Consumes: `einsum_operands(AstNode *stmt, TensorRef *ins[3], TensorRef **out)` (returns operand count, 0 for non-Einsum statements), `symtab_find_tensor`, `add_error`, `name_present(char *const names[], int n, const char *x)`, `TensorEntry.has_init`, `TensorEntry.gen_axis`, `SymbolTable.is_iterative` — all already in `src/semantic.c` / `src/semantic.h`.
- Produces: `static void validate_written_tensors(SemanticResult *res, AstNode *program)` (file-local). Later tasks rely only on its guarantees: in a valid generational program every Einsum output is generational; in a valid non-generational program no written tensor has an init and none is read before its first write.

- [x] **Step 1: Write the negative tests**

`tests/negative/intermediate_with_init/input.edge`:
```
// An intermediate starts empty (paper: output tensors are initially empty);
// its init used to be silently used as the reduce's starting value.
dim N = 2;
tensor A[R=N] : float, empty = 0.0;
tensor B[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0];
init B = [10.0, 10.0];
B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);
```
`tests/negative/intermediate_with_init/expected_errors.txt`:
```
tensor 'B' is written by an Einsum and must not have an init (output tensors start empty)
```

`tests/negative/output_with_init/input.edge`:
```
// The returned output's init used to be silently ignored.
dim N = 2;
tensor A[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0];
init Z = [10.0, 10.0];
Z[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
```
`tests/negative/output_with_init/expected_errors.txt`:
```
tensor 'Z' is written by an Einsum and must not have an init (output tensors start empty)
```

`tests/negative/read_before_write/input.edge`:
```
// B is read by the first Einsum but only written by the second (was a NameError).
dim N = 2;
tensor A[R=N] : float, empty = 0.0;
tensor B[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0];
Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);
B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
```
`tests/negative/read_before_write/expected_errors.txt`:
```
tensor 'B' is read before the Einsum that writes it
```

`tests/negative/self_update_nongen/input.edge`:
```
// A non-generational tensor read and written by the same Einsum: its init was
// silently discarded (D reset to empty before the read). Use a gen rank instead.
dim N = 3;
tensor G[R=N, C=N] : float, empty = inf;
tensor D[R=N] : float, empty = inf;
init G = [[inf, 1.0, inf], [inf, inf, 2.0], [inf, inf, inf]];
init D = [0.0, inf, inf];
D[d] = G[s,d] . D[s] :: map(add, intersect) reduce(min, union);
```
`tests/negative/self_update_nongen/expected_errors.txt`:
```
tensor 'D' is read before the Einsum that writes it
tensor 'D' is written by an Einsum and must not have an init (output tensors start empty)
```

`tests/negative/gen_nongen_write/input.edge`:
```
// A non-generational write inside a generational program ran every generation,
// accumulating into the caller's buffer. Rejected for now (hoisting is future work).
dim N = 4;
tensor G[R=N, C=N]  : float, empty = 0.0;
tensor G2[R=N, C=N] : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
init G = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]];
init F = [1.0, 0.0, 0.0, 0.0];
G2[s,d] = G[s,k] . G[k,d] :: map(mul, intersect) reduce(add, union);
F[i+1, d] = G2[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);
stop: stable(F);
```
`tests/negative/gen_nongen_write/expected_errors.txt`:
```
in a generational program every Einsum must write a generational tensor; 'G2' is not (compute it in a separate program)
```

- [x] **Step 2: Run them and confirm all five fail**

Run: `make -C build && python3 tests/negative/run_negative.py build/edgecc 2>&1 | grep -i "fail\|passed"`
Expected: `FAIL` lines for all five new tests (`expected failure but edgecc succeeded` or `expected error ... not found`), every older negative test still passing.

- [x] **Step 3: Rewrite the unit-test programs that violate the new rules**

Nine unit-test programs read and write the same non-generational tensor (and give it an init). Each test is about something else (seed codegen, fused classification), so give each a separate output tensor. All of these must pass before **and** after Step 4.

In `tests/unit/test_semantic.c`:
- `test_semantic_seed_rank1_ok` and in `tests/unit/test_codegen.c` `test_codegen_seed_init`: add the declaration line `"tensor Z[R=N]      : float, empty = inf;\n"` after the `tensor D` line, and change the Einsum line from `"D[d] = G[s,d] . D[s] :: …"` to `"Z[d] = G[s,d] . D[s] :: map(add, intersect) reduce(min, union);\n"`.
- `test_semantic_seed_2d_rejected`: add `"tensor Z[R=N, C=N] : float, empty = inf;\n"` after the `tensor D` line, and change the Einsum to `"Z[a,b] = G[a,b] . G[a,b] :: map(add, intersect) reduce(min, union);\n"` (now only the 2-D seed error fires).
- `test_semantic_fused_push_classification`, `test_semantic_fused_missing_reduce_errors`, `test_semantic_fused_bad_over_errors`: add `"tensor Z[R=N] : float, empty = 0.0;\n"` after the `tensor P` line and change the Einsum's leading `"F[d] = (G[s,d] .1 F[s]) .2 P[d] "` to `"Z[d] = (G[s,d] .1 F[s]) .2 P[d] "`.

In `tests/unit/test_codegen.c`:
- `test_codegen_fused_push`: add `"tensor Z[R=N] : float, empty = 0.0;\n"` after the `tensor P` line; change `"F[d] = (G[s,d] .1 F[s]) .2 !P[d] "` to `"Z[d] = (G[s,d] .1 F[s]) .2 !P[d] "`.
- `test_codegen_fused_pull`: add the same `tensor Z` line; change `"F[d] = (G[s,d] .1 !P[d]) .2 F[s] "` to `"Z[d] = (G[s,d] .1 !P[d]) .2 F[s] "`.

In `tests/unit/test_ir.c`:
- `test_ir_fused_push_node`: add `"tensor Z[R=N] : float, empty = 0.0;\n"` after the `tensor P` line; change `"F[d] = (G[s,d] .1 F[s]) .2 P[d] "` to `"Z[d] = (G[s,d] .1 F[s]) .2 P[d] "`.

In `examples/dotproduct.edge`: delete the line `init Z = [0];` (it was ignored anyway: output tensors start empty).

Run: `make -C build && ctest --test-dir build 2>&1 | tail -6`
Expected: 13 of 14 targets pass. The only failure is `negative_tests`, which still reports the five new negative tests from Step 1 (transiently red until Step 4). Every unit test passes.

- [x] **Step 4: Implement `validate_written_tensors`**

In `src/semantic.c`, insert directly after `validate_rank_offsets` (which sits just after `einsum_operands`):

```c
/* Output tensors start empty (paper, §Initialization): a tensor written by an
   Einsum is not an input. For non-generational tensors:
     E1  it may not have an init;
     E2  it may not be read before the first Einsum that writes it.
   E3: in a generational program every Einsum writes a generational tensor
   (hoisting loop-invariant Einsums out of the loop is future work). */
static int written_by_some_einsum(ProgramData *pd, const char *name) {
    for (int j = 0; j < pd->count; j++) {
        TensorRef *unused[3]; TensorRef *w;
        if (einsum_operands(pd->statements[j], unused, &w) &&
            strcmp(w->tensor_name, name) == 0)
            return 1;
    }
    return 0;
}

static void validate_written_tensors(SemanticResult *res, AstNode *program) {
    SymbolTable *st = res->symtab;
    ProgramData *pd = &program->data.program;
    char *seen[MAX_EINSUMS];   int nseen = 0;       /* written so far, source order */
    char *e2_done[MAX_TENSORS]; int ne2 = 0;        /* E2 reported once per tensor */
    for (int i = 0; i < pd->count; i++) {
        TensorRef *ins[3]; TensorRef *out;
        int nins = einsum_operands(pd->statements[i], ins, &out);
        if (nins == 0) continue;

        if (!st->is_iterative) {
            for (int k = 0; k < nins; k++) {
                char *nm = ins[k]->tensor_name;
                TensorEntry *te = symtab_find_tensor(st, nm);
                if (te == NULL || te->gen_axis >= 0) continue;
                if (name_present(seen, nseen, nm) || name_present(e2_done, ne2, nm)) continue;
                if (!written_by_some_einsum(pd, nm)) continue;
                if (ne2 < MAX_TENSORS) e2_done[ne2++] = nm;
                add_error(res, "tensor '%s' is read before the Einsum that writes it", nm);
            }
        }

        TensorEntry *ote = symtab_find_tensor(st, out->tensor_name);
        if (ote == NULL || ote->gen_axis >= 0) continue;
        if (st->is_iterative) {
            add_error(res, "in a generational program every Einsum must write a "
                           "generational tensor; '%s' is not (compute it in a "
                           "separate program)", out->tensor_name);
            continue;
        }
        if (name_present(seen, nseen, out->tensor_name)) continue;
        if (ote->has_init)
            add_error(res, "tensor '%s' is written by an Einsum and must not have an "
                           "init (output tensors start empty)", out->tensor_name);
        if (nseen < MAX_EINSUMS) seen[nseen++] = out->tensor_name;
    }
}
```

Note the E2 loop runs before the output is added to `seen`, so `D[d] = G[s,d] . D[s]` reports E2 (the read precedes its own write). `semantic_analyze` returns early when there are more than `MAX_EINSUMS` Einsums, so `seen` cannot overflow; the bounds checks are defensive.

In `semantic_analyze`, change

```c
    validate_rank_offsets(&res, program);
    validate_cascade_ordering(&res, program);
```
to
```c
    validate_rank_offsets(&res, program);
    validate_written_tensors(&res, program);
    validate_cascade_ordering(&res, program);
```

- [x] **Step 5: Run everything**

Run: `make -C build 2>&1 | grep -E "error|warning:" ; python3 tests/negative/run_negative.py build/edgecc 2>&1 | grep -i "fail\|passed"; ctest --test-dir build 2>&1 | tail -3`
Expected: no compiler warnings; all negative tests pass (72/72); `100% tests passed, 0 tests failed out of 14`.

Also confirm no real program is newly rejected:
```bash
for f in examples/*.edge tests/e2e/*/input.edge tests/e2e/iterative/*/kernel.edge tests/reference/kernels/*.edge; do build/edgecc "$f" -o /dev/null >/dev/null 2>&1 || echo "REJECTED $f"; done
```
Expected: no output.

- [x] **Step 6: Commit**

```bash
git add src/semantic.c tests/unit tests/negative examples/dotproduct.edge
git commit -m "semantic: written tensors start empty — no init, no read before write, no non-gen write in a gen program

E1/E2/E3 from the intermediates-out-of-ABI spec. Unit-test programs that read
and wrote the same non-generational tensor (which silently discarded the init)
now write a separate output tensor; dotproduct.edge drops its ignored init."
```

---

### Task 2: Codegen — intermediates allocated in `run()`, not parameters

**Files:**
- Modify: `src/codegen.c` (`collect_io`, the input filtering next to the generational intermediates filter, the `IR_TENSOR_ALLOC` case inside `run()`, cleanup at the end of `codegen_python_ex`)
- Modify: `tests/e2e/run_e2e.py` (optional `run_twice` marker)
- Modify: `tests/unit/test_codegen.c` (two new tests)
- Create: `tests/e2e/intermediate_repeatable/`, `tests/e2e/intermediate_inferred_dim/`, `tests/e2e/intermediate_dead_write/`, `tests/e2e/intermediate_double_write/`
- Replace: `tests/e2e/many_inputs_chain/{input.edge,expected_output.txt}`

**Interfaces:**
- Consumes: Task 1's guarantee that a valid generational program writes only generational tensors.
- Produces: `static void collect_io(IRProgram *ir, TensorNameList *inputs, TensorNameList *written, char **output)` (new `written` out-parameter, caller frees it with `free_tensor_name_list`), and `static int name_in_tnl(const TensorNameList *l, const char *name)`.

- [x] **Step 1: Add the `run_twice` check to the e2e harness**

In `tests/e2e/run_e2e.py`, insert immediately before the comment `# Round-trip check: re-read a saved .mtx and compare to expected values.`:

```python
        # Optional repeatability check: a `run_twice` marker re-runs the program's
        # __main__ in-process (output captured), then calls run() again on the
        # same input objects and requires the same result, so run() must not
        # keep state in its arguments.
        if os.path.exists(os.path.join(test_dir, "run_twice")):
            snippet = (
                "import contextlib, io, runpy, sys\n"
                "path = sys.argv[1]\n"
                "sys.argv = sys.argv[1:]\n"
                "with contextlib.redirect_stdout(io.StringIO()):\n"
                "    g = runpy.run_path(path, run_name='__main__')\n"
                "m = g['__edgecc_meta__']\n"
                "first = g[m['output']]\n"
                "second = g['run'](*[g[n] for n in m['inputs']])\n"
                "if first != second:\n"
                "    sys.exit('run() is not repeatable: %r then %r' % (first, second))\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", snippet, tmp_py, *extra_args],
                capture_output=True, text=True, timeout=20, cwd=test_dir
            )
            if result.returncode != 0:
                print(f"  FAIL {name}: {result.stderr.strip()}")
                return False
```

- [x] **Step 2: Write the e2e tests**

`tests/e2e/intermediate_repeatable/input.edge` (plus an empty marker file `tests/e2e/intermediate_repeatable/run_twice`):
```
// B is an intermediate: allocated inside run(), not a parameter (G5), so a
// second run() call does not accumulate into the caller's buffer.
dim N = 2;
tensor A[R=N] : float, empty = 0.0;
tensor B[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0];
B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);
```
`expected_output.txt`: `[2.0, 6.0]`

`tests/e2e/intermediate_inferred_dim/` — files `input.edge`, `g.mtx`, `expected_output.txt`, empty markers `needs_scipy` and `run_twice`:
```
// dim N = * plus an intermediate: __main__ used to allocate P with N before N
// existed (G6, NameError). P is now allocated inside run(), after N = len(G).
dim N = *;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor P[R=N, C=N] : float, empty = 0.0;
tensor Z[R=N, C=N] : float, empty = 0.0;
init G = load("g.mtx");
P[r,c] = G[r,k] . G[k,c] :: map(mul, intersect) reduce(add, union);
Z[r,c] = P[r,k] . G[k,c] :: map(mul, intersect) reduce(add, union);
```
`g.mtx`:
```
%%MatrixMarket matrix array real general
2 2
1.0
0.0
1.0
1.0
```
(column-major: G = [[1, 1], [0, 1]], so G³ = [[1, 3], [0, 1]].)
`expected_output.txt`: `[[1.0, 3.0], [0.0, 1.0]]`

`tests/e2e/intermediate_dead_write/input.edge`:
```
// B is written but never read and is not the returned tensor. It used to be
// left unallocated (NameError); it is now allocated and computed.
dim N = 2;
tensor A[R=N] : float, empty = 0.0;
tensor B[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0];
B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
Z[i] = A[i] . A[i] :: map(add, intersect) reduce(add, union);
```
`expected_output.txt`: `[2.0, 4.0]`

`tests/e2e/intermediate_double_write/input.edge`:
```
// Two Einsums write B (the paper allows several Einsums per output); C reads B
// between the writes, which is not a read-before-write. Both writes accumulate
// into one buffer: B = [1,4] then [2,8]; C = [2,6]; Z = C + B = [4,14].
// (Characterization test: passes before and after the change.)
dim N = 2;
tensor A[R=N] : float, empty = 0.0;
tensor B[R=N] : float, empty = 0.0;
tensor C[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0];
B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
C[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);
B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
Z[i] = C[i] . B[i] :: map(add, intersect) reduce(add, union);
```
`expected_output.txt`: `[4.0, 14.0]`

Replace `tests/e2e/many_inputs_chain/input.edge` (its chain tensors are about to become intermediates, which would leave only one input and stop testing G4) with nine *true* inputs:
```
// More than 8 true run() inputs must all survive codegen (DFT gap G4). The
// chain tensors T1..T7 are intermediates and are not parameters.
dim N = 2;
tensor X0[R=N] : float, empty = 0.0;
tensor X1[R=N] : float, empty = 0.0;
tensor X2[R=N] : float, empty = 0.0;
tensor X3[R=N] : float, empty = 0.0;
tensor X4[R=N] : float, empty = 0.0;
tensor X5[R=N] : float, empty = 0.0;
tensor X6[R=N] : float, empty = 0.0;
tensor X7[R=N] : float, empty = 0.0;
tensor X8[R=N] : float, empty = 0.0;
tensor T1[R=N] : float, empty = 0.0;
tensor T2[R=N] : float, empty = 0.0;
tensor T3[R=N] : float, empty = 0.0;
tensor T4[R=N] : float, empty = 0.0;
tensor T5[R=N] : float, empty = 0.0;
tensor T6[R=N] : float, empty = 0.0;
tensor T7[R=N] : float, empty = 0.0;
tensor Z[R=N]  : float, empty = 0.0;
init X0 = [1.0, 2.0];
init X1 = [1.0, 2.0];
init X2 = [1.0, 2.0];
init X3 = [1.0, 2.0];
init X4 = [1.0, 2.0];
init X5 = [1.0, 2.0];
init X6 = [1.0, 2.0];
init X7 = [1.0, 2.0];
init X8 = [1.0, 2.0];
T1[i] = X0[i] . X1[i] :: map(add, intersect) reduce(add, union);
T2[i] = T1[i] . X2[i] :: map(add, intersect) reduce(add, union);
T3[i] = T2[i] . X3[i] :: map(add, intersect) reduce(add, union);
T4[i] = T3[i] . X4[i] :: map(add, intersect) reduce(add, union);
T5[i] = T4[i] . X5[i] :: map(add, intersect) reduce(add, union);
T6[i] = T5[i] . X6[i] :: map(add, intersect) reduce(add, union);
T7[i] = T6[i] . X7[i] :: map(add, intersect) reduce(add, union);
Z[i]  = T7[i] . X8[i] :: map(add, intersect) reduce(add, union);
```
`expected_output.txt`: `[9.0, 18.0]`

- [x] **Step 3: Write the codegen unit tests**

Append to `tests/unit/test_codegen.c` before `int main`, and add `RUN_TEST(test_codegen_intermediate_not_a_param);` and `RUN_TEST(test_codegen_fused_intermediate_not_a_param);` at the end of `main`'s list:

```c
TEST(test_codegen_intermediate_not_a_param) {
    /* B is written by an Einsum: allocated empty inside run(), not a parameter,
       and not allocated by __main__ (spec: intermediates out of the run() ABI). */
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor A[R=N] : float, empty = 0.0;\n"
        "tensor B[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init A = [1.0, 2.0];\n"
        "B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);\n"
        "Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "\"inputs\": [\"A\"]") != NULL);
    ASSERT_TRUE(strstr(code, "def run(A):\n    B = [0.0] * N\n    Z = [0.0] * N\n") != NULL);
    ASSERT_TRUE(strstr(code, "Z = run(A)") != NULL);
    char *main_block = strstr(code, "if __name__ == \"__main__\":");
    ASSERT_NOT_NULL(main_block);
    ASSERT_TRUE(strstr(main_block, "B = ") == NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_fused_intermediate_not_a_param) {
    /* T is written by a fused Einsum and read by the next one. */
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "tensor T[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init G = [[0,1,0],[0,0,1],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "T[d] = (G[s,d] .1 F[s]) .2 !P[d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);\n"
        "Z[d] = T[d] . F[d] :: map(or, union) reduce(or, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "def run(G, F, P):") != NULL);
    ASSERT_TRUE(strstr(code, "    T = [0.0] * N\n") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```

- [x] **Step 4: Run the new tests and confirm they fail for the right reasons**

Run: `make -C build && ./build/test_codegen | tail -3; python3 tests/e2e/run_e2e.py build/edgecc 2>&1 | grep -i "fail\|passed"`
Expected:
- `test_codegen`: the two new tests FAIL (signature still `def run(A, B):` / `def run(G, F, P, T):` or similar).
- e2e: `intermediate_repeatable` FAILS with `run() is not repeatable: [2.0, 6.0] then [3.0, 10.0]`; `intermediate_inferred_dim` FAILS with `NameError: name 'N' is not defined`; `intermediate_dead_write` FAILS with `NameError: name 'B' is not defined`; `intermediate_double_write` and `many_inputs_chain` PASS (characterization).
These are the transiently red tests until Step 5.

- [x] **Step 5: Implement in `src/codegen.c`**

(a) Add directly after `add_input_name`:

```c
static int name_in_tnl(const TensorNameList *l, const char *name) {
    for (int i = 0; i < l->count; i++)
        if (strcmp(l->names[i], name) == 0) return 1;
    return 0;
}
```

(b) Replace `collect_io` with a version that also records every stored tensor:

```c
/* Read tensors (IR loads, in load order) go to `inputs`; every tensor an
   Einsum stores to goes to `written`; the last stored tensor is `output`.
   Caller frees both lists and the output string. */
static void collect_io(IRProgram *ir, TensorNameList *inputs, TensorNameList *written,
                       char **output) {
    inputs->names = NULL;  inputs->count = 0;  inputs->cap = 0;
    written->names = NULL; written->count = 0; written->cap = 0;
    *output = NULL;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type == IR_LOAD) {
            add_input_name(inputs, instr->data.load.tensor_name);
        } else if (instr->type == IR_STORE) {
            add_input_name(written, instr->data.store.tensor_name);
            if (*output) free(*output);
            *output = strdup(instr->data.store.tensor_name);
        } else if (instr->type == IR_FUSED) {
            IRFused *fu = &instr->data.fused;
            add_input_name(inputs, fu->inner1.name);
            add_input_name(inputs, fu->inner2.name);
            add_input_name(inputs, fu->outer.name);
            add_input_name(written, fu->out_name);
            if (*output) free(*output);
            *output = strdup(fu->out_name);
        }
    }
}
```

(c) In `codegen_python_ex`, change the declaration and call

```c
    TensorNameList inputs;
    char *output = NULL;
    collect_io(ir, &inputs, &output);
```
to
```c
    TensorNameList inputs, written;
    char *output = NULL;
    collect_io(ir, &inputs, &written, &output);
```

(d) Extend the cascade I/O block. It currently reads `if (iterative) { int kept = 0; … inputs.count = kept; if (watch) {…} }`. Add an `else` branch after its closing brace:

```c
    } else {
        /* Non-generational: a tensor written by an Einsum starts empty inside
           run() (paper: output tensors are initially empty), so it is never a
           parameter, even when it is also read. Semantic analysis rejects an
           init on it and a read before its first write. */
        int kept = 0;
        for (int i = 0; i < inputs.count; i++) {
            if (name_in_tnl(&written, inputs.names[i])) {
                free(inputs.names[i]); continue;
            }
            inputs.names[kept++] = inputs.names[i];
        }
        inputs.count = kept;
    }
```

(e) In the `IR_TENSOR_ALLOC` case inside `run()`, replace the condition

```c
                if (output && strcmp(instr->data.tensor_alloc.name, output) == 0 &&
                    !(iterative &&
                      (name_in_list(instr->data.tensor_alloc.name, cross_gens, num_cross_gens) ||
                       name_in_list(instr->data.tensor_alloc.name, intermediates, num_intermediates)))) {
```
with
```c
                if ((!iterative && name_in_tnl(&written, instr->data.tensor_alloc.name)) ||
                    (output && strcmp(instr->data.tensor_alloc.name, output) == 0 &&
                     !(iterative &&
                       (name_in_list(instr->data.tensor_alloc.name, cross_gens, num_cross_gens) ||
                        name_in_list(instr->data.tensor_alloc.name, intermediates, num_intermediates))))) {
```
and update the comment above it to: `/* Every tensor an Einsum writes is allocated (empty) inside run(); a generational tensor is instead built by the gen loop (from the caller's generation 0, or empty for an intermediate). */`

(f) At the end of `codegen_python_ex`, next to `free_tensor_name_list(&inputs);`, add `free_tensor_name_list(&written);`.

- [x] **Step 6: Run everything**

Run: `make -C build 2>&1 | grep -E "error|warning:"; ./build/test_codegen | tail -1; python3 tests/e2e/run_e2e.py build/edgecc 2>&1 | grep -i "fail\|passed"; ctest --test-dir build 2>&1 | tail -3`
Expected: no compiler warnings; `test_codegen` all pass (including the four existing generational signature asserts `def run(G, F):`, `def run(W, D):`, `def run(G, F, P):`, `def run(G, A, B):`); all e2e pass; `100% tests passed, 0 tests failed out of 14`.

- [x] **Step 7: Commit**

```bash
git add src/codegen.c tests/e2e tests/unit/test_codegen.c
git commit -m "codegen: intermediates are allocated inside run(), not parameters (DFT gaps G5/G6)

run()'s parameters are the tensors read but never written; every tensor an
Einsum writes starts empty inside run(), so __main__ never allocates one (fixes
the inferred-dim NameError) and repeated run() calls no longer accumulate.
e2e harness: optional run_twice marker checks run() is repeatable.
many_inputs_chain rebuilt with 9 true inputs so it still exercises G4."
```

---

### Task 3: Docs, DFT acceptance, and Finish

**Files:**
- Modify: `CLAUDE.md`, `docs/roadmap.md`, `docs/superpowers/specs/2026-10-06-edgecc-intermediates-out-of-abi-design.md` (deviations section), this plan (checkboxes)

**Interfaces:**
- Consumes: Tasks 1–2 on the branch.
- Produces: the merged `main`.

- [x] **Step 1: Update `CLAUDE.md`**

(a) In "## Generated Python ABI", after the sentence ending `…edgecc prints a warning for it.`, insert: `` `run()`'s parameters are exactly the tensors that are read but never written; every tensor an Einsum writes (the returned output and all intermediates) starts empty and is allocated inside `run()` (the paper: "all output tensors are initially empty"), so it may not have an `init` and may not be read before the first Einsum that writes it. Each `run()` call starts from fresh buffers. ``

(b) In the paragraph under "## EDGE Syntax (v2)" that begins `Numeric literals may be negative`, append: `` A tensor written by an Einsum may not have an `init` and may not be read before its first write (outputs start empty). ``

(c) In "## v2 Scope Limitations", after the size-limits paragraph, add a paragraph: `` In a generational program every Einsum must write a generational tensor; a loop-invariant precompute (e.g. `G2 = G·G`) must be done in a separate program for now (hoisting it before the loop is future work). ``

- [x] **Step 2: Update `docs/roadmap.md`**

Set `**Last updated:**` to the current date. Under "### Implemented", after the "Robustness (DFT gaps, batch A)" bullet, add: `` - **Intermediates out of the ABI (DFT gaps, batch B):** `run()` takes only read-but-never-written tensors; written tensors start empty inside `run()` (fixes `dim N = *` + intermediate); no `init` on, or read before, a written tensor; non-generational writes rejected in generational programs. `` Under "### Planned ❌" → "Language", add `hoisting loop-invariant Einsums out of a generational loop;`.

- [x] **Step 3: DFT acceptance (read-only use of the repo root)**

Mirror the DFT files into a scratch directory so `dft_ref.py` uses this branch's compiler (it locates `build/edgecc` relative to its own repo):

```bash
S=$(mktemp -d)
ROOT=/Users/porumbes/Dropbox/Home/Serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc
mkdir -p $S/repo/docs $S/repo/build
cp -R $ROOT/docs/dft $S/repo/docs/
ln -s "$PWD/build/edgecc" $S/repo/build/edgecc
cd $S/repo
K=docs/dft/kernels; C="python3 docs/dft/reference/dft_ref.py check"
$C $K/dft_naive_re.edge $K/dft_naive_im.edge --n 16                       # G6: no --direct
$C $K/dft_naive_n4_re.edge $K/dft_naive_n4_im.edge --x "1, 2+1j, 3j, 0.5+0.5j"
$C $K/dft_naive_n4_packed.edge --x "1, 2+1j, 3j, 0.5+0.5j"
$C $K/dft_stacked_flat.edge --layout interleaved --n 16
$C $K/dft_stacked.edge --layout cols --n 16 --direct
$C $K/dft_stacked_jx.edge --layout cols --n 16 --direct
$C $K/fft_split1.edge --layout split --n 16 --direct
```
Expected: every line ends `-> PASS`. The first line (no `--direct`) is the G6 acceptance test. The `--direct` runs may print a `[G4 hack]` line naming intermediates; that is the expected, harmless false positive described in the spec. If any DFT kernel is rejected by E1/E2/E3, stop and report it to the user rather than editing DFT files (they belong to the exploration session).

- [x] **Step 4: Pre-merge gate**

Run: `tools/premerge.sh`
Expected: `GATE PASSED (6 steps)`.

- [x] **Step 5: Record deviations, tick the plan, commit**

Append a `## Deviations and findings (recorded at Finish, <date>)` section to the spec listing anything that differed from it (write "None." if nothing did). Tick every checkbox in this plan. Then:

```bash
git add CLAUDE.md docs/roadmap.md docs/superpowers
git commit -m "docs: intermediates out of the run() ABI (batch B) — CLAUDE.md, roadmap, spec deviations, plan ticked"
```

- [ ] **Step 6: Merge from the `main` worktree**

```bash
cd ../premerge-gate          # .worktrees/premerge-gate has main checked out
git merge --no-ff feature/intermediates-abi -m "Merge feature/intermediates-abi: intermediates out of the run() ABI (DFT gaps G5/G6, batch B)"
make -C build && ctest --test-dir build 2>&1 | tail -3
```
Expected: `100% tests passed`.

- [ ] **Step 7: Memory and hand-off**

Update the auto-memory `project_resume_state.md` (main's new hash, batch B done, next = batch C multi-output `run()`), and tell the user that `main` moved so the DFT session can merge it, drop `--direct` where it only worked around G6, and retire the `[G4 hack]` in `dft_ref.py`.
