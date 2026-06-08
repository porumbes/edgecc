# Graphics — Per-Generation Visual Stepper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `edgecc --emit-trace` instrument generational kernels to write a per-generation JSON trace, and add a standalone `tools/edgecc_viz.py` that renders each generation as a PNG frame (graph + tensor panel, dark theme, cyan active edges).

**Architecture:** Compiler stays viz-free: `--emit-trace` adds stdlib-`json` trace capture to the generated `run()` (default-off → byte-identical otherwise). A separate Python tool (lazy matplotlib + networkx, skip-if-missing) consumes `{trace.json + adjacency .mtx}` and emits frames. The trace file is the contract between them.

**Tech Stack:** C99/Flex/Bison + CMake (compiler); generated Python 3 (`json` only for capture); `tools/edgecc_viz.py` with matplotlib + networkx + scipy; ctest.

**Spec:** `docs/superpowers/specs/2026-06-08-edgecc-graphics-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/codegen.h` / `src/codegen.c` | Add `codegen_python_ex(ir, emit_trace, kernel_name)`; emit trace capture for generational kernels | Modify |
| `src/main.c` | Parse `--emit-trace`; derive kernel name from input basename; call `_ex` | Modify |
| `tests/unit/test_codegen.c` | Unit-assert trace scaffolding emitted (and absent by default) | Modify |
| `tests/trace/run_trace_tests.py` | Compile a kernel with `--emit-trace`, run it, validate the JSON trace | Create |
| `tests/trace/reachability_bfs/` | Trace test fixture (`input.edge`, `graph.mtx`, `expected_trace.json`-ish checks) | Create |
| `tools/edgecc_viz.py` | Standalone renderer: load trace+graph, active-edge calc, draw frames | Create |
| `tests/viz/run_viz_tests.py` | Test pure functions always; full render skipped if matplotlib/networkx absent | Create |
| `tests/viz/sample/` | Committed `trace.json` + `graph.mtx` fixture for the renderer | Create |
| `CMakeLists.txt` | Wire `trace_tests` and `viz_tests` ctest targets | Modify |
| `CLAUDE.md` | Document `--emit-trace`, the trace format, and `tools/edgecc_viz.py` | Modify |

**Conventions to follow:**
- `CodeBuf` + `cb_append`/`cb_appendf`/`cb_indent`/`cb.indent` for emission.
- Emit helper blocks once near the top of the module (mirror `_edgecc_load`/`_edgecc_save` in `codegen.c`).
- Test runners are plain Python invoked by ctest (mirror `tests/e2e/run_e2e.py`); a missing optional dep prints `SKIP` and exits 0 for that suite.
- Generational metadata in `codegen_python` already available as locals: `iterative`, `cross_gens[]`/`num_cross_gens`, `intermediates[]`/`num_intermediates`, `watch`.

---

## Task 1: `--emit-trace` plumbing (default-off, byte-identical)

**Files:** Modify `src/codegen.h`, `src/codegen.c`, `src/main.c`, `tests/unit/test_codegen.c`

- [ ] **Step 1: Write failing unit tests**

Add to `tests/unit/test_codegen.c` (register in `main()`). They use the new `_ex` entry point:

```c
TEST(test_codegen_default_no_trace) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 3; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "init A = [[1, 2], [3, 4], [5, 6]];\n"
        "init B = [[7, 8], [9, 10], [11, 12]];\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python_ex(ir, 0, "gemm");
    ASSERT_TRUE(strstr(code, "_trace") == NULL);   /* no trace scaffolding when off */
    free(code);
    ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

TEST(test_codegen_trace_on_nongenerational_warns_no_scaffold) {
    /* emit_trace requested on a non-generational kernel: no _trace scaffolding. */
    AstNode *ast = parse(
        "dim M = 2; dim K = 3; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "init A = [[1, 2], [3, 4], [5, 6]];\n"
        "init B = [[7, 8], [9, 10], [11, 12]];\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python_ex(ir, 1, "gemm");   /* request trace, but not generational */
    ASSERT_TRUE(strstr(code, "_trace") == NULL);
    free(code);
    ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd build && make test_codegen 2>&1 | tail -15`
Expected: FAIL — `codegen_python_ex` undeclared.

- [ ] **Step 3: Add the `_ex` entry point**

In `src/codegen.h`:

```c
/* Generate Python source code from IR. Returns a malloc'd string. */
char *codegen_python(IRProgram *ir);

/* Extended form: emit_trace=1 adds per-generation trace capture for
   generational kernels (no-op for non-generational). kernel_name labels the
   trace and its output file (NULL -> "kernel"). */
char *codegen_python_ex(IRProgram *ir, int emit_trace, const char *kernel_name);
```

In `src/codegen.c`, rename the current `char *codegen_python(IRProgram *ir) {` body to the `_ex` form and add a thin wrapper. At the top of the function, gate trace on generational detection. Replace the function header (`codegen.c:458`):

```c
char *codegen_python(IRProgram *ir) {
    return codegen_python_ex(ir, 0, NULL);
}

char *codegen_python_ex(IRProgram *ir, int emit_trace, const char *kernel_name) {
    if (!ir) return NULL;
    if (!kernel_name) kernel_name = "kernel";
    CodeBuf cb;
    cb_init(&cb);
    ...
```

After the `iterative` flag is computed (search `int iterative = 0;` and the loop that sets it from `IR_GEN_LOOP_BEGIN`), add:

```c
    /* Trace capture is only meaningful for generational kernels. */
    if (emit_trace && !iterative) {
        fprintf(stderr,
            "edgecc: --emit-trace ignored: '%s' is not a generational kernel\n",
            kernel_name);
        emit_trace = 0;
    }
```

(`fprintf`/`stderr` need `#include <stdio.h>` — already present in codegen.c.)

For now, no other behavior: `emit_trace` is carried but unused until Task 2. This keeps Task 1 a pure plumbing change.

- [ ] **Step 4: Plumb the flag through main.c**

In `src/main.c`, add flag parsing next to `--dump-ir`:

```c
    int dump_ir = 0;
    int emit_trace = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--dump-ir") == 0) {
            dump_ir = 1;
        } else if (strcmp(argv[i], "--emit-trace") == 0) {
            emit_trace = 1;
        }
    }
```

Derive a kernel name from the input basename (strip directory and `.edge`) and call the `_ex` form. Replace `char *code = codegen_python(ir);` (`main.c:66`):

```c
    /* kernel name = input basename without directory or .edge extension */
    const char *slash = strrchr(input_file, '/');
    const char *base = slash ? slash + 1 : input_file;
    char kernel_name[256];
    snprintf(kernel_name, sizeof(kernel_name), "%s", base);
    char *dot = strstr(kernel_name, ".edge");
    if (dot) *dot = '\0';

    char *code = codegen_python_ex(ir, emit_trace, kernel_name);
```

- [ ] **Step 5: Run to verify pass + no regressions**

Run: `cd build && make 2>&1 | grep -iE "error|warning" | head; make test_codegen 2>&1 | tail -3 && ./test_codegen | tail -2 && ctest 2>&1 | tail -3`
Expected: builds clean; new tests PASS; all 8 suites PASS.

- [ ] **Step 6: Commit**

```bash
git add src/codegen.h src/codegen.c src/main.c tests/unit/test_codegen.c
git commit -m "feat(codegen): --emit-trace plumbing via codegen_python_ex (no-op until capture)"
```

---

## Task 2: Emit per-generation trace capture

**Files:** Modify `src/codegen.c`, `tests/unit/test_codegen.c`

Emits, only when `emit_trace` (and generational): a snapshot helper, a `_trace=None` param on `run()`, seed + per-generation snapshots, and a `__main__` envelope write.

- [ ] **Step 1: Write failing unit test**

Add to `tests/unit/test_codegen.c` (register in `main()`). Uses the reachability-BFS cascade (two cross-gen tensors F, P):

```c
TEST(test_codegen_trace_scaffolding) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0]];\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "init P = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python_ex(ir, 1, "reachability_bfs");
    ASSERT_TRUE(strstr(code, "def _edgecc_snapshot(") != NULL);
    ASSERT_TRUE(strstr(code, "_trace=None") != NULL);
    ASSERT_TRUE(strstr(code, "\"kind\": \"generation\"") != NULL);
    ASSERT_TRUE(strstr(code, "\"schema\": \"edgecc-trace/1\"") != NULL);
    ASSERT_TRUE(strstr(code, ".trace.json") != NULL);
    free(code);
    ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd build && make test_codegen 2>&1 | tail -8`
Expected: FAIL — scaffolding not emitted.

- [ ] **Step 3: Emit the snapshot helper**

In `codegen.c`, where `_edgecc_load`/`_edgecc_save` are conditionally emitted (search `uses_fileio`), add an analogous block after it:

```c
if (emit_trace) {
    cb_append(&cb,
        "def _edgecc_snapshot(x):\n"
        "    return [row[:] for row in x] if (x and isinstance(x[0], list)) else list(x)\n"
        "\n");
}
```

- [ ] **Step 4: Add the `_trace` parameter to run()**

Find where the `def run(` parameter list is closed (search `cb_append(&cb, \"):\\n\");` right after the inputs loop, ~line 487). Immediately before emitting `):`, append the trace param when enabled. Change the inputs emission so that, after the loop over `inputs`, you add:

```c
    if (emit_trace) cb_append(&cb, ", _trace=None");
    cb_append(&cb, "):\n");
    cb.indent = 1;
```

(Replace the existing `cb_append(&cb, "):\n"); cb.indent = 1;` with the two lines above.)

- [ ] **Step 5: Emit the seed snapshot (gen 0) in GEN_LOOP_BEGIN**

In the `IR_GEN_LOOP_BEGIN` case, after `_it = 0\n` is emitted and **before** `while True:\n` (between `codegen.c:659` and `:660`), insert the seed snapshot of the cross-generational tensors (from their `_prev` buffers):

```c
if (emit_trace) {
    cb_indent(&cb); cb_append(&cb, "if _trace is not None:\n");
    cb.indent++;
    cb_indent(&cb);
    cb_append(&cb, "_trace.append({\"kind\": \"generation\", \"gen\": 0, \"tensors\": {");
    for (int c = 0; c < gloop->num_cross_gens; c++) {
        if (c) cb_append(&cb, ", ");
        cb_appendf(&cb, "\"%s\": _edgecc_snapshot(%s_prev)",
                   gloop->cross_gens[c].name, gloop->cross_gens[c].name);
    }
    cb_append(&cb, "}})\n");
    cb.indent--;
}
```

- [ ] **Step 6: Emit the per-generation snapshot in GEN_LOOP_END**

At the very start of the `IR_GEN_LOOP_END` case (before the convergence check, `codegen.c:686`), snapshot the freshly computed generation: cross-gens from `_next`, intermediates from their plain variable. `_it` is the 0-based iteration, so this generation is `_it + 1`:

```c
if (emit_trace) {
    cb_indent(&cb); cb_append(&cb, "if _trace is not None:\n");
    cb.indent++;
    cb_indent(&cb);
    cb_append(&cb, "_trace.append({\"kind\": \"generation\", \"gen\": _it + 1, \"tensors\": {");
    int first = 1;
    for (int c = 0; c < num_cross_gens; c++) {
        if (!first) cb_append(&cb, ", ");
        cb_appendf(&cb, "\"%s\": _edgecc_snapshot(%s_next)", cross_gens[c], cross_gens[c]);
        first = 0;
    }
    for (int m = 0; m < num_intermediates; m++) {
        if (!first) cb_append(&cb, ", ");
        cb_appendf(&cb, "\"%s\": _edgecc_snapshot(%s)", intermediates[m], intermediates[m]);
        first = 0;
    }
    cb_append(&cb, "}})\n");
    cb.indent--;
}
```

(`cross_gens`, `num_cross_gens`, `intermediates`, `num_intermediates`, `watch` are the locals already populated near the top of `codegen_python_ex` from the gen-loop metadata.)

- [ ] **Step 7: Emit the __main__ envelope write**

In `__main__`, after the `run(...)` call and `print(output)` (search the `print(%s)` emission near `codegen.c:677`), when `emit_trace`, (a) pass `_trace=_tr` to the call and (b) write the envelope. First, change the call emission to thread `_trace` when enabled — locate the loop that emits `output = run(<inputs>)` and, before closing `)`, add `, _trace=_tr` when `emit_trace`. Just before that call, declare the list:

```c
if (emit_trace) { cb_indent(&cb); cb_append(&cb, "_tr = []\n"); }
```

After `print(%s)`, add:

```c
if (emit_trace) {
    cb_indent(&cb); cb_append(&cb, "import json\n");
    cb_indent(&cb);
    cb_appendf(&cb,
        "json.dump({\"schema\": \"edgecc-trace/1\", \"kernel\": \"%s\", "
        "\"dims\": {\"N\": len(%s)}, \"gen_tensors\": [",
        kernel_name, output);
    /* gen_tensors = cross_gens + intermediates */
    {
        int first = 1;
        for (int c = 0; c < num_cross_gens; c++) {
            if (!first) cb_append(&cb, ", ");
            cb_appendf(&cb, "\"%s\"", cross_gens[c]); first = 0;
        }
        for (int m = 0; m < num_intermediates; m++) {
            if (!first) cb_append(&cb, ", ");
            cb_appendf(&cb, "\"%s\"", intermediates[m]); first = 0;
        }
    }
    cb_appendf(&cb, "], \"watch\": \"%s\", \"steps\": _tr}, open(\"%s.trace.json\", \"w\"))\n",
               watch ? watch : "", kernel_name);
}
```

> The call-site change in (a): find the block that emits `%s = run(` … `)` and insert `if (emit_trace) cb_append(&cb, ", _trace=_tr");` immediately before the closing `cb_append(&cb, ")\n");`.

- [ ] **Step 8: Run to verify pass**

Run: `cd build && make test_codegen 2>&1 | tail -3 && ./test_codegen | tail -2`
Expected: PASS. Then full suite: `ctest 2>&1 | tail -3` — all pass (non-trace programs unaffected; `codegen_python(ir)` path sets emit_trace=0).

- [ ] **Step 9: Commit**

```bash
git add src/codegen.c tests/unit/test_codegen.c
git commit -m "feat(codegen): emit per-generation JSON trace capture under --emit-trace"
```

---

## Task 3: Trace e2e test (compile → run → validate JSON)

**Files:** Create `tests/trace/run_trace_tests.py`, `tests/trace/reachability_bfs/{input.edge,graph.mtx}`; Modify `CMakeLists.txt`

- [ ] **Step 1: Create the fixture**

`tests/trace/reachability_bfs/input.edge` (path 0→1→2→3; F frontier, P visited):

```
dim N = 4;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
tensor P[gen I, R=N] : float, empty = 0.0;
init G = [[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0]];
init F = [1.0, 0.0, 0.0, 0.0];
init P = [1.0, 0.0, 0.0, 0.0];
F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
stop: stable(P);
```

`tests/trace/reachability_bfs/graph.mtx` (same 4-node path, for the renderer later / reference):

```
%%MatrixMarket matrix coordinate real general
4 4 3
1 2 1.0
2 3 1.0
3 4 1.0
```

- [ ] **Step 2: Write the trace test runner**

Create `tests/trace/run_trace_tests.py`:

```python
#!/usr/bin/env python3
"""Trace test driver: compile a kernel with --emit-trace, run it, validate JSON."""
import json
import os
import subprocess
import sys
import tempfile

def run_test(test_dir, edgecc_bin):
    name = os.path.basename(test_dir)
    src = os.path.join(test_dir, "input.edge")
    if not os.path.exists(src):
        return None
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as t:
        tmp_py = t.name
    try:
        r = subprocess.run([edgecc_bin, src, "-o", tmp_py, "--emit-trace"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print(f"  FAIL {name}: edgecc failed: {r.stderr.strip()}")
            return False
        r = subprocess.run([sys.executable, tmp_py],
                           capture_output=True, text=True, timeout=10, cwd=test_dir)
        if r.returncode != 0:
            print(f"  FAIL {name}: run failed: {r.stderr.strip()}")
            return False
        trace_path = os.path.join(test_dir, f"{name}.trace.json")
        if not os.path.exists(trace_path):
            print(f"  FAIL {name}: no trace file written")
            return False
        with open(trace_path) as f:
            tr = json.load(f)
        os.unlink(trace_path)
        # Structural checks.
        if tr.get("schema") != "edgecc-trace/1":
            print(f"  FAIL {name}: bad schema {tr.get('schema')}"); return False
        if tr.get("watch") != "P" or "F" not in tr.get("gen_tensors", []):
            print(f"  FAIL {name}: bad metadata {tr.get('gen_tensors')}/{tr.get('watch')}")
            return False
        steps = tr.get("steps", [])
        # gen 0 seed + at least 3 computed gens (reaches node 3 in 3 hops).
        if len(steps) < 4:
            print(f"  FAIL {name}: too few steps ({len(steps)})"); return False
        if steps[0]["gen"] != 0 or steps[0]["tensors"]["P"] != [1.0, 0.0, 0.0, 0.0]:
            print(f"  FAIL {name}: bad seed {steps[0]}"); return False
        # Final visited set must be all-reached (path graph from node 0).
        if steps[-1]["tensors"]["P"] != [1.0, 1.0, 1.0, 1.0]:
            print(f"  FAIL {name}: final P {steps[-1]['tensors']['P']}"); return False
        print(f"  PASS {name}")
        return True
    finally:
        os.unlink(tmp_py)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    edgecc_bin = sys.argv[1] if len(sys.argv) > 1 else "edgecc"
    passed = failed = 0
    for entry in sorted(os.listdir(script_dir)):
        d = os.path.join(script_dir, entry)
        if not os.path.isdir(d):
            continue
        res = run_test(d, edgecc_bin)
        if res is None:
            continue
        passed += res
        failed += (not res)
    print(f"\n{passed}/{passed + failed} trace tests passed")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Wire into ctest**

In `CMakeLists.txt`, after the `negative_tests` `add_test(...)` block, add:

```cmake
add_test(
    NAME trace_tests
    COMMAND ${CMAKE_COMMAND} -E env python3
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/trace/run_trace_tests.py
        $<TARGET_FILE:edgecc>
)
```

(This matches the existing `e2e_tests`/`negative_tests` invocation style exactly.)

- [ ] **Step 4: Run**

Run: `cd build && cmake .. >/dev/null && ctest -R trace_tests --output-on-failure 2>&1 | tail -8`
Expected: PASS `reachability_bfs`.

- [ ] **Step 5: Commit**

```bash
git add tests/trace CMakeLists.txt
git commit -m "test(trace): compile with --emit-trace, run, validate per-generation JSON"
```

---

## Task 4: Renderer core — load, validate, active edges, theme (no plotting)

**Files:** Create `tools/edgecc_viz.py`; Create `tests/viz/run_viz_tests.py`, `tests/viz/sample/{trace.json,graph.mtx}`; Modify `CMakeLists.txt`

This task builds only the **pure** parts (no matplotlib/networkx import at module load), so they are testable everywhere.

- [ ] **Step 1: Write the sample fixture**

`tests/viz/sample/graph.mtx`:

```
%%MatrixMarket matrix coordinate real general
4 4 3
1 2 1.0
2 3 1.0
3 4 1.0
```

`tests/viz/sample/trace.json` (hand-authored BFS over that path; gen 0 seed + 3 steps):

```json
{
  "schema": "edgecc-trace/1",
  "kernel": "reachability_bfs",
  "dims": {"N": 4},
  "gen_tensors": ["F", "P"],
  "watch": "P",
  "steps": [
    {"kind": "generation", "gen": 0, "tensors": {"F": [1.0,0.0,0.0,0.0], "P": [1.0,0.0,0.0,0.0]}},
    {"kind": "generation", "gen": 1, "tensors": {"F": [0.0,1.0,0.0,0.0], "P": [1.0,1.0,0.0,0.0]}},
    {"kind": "generation", "gen": 2, "tensors": {"F": [0.0,0.0,1.0,0.0], "P": [1.0,1.0,1.0,0.0]}},
    {"kind": "generation", "gen": 3, "tensors": {"F": [0.0,0.0,0.0,1.0], "P": [1.0,1.0,1.0,1.0]}}
  ]
}
```

- [ ] **Step 2: Write failing pure-function tests**

Create `tests/viz/run_viz_tests.py`:

```python
#!/usr/bin/env python3
"""Viz tests. Pure-function tests always run; full rendering is skipped if
matplotlib/networkx/scipy are unavailable."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, TOOLS)

import edgecc_viz as viz   # must import WITHOUT importing matplotlib/networkx

results = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'} {name}")
    results.append(bool(cond))

def test_load_trace():
    tr = viz.load_trace(os.path.join(HERE, "sample", "trace.json"))
    check("load_trace schema", tr["schema"] == "edgecc-trace/1")
    check("load_trace steps", len(tr["steps"]) == 4)

def test_active_edges():
    # adjacency: 0->1->2->3 ; prev F lit at node1, curr F lit at node2
    adj = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]
    prev = {"F": [0.0,1.0,0.0,0.0]}
    curr = {"F": [0.0,0.0,1.0,0.0]}
    edges = viz.active_edges(adj, prev, curr, "F")
    # node1 was active and node2 changed -> edge (1,2) is active
    check("active_edges finds 1->2", (1, 2) in edges)
    check("active_edges excludes 0->1", (0, 1) not in edges)

def test_bad_schema():
    try:
        viz.validate_trace({"schema": "nope"})
        check("validate_trace rejects bad schema", False)
    except ValueError:
        check("validate_trace rejects bad schema", True)

test_load_trace(); test_active_edges(); test_bad_schema()
print(f"\n{sum(results)}/{len(results)} viz pure-function checks passed")
sys.exit(1 if not all(results) else 0)
```

- [ ] **Step 3: Run to verify failure**

Run: `python3 tests/viz/run_viz_tests.py`
Expected: FAIL — `tools/edgecc_viz.py` does not exist / functions undefined.

- [ ] **Step 4: Implement the pure core of `tools/edgecc_viz.py`**

Create `tools/edgecc_viz.py` (matplotlib/networkx imported **lazily inside render functions only**, added in Task 5):

```python
#!/usr/bin/env python3
"""edgecc visualization: render a per-generation trace to PNG frames.

Usage:
    python tools/edgecc_viz.py TRACE.json --graph ADJ.mtx -o FRAMES_DIR \
        [--color TENSOR] [--layout spring] [--theme dark]
"""
import argparse
import json
import os

SCHEMA = "edgecc-trace/1"

# --- Theme palette objects (only "dark" implemented; structured for more) ---
THEMES = {
    "dark": {
        "bg": "#000000",
        "node_cmap": "bone",        # cream->dark ramp
        "edge": "#444444",
        "active_edge": "#22d3ee",   # cyan
        "panel_cmap": "bone",
        "text": "#cccccc",
    },
}

def load_trace(path):
    with open(path) as f:
        tr = json.load(f)
    validate_trace(tr)
    return tr

def validate_trace(tr):
    if tr.get("schema") != SCHEMA:
        raise ValueError(f"edgecc_viz: unsupported trace schema {tr.get('schema')!r} "
                         f"(expected {SCHEMA!r})")
    if "steps" not in tr or not isinstance(tr["steps"], list):
        raise ValueError("edgecc_viz: trace has no 'steps' list")
    return tr

def load_graph(path):
    """Return a dense adjacency matrix (list of lists) from a Matrix Market file."""
    try:
        from scipy.io import mmread
    except ImportError:
        raise ImportError("edgecc_viz requires scipy: pip install scipy")
    m = mmread(path)
    arr = m.toarray() if hasattr(m, "toarray") else m
    return arr.tolist()

def _active(v):
    return v != 0.0

def active_edges(adj, prev_tensors, curr_tensors, color_tensor):
    """Edges (s, d) where s was active last generation and d's value changed."""
    prev = prev_tensors.get(color_tensor)
    curr = curr_tensors.get(color_tensor)
    n = len(adj)
    edges = set()
    if prev is None or curr is None:
        return edges
    for s in range(n):
        if not _active(prev[s]):
            continue
        for d in range(n):
            if adj[s][d] != 0.0 and curr[d] != prev[d]:
                edges.add((s, d))
    return edges

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render an edgecc trace to PNG frames.")
    p.add_argument("trace")
    p.add_argument("--graph", required=True)
    p.add_argument("-o", "--out", default="frames")
    p.add_argument("--color", default=None, help="tensor to color nodes by (default: watch)")
    p.add_argument("--layout", default="spring")
    p.add_argument("--theme", default="dark")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if args.theme not in THEMES:
        raise ValueError(f"edgecc_viz: unknown theme {args.theme!r} "
                         f"(available: {', '.join(sorted(THEMES))})")
    tr = load_trace(args.trace)
    adj = load_graph(args.graph)
    n = len(adj)
    if n != tr.get("dims", {}).get("N", n):
        raise ValueError(f"edgecc_viz: graph has {n} nodes but trace dims.N="
                         f"{tr['dims']['N']}")
    color = args.color or tr.get("watch")
    if color not in tr.get("gen_tensors", []):
        raise ValueError(f"edgecc_viz: --color {color!r} not in gen_tensors "
                         f"{tr.get('gen_tensors')}")
    from _render import render_all   # local import added in Task 5
    render_all(tr, adj, color, args.out, THEMES[args.theme], args.layout)

if __name__ == "__main__":
    main()
```

> The `from _render import render_all` line will be replaced in Task 5 by an in-file function; for Task 4 it is never reached by the pure tests. To keep Task 4 importable and green, define a stub at module scope now so the import in `main()` is the only forward reference:
> Add near the bottom (above `if __name__`):
> ```python
> def render_all(*a, **k):
>     raise NotImplementedError("rendering implemented in Task 5")
> ```
> and change `main()`'s `from _render import render_all` to just call the module-level `render_all(...)`.

- [ ] **Step 5: Run to verify pass**

Run: `python3 tests/viz/run_viz_tests.py`
Expected: PASS — 5/5 pure-function checks (import works without matplotlib).

- [ ] **Step 6: Wire viz_tests into ctest**

In `CMakeLists.txt`, after `trace_tests`:

```cmake
add_test(
    NAME viz_tests
    COMMAND ${CMAKE_COMMAND} -E env python3
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/viz/run_viz_tests.py
)
```

- [ ] **Step 7: Commit**

```bash
git add tools/edgecc_viz.py tests/viz CMakeLists.txt
git commit -m "feat(viz): renderer core — trace load/validate, active-edge calc, themes (no plotting)"
```

---

## Task 5: Renderer — draw frames (graph + tensor panel)

**Files:** Modify `tools/edgecc_viz.py`, `tests/viz/run_viz_tests.py`

- [ ] **Step 1: Add a render test (skipped if deps missing)**

Append to `tests/viz/run_viz_tests.py`, before the final summary `print`/`exit`:

```python
def test_render_frames():
    try:
        import matplotlib  # noqa: F401
        import networkx    # noqa: F401
        import scipy        # noqa: F401
    except ImportError:
        print("  SKIP test_render_frames: matplotlib/networkx/scipy not installed")
        return
    import tempfile, glob
    out = tempfile.mkdtemp()
    viz.main([os.path.join(HERE, "sample", "trace.json"),
              "--graph", os.path.join(HERE, "sample", "graph.mtx"),
              "-o", out])
    frames = sorted(glob.glob(os.path.join(out, "gen_*.png")))
    check("render produced 4 frames", len(frames) == 4)
    check("frames non-empty", all(os.path.getsize(f) > 0 for f in frames))

test_render_frames()
```

(Place this call with the other `test_*()` calls, before the summary.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/viz/run_viz_tests.py`
Expected: FAIL (if deps present) — `render_all` raises `NotImplementedError`; or SKIP if deps absent. If deps are absent locally, install them: `python3 -m pip install --user matplotlib networkx scipy`.

- [ ] **Step 3: Implement rendering**

In `tools/edgecc_viz.py`, replace the `render_all` stub with the real implementation (lazy imports inside):

```python
def _layout(adj, kind, seed=1):
    import networkx as nx
    n = len(adj)
    directed = any(adj[i][j] != adj[j][i] for i in range(n) for j in range(n))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i][j] != 0.0:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, seed=seed) if kind == "spring" else nx.circular_layout(G)
    return G, pos

def render_all(tr, adj, color, out_dir, theme, layout_kind):
    import matplotlib
    matplotlib.use("Agg")            # headless
    import matplotlib.pyplot as plt
    import networkx as nx

    os.makedirs(out_dir, exist_ok=True)
    G, pos = _layout(adj, layout_kind)
    steps = tr["steps"]
    gen_tensors = tr.get("gen_tensors", [])

    for idx, step in enumerate(steps):
        prev = steps[idx - 1]["tensors"] if idx > 0 else step["tensors"]
        cur = step["tensors"]
        act = active_edges(adj, prev, cur, color)

        fig = plt.figure(figsize=(10, 5), facecolor=theme["bg"])
        # Left: graph
        axg = fig.add_axes([0.02, 0.05, 0.62, 0.9]); axg.set_facecolor(theme["bg"])
        axg.axis("off")
        node_vals = cur.get(color, [0.0] * len(adj))
        nx.draw_networkx_edges(G, pos, ax=axg, edge_color=theme["edge"], width=0.6, alpha=0.5)
        if act:
            # cyan glow: a few strokes of increasing width / decreasing alpha
            for w, a in [(5.0, 0.15), (3.0, 0.3), (1.6, 0.9)]:
                nx.draw_networkx_edges(G, pos, ax=axg, edgelist=sorted(act),
                                       edge_color=theme["active_edge"], width=w, alpha=a)
        nx.draw_networkx_nodes(G, pos, ax=axg, node_color=node_vals,
                               cmap=plt.get_cmap(theme["node_cmap"]),
                               vmin=0.0, vmax=max(1.0, max(node_vals) if node_vals else 1.0),
                               edgecolors=theme["edge"], node_size=300)
        axg.set_title(f"gen {step['gen']}  (color: {color})", color=theme["text"])

        # Right: tensor panels, stacked
        k = max(1, len(gen_tensors))
        for ti, tname in enumerate(gen_tensors):
            axp = fig.add_axes([0.68, 0.05 + ti * (0.9 / k), 0.3, (0.9 / k) * 0.8])
            vals = cur.get(tname, [])
            row = [vals] if (vals and not isinstance(vals[0], list)) else vals
            axp.imshow(row, cmap=plt.get_cmap(theme["panel_cmap"]),
                       vmin=0.0, vmax=1.0, aspect="auto")
            axp.set_yticks([]); axp.set_xticks([])
            axp.set_ylabel(tname, color=theme["text"], rotation=0, labelpad=12)
            for s in axp.spines.values():
                s.set_color(theme["text"])

        fig.savefig(os.path.join(out_dir, f"gen_{idx:03d}.png"),
                    facecolor=theme["bg"], dpi=110)
        plt.close(fig)
```

Remove the `NotImplementedError` stub.

- [ ] **Step 4: Run to verify pass**

Run: `python3 tests/viz/run_viz_tests.py`
Expected: PASS (or SKIP for the render test if deps missing). With deps: "render produced 4 frames" + "frames non-empty" pass.

- [ ] **Step 5: Manual smoke (optional, if deps installed)**

Run:
```bash
python3 tools/edgecc_viz.py tests/viz/sample/trace.json --graph tests/viz/sample/graph.mtx -o /tmp/frames
ls /tmp/frames    # gen_000.png .. gen_003.png
```
Expected: four PNGs; opening them shows the frontier moving 0→1→2→3 with cyan active edges.

- [ ] **Step 6: Commit**

```bash
git add tools/edgecc_viz.py tests/viz/run_viz_tests.py
git commit -m "feat(viz): render per-generation frames (graph + tensor panel, dark theme, cyan active edges)"
```

---

## Task 6: Negative-path coverage for the renderer

**Files:** Modify `tests/viz/run_viz_tests.py`

- [ ] **Step 1: Add negative tests**

Append before the summary in `tests/viz/run_viz_tests.py`:

```python
def test_size_mismatch():
    # trace dims.N=4 but a 3-node graph
    import tempfile
    g = tempfile.NamedTemporaryFile("w", suffix=".mtx", delete=False)
    g.write("%%MatrixMarket matrix coordinate real general\n3 3 2\n1 2 1.0\n2 3 1.0\n")
    g.close()
    try:
        viz.main([os.path.join(HERE, "sample", "trace.json"), "--graph", g.name, "-o", "/tmp/x"])
        check("size mismatch raises", False)
    except ValueError:
        check("size mismatch raises", True)
    except ImportError:
        print("  SKIP test_size_mismatch: scipy not installed")

def test_bad_color():
    try:
        viz.main([os.path.join(HERE, "sample", "trace.json"),
                  "--graph", os.path.join(HERE, "sample", "graph.mtx"),
                  "--color", "ZZZ", "-o", "/tmp/x"])
        check("bad --color raises", False)
    except ValueError:
        check("bad --color raises", True)
    except ImportError:
        print("  SKIP test_bad_color: scipy not installed")

test_size_mismatch(); test_bad_color()
```

> Note: `test_bad_color` reaches the `--color` validation only after `load_graph` (needs scipy). Both guard `ImportError` and SKIP if scipy is absent.

- [ ] **Step 2: Run + commit**

Run: `python3 tests/viz/run_viz_tests.py`
Expected: PASS (or SKIP where scipy absent).

```bash
git add tests/viz/run_viz_tests.py
git commit -m "test(viz): negative paths — size mismatch and unknown --color"
```

---

## Task 7: Docs + example

**Files:** Modify `CLAUDE.md`

- [ ] **Step 1: Document the feature**

Add a new subsection after the "File input/output" subsection in CLAUDE.md's **EDGE Syntax (v2)** area:

```markdown
### Visualization (per-generation trace + renderer)

Compile a generational kernel with `--emit-trace` to record a per-generation
JSON trace, then render it to PNG frames:

```
./build/edgecc kernel.edge -o kernel.py --emit-trace
python3 kernel.py                       # writes <kernel>.trace.json
python3 tools/edgecc_viz.py kernel.trace.json --graph graph.mtx -o frames/
```

`--emit-trace` is **default-off and only affects generational kernels** (a clear
warning is printed and ignored otherwise); without it the generated Python is
byte-identical to before. The trace (`edgecc-trace/1`) records every
generational tensor's value per generation; the schema is an ordered, tagged
`steps` list so finer (per-scalar) tracing can be added later.

`tools/edgecc_viz.py` reads `{trace.json + adjacency .mtx}` and emits one frame
per generation: the graph (nodes colored by the watched tensor, cyan "active"
edges that propagated this generation) beside a heatmap panel of the tensor
values. It lazily imports `matplotlib`, `networkx`, and `scipy`
(`pip install matplotlib networkx scipy`); `viz_tests` skip the rendering checks
when those are absent. Only the `dark` theme ships; `--theme` is structured for
print themes later. See `docs/superpowers/specs/2026-06-08-edgecc-graphics-design.md`.
```

In **v2 Scope Limitations**, append: *Per-generation visualization is available
via `--emit-trace` + `tools/edgecc_viz.py` (generational kernels only); per-scalar
stepping, interactive scrubbing, animation export, and print themes are deferred.*

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document --emit-trace and tools/edgecc_viz.py"
```

---

## Task 8: Full verification

- [ ] **Step 1: Clean build + full suite**

Run:
```bash
rm -rf build && mkdir build && cd build && cmake .. >/dev/null && make 2>&1 | tail -3 && ctest --output-on-failure 2>&1 | tail -20
```
Expected: builds clean; all suites pass — `trace_tests` PASS; `viz_tests` PASS (rendering checks SKIP if matplotlib/networkx/scipy absent).

- [ ] **Step 2: Confirm non-trace codegen unchanged**

Run:
```bash
cd /Users/porumbes/.../edgecc
for f in examples/reachability_bfs.edge examples/gemm.edge; do
  ./build/edgecc "$f" -o /tmp/a.py 2>/dev/null
  ./build/edgecc "$f" -o /tmp/b.py 2>/dev/null   # same path, no --emit-trace
  diff /tmp/a.py /tmp/b.py && echo "OK $f unchanged"
done
```
Expected: identical (the default `codegen_python` path is untouched).

- [ ] **Step 3: Status check**

```bash
git status   # clean
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- Per-generation trace capture, behind `--emit-trace`, generational-only, byte-identical otherwise → Tasks 1–2. ✓
- Trace format `edgecc-trace/1`, ordered tagged `steps`, all gen tensors per step, adjacency not duplicated, consecutive snapshots → Task 2 + Task 4 sample. ✓
- Standalone renderer, lazy deps, graph + tensor panel, dark theme palette object, cyan active edges from prev/next, layout-once → Tasks 4–5. ✓
- `--color` default = watch; size-mismatch / bad-color / bad-schema / missing-deps errors → Tasks 4–6 (Section 6 of spec). ✓
- Testing: capture tested without plotting deps (Task 3), renderer pure-funcs always + render skipped-if-missing (Tasks 4–6) → ✓
- Deferred items (per-scalar, interactive, animation, contact sheet, print themes, non-generational) → documented, no tasks. ✓

**Gap surfaced (resolved):** the trace envelope's `dims.N` is emitted as `len(<output>)` in `__main__` (not a module constant), so it is correct for inferred-dim generational kernels where `N` is only known inside `run()`. ✓

**Placeholder scan:** no TBD/TODO; all steps carry concrete code/commands. The Task 4 `from _render import` note is explicitly replaced by a module-level `render_all` stub in the same step. ✓

**Type/name consistency:** `codegen_python_ex`, `_edgecc_snapshot`, `_trace`, `edgecc-trace/1`, `gen_tensors`/`watch`/`steps`, `active_edges`, `load_trace`/`validate_trace`/`load_graph`/`render_all`, `THEMES["dark"]` used consistently across tasks. ✓
