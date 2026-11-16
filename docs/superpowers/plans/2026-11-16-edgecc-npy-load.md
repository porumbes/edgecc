# `.npy` load and save (1-D and N-D tensors) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `load` and `save:` accept numpy `.npy` files of any rank (closing DFT gaps G8 and G9), unknown file extensions become compile errors, and every loaded file's shape is checked against its declaration before `run()` is called.

**Architecture:** Semantic analysis validates the path extension of each `load`/`save:` and keeps the 2-D limits only for `.mtx`. Codegen extends the `_edgecc_load` / `_edgecc_save` helpers with a `.npy` branch, passes each load the tensor name and expected shape so the helper checks it, prints real extensions in `-h`, and derives inferred dims from any axis. The e2e harness learns `needs_numpy`, expected run-time failures, and `.npy` round-trips.

**Tech Stack:** C99, Flex/Bison 2.3, CMake/ctest; Python 3.8 + numpy (and scipy for `.mtx`) in generated code and harnesses.

**Spec:** `docs/superpowers/specs/2026-11-16-edgecc-npy-load-design.md`

## Global Constraints

- Work only in the worktree `.worktrees/npy-load` (branch `feature/npy-load`). Never `git checkout`, edit, or build in the repo root; it belongs to the DFT exploration session (branch `dft-exploration`). Reading `dft-exploration` files with `git show dft-exploration:<path>` is fine.
- Build: `cmake -S . -B build && make -C build` (no compiler warnings). Tests: `ctest --test-dir build --output-on-failure`.
- Generated Python and harnesses run on Python 3.8 (no `dict |`, `match`, `list[int]`).
- No `Co-Authored-By` trailer in commit messages. Edit files with an editor, not `sed -i`.
- Format is chosen by the path's extension, case-sensitive: `.mtx` → Matrix Market, `.npy` → numpy.
- `.npy` loads use `numpy.load(path, allow_pickle=False)`. numpy is imported lazily inside the `.npy` branch; `.mtx` keeps requiring scipy.
- Loaded values are Python floats whatever the declared type (int/bool arrays are cast); `.npy` saves write `numpy.array(tensor, dtype=float)` with the tensor's full shape (generational: leading generations axis; 1-D stays 1-D).
- Programs that use no `load`/`save` must produce byte-identical Python to `main`.
- Exact compile-time messages (tests match substrings):
  - `load: unsupported file extension in "<P>" (supported: .mtx, .npy)`
  - `save: unsupported file extension in "<P>" (supported: .mtx, .npy)`
  - `load: tensor '<T>' must be 2-D (Matrix Market is 2-D), got <k>-D` (unchanged; now `.mtx` only)
  - `save: '<T>' would be <k>-D; Matrix Market stores at most 2-D (save to a .npy path)` (`.mtx` only; replaces `save: generational tensor '<T>' has more than one data rank …`)
- Exact run-time messages (raised as `ValueError` / `ImportError` by the generated helpers):
  - `edgecc load: "<path>" for '<T>' has shape <actual>, expected <expected>` where a shape renders as `"(" + ", ".join(str(x) for x in shape) + ")"` — fixed dims as numbers, inferred dims by name, e.g. `(2, 2)` vs `(H, 2, 2)`, and `(3)` for 1-D.
  - `edgecc load: "<path>" for '<T>' has dtype <dtype>; edgecc tensors are real (split Re/Im into separate tensors or a stacked axis)` for any dtype whose `kind` is not one of `b`, `i`, `u`, `f`.
  - `edgecc .npy load/save requires numpy: pip install numpy`
  - `edgecc load: unsupported file extension for %r (supported: .mtx, .npy)` and the same with `save` (existing run-time messages, list extended).

## Review Focus

- **A generational tensor loaded from `.npy`** (its file is generation 0): the expected shape must skip the `gen` axis. Task 2 pins this with `test_codegen_load_shape_skips_gen_axis`.
- **Integer `.npy` data** must load as floats, or outputs print `1` instead of `1.0`. Task 2's `npy_load_save_3d` uses an int64 input.
- **Existing `.mtx` programs** must still pass the new shape check; if one fails it was silently truncating (fix the data, never the check). Task 2 Step 6 runs the whole suite.
- **Callers that pass lists to `run()` directly** (iterative driver, reference runner, `dft_ref.py --direct`) must not be affected: the check lives in `__main__` only. Task 2 Step 6 keeps `e2e_iterative` and `reference_tests` green.
- **An argv override with another extension** than its default must load in the override's format. Task 2's `npy_argv_override` switches `.mtx` → `.npy`.

---

### Task 1: Semantic rules — extensions and `.mtx`-only limits

**Files:**
- Modify: `src/semantic.c` (`analyze_init` load branch; the `save:` loop after the `output:` block)
- Test: `tests/unit/test_semantic.c`
- Create: `tests/negative/{load_unsupported_ext,save_unsupported_ext,save_mtx_3d}/{input.edge,expected_errors.txt}`
- Modify (reconciliation): `tests/negative/save_generational_2d/expected_errors.txt`

**Interfaces:**
- Produces: `static int path_has_ext(const char *path, const char *ext)` in `src/semantic.c` (1 if `path` ends with `ext`). `.npy` loads of any data rank and `.npy` saves of any rank pass semantic analysis; codegen (Task 2) supplies the run-time support.

- [x] **Step 1: Write the negative tests**

Each `input.edge` starts with the `//` comment shown.

`load_unsupported_ext` (`// load: the file extension must be .mtx or .npy (unknown-value test for the extension slot).`):
```
dim N = 2;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor Z[R=N]      : float, empty = 0.0;
init G = load("g.csv");
Z[i] = G[i,k] . G[k,i] :: map(mul, intersect) reduce(add, union);
```
→ `load: unsupported file extension in "g.csv" (supported: .mtx, .npy)`

`save_unsupported_ext` (`// save: the file extension must be .mtx or .npy.`):
```
dim N = 2;
tensor A[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0];
Z[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
save: Z -> "z.csv";
```
→ `save: unsupported file extension in "z.csv" (supported: .mtx, .npy)`

`save_mtx_3d` (`// A 3-D tensor cannot be stored as Matrix Market; .npy can.`):
```
dim H = 2;
dim TWO = 2;
tensor A[M=H, B=TWO, D=TWO] : float, empty = 0.0;
tensor Z[M=H, B=TWO, D=TWO] : float, empty = 0.0;
init A = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
Z[m,b,d] = A[m,b,d] . A[m,b,d] :: map(mul, intersect) reduce(add, union);
save: Z -> "z.mtx";
```
→ `save: 'Z' would be 3-D; Matrix Market stores at most 2-D (save to a .npy path)`

Reconciliation: `save_generational_2d/expected_errors.txt` becomes
`save: 'T' would be 3-D; Matrix Market stores at most 2-D (save to a .npy path)`.
`load_1d` and `load_3d` stay unchanged: they now defend the `.mtx`-only 2-D load rule.

- [x] **Step 2: Write the unit tests** (in `tests/unit/test_semantic.c`, with `RUN_TEST` lines)

- `test_semantic_npy_load_any_rank` → `ASSERT_INT_EQ(0, sem.error_count)` for:
  ```
  dim N = *;
  dim TWO = 2;
  tensor F[R=N]               : float, empty = 0.0;
  tensor X[R=N, C=TWO, D=TWO] : float, empty = 0.0;
  tensor Z[R=N]               : float, empty = 0.0;
  init F = load("f.npy");
  init X = load("x.npy");
  Z[i] = X[i,c,d] . F[i] :: map(mul, intersect) reduce(add, union);
  ```
- `test_semantic_mtx_load_still_2d`: the same `X` with `init X = load("x.mtx")` → `error_count > 0` and the first error contains `must be 2-D (Matrix Market is 2-D), got 3-D` (check how existing tests read error text, e.g. `test_semantic_unpinned_inferred_dim_errors`, and do the same).
- `test_semantic_npy_save_generational_2d_ok`: the `save_generational_2d` negative's program with `save: T -> "t.npy";` → 0 errors.
- `test_semantic_unsupported_extension`: `load("g.txt")` → error containing `load: unsupported file extension in "g.txt"`.

- [x] **Step 3: Run and confirm red**

Run: `make -C build 2>&1 | grep -E "error|warning:"; ./build/test_semantic | tail -3; python3 tests/negative/run_negative.py build/edgecc 2>&1 | grep -E "FAIL|passed"`
Expected: `test_semantic_npy_load_any_rank`, `test_semantic_npy_save_generational_2d_ok` and `test_semantic_unsupported_extension` FAIL (`test_semantic_mtx_load_still_2d` passes — a guard for unchanged behavior); negatives `load_unsupported_ext`, `save_unsupported_ext`, `save_mtx_3d` FAIL ("expected failure but edgecc succeeded") and `save_generational_2d` FAILS (old message).

- [x] **Step 4: Implement**

In `analyze_init`'s `is_load` branch: if the path ends with neither `.mtx` nor `.npy`, report the `load: unsupported file extension …` error and return; apply the existing 2-D check only when it ends with `.mtx`.

In the `save:` loop: right after the `save: unknown tensor` check, report `save: unsupported file extension …` (and `continue`) for a path that is neither; replace the generational `more than one data rank` check with a `.mtx`-only rule — `k = num_ranks` (the generations axis counts when the tensor is generational, data ranks otherwise, i.e. `k = ste->num_ranks`) and `k > 2` → the `would be <k>-D` error.

- [x] **Step 5: Run everything**

Run: `make -C build 2>&1 | grep -E "error|warning:"; ./build/test_semantic | tail -1; python3 tests/negative/run_negative.py build/edgecc 2>&1 | tail -1; ctest --test-dir build 2>&1 | tail -3`
Expected: no warnings; all unit and negative tests pass; `100% tests passed`. Nothing real is newly rejected:
```bash
for f in examples/*.edge tests/e2e/*/input.edge tests/e2e/iterative/*/kernel.edge tests/reference/kernels/*.edge; do build/edgecc "$f" -o /dev/null >/dev/null 2>&1 || echo "REJECTED $f"; done
```
Expected: no output.

- [x] **Step 6: Commit**

```bash
git add src/semantic.c tests/unit/test_semantic.c tests/negative
git commit -m "semantic: load/save file extension must be .mtx or .npy; 2-D limits apply to .mtx only"
```

---

### Task 2: Codegen `.npy` support, load-time shape check, harness and e2e

**Files:**
- Modify: `src/codegen.c` (`_edgecc_load`/`_edgecc_save` helper text; `__main__` load lines and `-h` usage; `emit_inferred_dim_derivation`, `emit_inferred_dim_asserts`, `emit_seed_init` length expressions)
- Modify: `tests/e2e/run_e2e.py`
- Test: `tests/unit/test_codegen.c`
- Create: `tests/e2e/npy_fixtures.py`; `tests/e2e/{npy_load_1d,npy_load_save_3d,npy_inferred_dim_axis2,npy_save_generational_2d,npy_argv_override,npy_inf_minplus,npy_wrong_ndim,npy_wrong_size,mtx_wrong_size,npy_complex_rejected,npy_fft_split_n4}/`

**Interfaces:**
- Consumes: Task 1 (`.npy` programs pass semantic analysis).
- Produces (generated Python): `def _edgecc_load(path, name, shape):` returns nested lists of floats after checking dtype and shape (`shape` is a tuple with one entry per data axis: an `int` for a fixed dim, the dim's name as a `str` for an inferred dim — str entries are not checked); `def _edgecc_save(tensor, path):`. Each loaded input is emitted as
  `X = _edgecc_load(_argv[J] if len(_argv) > J else "<default path>", "X", <shape tuple>)` (a 1-element tuple is written `(4,)` / `("N",)`).
- Produces (C): `static void len_expr(char *buf, size_t n, const char *name, int axis)` → `len(X)`, `len(X[0])`, `len(X[0][0])`, …; `static void emit_load_shape(CodeBuf *cb, IRProgram *ir, const char *tensor_name)` → the shape tuple literal (skipping the generational axis).

- [x] **Step 1: Harness support in `tests/e2e/run_e2e.py`**

- `HAVE_NUMPY` (like `HAVE_SCIPY`) and a `needs_numpy` marker that prints `SKIP <name>: numpy not installed`.
- A test dir may hold `expected_runtime_error.txt` instead of `expected_output.txt`: the program must compile, then exit non-zero, and each non-empty line of the file must appear in its stderr (`FAIL <name>: expected a run-time error …` otherwise). stdout is not compared.
- `roundtrip_check`: a `.npy` file is read with `numpy.load` (`.mtx` with `mmread`, as now); values compared flattened. The block must not require scipy for `.npy`-only checks.

- [x] **Step 2: Write the fixtures script and the e2e tests**

`tests/e2e/npy_fixtures.py` (Python 3.8, numpy only; `python3 tests/e2e/npy_fixtures.py` regenerates every `.npy` below and `npy_fft_split_n4/expected_output.txt`; run it once and commit its outputs). Each `input.edge` names the script in its first comment line.

| Test dir | markers | `input.edge` (after the comment) | fixture(s) | expected |
|---|---|---|---|---|
| `npy_load_1d` (G8) | `needs_numpy` | `dim N = *;` `G[R=N,C=N]`, `F[R=N]`, `Z[R=N]` (float, empty 0.0); `init G = [[0.0, 1.0, 2.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]];` `init F = load("f.npy");` `Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);` | `f.npy` = float64 `[1, 2, 3]` | `[6.0, 13.0, 2.0]` |
| `npy_load_save_3d` (G9) | `needs_numpy`, `roundtrip_check` | `dim H = 2; dim TWO = 2;` `A[M=H,B=TWO,D=TWO]`, `Z[M=H,B=TWO,D=TWO]`; `init A = load("a.npy");` `Z[m,b,d] = A[m,b,d] . A[m,b,d] :: map(mul, intersect) reduce(add, union);` `save: Z -> "z.npy";` | `a.npy` = **int64** `[[[1,2],[3,4]],[[5,6],[7,8]]]` | `[[[1.0, 4.0], [9.0, 16.0]], [[25.0, 36.0], [49.0, 64.0]]]`; roundtrip `z.npy: 1.0 4.0 9.0 16.0 25.0 36.0 49.0 64.0` |
| `npy_inferred_dim_axis2` | `needs_numpy` | `dim N = *; dim TWO = 2;` `X[R=TWO,C=TWO,K=N]`, `Z[K=N]`; `init X = load("x.npy");` `Z[k] = X[r,c,k] . X[r,c,k] :: map(mul, intersect) reduce(add, union);` | `x.npy` = `np.arange(1, 13, dtype=float).reshape(2, 2, 3)` | `[166.0, 214.0, 270.0]` |
| `npy_save_generational_2d` | `needs_numpy`, `roundtrip_check` | the `tests/negative/save_generational_2d` program with `save: T -> "t.npy";` | — | the line below; roundtrip `t.npy:` the 36 values of that line, flattened |
| `npy_argv_override` | `needs_numpy`, `needs_scipy`, `args` = `g_arg.npy` | `dim N = *;` `G[R=N,C=N]`, `Z[R=N]`; `init G = load("g.mtx");` `Z[i] = G[i,k] . G[k,i] :: map(mul, intersect) reduce(add, union);` | `g_arg.npy` = `[[1,2],[3,4]]`; `g.mtx` (text, below) = identity 2×2 | `[7.0, 22.0]` (the override, not the default's `[1.0, 1.0]`) |
| `npy_inf_minplus` | `needs_numpy` | `dim N = *;` `W[R=N,C=N]`, `D[R=N]`, `Z[R=N]`, all `empty = inf`; `init W = load("w.npy");` `init D = [0.0, inf, inf];` `Z[d] = W[s,d] . D[s] :: map(add, intersect) reduce(min, union);` | `w.npy` = `[[inf,1,4],[inf,inf,2],[inf,inf,inf]]` | `[inf, 1.0, 4.0]` |
| `npy_wrong_ndim` | `needs_numpy` | `dim H = *; dim TWO = 2;` `A[M=H,B=TWO,D=TWO]`, `Z[M=H,B=TWO]`; `init A = load("a.npy");` `Z[m,b] = A[m,b,d] . A[m,b,d] :: map(mul, intersect) reduce(add, union);` | `a.npy` = 2×2 `[[1,2],[3,4]]` | `expected_runtime_error.txt`: `edgecc load: "a.npy" for 'A' has shape (2, 2), expected (H, 2, 2)` |
| `npy_wrong_size` | `needs_numpy` | `dim N = 3;` `G[R=N,C=N]`, `Z[R=N]`; `init G = load("g.npy");` `Z[i] = G[i,k] . G[k,i] :: …(add, union);` | `g.npy` = `np.eye(4)` | `edgecc load: "g.npy" for 'G' has shape (4, 4), expected (3, 3)` |
| `mtx_wrong_size` | `needs_scipy` | as `npy_wrong_size` with `load("g.mtx")` | `g.mtx` (text) = identity 4×4 | `edgecc load: "g.mtx" for 'G' has shape (4, 4), expected (3, 3)` |
| `npy_complex_rejected` | `needs_numpy` | `dim N = *;` `C[R=N]`, `Z[R=N]`; `init C = load("c.npy");` `Z[i] = C[i] . C[i] :: map(mul, intersect) reduce(add, union);` | `c.npy` = `np.array([1+2j, 3-1j])` | `edgecc load: "c.npy" for 'C' has dtype complex128; edgecc tensors are real` |
| `npy_fft_split_n4` (independent oracle) | `needs_numpy` | see below | `xs.npy`; `expected_output.txt` written by the script from `numpy.fft.fft` | `[[[2.0, 6.0], [0.0, 0.0]], [[0.0, -6.0], [2.0, 0.0]]]` |

`npy_save_generational_2d` expected stdout (hand-computed: `T0 = G`, `T1 = G²`, `T2 = T3 = 0`):
```
[[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
```

Matrix Market text files (identity n×n, `n` = 2 for `npy_argv_override/g.mtx`, 4 for `mtx_wrong_size/g.mtx`):
```
%%MatrixMarket matrix coordinate real general
n n n
1 1 1.0
2 2 1.0
...
```

`npy_fft_split_n4/input.edge`: comment lines `// Independent oracle: one radix-2 FFT split (N = 4) with a 3-D input loaded from .npy;` and `// expected_output.txt comes from numpy.fft.fft (tests/e2e/npy_fixtures.py), x = [1, 2+1j, 3j, -1+2j].`, then the tensor declarations and the 11 Einsums of `fft_split1.edge` (`git show dft-exploration:docs/dft/kernels/fft_split1.edge`) with these changes: `dim H = *;` (pinned by `Xs`), `init Xs = load("xs.npy");`, and inline twiddles for N = 4:
```
init Whr = [[1.0, 1.0], [1.0, -1.0]];
init Whi = [[0.0, 0.0], [0.0, 0.0]];
init Twr = [[1.0, 1.0], [1.0, 0.0]];
init Twi = [[0.0, 0.0], [0.0, -1.0]];
init J   = [[0.0, -1.0], [1.0, 0.0]];
init F2  = [[1.0, 1.0], [1.0, -1.0]];
```
The script writes `xs.npy` with `Xs[m][b][d] = (x[2m+b].real, x[2m+b].imag)` and `expected_output.txt` as `repr` of `Y[k1][k2][c] = (X[k1 + 2*k2].real, X[k1 + 2*k2].imag)` where `X = numpy.fft.fft(x)`, each value rounded to 9 decimals and `+ 0.0` (no `-0.0`). It must not run edgecc.

- [x] **Step 3: Write the codegen unit tests** (in `tests/unit/test_codegen.c`, with `RUN_TEST` lines)

- `test_codegen_npy_load_helper`: `dim N = *;` `F[R=N]`, `Z[R=N]`, `init F = load("f.npy");` `Z[i] = F[i] . F[i] …` → code contains `def _edgecc_load(path, name, shape):`, `allow_pickle=False`, `(supported: .mtx, .npy)`, and `F = _edgecc_load(_argv[0] if len(_argv) > 0 else "f.npy", "F", ("N",))`.
- `test_codegen_npy_save`: the `test_codegen_emits_save` program with `save: Z -> "out.npy";` → contains `_edgecc_save(Z, "out.npy")` and `np.save(`.
- `test_codegen_inferred_dim_axis2`: the `npy_inferred_dim_axis2` program → contains `N = len(X[0][0])`.
- `test_codegen_usage_shows_extension`: `dim N = *;` `G[R=N,C=N]`, `F[R=N]`, `Z[R=N]`; `init G = load("g.mtx"); init F = load("f.npy");` `Z[d] = G[s,d] . F[s] …` → contains `usage: <prog> [G.mtx F.npy]`.
- `test_codegen_load_shape_skips_gen_axis`: `dim N = 4;` `G[R=N,C=N]` (literal), `F[gen I, R=N]` with `init F = load("f.npy");`, `F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union); stop: stable(F);` → contains `"F", (4,))`.

Reconciliation (the load call form changes): in `test_codegen_emits_load_helper` replace the `_edgecc_load(\"g.mtx\")` assert with `G = _edgecc_load(_argv[0] if len(_argv) > 0 else \"g.mtx\", \"G\", (\"N\", \"N\"))`; in the argv-order test (~line 625) the usage assert stays (`[B.mtx A.mtx]`) and the two `_edgecc_load(_argv[J])` asserts become `B = _edgecc_load(_argv[0] if len(_argv) > 0 else` and `A = _edgecc_load(_argv[1] if len(_argv) > 1 else`.

- [x] **Step 4: Run and confirm red**

Run: `make -C build && ./build/test_codegen | tail -3; python3 tests/e2e/run_e2e.py build/edgecc 2>&1 | grep -E "FAIL|SKIP|passed"`
Expected: the five new codegen tests and the reconciled asserts FAIL; every new e2e test FAILS (`.npy` unsupported at run time, or — for `mtx_wrong_size` — the program exits 0 after silently truncating); all other e2e tests pass. These are the transiently red tests until Step 6.

- [x] **Step 5: Implement in `src/codegen.c`**

- Helper text: `_edgecc_load(path, name, shape)` — read `.mtx` (scipy `mmread` → dense) or `.npy` (`numpy.load(path, allow_pickle=False)`, dtype-kind check), else the unsupported-extension `ValueError`; compare `ndim` and every `int` entry of `shape` with the array's shape, raising the exact shape message on mismatch; return `array.astype(float).tolist()`. `_edgecc_save(tensor, path)` gains the `.npy` branch (`np.save(path, np.array(tensor, dtype=float))`). Missing numpy raises the exact `ImportError`.
- `__main__`: emit each load in the call form from Interfaces, using `emit_load_shape`; the `-h` usage prints `<name><extension of its default path>` (`strrchr(path, '.')`; Task 1 guarantees one of the two).
- `len_expr` replaces the three `len(%s)` / `len(%s[0])` branches (derivation, asserts, seed sizing), using the axis from `python_axis_of`. Update `python_axis_of`'s comment (it no longer returns only 0 or 1).

- [x] **Step 6: Run everything**

Run: `make -C build 2>&1 | grep -E "error|warning:"; ./build/test_codegen | tail -1; python3 tests/e2e/run_e2e.py build/edgecc 2>&1 | grep -E "FAIL|passed"; ctest --test-dir build 2>&1 | tail -3; git status --short`
Expected: no warnings; all codegen tests pass; all e2e pass (the 11 new ones included); `100% tests passed` (iterative and reference tests included); no stray `.npy`/`.mtx` left behind.

Byte-identity for programs without file I/O — compile every corpus program with `main`'s compiler and this branch's, and diff the outputs of those that contain no `load(`/`save:`:
```bash
MAIN=../premerge-gate/build/edgecc; S=$(mktemp -d)
for f in examples/*.edge tests/e2e/*/input.edge tests/e2e/iterative/*/kernel.edge; do
  grep -qE 'load\(|save:' "$f" && continue
  $MAIN "$f" -o $S/a.py >/dev/null 2>&1; build/edgecc "$f" -o $S/b.py >/dev/null 2>&1
  cmp -s $S/a.py $S/b.py || echo "DIFFERS $f"
done
```
Expected: no output. (`../premerge-gate/build/edgecc` is `main`'s build; run `make -C ../premerge-gate/build edgecc` first if it is missing — that worktree is `main`, building there is fine.)

- [x] **Step 7: Commit**

```bash
git add src/codegen.c tests/unit/test_codegen.c tests/e2e
git commit -m "codegen: .npy load/save for any rank, load-time shape check, inferred dims on any axis (DFT gaps G8/G9)"
```

---

### Task 3: Docs, DFT acceptance, and Finish

**Files:**
- Modify: `CLAUDE.md`, `docs/roadmap.md`, the spec (deviations), this plan (checkboxes)

- [x] **Step 1: `CLAUDE.md`**

- "File input/output": `load` reads `.mtx` (2-D, densified) or `.npy` (any rank, dense values as-is — e.g. `inf` for a min-plus `empty = inf` operand); format by extension, unsupported extensions are compile errors; every loaded file's shape is checked in `__main__` before `run()` (fixed dims by size, inferred dims derived in `run()` from any axis); loaded values are floats; `save:` to `.npy` keeps the full shape (generational: leading generations axis), `.mtx` saves are at most 2-D; `.npy` needs numpy, `.mtx` needs scipy. Replace "1-D and N-D loading are future work" and the "tensors with more than one data rank cannot be saved" clause accordingly.
- "## Test", e2e line: add the `needs_numpy` marker and `expected_runtime_error.txt` (and that `roundtrip_check` reads `.npy` too); mention `tests/e2e/npy_fixtures.py`.
- "## Generated Python ABI": the `_edgecc_load(path, name, shape)` call form.
- "## v2 Scope Limitations": drop "loading non-`.mtx` formats" from the deferred list (now: other formats beyond `.mtx`/`.npy`).

- [x] **Step 2: `docs/roadmap.md`**

Last-updated date; Implemented entry `- **.npy load/save (DFT gaps G8/G9, batch D):** any-rank .npy load and save, format by extension, load-time shape check for .mtx and .npy.`; remove/adjust the 1-D `.mtx` load and N-D load items; add under future/open design: **General data loading in EDGE** — how EDGE should load data in general (formats, a standard or utility library of loaders, user-supplied loaders, mapping a file's layout onto a tensor's ranks); batch D's `.npy` support is a stopgap. Add unfixed review findings to "Known issues & tech debt", tagged `(batch D)`.

- [x] **Step 3: DFT acceptance (read-only use of the repo root, from a scratch copy)**

As in batch C: mirror `$ROOT/docs/dft` into a scratch repo with `build/edgecc` symlinked to this branch's binary, run the `$ROOT/docs/dft/README.md` quickstart under `bash`; every line ends `-> PASS`.

Then, in the scratch copy only: add `init Xs = load("xs.npy");` to `docs/dft/kernels/fft_split1.edge` (N = 8: it uses the committed `twiddle_n8_*.mtx` defaults), write `xs.npy` from a random complex `x` of length 8 in split layout (`Xs[m][b][d] = (x[2m+b].real, x[2m+b].imag)`), compile it, run it **without `--direct`** from the scratch repo root, parse the printed `Y` with `ast.literal_eval`, map `y[k1 + 4*k2] = Y[k1][k2][0] + 1j*Y[k1][k2][1]`, and compare with `numpy.fft.fft(x)` (max abs error < 1e-12). Paste the command and result in the report. Never edit the real DFT files.

- [x] **Step 4: Pre-merge gate**

Run: `tools/premerge.sh` (long timeout). Expected: `GATE PASSED`. Then `ps -axo pid,etime,pcpu,command | grep -E 'edgecc|run_fuzz|build/test_|build-ubsan' | grep -v grep` and kill strays by PID.

- [x] **Step 5: Record deviations, tick the plan, commit**

Append `## Deviations and findings (recorded at Finish, <date>)` to the spec. Known deviation: the independent FFT oracle e2e uses N = 4 (not 8) so the expected values are exact and the harness can compare stdout exactly; N = 8 is covered against numpy by the Step 3 acceptance run. Tick Tasks 1–2 and Task 3 Steps 1–5.
```bash
git status --short
git add CLAUDE.md docs/roadmap.md docs/superpowers
git commit -m "docs: .npy load/save (batch D) — CLAUDE.md, roadmap, spec deviations, plan ticked"
```

- [x] **Step 6: Merge from the `main` worktree** (controller, after the final review)

```bash
cd ../premerge-gate
git merge --no-ff feature/npy-load -m "Merge feature/npy-load: .npy load/save for any rank (DFT gaps G8/G9, batch D)"
make -C build && ctest --test-dir build 2>&1 | grep "tests passed"
```

- [x] **Step 7: Memory and hand-off** (controller)

Update `project_resume_state.md` (main's new hash, batch D done) and tell the user the DFT session can merge `main`: `init Xs = load("xs.npy")` replaces the `--direct` injection for 3-D inputs, vectors load 1-D from `.npy`, and 3-D results can be saved to `.npy`.
