# edgecc `.npy` load and save (1-D and N-D tensors) — design

**Date:** 2026-11-16
**Status:** agreed in brainstorming (2026-11-16); builder check done; pending your review
**Branch:** `feature/npy-load` (worktree `.worktrees/npy-load`)
**Origin:** DFT exploration gaps G8 and G9 (`docs/dft/compiler-gaps.md` on
`dft-exploration`), batch D of the agreed A → B → C → D order.

## Intent

Batch D removes the last compiler gaps that block the DFT exploration session.
Both are about getting data into (and out of) a program:

- **G8:** a vector cannot be loaded as a vector. Matrix Market (`.mtx`) is
  always 2-D, so a loaded length-N vector must be declared N×1 and every Einsum
  that touches it carries a size-1 plumbing rank (`dim ONE = 1;`, index `o`).
- **G9:** a tensor with three or more data ranks cannot be loaded or saved
  (since batch A a 3-D `load` is a compile error). The FFT's natural layout is
  3-D — `Xs[m,b,d]` (index split × re/im) in `fft_split1.edge` — so such inputs
  are declared without `init` and the kernel only runs through
  `dft_ref.py --direct`, which injects the data into `run()` by hand.

**Success:** a DFT kernel like `fft_split1.edge` can declare
`init Xs = load("xs.npy");`, run through its own generated `__main__` with its
data in files (no `--direct`), and save a 3-D result; vectors load as 1-D
tensors without a plumbing rank.

## Decisions (from the brainstorming)

- `load` gains numpy's **`.npy`** format, which stores arrays of any rank
  natively. This closes G8 and G9 in one move. The user framed the broader
  question — how EDGE should load data in general — as a standard-library /
  utility-library concern; it is **not** solved here and goes on the roadmap as
  an open design question (see Docs).
- `save:` writes `.npy` too.
- `.mtx` stays 2-D only (no "squeeze N×1 into a vector" rule); `.npy` covers
  vectors.
- A loaded array's shape is checked against the tensor's declaration when the
  program runs.
- `.npy` needs only numpy (already a scipy dependency), imported lazily.
- Out of this batch: other file formats, complex-valued files, and the general
  data-loading design.

## Semantics

### Format selection

The format is chosen by the path's extension, case-sensitive like today:
`.mtx` → Matrix Market, `.npy` → numpy. This holds for `load("…")` paths, for
`save: T -> "…"` paths, and at run time for paths passed as argv overrides (an
override may use a different format than the default path; the extension of the
path actually opened decides).

A `load("…")` or `save:` path whose extension is neither is a compile error
(today it compiles and fails at run time):

- `load: unsupported file extension in "g.csv" (supported: .mtx, .npy)`
- `save: unsupported file extension in "z.csv" (supported: .mtx, .npy)`

An argv override with an unsupported extension still fails at run time with the
existing `ValueError` (its message lists both supported extensions).

### `load` from `.npy`

- Any tensor whose data rank is ≥ 1 may be loaded from `.npy`: 1-D, 2-D, 3-D and
  up (up to the existing 8-rank limit). For a generational tensor the file
  supplies generation 0 and its shape excludes the generational axis, exactly as
  for `.mtx` today.
- The file is read with `numpy.load(path, allow_pickle=False)` (a data file must
  never execute code) and converted to the normal nested-list representation.
- **Values are dense and taken as-is.** A value equal to the tensor's `empty`
  is empty, as with literal inits. Unlike `.mtx` (whose absent entries densify
  to `0.0`), a `.npy` file can hold `inf` directly, so a min-plus weight matrix
  can be loaded with `empty = inf`.
- Element type (my call): loaded values become Python floats whatever the
  tensor's declared type, as `.mtx` loads already do (edgecc computes in floats:
  even an `integer` tensor's outputs are allocated as `0.0`). Integer and
  boolean `.npy` arrays are cast to float. A complex, string, object or other
  non-real dtype is a load-time error naming the tensor, the path and the dtype
  (complex data must be split into Re/Im tensors or a stacked axis).
- `.mtx` loading is unchanged: the target must still be 2-D
  (`load: tensor 'A' must be 2-D (Matrix Market is 2-D), got 3-D`); the check now
  applies only to `.mtx` paths.

### Shape check at load time (both formats)

When `__main__` loads a file, it checks the array against the declaration
before `run()` is called:

- the number of dimensions must equal the tensor's data rank;
- each axis whose dim has a fixed size must match that size;
- axes with an inferred dim (`dim N = *`) are not checked here; they are
  derived and cross-checked inside `run()` as today.

A mismatch raises an error naming the tensor, the path, the file's shape and the
expected shape (fixed sizes as numbers, inferred dims by name).

This applies to `.mtx` loads as well (my call — see "Calls made in writing
this up"): today an oversized `.mtx` is silently truncated by the loops and an
undersized one fails with an `IndexError`. Every existing test, example and
reference kernel that loads a `.mtx` must still pass; if one now fails the
check, it was silently truncating — record it as a found bug and fix the test
data, never weaken the check.

`run()` itself is unchanged by this: callers that pass lists directly (iterative
drivers, `dft_ref.py --direct`) are not checked. The check lives where files are
read.

### Inferred dims (`dim N = *`)

An inferred dim may be pinned by any loaded tensor on any axis, including the
third and later axes (today the derivation and the cross-input asserts only
handle the first two axes: `len(X)` / `len(X[0])`). For example
`init Xs = load("xs.npy");` with `Xs[M=H, B=TWO, D=TWO]` pins `H` on axis 0, and
a tensor `[R=TWO, C=H, K=TWO]` would pin `H` on axis 1. The rule "an inferred dim
must be pinned by at least one `load(...)` tensor" is unchanged; a `.npy` load
counts. `seed(...)` works unchanged with a dim pinned by a `.npy` load.

### `save:` to `.npy`

- Writes `numpy.array(tensor, dtype=float)` (float64, as the `.mtx` save already
  does) with the tensor's full shape: a 1-D tensor stays
  1-D (no column reshape), an N-D tensor stays N-D, and a generational output is
  written with a leading generations axis (all generations).
- Therefore the rule "a generational tensor with more than one data rank cannot
  be saved" now applies only to `.mtx` paths.
- `.mtx` save: the array written (data ranks, plus one if generational) must be
  at most 2-D. A non-generational 3-D tensor saved to `.mtx` becomes a compile
  error (today it fails at run time inside `mmwrite`). Both cases use one message
  that points to `.npy`, e.g.
  `save: 'Y' would be 3-D; Matrix Market stores at most 2-D (save to a .npy path)`.
  The existing negative `save_generational_2d` keeps defending the `.mtx` rule
  (update its expected text if the message changes).
- Batch C's rules are unchanged: each `save:` target is an output of `run()`,
  at most once per tensor and per path.

## Generated code

- The `_edgecc_load` / `_edgecc_save` helpers gain a `.npy` branch (numpy
  imported lazily inside it, with the error
  `edgecc .npy load/save requires numpy: pip install numpy` if missing). The
  `.mtx` branch keeps requiring scipy.
- The load shape check is emitted in `__main__`, next to each load (a shared
  helper is fine). The helper names stay `_`-prefixed so the reserved-name sync
  test is unaffected.
- The `-h` usage string shows each file input with the extension of its default
  path (today it always prints `<name>.mtx`), e.g. `usage: <prog> [Whr.mtx Xs.npy]`.
- Programs without `load`/`save` are byte-identical to today. Programs with
  `.mtx` loads change only by the added shape check (and its helper).
- `__edgecc_meta__["file_inputs"]` and the argv order (init order) are
  unchanged.

## Harnesses and fixtures

- `tests/e2e/run_e2e.py`: a `needs_numpy` marker skips a test when numpy is
  missing (like `needs_scipy`); `roundtrip_check` reads `.npy` files with
  `numpy.load` as well as `.mtx` (values compared flattened, as today).
- `.npy` test inputs are small committed binary fixtures in the test
  directories, with the numpy commands that produced them recorded next to them
  (a comment in `input.edge` or a short script in the test dir), so they can be
  regenerated.

## Testing

Test-first throughout (each new test fails on `main` first, except guards for
behavior that is already correct, which must be named as such in the plan).

- **Negative tests** (unknown-value tests for the new named slot, the file
  extension): `load` with an unsupported extension; `save:` with an unsupported
  extension; `.mtx` save of a non-generational 3-D tensor. Existing
  `load_1d`, `load_3d` (both `.mtx`) and `save_generational_2d` stay — they now
  defend the `.mtx`-only limits.
- **Unit tests:** semantic acceptance of 1-D and 3-D `.npy` loads and of a
  generational `.npy` save with two data ranks; codegen of the `.npy` helper
  branch, of an inferred dim pinned on axis 2 (`len(X[0][0])`), and of the
  usage string.
- **e2e:**
  - G8: a 1-D vector loaded from `.npy` used without a plumbing rank.
  - G9: a 3-D tensor loaded from `.npy` and a 3-D result saved to `.npy`,
    read back (`roundtrip_check`).
  - An inferred dim pinned only on axis 2 of a `.npy` load.
  - A generational output with two data ranks saved to `.npy`.
  - An argv override that switches a `.mtx` default to a `.npy` file.
  - A `.npy` file holding `inf` loaded into an `empty = inf` tensor (min-plus).
  - Run-time rejection of a wrong-shape file (wrong number of dimensions, and a
    wrong fixed size) for both `.npy` and `.mtx`, with the error text checked.
    (The current e2e harness compares stdout of successful runs; the plan picks
    how to express an expected run-time failure.)
- **Independent oracle:** a small FFT split step (N = 8, the `fft_split1`
  structure) whose 3-D input `Xs` comes from `.npy`, with expected output
  computed by `numpy.fft.fft` — not by another edgecc form.
- **No `tests/reference/` case** (my call): this is file-I/O plumbing with no
  graph-algorithm meaning of its own; the numpy FFT oracle is the independent
  check.
- **DFT acceptance** (read-only use of the repo root, from a scratch copy as in
  batches A–C): the `docs/dft/README.md` quickstart still passes; and, in the
  scratch copy only, `fft_split1.edge` with `init Xs = load("….npy")` added runs
  through its own `__main__` without `--direct` and matches numpy. The DFT
  session makes the real kernel edits after merging.
- `tools/premerge.sh` before merge.

## Docs

- `CLAUDE.md` "File input/output": `.npy` load/save, format by extension, the
  load-time shape check, `.mtx` limits, inferred dims on any axis, numpy as the
  `.npy` dependency, the `needs_numpy` marker.
- `docs/roadmap.md`: Implemented entry (batch D, G8/G9); remove or update the
  "1-D vector `.mtx` load" / N-D load items; add an open design question:
  **"General data loading in EDGE"** — how EDGE should load data in general
  (which formats, a standard or utility library of loaders, user-supplied
  loaders, mapping a file's layout onto a tensor's ranks). Batch D's `.npy`
  support is a stopgap that unblocks the DFT work, not that design.
- Spec deviations recorded here at Finish; unfixed review findings go to
  "Known issues & tech debt".

## Non-goals

- Other formats (`.npz`, `.csv`, HDF5, …), memory-mapped or sparse `.npy`.
- Complex-valued data.
- A `shape=` argument to `load` or any reshape of a file's layout.
- Squeezing an N×1 `.mtx` into a 1-D tensor.
- Shape checks inside `run()` for callers that pass lists directly.
- The general data-loading design (roadmap item only).

## Anti-goals

- A `.npy` file that runs code when loaded (pickle must stay disabled).
- A wrong-shape file that silently computes something (the shape check exists
  to make that impossible for loaded files).
- Any change to programs that do not use `load`/`save`.

## Left to the builder

- Exact wording of the run-time error messages (content as specified above).
- How the shape check is factored (inline vs. a helper) and how an expected
  run-time failure is expressed in the e2e harness.
- Fixture generation details.

## Calls made in writing this up

- Unsupported extensions in `load`/`save:` become compile errors (they are a new
  named slot and need an unknown-value negative test).
- The load-time shape check also covers `.mtx` (oversized files are silently
  truncated today).
- A non-generational 3-D `.mtx` save becomes a compile error, sharing one
  "save to a .npy path" message with the generational case.
- The `-h` usage string shows each default path's real extension.
- No `tests/reference/` case; the numpy FFT oracle is the independent check.
- Loaded values are always floats (int/bool `.npy` arrays are cast), and
  `.npy` saves are float64 — matching `.mtx` and edgecc's float computation.
- Pickle stays disabled for `.npy` loads (`allow_pickle=False`).
