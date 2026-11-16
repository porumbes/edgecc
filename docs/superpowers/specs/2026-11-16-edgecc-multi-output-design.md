# edgecc multi-output `run()` and multiple `save:` — design

**Date:** 2026-11-16
**Status:** approved in brainstorming (2026-10-28 – 2026-11-16); pending written-spec review
**Branch:** `feature/multi-output`
**Origin:** DFT exploration gap G3 (`docs/dft/compiler-gaps.md` on
`dft-exploration`), batch C of the agreed A → B → C order. Also the
"multi-output return" item of the full paper-faithful BFS milestone.

## Motivation

A program returns exactly one tensor: the last Einsum's output, or the
stop-watched tensor in a generational program. `save:` may appear only once and
must target that tensor:

```
B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);
save: B -> "b.mtx";     // error: save: 'B' is not the tensor run() returns ('Z')
save: Z -> "z.mtx";
```

A complex result needs two real tensors (Re, Im), so the DFT is split into two
programs today. The paper's full BFS wants several results too (depth and
parent).

## Semantics

The paper has no notion of program outputs: every Einsum's left-hand side is an
"output tensor" (`tex/6-syntax.tex`, §Initialization), and batch B already
allocates every written tensor inside `run()`. Which of them `run()` hands back
is an ABI choice, so the program states it explicitly.

### `output:` directive

```
output: B, Z;
```

- A directive like `stop:` and `save:`: at most one per program, anywhere among
  the statements. `output` becomes a lexer keyword (no `.edge` file in the repo
  or on `dft-exploration` uses it as a name).
- Lists the tensors `run()` returns, in return order. The list is non-empty
  (`output: ;` is a parse error).
- Without `output:`, the default is unchanged: the last Einsum's output, or the
  stop-watched tensor in a generational program.
- A generational tensor in the list is returned with every generation, exactly
  like the watched tensor today. Any written generational tensor may be listed
  (cross-generational or intermediate); the watched tensor need not be listed.

### Validation (new compile errors)

- `output: unknown tensor 'X'` — not a declared tensor.
- `output: 'A' is not written by any Einsum (inputs are not outputs)`.
- `output: 'B' listed twice`.
- `output: only one output directive is allowed`.

### `save:`

`save:` may appear any number of times.

- Each target must be one of `run()`'s outputs (the `output:` list, or the
  default output when there is no list):
  `save: 'B' is not an output of run(); add it to the output: list`.
  This replaces today's `save: 'B' is not the tensor run() returns ('Z')`.
- `save: 'B' is saved twice` — each tensor at most once.
- `save: path "b.mtx" is used twice` — each path at most once.
- Unchanged: `save: unknown tensor`, and a generational tensor with more than
  one data rank cannot be saved (checked per target).

## Generated code

- **`run()`** ends with `return (B, Z)` in the order of the `output:` line when
  there are several outputs, and `return Z` when there is one (whether or not
  `output:` is written). Allocation is unchanged (batch B).
- **`__edgecc_meta__`** gains `"outputs": [...]` in every program (names in
  return order). `"output"` is kept only when there is exactly one output, so a
  consumer that assumes one output fails with `KeyError` on a multi-output
  program instead of silently using one of the tensors. Every generated file
  changes by that one key; single-output code is otherwise unchanged.
- **`__main__`** unpacks (`B, Z = run(A)`), prints **one line per output** in
  listed order (each line a Python literal), then runs every `save:` in source
  order.
- **`--emit-trace`** is unchanged: the trace already records every generational
  tensor and its `"watch"` field stays the stop-watched tensor.

For example, the program from the motivation with `output: B, Z;` becomes:

```python
__edgecc_meta__ = {"inputs": ["A"], "outputs": ["B", "Z"], "file_inputs": []}

def run(A):
    B = [0.0] * N
    Z = [0.0] * N
    ...
    return (B, Z)

if __name__ == "__main__":
    A = [1.0, 2.0]
    B, Z = run(A)
    print(B)
    print(Z)
    _edgecc_save(B, "b.mtx")
    _edgecc_save(Z, "z.mtx")
```

(The exact key order inside `__edgecc_meta__` follows the existing emitter.)

## Harnesses and tools

- `tests/e2e/run_e2e.py` `run_twice`: compares the whole return value — the
  tuple of `meta["outputs"]` values when there are several outputs.
- `tests/e2e/iterative/run_iterative.py`: feedback needs a single result; it
  stops with a clear error when `meta["outputs"]` has more than one entry.
- `tools/edgecc_viz.py`: reads only the trace; unaffected.
- VS Code extension (`editors/vscode/`): highlight `output` as a directive
  keyword, with a grammar test.
- DFT session (hand-off note only; its files are not edited here):
  `dft_ref.py --direct` receives a tuple from a multi-output kernel, and its
  printed-output decoder must read one line per output. Single-output kernels
  are unaffected.

## Testing

Test-first throughout (each new test fails on `main` first).

- **Lexer/parser unit tests:** `output` keyword; `output: B, Z;` parses to the
  list in order.
- **Semantic unit tests:** the outputs list resolves to the default when absent;
  each new error.
- **Codegen unit tests:** `return (B, Z)`, `"outputs": ["B", "Z"]`, no
  `"output"` key with several outputs, `B, Z = run(A)` and one `print` per
  output; a single-output program keeps `"output"` and a bare `return`.
- **Negative tests:** unknown tensor in `output:` (the unknown-value test for the
  new named slot); an input in `output:`; a name listed twice; two `output:`
  directives; `output: ;`; `save:` of a non-output; the same tensor saved twice;
  the same path saved twice.
- **e2e:**
  - G3 repro: two outputs, two `save:`, both files read back and checked
    (`roundtrip_check`).
  - `output: Z;` gives the same result as no directive.
  - `output: Z, B;` returns and prints `Z` before `B` although `B` is written
    first.
  - An intermediate listed as an output.
  - Generational: reachability BFS returning frontier `F` and visited `P`, each
    with every generation (expected values worked out by hand).
  - `run_twice` on a multi-output program.
  - DFT N = 4, Re and Im from one kernel, expected output from numpy's FFT
    values (independent oracle).
- **Iterative driver:** a multi-output kernel is rejected with the clear error.
- **Reconciliation** (lifted restrictions):
  - `tests/negative/save_not_returned_{intermediate,nongen,gen}` and
    `save_not_output`: still invalid without an `output:` list; update their
    expected message to the new `is not an output of run()` wording.
  - Any unit test asserting the old messages (`only one save directive`,
    `is not the tensor run() returns`).
- **No `tests/reference/` case:** multi-output is ABI plumbing with no
  graph-algorithm meaning of its own. The independent checks are the numpy DFT
  values and the hand-computed BFS values.
- **DFT acceptance:** the `docs/dft/README.md` quickstart still passes (run
  read-only against the repo-root DFT files from a scratch copy, as in batches A
  and B).
- `tools/premerge.sh` before merge.

## Docs

- `CLAUDE.md`: `output:` syntax and the `save:` rules (EDGE Syntax, File
  input/output); Generated Python ABI (tuple return, `meta["outputs"]`, one
  printed line per output); scope limitations (drop "save: of any tensor other
  than the one run() returns" and "multi-output return" from the BFS gaps).
- `docs/roadmap.md`: Implemented entry; update the BFS Partial line; move any
  unfixed review findings into "Known issues & tech debt".

## Out of scope

- Multi-output feedback in the iterative driver.
- A dict return, or an `output: *` shorthand.
- Saving a tensor that is not an output.
- Trace format changes.
- `ANY` compute and parent-pointer (coordinate-valued) outputs.
- 1-D and N-D `load` (batch D, gaps G8/G9).

## Deviations and findings (recorded at Finish, 2026-11-16)

- Task 4 also highlights `save` and `stop` (none of the three directives was highlighted before), not only `output`.
- Extra commit `8b9d3d6`: with `--emit-trace`, when `output:` omits the stop-watched tensor, the plan's trace header `len(<watch>[0])` raised `TypeError` (in `__main__` the watched name holds only its generation-0 input). The header now takes `N` from the trace's last step in that case (`len(_tr[-1]["tensors"][watch])`); when the watched tensor is an output the header is unchanged. The trace format is unchanged. Test: `tests/trace/watch_not_output`.
- Plan defect: the iterative harness command in the plan (Task 3 Steps 4/6) needs `--build-dir build --tests-dir tests/e2e/iterative`.
- `test_codegen_saves_follow_prints` passed before the codegen change (the plan expected it red); the e2e `multi_output_save` test is the red-to-green guard for saving a non-last output.
- The spec's Testing list asked for semantic unit tests for each new error; those errors are covered by negative tests (exact messages) instead. The semantic unit tests cover the outputs list (default, listed order, default watch) and two saves in source order.
- `tests/negative/output_empty_list` and `test_parse_output_empty_list_fails` could not fail on main (`output: ;` was already a syntax error when `output` lexed as an identifier); they are regression guards, an exception to "each new test fails on main first".
- Final-review fix wave: added e2e `output_gen_intermediate` and `output_watch_not_listed`; `run_e2e.py` `roundtrip_check` now cleans up saved files on failure and rejects mixed formats cleanly.
