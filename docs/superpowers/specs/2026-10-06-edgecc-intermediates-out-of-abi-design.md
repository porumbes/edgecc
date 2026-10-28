# edgecc intermediates out of the `run()` ABI — design

**Date:** 2026-10-06
**Status:** approved in brainstorming (2026-10-04 – 2026-10-06); pending written-spec review
**Branch:** `feature/intermediates-abi`
**Origin:** DFT exploration gaps G5 and G6 (`docs/dft/compiler-gaps.md` on
`dft-exploration`), batch B of the agreed A → B → C order.

## Motivation

In a non-generational program with several Einsums, a tensor written by one
Einsum and read by a later one (an *intermediate*) is treated as an input:

```
B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);
```

generates `def run(A, B)` and `"inputs": ["A", "B"]`. `__main__` allocates `B`
and passes it in, and the reduce seeds from whatever the caller supplied
(`state = B[i]`). Consequences found so far:

| Case | Today |
|---|---|
| G5: intermediates are `run()` parameters | ABI lists non-inputs; `run()` mutates the caller's lists |
| G6: `dim N = *` plus an intermediate | `__main__` allocates `B` at module scope using `N`, which only exists inside `run()` → `NameError` |
| Calling `run()` twice with the same `B` | `B` accumulates across calls |
| Intermediate with an `init` | the init is silently used as the reduce's starting value |
| The returned output with an `init` | the init is silently ignored |
| Tensor written but never read (not the last Einsum) | never allocated → `NameError` |
| Read before the Einsum that writes it | `NameError` |
| Generational program with a non-generational write (`G2 = G·G` before the loop) | runs every generation, accumulating into the caller's `G2`: right for `max`, silently wrong for `add` |

## Semantics (paper-faithful)

The paper (`tex/6-syntax.tex`, §Initialization): "The initialization region
specifies the values of input tensors. Any input tensor not initialized in this
section is assumed user-specified. All *output* tensors are initially empty."
An output tensor is any Einsum's left-hand side. The paper also allows several
Einsums to define the same output (`tex/6-syntax.tex:405`, different regions of
the iteration space).

edgecc therefore classifies every tensor an Einsum touches as:

- **written** — the left-hand side of at least one Einsum. Starts empty, is
  allocated inside `run()`, and is never a parameter. Includes the returned
  output and all intermediates.
- **input** — read by some Einsum and written by none. A `run()` parameter,
  supplied by `__main__` from its `init` (or empty, with the G10 warning).

What `run()` returns is unchanged: the last Einsum's output (non-generational)
or the stop-watched tensor (generational). Returning several tensors is batch C.

Two Einsums writing the same tensor keep today's behaviour: one buffer,
allocated once, both Einsums write into it in source order.

## Validation (new compile errors)

- **E1: `init` on a written tensor.** `tensor 'B' is written by an Einsum and
  must not have an init (output tensors start empty)`. Applies to
  non-generational tensors and to every kind of init (literal, `load(...)`,
  `seed(...)`). Generational intermediates already have an
  equivalent rule, and cross-generational tensors still require their `init`
  (generation 0). `examples/dotproduct.edge` (`init Z = [0]` on its output)
  loses that line.
- **E2: read before write.** `tensor 'B' is read before the Einsum that writes
  it`, in source order, for non-generational programs. A read between two writes
  of the same tensor is fine (it follows the first write). Generational cascades
  already have their own read-before-write checks.
- **E3: non-generational write in a generational program.** `in a generational
  program every Einsum must write a generational tensor; 'G2' is not (compute it
  in a separate program)`.

**Decision (2026-10-06, user):** E3 rejects rather than (B) recomputing the
tensor fresh every generation or (C) hoisting loop-invariant Einsums before the
loop. Hoisting is the most useful option, but it adds a scheduling rule the
paper does not spell out, so it is recorded as future work.

## Generated code

Approach (chosen over carrying a role flag through the IR): codegen already
derives `run()`'s inputs from the IR (`collect_io`: every loaded tensor). It now
also collects every stored tensor (`IR_STORE` and fused outputs), and:

- `inputs` = loaded − written. This drives `def run(...)`, `meta["inputs"]`,
  the `__main__` call, `meta["file_inputs"]` and inferred-dim derivation, all of
  which already use `inputs`.
- Inside `run()`, every written non-generational tensor is allocated empty at
  its `IR_TENSOR_ALLOC` position, after inferred dims are derived. This is the
  existing output-allocation path, extended from "the output" to "every written
  tensor". That fixes G6: `__main__` never allocates an intermediate, so it never
  needs an inferred dim.
- `__main__` allocates and initializes inputs only (as today).

For example, the program above becomes:

```python
__edgecc_meta__ = {"inputs": ["A"], "output": "Z", "file_inputs": []}

def run(A):
    B = [0.0] * N
    Z = [0.0] * N
    ...
    return Z
```

Generational programs are unchanged: their intermediates and cross-generational
tensors already follow this rule, and E3 rules out the non-generational case.

## Testing

Test-first throughout (each new test fails on `main` first).

- **e2e**
  - G6 repro: `dim N = *`, a loaded matrix, an intermediate; runs through its own
    `__main__`.
  - Written-but-never-read tensor (today a `NameError`).
  - Two Einsums writing the same tensor (behaviour preserved).
- **codegen unit test:** the example above generates `def run(A):`,
  `"inputs": ["A"]`, and allocates `B` inside `run()`.
- **Python-level check:** calling `run()` twice gives the same result (the
  module is imported, not just run through `__main__`).
- **negative tests:** E1 on an intermediate, E1 on the returned output, E2, E3.
- **Reconciliation:**
  - `examples/dotproduct.edge`: drop `init Z = [0]`.
  - `tests/e2e/many_inputs_chain` (the G4 test) would silently stop testing G4,
    because its chain tensors become intermediates and only `A0` stays an input.
    Rebuild it with at least 9 *true* inputs.
  - Any unit test asserting the old signature with intermediates as parameters.
- **DFT acceptance** (run against the repo-root DFT files from a scratch copy,
  as in batch A):
  - `dft_naive_re.edge` + `dft_naive_im.edge --n 16` pass **without** `--direct`.
  - `dft_naive_n4_packed.edge` and the rest of the `docs/dft/README.md`
    quickstart still pass.
  - Note for the DFT session: `dft_ref.py --direct` injects tensors missing from
    `run()`'s signature as module globals. After this change, intermediates count
    as "missing", so its `[G4 hack]` message fires falsely (harmless: `run()`'s
    locals shadow the globals) until the hack is retired.
- `tools/premerge.sh` before merge.

## Docs

- `CLAUDE.md`: Generated Python ABI (inputs = read but never written; written
  tensors start empty inside `run()`); scope limitations (E3 restriction,
  hoisting as future work); E1/E2 rules next to the other validation rules.
- `docs/roadmap.md`.

## Out of scope

- Multi-output `run()` / several `save:` directives (batch C, gap G3).
- Hoisting loop-invariant Einsums out of a generational loop (E3's future
  alternative).
- The `<<` update operator (accumulating into an existing tensor), which is the
  paper's sanctioned way to start from non-empty values.
- Sparse storage, memory reuse of intermediates (liveness).

## Deviations and findings (recorded at Finish, 2026-10-28)

- Task 1 also reconciled `test_semantic_fused_pull_classification` (a tenth unit-test program, not in the plan's list) the same way as the nine listed: its written tensors no longer carry an `init`.
- The spec called for a "Python-level check" that `run()` starts from fresh buffers without naming a mechanism; Task 2 added an optional `run_twice` marker file to the e2e harness (`tests/e2e/run_e2e.py`) that calls `run()` twice and compares the results.
- DFT acceptance (all seven kernel invocations, with the first run without `--direct` as the G6 test) passed against this branch's compiler; no kernel was rejected by E1/E2/E3 and no `[G4 hack]` line appeared.
