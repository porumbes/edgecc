# edgecc File Input/Output (Matrix Market, densify-on-load) — Design

**Date:** 2026-06-07
**Status:** Approved (brainstorming complete; ready for implementation plan)

## Goal

Let EDGE programs load larger, real graphs from disk and save result tensors
back to disk, so the compiler can be exercised on existing datasets rather than
only inline `init` literals.

The first cut **densifies on load**: a `.mtx` file is read (via scipy) and
expanded into edgecc's existing dense nested-list representation. The generated
*compute* is unchanged — still dense `for` loops. File I/O is a thin
loader/writer plus inferred dimensions; it does **not** introduce sparse
storage.

## Motivation

Today every input is a literal baked into the generated `__main__`
(`G = [[...], ...]`). Testing against real graphs means hand-transcribing them,
and the size is fixed in source. We want to drop in standard graph files
(Matrix Market / SuiteSparse collection) and run, ideally sweeping many graphs
without recompiling.

## Decisions (and rationale)

1. **Format: Matrix Market (`.mtx`).** The SuiteSparse/NIST standard for sparse
   matrices and graphs; large existing collection. Matrix Market is a *file
   format* (its `coordinate` variant is COO on disk); COO/CSR/CSC are in-memory
   representations and belong to the deferred sparse-compute project, not here.

2. **Densify on load.** Read the `.mtx` and expand into the existing dense
   nested-list representation. Generated compute is untouched. Small, isolated,
   decoupled from the large sparse-codegen effort. Trades memory (dense) for a
   tiny blast radius. See *Deferred* for the sparse-compute note.

3. **Exposure: language-level `init`-from-file.** A new `load("path")`
   init-expression flows through the whole pipeline. This is the long-anticipated
   "file-based init" item. The compiler core only ever carries an **opaque path
   string**; all format knowledge lives in a Python runtime helper.

4. **Generic loader, extension dispatch.** One builtin, `load`, not a
   per-format keyword. The runtime helper dispatches on file extension (`.mtx`
   now; CSV/edge-list later with **no compiler change**). Avoids lexer/parser
   churn per format. (`mtx`-style examples in source are illustrative; the
   builtin name is `load`.)

5. **Name: `load` / `save`.** Idiomatic (numpy/torch), terse, matches EDGE
   builtin style (`mtx`, `add`, `intersect`), and generalizes beyond graphs —
   EDGE tensors are not always graphs (e.g. a GEMM operand), so graph-specific
   names (`gload`, `load_graph`) were rejected as over-narrowing. `save` is the
   natural output partner.

6. **Dimensions: inferred from file.** `dim N = *;` marks a dim whose size is
   set at runtime from the loaded tensor's shape; shared dims (e.g. `F[R=N]`)
   follow. One compiled program then runs on any graph. A concrete
   `dim N = 4;` still works (required by inline-literal inits). Cost: `N` becomes
   a runtime value, and two loads that disagree on a shared dim must error.

7. **Output: `save:` directive in source.** A top-level directive parallel to
   `stop:` — `save: F1 -> "result.mtx";`. Codegen writes the output via a
   `_edgecc_save` helper after `run()`. Keeps everything declared in the `.edge`.

8. **Path override via argv (documented `__main__` convention).** The in-source
   `load("g.mtx")` path is a *default*; positional argv overrides the
   **file-backed inputs** (in `__edgecc_meta__` order), so you compile once and
   sweep many graphs. Mitigations against "hidden behavior":
   - only `load`-backed inputs are overridable (inline literals are not);
   - `__main__` emits a usage comment and a `-h`/`--help` line documenting the
     argument order;
   - wrong argument count errors clearly.

   A proper EDGE-native CLI/argv syntax is deferred future work.

## Language surface

```
dim N = *;                                  // inferred from a loaded tensor
tensor G[R=N, C=N] : float, empty = 0.0;
tensor F[R=N]      : float, empty = 0.0;
init G = load("graph.mtx");                 // dense-materialized from file
init F = [1.0, 0.0, 0.0, 0.0];              // inline literals still work
F1[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);
save: F1 -> "result.mtx";                   // write output after run()
```

- `load(STRING)` is an init-expression — valid **only** on the RHS of `init`.
- `save:` is a top-level directive, parallel to `stop:`.
- `dim N = *;` marks an inferred dim; concrete `dim N = 4;` still works.

## How it flows through the pipeline

The compiler core moves only a **path string**. Format knowledge sits in the
runtime helper.

| Stage | Change |
|-------|--------|
| **Lexer** (`lexer.l`) | New **string-literal token** (`TOK_STRING_LIT`) — none exists today; this is the one genuinely new lexical primitive. New `load`, `save` tokens. `*` for inferred dims. |
| **Parser** (`parser.y`) | `init` RHS gains a `load ( STRING )` alternative beside the literal. New `save : IDENT -> STRING ;` directive. New `dim NAME = * ;` alternative. |
| **AST** (`ast.h/c`) | `InitData` becomes a tagged variant: `INIT_LITERAL { LiteralData* }` or `INIT_LOAD { char *path }`. `DimDeclData` gains `is_inferred`. New `SaveData { char *tensor; char *path; }` node + constructor/destructor. |
| **Semantic** (`semantic.h/c`) | Validate: `load` only inside `init`; `save:` names a real output tensor; every inferred dim is *pinned* by at least one `load`-backed tensor. Record the shared-dim agreement constraint (sizes unknown at compile time → enforced at runtime). |
| **IR** (`ir.h/c`) | `tensor_init` carries a discriminator: literal data **or** a path. New `SAVE` instruction (tensor + path). Inferred dims emit no compile-time constant. |
| **Codegen** (`codegen.c`) | `__main__` emits `G = _edgecc_load("graph.mtx")` instead of a baked literal; inferred dims assigned from loaded shape (`N = len(G)`, `M = len(G[0])` when non-square) before `run()`; after `run()`, emit `_edgecc_save(F1, "result.mtx")`. Plus argv-override block + usage comment. The `run()` body and inner-loop emitter are **untouched** — still dense. |

## Runtime helpers & ABI

Emitted into the generated file **only when** a `load`/`save` is present (keeps
simple programs clean). Imports are lazy so programs that don't use file I/O do
not require scipy.

```python
def _edgecc_load(path):
    # dispatch on extension; .mtx now, others later
    if path.endswith(".mtx"):
        try:
            from scipy.io import mmread
        except ImportError:
            raise ImportError("edgecc load/save requires scipy: pip install scipy")
        return mmread(path).toarray().tolist()   # densify COO -> nested lists
    raise ValueError(f"edgecc load: unsupported file extension for {path!r} (supported: .mtx)")

def _edgecc_save(tensor, path):
    if path.endswith(".mtx"):
        try:
            from scipy.io import mmwrite
            import numpy as np
        except ImportError:
            raise ImportError("edgecc load/save requires scipy: pip install scipy")
        mmwrite(path, np.array(tensor))
        return
    raise ValueError(f"edgecc save: unsupported file extension for {path!r} (supported: .mtx)")
```

**`__main__` changes:**
- Inferred dims assigned from the loaded tensor's shape before `run()`.
- argv override: file-backed inputs (in `__edgecc_meta__` order) replaced by
  positional argv paths; `-h`/`--help` prints usage; wrong count errors.
- `__edgecc_meta__` gains a `"file_inputs"` list so the override mapping is
  explicit and inspectable.

**Dependency:** scipy, imported lazily inside the helpers. CLAUDE.md's
"Generated Python ABI" section documents this runtime dependency.

## Error handling

| Condition | When | Behavior |
|-----------|------|----------|
| `load` used outside `init` | compile | semantic error |
| `save:` names a non-output / unknown tensor | compile | semantic error |
| Inferred `dim N = *` never pinned by any `load` | compile | semantic error |
| File missing / unreadable | runtime | `FileNotFoundError` with the path |
| Two loads disagree on a shared dim | runtime | `ValueError` naming both tensors and sizes |
| `.mtx` shape mismatches tensor rank (matrix into vector slot, etc.) | runtime | `ValueError` |
| Unknown / unimplemented file extension | runtime | `ValueError` listing supported formats |
| scipy not installed | runtime | `ImportError` with `pip install scipy` guidance |
| Wrong number of argv override paths | runtime | clear error + usage |

Compile-time where structural, runtime where the data decides (sizes).

## Testing

- **Unit (per stage):**
  - lexer: string-literal token, `load`/`save`/`*` tokens.
  - parser: init-from-`load`, `save:` directive, `dim = *`.
  - semantic: the three compile-time errors above.
  - ir: init discriminator (literal vs path) + `SAVE` instruction.
  - codegen: emits `_edgecc_load`/`_edgecc_save`, inferred-dim assignment, argv block.
- **E2E** (skipped if scipy missing, mirroring the PyYAML pattern):
  - small committed `.mtx` fixture → compile → run → assert output.
  - round-trip: `load → run → save`, then re-read the saved file and compare.
- **Negative:** `load` outside `init`; unpinned inferred dim; `save:` on unknown
  tensor.
- **Sweep smoke test:** one compiled program run over two different-sized `.mtx`
  fixtures via argv, proving inferred dims + override work together.

## Deferred / future work

- **Sparse end-to-end compute.** Introduce sparse storage (CSR/CSC/COO) into the
  IR and codegen so generated loops iterate only nonzeros — a major rewrite of
  the iteration model, semiring empty-value handling, and codegen. This is the
  existing *sparse tensor support* milestone; densify-on-load here is a
  deliberate stepping stone, and the `_edgecc_load` boundary is chosen so a
  future sparse backend can reuse the file reader. **(Append this note to the
  sparse tensor support project.)**
- **Additional formats** (CSV, edge-list) via runtime-helper extension dispatch —
  no compiler change required.
- **Proper EDGE-native CLI/argv syntax** to replace the positional `__main__`
  override convention.
- **Generated-code dependency management in the build system** — e.g. a
  `requirements.txt`/`pyproject.toml` for runtime deps, a CMake
  `find_package(Python)` + `python -c "import scipy"` preflight, or an
  `edgecc --check-deps` doctor. Broader than this feature (any future emitted
  dependency benefits); design once, deliberately.
- **VS Code extension support** — `load`/`save`/`*` keywords + string-literal
  highlighting in the TextMate grammar, and static autocomplete slots (Tier 1).

## Scope boundaries restated

- Dense compute only; `.mtx` only; one generic `load`/`save`; inferred + concrete
  dims; argv override as a documented `__main__` convention.
- No sparse storage, no non-`.mtx` formats, no EDGE-native CLI, no build-system
  dependency enforcement in this cut.
