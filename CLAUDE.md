# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is edgecc?

edgecc is a compiler/transpiler for the EDGE language, a declarative index-driven language that extends Einstein summation (Einsum) notation with alternative semirings. EDGE expressions describe tensor computations using Map (binary op on paired elements) and Reduce (aggregation). The compiler targets Python — generating pure nested for-loops with scalar operations.

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

Requires: C compiler (C99), CMake >= 3.15, Flex, Bison, Python 3.

Note: macOS ships Bison 2.3. The parser.y uses void* casts in the %union to stay compatible. If upgrading to Bison 3.x, the %union can use actual struct types.

## Run

```bash
./build/edgecc input.edge -o output.py
python3 output.py
```

## Test

```bash
cd build && ctest --output-on-failure
```

This runs:
- **Unit tests** per pipeline stage: `test_lexer`, `test_parser`, `test_semantic`, `test_ir`, `test_codegen`
- **E2E tests** (`tests/e2e/run_e2e.py`): compiles `.edge` files, runs generated Python, compares stdout against expected output
- **Negative tests** (`tests/negative/run_negative.py`): invalid programs that must produce specific error messages
- **Iterative E2E tests** (`tests/e2e/iterative/run_iterative.py`): drives a generated kernel in a loop with feedback until a watched tensor stabilizes (e.g., BFS). Requires PyYAML (`pip install pyyaml`); skipped if missing.

Run a single unit test: `./build/test_lexer`

## Architecture

Pipeline: `.edge source → Lexer (Flex) → Parser (Bison) → AST → Semantic Analysis → IR → Python Codegen → .py`

- `src/lexer.l` — Flex tokenizer (keywords, identifiers, literals, operators, `//` comments).
- `src/parser.y` — Bison grammar, builds AST via constructors in ast.h. Uses void* casts for Bison 2.3 compat.
- `src/ast.h/c` — AST node types (tagged union structs), constructors, destructors. Forward typedefs + struct definitions for C99 compat. Defines `MergeOp` (9-value enum) alongside `OperatorType`.
- `src/semantic.h/c` — Validates index consistency, resolves dimensions, classifies indices as free (output) vs reduction (inner loop). Produces a SymbolTable. Provides `compute_op_identity` (identity per built-in compute op) and emits a `nopass` warning to stderr.
- `src/ir.h/c` — Flat VM-style instruction list (LOOP_BEGIN, LOAD, MAP, REDUCE, STORE, LOOP_END). Lowered from AST + SymbolTable. Each MAP/REDUCE carries `merge_op`, `identity`, and operand empty values for codegen.
- `src/codegen.h/c` — Walks IR and emits Python wrapped in `def run(<inputs>) -> <output>` plus a `__main__` block. The inner-loop body emits the general merge skeleton (existence checks → merge guard → compute → store-or-empty). No special-case fast path — all merge configurations go through the same emitter.
- `src/main.c` — CLI driver: reads file, runs all pipeline stages, writes output. Supports `--dump-ir` to print the IR (including merge keyword and identity values) to stdout for debugging.

## EDGE Syntax (v2)

```
dim M = 4;
tensor Z[R=M, C=N] : float, empty = 0.0;
init A = [[1, 2], [3, 4]];
Z[m,n] = A[k,m] . B[k,n] :: map(mul, intersect) reduce(add, union);
```

### Generational (iterative) ranks

A single Einsum may iterate via a **generational rank** declared with `gen` (no
dimension). The Einsum reads generation `i` and writes generation `i+1`; the
loop runs until the watched tensor is stable:

```
dim N = 4;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
init F = [1.0, 0.0, 0.0, 0.0];           // generation 0
F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);
stop: stable(F);
```

Generated `run()` loops internally with a **fresh (empty) output slice each
generation** (the paper's default populate; no cross-generation accumulation),
double-buffered as `F_prev`/`F_next`, and returns the final state. A fixed
safety cap (`__EDGECC_MAX_GEN__ = 1000`) raises `RuntimeError` if the watched
tensor never stabilizes. Limitations: one generational tensor / one
generational rank, offsets `i` and `i+1` only, `stable` stop only, single
Einsum, no `<<` accumulation. See `docs/superpowers/specs/2026-06-01-edgecc-generational-ranks-a-design.md`.

Rank variables in expressions bind to declared rank names by position. Built-in compute operators: `add`, `sub`, `mul`, `min`, `max`, `and`, `or`. Built-in merge operators (optional second arg, default `passthrough`): `passthrough`, `nopass`, `intersect`, `take_left`, `take_left_only`, `take_right`, `take_right_only`, `xor`, `union`. Omitting the merge argument preserves v1 source-program semantics.

## Generated Python ABI

```python
import math

S = 4
D = 4

__edgecc_meta__ = {"inputs": ["G", "F"], "output": "F1"}

def run(G, F):
    F1 = ...
    for d in range(D):
        ...
    return F1

if __name__ == "__main__":
    G = ...
    F = ...
    F1 = run(G, F)
    print(F1)
```

The `if __name__ == "__main__"` block constructs inputs from `init` declarations and prints the output. Iterative drivers import the module and call `run()` repeatedly with feedback applied between iterations (see `tests/e2e/iterative/run_iterative.py`).

## Iterative Tests

Tests under `tests/e2e/iterative/<name>/` consist of `kernel.edge` plus `driver.yaml`. The driver YAML has this shape:

```yaml
inputs:
  G: [...]
  F: [0.0, .inf, .inf, .inf]

feedback:
  to: F                # input slot updated by run()'s output each iteration
  combine: min         # one of: replace (default), min, max, add, or, and

termination:
  type: stable         # only "stable" supported in v2
  watch: F             # watched tensor; iteration stops when its value is unchanged

max_iterations: 10

expected:
  final:
    F: [0.0, 1.0, 1.0, 2.0]
```

Note: the field is `watch:` (not `on:`) because YAML 1.1 (PyYAML's default) parses bare `on` as boolean `True`.

Iterative tests are wired into ctest as the `e2e_iterative` target. Missing PyYAML causes only iterative tests to be skipped; everything else runs.

## VS Code extension

Lives in `editors/vscode/`. Provides syntax highlighting and static-list autocompletion for `.edge` files.

Build: `cd editors/vscode && npm install && npm run package`
Install: `code --install-extension edgecc-0.1.0.vsix`
Test: `cd editors/vscode && npm test` (full suite) or `npm run test:grammar` (grammar only, fast)
Develop: open `editors/vscode/` in VS Code and press F5 (Extension Development Host).
Requires Node.js 18+.

Coverage is Tier 1 only — static dropdowns for compute / merge / type / empty-value slots; no parsing, no diagnostics, no go-to-definition. Tier 2 features (context-aware completions) and Neovim port are deferred to future work.

## v2 Scope Limitations

Not yet supported: multi-operation expressions with compute labels, multi-Einsum generational cascades (the `<<` update/accumulation populate, and chained Einsums sharing a generational rank — single-Einsum generational ranks are supported; see "Generational (iterative) ranks"), anonymous tensors / parenthesized sub-expressions, unary functions, the 7 (No,No)→Yes merge operators (NOR, equivalence, NAND, ¬→, ¬←, ⇐, ⇒), file-based init, sparse tensor storage, optimizations of any kind, user-defined compute operators.
