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

Run a single unit test: `./build/test_lexer`

## Architecture

Pipeline: `.edge source → Lexer (Flex) → Parser (Bison) → AST → Semantic Analysis → IR → Python Codegen → .py`

- `src/lexer.l` — Flex tokenizer (keywords, identifiers, literals, operators, `//` comments)
- `src/parser.y` — Bison grammar, builds AST via constructors in ast.h. Uses void* casts for Bison 2.3 compat.
- `src/ast.h/c` — AST node types (tagged union structs), constructors, destructors. Forward typedefs + struct definitions for C99 compat.
- `src/semantic.h/c` — Validates index consistency, resolves dimensions, classifies indices as free (output) vs reduction (inner loop). Produces a SymbolTable.
- `src/ir.h/c` — Flat VM-style instruction list (LOOP_BEGIN, LOAD, MAP, REDUCE, STORE, LOOP_END). Lowered from AST + SymbolTable. Free indices emit as outer loops, reduction indices as inner loops.
- `src/codegen.h/c` — Walks IR sequentially, emits Python. LOAD+MAP+REDUCE+STORE are combined into single Python assignment lines at STORE time. Uses CodeBuf (dynamic string buffer with indent tracking).
- `src/main.c` — CLI driver: reads file, runs all pipeline stages, writes output.

## EDGE Syntax (v1)

```
dim M = 4;
tensor Z[R=M, C=N] : float, empty = 0.0;
init A = [[1, 2], [3, 4]];
Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);
```

Rank variables in expressions bind to declared rank names by position. Built-in operators: add, sub, mul, min, max, and, or.

## v1 Scope Limitations

Not yet supported: merge/coordinate operators, generational/iterative ranks, multi-operation expressions with compute labels, Populate action, unary functions, file-based init, sparse tensors, optimizations.
