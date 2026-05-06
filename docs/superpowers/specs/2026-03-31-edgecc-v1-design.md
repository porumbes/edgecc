# edgecc v1 Design Spec

EDGE language compiler — first pass. Targets basic Map+Reduce Einsum expressions, transpiling to Python.

## Background

EDGE is a declarative, index-driven language that extends Einstein summation (Einsum) notation with alternative semirings. It originates from research by the Owens Group (researcher: Toluwanimi Odemuyiwa) and is described in a ~100-page draft paper with formal BNF grammar and semantics.

EDGE has three core operations:

- **Map**: a binary operation applied to paired tensor elements (e.g., multiply, boolean AND)
- **Reduce**: an aggregation operation applied to accumulated results (e.g., add, min)
- **Populate**: a fiber-level operation that considers a collection of values to decide output (deferred to a later version)

Indices drive iteration: they latch together coordinates across tensors, determining which elements are paired and how loops are structured.

## Scope (v1)

**In scope:**
- Tensor declarations with rank names, shapes, datatypes, and empty values
- Dimension declarations
- Tensor initialization via inline literals or default (empty value)
- Single Einsum expressions with one Map and one Reduce operation
- Two input tensors paired with `.`, one output tensor
- Built-in binary operators: `add`, `sub`, `mul`, `min`, `max`, `and`, `or`
- Python code generation (pure nested for-loops, scalar operations)

**Out of scope (future work):**
- Merge/coordinate operators (intersection, union)
- Generational/iterative ranks and termination conditions
- Multi-operation expressions with compute labels
- Sub-expressions and anonymous tensors
- Populate action
- Unary functions
- File-based tensor initialization
- Sparse tensor support
- Optimizations of any kind

## Syntax

### Dimension declarations

```
dim M = 4;
dim N = 3;
dim K = 2;
```

Declares named dimension sizes as positive integer literals.

### Tensor declarations

```
tensor Z[R=M, C=N] : float, empty = 0.0;
tensor A[R=K, C=M] : float, empty = 0.0;
tensor B[R=K, C=N] : float, empty = 0.0;
```

Each tensor declaration specifies:
- **Tensor name**: an identifier (e.g., `Z`)
- **Rank names and shapes**: `rankname = dimname` pairs in square brackets. Rank names are identifiers declared here; dim names refer to previously declared dimensions.
- **Datatype**: one of `integer`, `float`, `boolean`
- **Empty value**: the value used to represent "empty" (e.g., `0`, `0.0`, `inf`, `false`)

### Tensor initialization

```
init A = [[1, 2], [3, 4], [5, 6]];
init B = [[7, 8], [9, 10], [11, 12]];
```

- Omitting `init` means the tensor is filled with its empty value.
- Inline initialization uses nested lists in row-major order. Nesting depth must match the number of ranks.

### Einsum expressions

```
Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);
```

Structure:
- **Output**: `TensorName[rank_var, ...]` — the target tensor with rank variables
- **`=`** — assignment
- **Inputs**: `TensorName[rank_var, ...] . TensorName[rank_var, ...]` — two input tensors paired with `.`
- **`::`** — separates tensor expression from action specification
- **Map action**: `map(operator)` — the binary operation applied to each pair
- **Reduce action**: `reduce(operator)` — the binary operation used for aggregation
- **`;`** — statement terminator

Rank variables (lowercase, in subscript/bracket position of expressions) bind to rank names (in declarations) by position, not by name. For example, in `A[k,m]`, `k` binds to A's first declared rank and `m` binds to A's second declared rank.

Index semantics:
- Indices appearing in both inputs but not the output are **reduction indices** (they form the inner loop)
- Indices appearing in the output are **free indices** (they form the outer loops)
- Every index in the expression must be traceable to a declared rank with a known dimension size

## Complete Example: GEMM

```
dim M = 2;
dim K = 3;
dim N = 2;

tensor Z[R=M, C=N] : float, empty = 0.0;
tensor A[R=K, C=M] : float, empty = 0.0;
tensor B[R=K, C=N] : float, empty = 0.0;

init A = [[1, 2], [3, 4], [5, 6]];
init B = [[7, 8], [9, 10], [11, 12]];

Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);
```

Expected generated Python:

```python
M = 2
K = 3
N = 2

Z = [[0.0] * N for _ in range(M)]
A = [[1, 2], [3, 4], [5, 6]]
B = [[7, 8], [9, 10], [11, 12]]

for m in range(M):
    for n in range(N):
        for k in range(K):
            Z[m][n] = Z[m][n] + A[k][m] * B[k][n]

print(Z)
```

## Compiler Pipeline

```
.edge source → Lexer → Parser → AST → Semantic Analysis → IR → Python Codegen → .py output
```

### Lexer (Flex)

Tokenizes:
- Keywords: `dim`, `tensor`, `init`, `map`, `reduce`, `integer`, `float`, `boolean`, `empty`, `true`, `false`, `inf`
- Identifiers: `[a-zA-Z_][a-zA-Z0-9_]*`
- Integer literals: `[0-9]+`
- Float literals: `[0-9]+\.[0-9]+`
- Operators/punctuation: `=`, `.`, `::`, `:`, `,`, `;`, `[`, `]`, `(`, `)`
- Whitespace: skipped
- Comments: TBD (suggest `//` line comments for v1)

### Parser (Bison)

Produces an AST. Grammar supports a program as a sequence of statements:
- `dim_decl`: `dim IDENT = INT_LIT ;`
- `tensor_decl`: `tensor IDENT [ rank_list ] : type , empty = literal ;`
- `init_stmt`: `init IDENT = nested_list ;`
- `einsum_stmt`: `output = input . input :: map ( IDENT ) reduce ( IDENT ) ;`

### AST

Node types:
- `ProgramNode` — list of statements
- `DimDeclNode` — name, size (integer)
- `TensorDeclNode` — name, list of (rank_name, dim_name) pairs, datatype, empty value
- `InitNode` — tensor name, nested literal data
- `EinsumNode` — output tensor ref, two input tensor refs, map operator, reduce operator
- `TensorRefNode` — tensor name, list of rank variable names

The AST is represented as C structs with tagged unions (or a simple enum + union approach).

### Semantic Analysis

Validates:
1. All dimension names in tensor declarations refer to declared `dim` values
2. All tensor names in `init` and `einsum` statements refer to declared tensors
3. Rank variable consistency: every rank variable in an expression maps to exactly one dimension size. If `m` appears in both `A[k,m]` and `Z[m,n]`, the corresponding ranks must have the same dimension.
4. Reduction index identification: indices in both inputs but not the output are reduction indices
5. No undeclared or orphan indices
6. Operator names are valid built-in operators
7. Init data shape matches the tensor's declared dimensions

Produces an annotated AST or symbol table with resolved dimension sizes and index classifications.

### IR

A flat list of VM-style instructions representing the computation:

- `IR_DIM_DEF(name, size)` — define a dimension
- `IR_TENSOR_ALLOC(name, dims[], ndims, empty_val)` — allocate and initialize tensor
- `IR_TENSOR_INIT(name, data)` — load literal data into tensor
- `IR_LOOP_BEGIN(index_var, range)` — begin a loop over an index
- `IR_LOOP_END` — close innermost loop
- `IR_LOAD(dst_reg, tensor, coords[])` — load a scalar from a tensor
- `IR_MAP(dst_reg, op, src_reg1, src_reg2)` — apply binary map operation
- `IR_REDUCE(dst_reg, op, src_reg, acc_reg)` — apply binary reduce operation
- `IR_STORE(tensor, coords[], src_reg)` — store a scalar to a tensor

The IR is generated by:
1. Emitting dimension definitions and tensor allocations/inits
2. Emitting nested loops: free indices (output indices) as outer loops, reduction indices as inner loops
3. Inside the innermost loop: LOAD both inputs, MAP, REDUCE into the output location

### Python Code Generation

Walks the IR list sequentially and emits Python source:
- `IR_DIM_DEF` → `M = 4`
- `IR_TENSOR_ALLOC` → nested list comprehension initialized to empty value
- `IR_TENSOR_INIT` → literal assignment
- `IR_LOOP_BEGIN` → `for idx in range(size):` with indentation tracking
- `IR_LOOP_END` → decrease indentation
- `IR_LOAD/MAP/REDUCE/STORE` → corresponding Python expressions inside the loop body

The output is a self-contained `.py` file that can be run with `python3`.

## Project Structure

```
edgecc/
├── CMakeLists.txt
├── CLAUDE.md
├── src/
│   ├── main.c            # Driver: CLI arg parsing, runs pipeline, writes output
│   ├── lexer.l           # Flex lexer specification
│   ├── parser.y          # Bison grammar specification
│   ├── ast.h             # AST node type definitions
│   ├── ast.c             # AST constructors and destructors
│   ├── semantic.h        # Semantic analysis interface
│   ├── semantic.c        # Semantic analysis implementation
│   ├── ir.h              # IR instruction types and builder interface
│   ├── ir.c              # AST-to-IR lowering
│   ├── codegen.h         # Python code generator interface
│   └── codegen.c         # IR-to-Python emission
├── tests/
│   ├── unit/
│   │   ├── test_lexer.c
│   │   ├── test_parser.c
│   │   ├── test_semantic.c
│   │   ├── test_ir.c
│   │   └── test_codegen.c
│   ├── e2e/
│   │   ├── gemm/
│   │   │   ├── input.edge
│   │   │   ├── expected.py
│   │   │   └── expected_output.txt
│   │   └── ...
│   ├── negative/
│   │   ├── bad_syntax/
│   │   │   ├── input.edge
│   │   │   └── expected_errors.txt
│   │   └── ...
│   └── run_e2e.py        # End-to-end test driver script
├── examples/
│   └── gemm.edge         # Example EDGE program
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-03-31-edgecc-v1-design.md
```

## Build System

CMake with `find_package(FLEX)` and `find_package(BISON)` for cross-platform macOS/Linux support.

Build commands:
```
mkdir build && cd build
cmake ..
make
```

Run:
```
./edgecc ../examples/gemm.edge -o output.py
python3 output.py
```

## Testing Strategy

### Unit tests (per pipeline stage)

All unit tests use CTest as the runner. A minimal single-header C test framework (either custom or Unity) provides assertion macros.

- **Lexer tests** (`test_lexer.c`): Feed source strings to the lexer, assert correct token types and values in sequence.
- **Parser tests** (`test_parser.c`): Feed source strings through lexer+parser, assert correct AST structure (node types, children, values).
- **Semantic tests** (`test_semantic.c`): Construct or parse ASTs, run semantic analysis, assert pass/fail and specific error messages for invalid programs.
- **IR tests** (`test_ir.c`): Feed validated ASTs through IR lowering, assert correct instruction sequences (loop nesting, operation ordering).
- **Codegen tests** (`test_codegen.c`): Feed IR instruction lists to the code generator, compare emitted Python source against golden strings.

### End-to-end tests

Each test case is a directory under `tests/e2e/` containing:
- `input.edge` — the EDGE source program
- `expected.py` — expected generated Python (optional, for verifying codegen)
- `expected_output.txt` — expected stdout when running the generated Python

The `run_e2e.py` driver:
1. Runs `edgecc` on `input.edge`
2. Optionally diffs generated Python against `expected.py`
3. Runs the generated Python with `python3`
4. Diffs stdout against `expected_output.txt`
5. Reports pass/fail per test case

### Negative tests

Each test case is a directory under `tests/negative/` containing:
- `input.edge` — an invalid EDGE program
- `expected_errors.txt` — expected error message substrings (one per line)

The driver runs `edgecc`, captures stderr, and verifies each expected error substring appears.

### Running all tests

```
cmake --build build && ctest --test-dir build
```

CTest discovers and runs unit tests as individual executables and invokes the e2e/negative test drivers.

## Dependencies

- C compiler (gcc or clang)
- CMake >= 3.15
- Flex
- Bison
- Python 3 (for running generated code and e2e test driver)

No other external libraries.
