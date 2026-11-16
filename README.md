# edgecc

A compiler for **EDGE**, a declarative, index-driven tensor language that extends
Einstein summation (Einsum) notation with alternative semirings. Each EDGE
statement is a *map* (a binary operation on paired elements) and a *reduce* (an
aggregation), each parameterized by a compute operator (`mul`, `add`, `min`,
`and`, …) and a merge operator (`intersect`, `union`, `take_left_only`, …).
The same notation expresses linear algebra (GEMM), graph algorithms (BFS,
shortest paths, reachability) and transforms such as the DFT.

edgecc compiles a `.edge` program to plain Python — nested loops over scalar
operations — so a program's meaning can be run and checked against independent
reference implementations. It is a reference compiler: there are no
optimizations yet, and tensors are dense.

## Example

Bellman-Ford shortest paths as a two-Einsum cascade that repeats until the
distances stop changing (`examples/bellman_ford.edge`):

```
dim N = 4;
tensor W[R=N, C=N]   : float, empty = inf;
tensor T[gen I, R=N] : float, empty = inf;   // relaxed candidates
tensor D[gen I, R=N] : float, empty = inf;   // distances, one row per generation
init W = [[inf, 1.0, 4.0, inf], [inf, inf, 2.0, 6.0], [inf, inf, inf, 3.0], [inf, inf, inf, inf]];
init D = [0.0, inf, inf, inf];
T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);   // relax
D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);      // combine
stop: stable(D);
```

```bash
./build/edgecc examples/bellman_ford.edge -o out.py
python3 out.py        # every generation of D; the last is [0.0, 1.0, 3.0, 6.0]
```

## Build and test

Requires a C99 compiler, CMake ≥ 3.15, Flex, Bison (2.3 or later) and Python 3.
Some tests also use numpy, scipy and PyYAML; they are skipped when missing.

```bash
cmake -S . -B build && make -C build
ctest --test-dir build --output-on-failure
```

## Language and docs

- [`CLAUDE.md`](CLAUDE.md) — the full working guide: syntax (dims, tensors,
  `init`, Einsums, merge and compute operators, generational ranks and stop
  predicates, fused Einsums, `output:`), file I/O (`load` / `save:` for Matrix
  Market), the generated Python ABI, the test suites and the pre-merge gate.
- [`docs/roadmap.md`](docs/roadmap.md) — what is implemented, what is next, and
  known issues.
- [`docs/superpowers/specs/`](docs/superpowers/specs) — the design of each
  feature; [`docs/superpowers/plans/`](docs/superpowers/plans) — how each was
  built.
- [`docs/visualization.md`](docs/visualization.md) — per-generation traces and
  rendered frames for generational kernels.
- [`examples/`](examples) — small programs: GEMM, BFS variants, reachability,
  shortest paths, widest path, connected components, transitive closure.
- [`editors/vscode/`](editors/vscode) — syntax highlighting and completion for
  `.edge` files.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
