# edgecc SuiteSparse test-set — design

**Date:** 2026-06-08
**Status:** Draft for review
**Scope:** Data setup only (acquisition + layout + tests of existing capabilities). The
seed-vector feature and the data-sweep runner are explicitly out of scope (see Future Work).

## Motivation

edgecc's `load`/`save` file I/O (merged 2026-06-07) lets EDGE programs read real graphs
from Matrix Market `.mtx` files, but every `.mtx` currently in the repo is a hand-authored
4×4 fixture (68–92 bytes). We want a small, curated set of **real** graphs from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu) so the loader and dense-compute
path are exercised against non-trivial input, with results that are still hand-verifiable.

### Gating constraint

edgecc **densifies on load** and runs interpreted nested loops, so graph size is bounded:

| Computation | Complexity | Practical N |
|-------------|-----------|-------------|
| single map/reduce push step | O(N²) | ≤ ~1000 |
| iterative BFS/SSSP | O(N²)·gens | ≤ ~300 |
| GEMM | O(N³) | ≤ ~150 |

We pick the smallest classic graphs so outputs are hand-verifiable.

### What this work does NOT do

- **No seed vector.** BFS/SSSP need a source vector `F` of length N; today that is an
  inline literal that cannot match an arbitrarily-sized loaded graph (the runtime
  shared-dim assert rejects it). Solving that is a separate design. Consequently the tests
  here use only kernels whose **sole input is the graph itself** (no length-N companion).
- **No diagonal/reflexive-adjacency preprocessing** (adding `I` for reachability/transitive
  closure). Also deferred.
- **No data-sweep runner** (one kernel × many graphs). Deferred; this design chooses the
  argv-override path-resolution mechanism specifically so that runner is cheap to add later.

## Candidate graphs

All ship as Matrix Market `.mtx`, so they drop straight into `load(...)`.

| Tier | Graph | N | Use | Semiring family |
|------|-------|---|-----|-----------------|
| 1 | `Newman/karate` | 34 | Zachary's club — start here | boolean / count |
| 1 | `Newman/dolphins` | 62 | symmetric pattern | boolean |
| 1 | `Newman/polbooks` | 105 | | boolean |
| 1 | `Newman/football` | 115 | | boolean |
| 2 | `Newman/lesmis` | 77 | real weights, symmetric | weighted (min-plus, max-min) |
| 2 | `Pajek/Ragusa16` / `Ragusa18` | 24 / 23 | tiny directed weighted, integer | weighted |
| 3 | `Pajek/GD98_a` | 38 | directed | directed |
| 3 | `Pajek/GD96_b` | 111 | directed | directed |
| 3 | `Newman/celegansneural` | 297 | directed real — near iterative ceiling | weighted-directed |

**Vendored first-three** (cover all three semiring families, all a few KB):
`Newman/karate`, `Newman/lesmis`, `Pajek/GD98_a`. The rest are fetch-on-demand.

## Architecture

### A1. Directory layout

A new top-level `data/` directory (peer of `examples/`, `tests/`) so examples and the viz
tool can use the graphs too, not just tests.

```
data/
└── suitesparse/
    ├── manifest.json          # curated set: provenance + integrity
    ├── README.md              # attribution / licensing
    ├── karate.mtx             # vendored (Newman/karate, N=34)
    ├── lesmis.mtx             # vendored (Newman/lesmis, N=77, weighted)
    └── GD98_a.mtx             # vendored (Pajek/GD98_a, N=38, directed)
```

Fetched (non-vendored) graphs also land in `data/suitesparse/` but are gitignored. Add to
the repo `.gitignore`:

```gitignore
data/suitesparse/*.mtx
!data/suitesparse/karate.mtx
!data/suitesparse/lesmis.mtx
!data/suitesparse/GD98_a.mtx
```

The existing tiny 4×4 per-test fixtures stay as-is (self-contained unit-style cases).

### A2. Manifest (`data/suitesparse/manifest.json`)

One entry per curated graph. `sha256` is recorded and verified for **all** graphs,
including vendored ones (guards against corruption).

```json
{
  "karate": {
    "group": "Newman", "name": "karate", "n": 34,
    "vendored": true, "sha256": "<sha256 of karate.mtx>",
    "kind": "undirected-pattern", "semiring": "boolean",
    "source": "Zachary 1977 karate club",
    "url": "https://sparse.tamu.edu/Newman/karate"
  },
  "lesmis": {
    "group": "Newman", "name": "lesmis", "n": 77,
    "vendored": true, "sha256": "<...>",
    "kind": "undirected-weighted", "semiring": "weighted",
    "source": "Knuth, Les Misérables co-appearance",
    "url": "https://sparse.tamu.edu/Newman/lesmis"
  },
  "GD98_a": {
    "group": "Pajek", "name": "GD98_a", "n": 38,
    "vendored": true, "sha256": "<...>",
    "kind": "directed", "semiring": "directed",
    "source": "Graph Drawing 1998 contest",
    "url": "https://sparse.tamu.edu/Pajek/GD98_a"
  },
  "dolphins":       { "group": "Newman", "name": "dolphins",       "n": 62,  "vendored": false, "sha256": "<...>", "kind": "undirected-pattern", "semiring": "boolean",          "url": "https://sparse.tamu.edu/Newman/dolphins" },
  "polbooks":       { "group": "Newman", "name": "polbooks",       "n": 105, "vendored": false, "sha256": "<...>", "kind": "undirected-pattern", "semiring": "boolean",          "url": "https://sparse.tamu.edu/Newman/polbooks" },
  "football":       { "group": "Newman", "name": "football",       "n": 115, "vendored": false, "sha256": "<...>", "kind": "undirected-pattern", "semiring": "boolean",          "url": "https://sparse.tamu.edu/Newman/football" },
  "Ragusa16":       { "group": "Pajek",  "name": "Ragusa16",       "n": 24,  "vendored": false, "sha256": "<...>", "kind": "directed-weighted",  "semiring": "weighted",         "url": "https://sparse.tamu.edu/Pajek/Ragusa16" },
  "Ragusa18":       { "group": "Pajek",  "name": "Ragusa18",       "n": 23,  "vendored": false, "sha256": "<...>", "kind": "directed-weighted",  "semiring": "weighted",         "url": "https://sparse.tamu.edu/Pajek/Ragusa18" },
  "GD96_b":         { "group": "Pajek",  "name": "GD96_b",         "n": 111, "vendored": false, "sha256": "<...>", "kind": "directed",           "semiring": "directed",         "url": "https://sparse.tamu.edu/Pajek/GD96_b" },
  "celegansneural": { "group": "Newman", "name": "celegansneural", "n": 297, "vendored": false, "sha256": "<...>", "kind": "directed-weighted",  "semiring": "weighted-directed", "url": "https://sparse.tamu.edu/Newman/celegansneural" }
}
```

> Implementation note: the `<...>` sha256 values are placeholders filled in during
> implementation by hashing the actual extracted `.mtx` files.

### A3. Fetch script (`tools/fetch_suitesparse.py`)

- **Stdlib only** (`urllib`, `tarfile`, `hashlib`, `json`, `argparse`) — no new runtime
  dependency.
- Usage: `fetch_suitesparse.py <name>` or `fetch_suitesparse.py --all`.
- For each requested graph: read the manifest entry → download
  `https://sparse.tamu.edu/MM/<group>/<name>.tar.gz` → extract the inner `<name>.mtx` from
  the tarball → verify sha256 against the manifest → write `data/suitesparse/<name>.mtx`.
- Vendored graphs already present are checksum-verified and skipped unless `--force`.
- Clear, actionable errors on network failure, missing tarball member, or checksum
  mismatch (mismatch is a hard failure, not a warning).

### A4. Provenance / licensing (`data/suitesparse/README.md`)

Credits the SuiteSparse Matrix Collection (Davis & Hu) and the per-graph original sources
(Zachary's karate club; Knuth's *Les Misérables* co-appearance network; the GD98
graph-drawing contest). SuiteSparse matrices are freely redistributable, so vendoring the
three tiny graphs with attribution is fine. The README also documents
`tools/fetch_suitesparse.py` for the non-vendored graphs.

## Tests

### Path resolution — argv override

The e2e harness runs the generated program with `cwd=<test_dir>`, so a bare `load("x.mtx")`
resolves relative to the test dir, not to `data/`. We resolve this via the generated
program's existing **positional-argv file-input override** ABI:

- Extend `tests/e2e/run_e2e.py`: if a test dir contains an `args` file, read its
  whitespace-separated tokens, replace a `{DATA}` token with the absolute path to the
  repo-root `data/` directory, and pass the result as positional arguments when running the
  generated program. Tests without an `args` file are unaffected.
- Each test's `input.edge` keeps a plain default `load("<name>.mtx")`; the argv override
  always supplies the real path, so the default is never used at test time.

This decouples the kernel `.edge` from the data layout and gives the argv-override ABI its
first end-to-end coverage. It is also the primitive a future data-sweep runner needs.

### Kernels (no seed vector required)

Each kernel's only input is the graph; output is a **length-N vector** (not N×N), keeping
`expected_output.txt` small and hand-verifiable. The key trick is the diagonal of G²:
`Z[i] = G[i,k] · G[k,i]`.

| Test dir | Kernel | Graph | Computes | Output |
|----------|--------|-------|----------|--------|
| `load_karate_degree` | `Z[i] = G[i,k] . G[k,i] :: map(mul,intersect) reduce(add,union)` | karate (34) | 2-cycle count = **degree** for a simple undirected graph | length-34 vector |
| `load_gd98a_2cycles` | same kernel | GD98_a (38) | 2-cycles through each node; exercises directed asymmetry | length-38 vector |
| `load_lesmis_minplus` | `Z[i] = W[i,k] . W[k,i] :: map(add,intersect) reduce(min,union)` | lesmis (77) | cheapest 2-edge round-trip weight; exercises the weighted semiring | length-77 vector |

Notes:
- lesmis weights are co-appearance counts, so min-plus is semantically arbitrary there.
  That is acceptable — these tests validate the **loader + dense-compute path**, not graph
  semantics. The test will carry a comment saying so.
- The karate output is eyeballed against the known degree sequence of Zachary's club; all
  three `expected_output.txt` files are generated once via an independent scipy computation.

### Test layout

```
tests/e2e/load_karate_degree/
├── input.edge          # diagonal-of-G² degree kernel + `save: Z -> "result.mtx";`
├── args                # {DATA}/suitesparse/karate.mtx
├── expected_output.txt # length-34 degree vector
├── roundtrip_check     # expected re-read values from result.mtx (exercises save)
└── needs_scipy
tests/e2e/load_gd98a_2cycles/
├── input.edge
├── args                # {DATA}/suitesparse/GD98_a.mtx
├── expected_output.txt
└── needs_scipy
tests/e2e/load_lesmis_minplus/
├── input.edge
├── args                # {DATA}/suitesparse/lesmis.mtx
├── expected_output.txt
└── needs_scipy
```

- The **karate** test additionally exercises `save` on a real graph via the harness's
  existing `roundtrip_check` mechanism (re-read `result.mtx`, compare flattened values).
- All three carry `needs_scipy` and skip cleanly when scipy is absent (consistent with the
  existing load tests).

## Error handling

- **Fetch:** network failure, missing tarball member, and sha256 mismatch are all hard
  failures with clear messages. A checksum mismatch never silently proceeds.
- **Tests:** absence of scipy → skip (existing behavior). Absence of a vendored graph would
  fail the test loudly (it should never happen — they are committed).
- **Missing `args` token:** if `run_e2e.py` finds `{DATA}` but cannot resolve the data dir,
  it fails the test with a clear message rather than passing a literal `{DATA}` string.

## Testing strategy

- The three new e2e tests run under the existing `ctest` e2e target; they skip without
  scipy, like the current load tests.
- `tools/fetch_suitesparse.py` is verified manually during implementation by fetching at
  least one non-vendored graph and confirming checksum + placement. (A networked unit test
  is out of scope; CI should not depend on sparse.tamu.edu reachability.)
- The vendored graphs' checksums are recorded in the manifest and re-verified by the fetch
  script's `--force`-less default path.

## Future work

- **Seed vector.** Load an N×1 seed `.mtx`, or add a source/unit-vector seeding construct,
  so BFS/SSSP kernels run end-to-end on these graphs. This is the most likely next feature
  the expanded set motivates.
- **Diagonal / reflexive adjacency.** A small preprocessing pass to add `I` for
  reachability / transitive-closure formulations (SuiteSparse graphs have a zero diagonal).
- **Data-sweep runner.** A parameterized harness running one kernel over many graphs with
  per-graph expected outputs. The argv-override path resolution chosen here is the enabling
  primitive; the sweep runner itself is a separate, larger test-harness capability.
- **Larger / non-`.mtx` datasets and sparse compute** remain part of the deferred
  sparse-tensor-storage milestone.

## References

- File I/O design: `docs/superpowers/specs/2026-06-07-edgecc-file-io-design.md`
- SuiteSparse Matrix Collection: https://sparse.tamu.edu
- Existing load tests: `tests/e2e/load_mtx/`, `tests/e2e/load_save_roundtrip/`,
  `tests/e2e/load_two_inferred_dims/`
