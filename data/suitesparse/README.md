# SuiteSparse test graphs

Real graphs from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu)
(Davis & Hu), used to exercise edgecc's `load`/`save` and dense-compute path on
non-trivial input. Each ships as Matrix Market (`.mtx`) and drops straight into
`load(...)`.

## Vendored (committed) — the curated three

| File | SuiteSparse | N | Semiring use | Original source |
|------|-------------|---|--------------|-----------------|
| `karate.mtx` | Newman/karate | 34 | boolean / count | Zachary 1977 karate club |
| `lesmis.mtx` | Newman/lesmis | 77 | weighted | Knuth, *Les Misérables* co-appearance |
| `GD98_a.mtx` | Pajek/GD98_a | 38 | directed | Graph Drawing 1998 contest |

## Fetch-on-demand (gitignored)

The rest of `manifest.json` (dolphins, polbooks, football, Ragusa16/18, GD96_b,
celegansneural) is not committed. Download into this directory with:

    python3 tools/fetch_suitesparse.py <name>      # e.g. dolphins
    python3 tools/fetch_suitesparse.py --all       # everything in the manifest

The script verifies each file's sha256 against `manifest.json`.

## Integrity / provenance

`manifest.json` records `group`, `name`, `n`, `sha256`, `kind`, `semiring`,
`source`, and `url` per graph. SuiteSparse matrices are freely redistributable;
please credit the collection and the original authors above when reusing.
