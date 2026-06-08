# SuiteSparse Test-Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a curated set of real SuiteSparse `.mtx` graphs (vendored karate/lesmis/GD98_a + a stdlib fetch script for the rest) and three no-seed e2e tests that exercise edgecc's loader and dense-compute path on them.

**Architecture:** A new top-level `data/suitesparse/` holds vendored graphs plus a `manifest.json` (provenance + sha256). `tools/fetch_suitesparse.py` (stdlib only) downloads non-vendored graphs from sparse.tamu.edu. The e2e harness gains a `{DATA}` argv-override so tests source graphs from `data/` via the generated program's existing positional file-input ABI. Tests use the diagonal-of-G² trick (`Z[i] = G[i,k]·G[k,i]`) so output is a small, hand-verifiable length-N vector needing no seed vector.

**Tech Stack:** C (edgecc, unchanged), Python 3 (generated code, fetch script, test harness), scipy (load/save, test-only), CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-06-08-edgecc-suitesparse-test-set-design.md`

---

## File Structure

**Create:**
- `data/suitesparse/manifest.json` — curated graph metadata + sha256
- `data/suitesparse/README.md` — provenance / attribution / fetch usage
- `data/suitesparse/karate.mtx`, `lesmis.mtx`, `GD98_a.mtx` — vendored (fetched then committed)
- `tools/fetch_suitesparse.py` — stdlib downloader/extractor/verifier
- `tests/tools/test_fetch_suitesparse.py` — offline unit test for fetch helpers
- `tests/e2e/load_karate_degree/` — `input.edge`, `args`, `expected_output.txt`, `roundtrip_check`, `needs_scipy`
- `tests/e2e/load_gd98a_2cycles/` — `input.edge`, `args`, `expected_output.txt`, `needs_scipy`
- `tests/e2e/load_lesmis_minplus/` — `input.edge`, `args`, `expected_output.txt`, `needs_scipy`

**Modify:**
- `.gitignore` — ignore fetched `.mtx`, force-include vendored three
- `tests/e2e/run_e2e.py` — `{DATA}` args-file argv override
- `CMakeLists.txt` — register `fetch_helper_tests`
- `CLAUDE.md` — document `data/` + fetch script

---

## Task 1: Manifest with metadata (sha256 placeholders)

**Files:**
- Create: `data/suitesparse/manifest.json`

- [ ] **Step 1: Create the manifest with metadata; sha256 null for now**

Create `data/suitesparse/manifest.json`. The `sha256` fields are `null` and will be populated by the fetch script in Task 3.

```json
{
  "karate":         { "group": "Newman", "name": "karate",         "n": 34,  "vendored": true,  "sha256": null, "kind": "undirected-pattern", "semiring": "boolean",          "source": "Zachary 1977 karate club",            "url": "https://sparse.tamu.edu/Newman/karate" },
  "lesmis":         { "group": "Newman", "name": "lesmis",         "n": 77,  "vendored": true,  "sha256": null, "kind": "undirected-weighted", "semiring": "weighted",         "source": "Knuth, Les Miserables co-appearance", "url": "https://sparse.tamu.edu/Newman/lesmis" },
  "GD98_a":         { "group": "Pajek",  "name": "GD98_a",         "n": 38,  "vendored": true,  "sha256": null, "kind": "directed",            "semiring": "directed",         "source": "Graph Drawing 1998 contest",          "url": "https://sparse.tamu.edu/Pajek/GD98_a" },
  "dolphins":       { "group": "Newman", "name": "dolphins",       "n": 62,  "vendored": false, "sha256": null, "kind": "undirected-pattern",  "semiring": "boolean",          "source": "Lusseau et al. dolphin social network","url": "https://sparse.tamu.edu/Newman/dolphins" },
  "polbooks":       { "group": "Newman", "name": "polbooks",       "n": 105, "vendored": false, "sha256": null, "kind": "undirected-pattern",  "semiring": "boolean",          "source": "Krebs political books",               "url": "https://sparse.tamu.edu/Newman/polbooks" },
  "football":       { "group": "Newman", "name": "football",       "n": 115, "vendored": false, "sha256": null, "kind": "undirected-pattern",  "semiring": "boolean",          "source": "Girvan-Newman college football",      "url": "https://sparse.tamu.edu/Newman/football" },
  "Ragusa16":       { "group": "Pajek",  "name": "Ragusa16",       "n": 24,  "vendored": false, "sha256": null, "kind": "directed-weighted",   "semiring": "weighted",         "source": "Pajek Ragusa16",                      "url": "https://sparse.tamu.edu/Pajek/Ragusa16" },
  "Ragusa18":       { "group": "Pajek",  "name": "Ragusa18",       "n": 23,  "vendored": false, "sha256": null, "kind": "directed-weighted",   "semiring": "weighted",         "source": "Pajek Ragusa18",                      "url": "https://sparse.tamu.edu/Pajek/Ragusa18" },
  "GD96_b":         { "group": "Pajek",  "name": "GD96_b",         "n": 111, "vendored": false, "sha256": null, "kind": "directed",            "semiring": "directed",         "source": "Graph Drawing 1996 contest",          "url": "https://sparse.tamu.edu/Pajek/GD96_b" },
  "celegansneural": { "group": "Newman", "name": "celegansneural", "n": 297, "vendored": false, "sha256": null, "kind": "directed-weighted",   "semiring": "weighted-directed","source": "C. elegans neural network",           "url": "https://sparse.tamu.edu/Newman/celegansneural" }
}
```

- [ ] **Step 2: Commit**

```bash
git add data/suitesparse/manifest.json
git commit -m "data: SuiteSparse manifest (metadata; sha256 to be filled)"
```

---

## Task 2: Fetch script + offline helper tests

**Files:**
- Create: `tools/fetch_suitesparse.py`
- Create: `tests/tools/test_fetch_suitesparse.py`
- Modify: `CMakeLists.txt` (after the `viz_tests` block, around line 118)

- [ ] **Step 1: Write the failing test for the pure helpers**

Create `tests/tools/test_fetch_suitesparse.py`. It builds a tar.gz in memory (no network) and checks `extract_mtx` + `sha256_bytes`, mirroring the project's `check()`/`results` test style.

```python
#!/usr/bin/env python3
"""Offline unit tests for tools/fetch_suitesparse.py helpers (no network)."""
import io
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, TOOLS)

import fetch_suitesparse as fs

results = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'} {name}")
    results.append(bool(cond))

def _make_tarball(member_name, payload):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()

def test_extract_mtx_nested():
    payload = b"%%MatrixMarket matrix coordinate pattern symmetric\n2 2 1\n1 2\n"
    blob = _make_tarball("karate/karate.mtx", payload)
    got = fs.extract_mtx(blob, "karate")
    check("extract_mtx finds nested member", got == payload)

def test_extract_mtx_missing():
    blob = _make_tarball("karate/karate_other.txt", b"x")
    try:
        fs.extract_mtx(blob, "karate")
        check("extract_mtx raises on missing member", False)
    except ValueError:
        check("extract_mtx raises on missing member", True)

def test_sha256_bytes():
    # sha256("") is the well-known empty-string digest
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    check("sha256_bytes empty", fs.sha256_bytes(b"") == empty)

test_extract_mtx_nested()
test_extract_mtx_missing()
test_sha256_bytes()

print(f"\n{sum(results)}/{len(results)} fetch-helper tests passed")
sys.exit(0 if all(results) else 1)
```

- [ ] **Step 2: Run the test to verify it fails (module missing)**

Run: `python3 tests/tools/test_fetch_suitesparse.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'fetch_suitesparse'`

- [ ] **Step 3: Write the fetch script**

Create `tools/fetch_suitesparse.py`:

```python
#!/usr/bin/env python3
"""Fetch curated SuiteSparse .mtx graphs for edgecc's test set.

Reads data/suitesparse/manifest.json, downloads each requested graph's Matrix
Market tarball from sparse.tamu.edu, extracts the inner <name>.mtx, verifies
its sha256 against the manifest, and writes data/suitesparse/<name>.mtx.
Stdlib only (no new runtime dependency).

Usage:
  fetch_suitesparse.py <name> [<name> ...]   # fetch specific graphs
  fetch_suitesparse.py --all                 # fetch every graph in the manifest
  fetch_suitesparse.py --all --update-hashes # (re)compute sha256 into manifest
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "suitesparse")
MANIFEST = os.path.join(DATA_DIR, "manifest.json")
BASE_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"


def load_manifest(path=MANIFEST):
    with open(path) as f:
        return json.load(f)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def extract_mtx(tar_bytes, name):
    """Return the bytes of <name>.mtx inside a SuiteSparse tar.gz blob."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("%s.mtx" % name):
                f = tf.extractfile(m)
                if f is not None:
                    return f.read()
        raise ValueError("no %s.mtx found in tarball" % name)


def download(group, name):
    url = BASE_URL.format(group=group, name=name)
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def fetch_one(entry, update_hashes=False, force=False):
    name = entry["name"]
    dest = os.path.join(DATA_DIR, "%s.mtx" % name)

    if os.path.exists(dest) and not force and not update_hashes:
        with open(dest, "rb") as f:
            have = sha256_bytes(f.read())
        want = entry.get("sha256")
        if want and have != want:
            raise SystemExit("checksum mismatch for %s: have %s want %s" % (name, have, want))
        print("ok (cached): %s" % name)
        return

    blob = download(entry["group"], name)
    mtx = extract_mtx(blob, name)
    digest = sha256_bytes(mtx)

    if update_hashes:
        entry["sha256"] = digest
        print("hash %s: %s" % (name, digest))
    else:
        want = entry.get("sha256")
        if not want:
            raise SystemExit("manifest has no sha256 for %s; run with --update-hashes first" % name)
        if digest != want:
            raise SystemExit("checksum mismatch for %s: got %s want %s" % (name, digest, want))

    with open(dest, "wb") as f:
        f.write(mtx)
    print("wrote %s (%d bytes)" % (dest, len(mtx)))


def main():
    ap = argparse.ArgumentParser(description="Fetch SuiteSparse graphs for edgecc tests.")
    ap.add_argument("names", nargs="*", help="manifest keys to fetch")
    ap.add_argument("--all", action="store_true", help="fetch every graph in the manifest")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--update-hashes", action="store_true",
                    help="download and write computed sha256 back into the manifest")
    args = ap.parse_args()

    manifest = load_manifest()
    if args.all or args.update_hashes:
        keys = list(manifest)
    elif args.names:
        keys = args.names
    else:
        ap.error("specify graph names, or --all")

    os.makedirs(DATA_DIR, exist_ok=True)
    for k in keys:
        if k not in manifest:
            raise SystemExit("unknown graph %r (not in manifest)" % k)
        fetch_one(manifest[k], update_hashes=args.update_hashes, force=args.force)

    if args.update_hashes:
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print("manifest updated")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 tests/tools/test_fetch_suitesparse.py`
Expected: `3/3 fetch-helper tests passed`

- [ ] **Step 5: Register the test in ctest**

In `CMakeLists.txt`, immediately after the `viz_tests` `add_test(...)` block (ends ~line 118), add:

```cmake
add_test(
    NAME fetch_helper_tests
    COMMAND ${CMAKE_COMMAND} -E env python3
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_fetch_suitesparse.py
)
```

- [ ] **Step 6: Build and run via ctest**

Run: `cd build && cmake .. >/dev/null && ctest -R fetch_helper_tests --output-on-failure`
Expected: `fetch_helper_tests ... Passed`

- [ ] **Step 7: Commit**

```bash
git add tools/fetch_suitesparse.py tests/tools/test_fetch_suitesparse.py CMakeLists.txt
git commit -m "tools: stdlib SuiteSparse fetch script + offline helper tests"
```

---

## Task 3: Fetch graphs, fill checksums, vendor the three

**Files:**
- Modify: `data/suitesparse/manifest.json` (sha256 populated)
- Create: `data/suitesparse/karate.mtx`, `lesmis.mtx`, `GD98_a.mtx`
- Modify: `.gitignore`

**Note:** This task needs network access to sparse.tamu.edu. If unavailable, stop and flag it — the rest of the plan depends on having the real files.

- [ ] **Step 1: Fetch all graphs and write computed checksums into the manifest**

Run: `python3 tools/fetch_suitesparse.py --all --update-hashes`
Expected: prints `wrote .../karate.mtx (...)` etc. for all nine graphs and `manifest updated`. Every `sha256` in `data/suitesparse/manifest.json` is now a 64-hex-char string.

- [ ] **Step 2: Sanity-check the vendored three loaded correctly**

Run:
```bash
python3 -c "from scipy.io import mmread; m=mmread('data/suitesparse/karate.mtx'); a=m.toarray(); print('karate', a.shape, int(a.sum()))"
```
Expected: `karate (34, 34) 156` (Zachary's club has 78 undirected edges → 156 nonzeros when expanded; confirms symmetric expansion).

- [ ] **Step 3: Add .gitignore rules — ignore fetched, keep vendored three**

Append to `.gitignore` (create the file if absent):

```gitignore
# SuiteSparse: ignore fetched graphs, but vendor the curated three
data/suitesparse/*.mtx
!data/suitesparse/karate.mtx
!data/suitesparse/lesmis.mtx
!data/suitesparse/GD98_a.mtx
```

- [ ] **Step 4: Verify only the three are tracked**

Run: `git add -A data/suitesparse .gitignore && git status --short data/suitesparse`
Expected: shows `karate.mtx`, `lesmis.mtx`, `GD98_a.mtx`, `manifest.json` staged — NOT dolphins/polbooks/football/etc.

- [ ] **Step 5: Commit**

```bash
git commit -m "data: vendor karate/lesmis/GD98_a + fill manifest checksums; gitignore fetched"
```

---

## Task 4: Provenance README

**Files:**
- Create: `data/suitesparse/README.md`

- [ ] **Step 1: Write the provenance/attribution README**

Create `data/suitesparse/README.md`:

```markdown
# SuiteSparse test graphs

Real graphs from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu)
(Davis & Hu), used to exercise edgecc's `load`/`save` and dense-compute path on
non-trivial input. Each ships as Matrix Market (`.mtx`) and drops straight into
`load(...)`.

## Vendored (committed) — the curated three

| File | SuiteSparse | N | Semiring use | Original source |
|------|-------------|---|--------------|-----------------|
| `karate.mtx` | Newman/karate | 34 | boolean / count | Zachary 1977 karate club |
| `lesmis.mtx` | Newman/lesmis | 77 | weighted | Knuth, *Les Miserables* co-appearance |
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
```

- [ ] **Step 2: Commit**

```bash
git add data/suitesparse/README.md
git commit -m "docs: provenance README for SuiteSparse test graphs"
```

---

## Task 5: Extend the e2e harness with `{DATA}` argv override

**Files:**
- Modify: `tests/e2e/run_e2e.py:39-45`

- [ ] **Step 1: Add data-dir resolution + args-file handling**

In `tests/e2e/run_e2e.py`, replace the generated-program invocation block (currently lines 39–45):

```python
        result = subprocess.run(
            [sys.executable, tmp_py],
            capture_output=True, text=True, timeout=10, cwd=test_dir
        )
        if result.returncode != 0:
            print(f"  FAIL {name}: python failed: {result.stderr.strip()}")
            return False
```

with:

```python
        # Optional argv override: an `args` file supplies positional input paths
        # to the generated program. The token {DATA} resolves to the repo-root
        # data/ directory, so tests can source graphs from data/suitesparse/.
        extra_args = []
        args_file = os.path.join(test_dir, "args")
        if os.path.exists(args_file):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            data_dir = os.path.join(repo_root, "data")
            with open(args_file) as f:
                for tok in f.read().split():
                    if "{DATA}" in tok and not os.path.isdir(data_dir):
                        print(f"  FAIL {name}: cannot resolve {{DATA}} (no {data_dir})")
                        return False
                    extra_args.append(tok.replace("{DATA}", data_dir))

        result = subprocess.run(
            [sys.executable, tmp_py, *extra_args],
            capture_output=True, text=True, timeout=20, cwd=test_dir
        )
        if result.returncode != 0:
            print(f"  FAIL {name}: python failed: {result.stderr.strip()}")
            return False
```

(`os` is already imported at the top of the file. Timeout bumped to 20s for the larger graphs.)

- [ ] **Step 2: Verify existing e2e tests still pass (no `args` files yet)**

Run: `cd build && ctest -R "^e2e_tests$" --output-on-failure`
Expected: all existing e2e tests PASS (the new branch is inert without an `args` file).

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/run_e2e.py
git commit -m "test(e2e): {DATA} argv-override so tests can source data/ graphs"
```

---

## Task 6: karate degree e2e test (with save + roundtrip)

**Files:**
- Create: `tests/e2e/load_karate_degree/input.edge`
- Create: `tests/e2e/load_karate_degree/args`
- Create: `tests/e2e/load_karate_degree/expected_output.txt`
- Create: `tests/e2e/load_karate_degree/roundtrip_check`
- Create: `tests/e2e/load_karate_degree/needs_scipy`

- [ ] **Step 1: Create the kernel, args, and needs_scipy marker**

`tests/e2e/load_karate_degree/input.edge`:

```
// Diagonal of G^2 over the karate graph: Z[i] = sum_k G[i,k]*G[k,i].
// For a simple undirected graph this equals the degree of node i.
// Validates load() on a real graph + dense compute + save. (No seed vector.)
dim N = *;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor Z[R=N]      : float, empty = 0.0;
init G = load("graph.mtx");
Z[i] = G[i,k] . G[k,i] :: map(mul, intersect) reduce(add, union);
save: Z -> "result.mtx";
```

`tests/e2e/load_karate_degree/args` (single line):

```
{DATA}/suitesparse/karate.mtx
```

`tests/e2e/load_karate_degree/needs_scipy` — empty file:

```bash
touch tests/e2e/load_karate_degree/needs_scipy
```

- [ ] **Step 2: Generate expected_output.txt by running the kernel**

Run (writes Python list repr to the expected file):
```bash
cd build && make >/dev/null
./build/edgecc tests/e2e/load_karate_degree/input.edge -o /tmp/karate.py
python3 /tmp/karate.py data/suitesparse/karate.mtx > tests/e2e/load_karate_degree/expected_output.txt
cat tests/e2e/load_karate_degree/expected_output.txt
```
Expected: a 34-element list of floats, e.g. starting `[16.0, 9.0, 10.0, ...]`.

- [ ] **Step 3: Independently verify the values against the known degree oracle**

The diagonal of G² for a simple undirected graph IS the degree sequence. Verify with an independent scipy computation (NOT via edgecc):

```bash
python3 - <<'PY'
from scipy.io import mmread
import ast
m = mmread("data/suitesparse/karate.mtx")
a = (m.toarray() if hasattr(m, "toarray") else m)
deg = [float((a[i] != 0).sum()) for i in range(a.shape[0])]
got = ast.literal_eval(open("tests/e2e/load_karate_degree/expected_output.txt").read())
assert got == deg, ("MISMATCH", list(zip(got, deg)))
# Spot-check well-known karate degrees: node 0 -> 16, node 33 -> 17
assert deg[0] == 16.0 and deg[33] == 17.0, deg
print("oracle OK; degrees match (node0=16, node33=17)")
PY
```
Expected: `oracle OK; degrees match (node0=16, node33=17)`. If this MISMATCHES, do NOT overwrite — investigate the loader/codegen before proceeding.

- [ ] **Step 4: Create roundtrip_check (saved Z re-read as a column)**

`save` writes the length-34 vector as a 34×1 column; `roundtrip_check` holds the same 34 values whitespace-separated. Generate it from the verified expected:

```bash
python3 - <<'PY'
import ast
vals = ast.literal_eval(open("tests/e2e/load_karate_degree/expected_output.txt").read())
open("tests/e2e/load_karate_degree/roundtrip_check", "w").write(" ".join(str(v) for v in vals) + "\n")
PY
cat tests/e2e/load_karate_degree/roundtrip_check
```
Expected: `16.0 9.0 10.0 ... 17.0` (34 values).

- [ ] **Step 5: Run the test through the harness**

Run: `cd build && ctest -R "^e2e_tests$" --output-on-failure 2>&1 | grep -E "karate|tests passed"`
Expected: `PASS load_karate_degree` (and the overall e2e count increased by 1). The harness runs the kernel with the `{DATA}` path, matches stdout to `expected_output.txt`, then re-reads the saved `result.mtx` and matches `roundtrip_check`.

- [ ] **Step 6: Commit**

```bash
git add tests/e2e/load_karate_degree
git commit -m "test(e2e): karate degree (diagonal of G^2) on real graph + save roundtrip"
```

---

## Task 7: GD98_a directed 2-cycles e2e test

**Files:**
- Create: `tests/e2e/load_gd98a_2cycles/input.edge`
- Create: `tests/e2e/load_gd98a_2cycles/args`
- Create: `tests/e2e/load_gd98a_2cycles/expected_output.txt`
- Create: `tests/e2e/load_gd98a_2cycles/needs_scipy`

- [ ] **Step 1: Create the kernel, args, and marker**

`tests/e2e/load_gd98a_2cycles/input.edge`:

```
// Diagonal of G^2 over a directed graph: Z[i] = sum_k G[i,k]*G[k,i]
// = number of 2-cycles through node i (i->k->i). Exercises directed asymmetry.
dim N = *;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor Z[R=N]      : float, empty = 0.0;
init G = load("graph.mtx");
Z[i] = G[i,k] . G[k,i] :: map(mul, intersect) reduce(add, union);
```

`tests/e2e/load_gd98a_2cycles/args`:

```
{DATA}/suitesparse/GD98_a.mtx
```

```bash
touch tests/e2e/load_gd98a_2cycles/needs_scipy
```

- [ ] **Step 2: Generate expected_output.txt**

```bash
./build/edgecc tests/e2e/load_gd98a_2cycles/input.edge -o /tmp/gd98a.py
python3 /tmp/gd98a.py data/suitesparse/GD98_a.mtx > tests/e2e/load_gd98a_2cycles/expected_output.txt
cat tests/e2e/load_gd98a_2cycles/expected_output.txt
```
Expected: a 38-element list of floats.

- [ ] **Step 3: Independently verify against a scipy oracle**

```bash
python3 - <<'PY'
from scipy.io import mmread
import ast
m = mmread("data/suitesparse/GD98_a.mtx")
a = (m.toarray() if hasattr(m, "toarray") else m)
N = a.shape[0]
# diagonal of G^2 with intersect-mul/union-add semantics (matches the kernel)
Z = []
for i in range(N):
    s = 0.0
    for k in range(N):
        if a[i][k] != 0.0 and a[k][i] != 0.0:
            s += float(a[i][k]) * float(a[k][i])
    Z.append(s)
got = ast.literal_eval(open("tests/e2e/load_gd98a_2cycles/expected_output.txt").read())
assert got == Z, ("MISMATCH", list(zip(got, Z))[:10])
print("oracle OK; GD98_a 2-cycle counts match")
PY
```
Expected: `oracle OK; GD98_a 2-cycle counts match`. On mismatch, investigate before proceeding.

- [ ] **Step 4: Run through the harness**

Run: `cd build && ctest -R "^e2e_tests$" --output-on-failure 2>&1 | grep -E "gd98a|tests passed"`
Expected: `PASS load_gd98a_2cycles`.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/load_gd98a_2cycles
git commit -m "test(e2e): GD98_a directed 2-cycles (diagonal of G^2) on real graph"
```

---

## Task 8: lesmis weighted min-plus e2e test

**Files:**
- Create: `tests/e2e/load_lesmis_minplus/input.edge`
- Create: `tests/e2e/load_lesmis_minplus/args`
- Create: `tests/e2e/load_lesmis_minplus/expected_output.txt`
- Create: `tests/e2e/load_lesmis_minplus/needs_scipy`

- [ ] **Step 1: Create the kernel, args, and marker**

`tests/e2e/load_lesmis_minplus/input.edge`:

```
// Weighted min-plus over the lesmis graph: Z[i] = min_k (W[i,k] + W[k,i]).
// Exercises the (add, min) semiring on real edge weights. NOTE: lesmis weights
// are co-appearance counts, so min-plus here is semantically arbitrary -- this
// test validates the loader + weighted dense-compute path, not graph meaning.
dim N = *;
tensor W[R=N, C=N] : float, empty = 0.0;
tensor Z[R=N]      : float, empty = 0.0;
init W = load("graph.mtx");
Z[i] = W[i,k] . W[k,i] :: map(add, intersect) reduce(min, union);
```

`tests/e2e/load_lesmis_minplus/args`:

```
{DATA}/suitesparse/lesmis.mtx
```

```bash
touch tests/e2e/load_lesmis_minplus/needs_scipy
```

- [ ] **Step 2: Generate expected_output.txt**

```bash
./build/edgecc tests/e2e/load_lesmis_minplus/input.edge -o /tmp/lesmis.py
python3 /tmp/lesmis.py data/suitesparse/lesmis.mtx > tests/e2e/load_lesmis_minplus/expected_output.txt
cat tests/e2e/load_lesmis_minplus/expected_output.txt
```
Expected: a 77-element list of floats.

- [ ] **Step 3: Independently verify against a scipy oracle**

This oracle replicates the kernel's reduce(min)/intersect semantics. Note the empty-value rule: a node with no qualifying `k` keeps the tensor empty (`0.0`).

```bash
python3 - <<'PY'
from scipy.io import mmread
import ast, math
m = mmread("data/suitesparse/lesmis.mtx")
a = (m.toarray() if hasattr(m, "toarray") else m)
N = a.shape[0]
Z = []
for i in range(N):
    acc = math.inf
    for k in range(N):
        if a[i][k] != 0.0 and a[k][i] != 0.0:        # intersect
            acc = min(acc, float(a[i][k]) + float(a[k][i]))   # map=add, reduce=min
    Z.append(acc if acc != math.inf else 0.0)        # empty if nothing contributed
got = ast.literal_eval(open("tests/e2e/load_lesmis_minplus/expected_output.txt").read())
assert got == Z, ("MISMATCH", [(j, got[j], Z[j]) for j in range(N) if got[j] != Z[j]][:10])
print("oracle OK; lesmis min-plus matches")
PY
```
Expected: `oracle OK; lesmis min-plus matches`. If this mismatches, the discrepancy is meaningful (likely a min-reduce identity/empty subtlety in codegen) — investigate, do not just overwrite the expected.

- [ ] **Step 4: Run through the harness**

Run: `cd build && ctest -R "^e2e_tests$" --output-on-failure 2>&1 | grep -E "lesmis|tests passed"`
Expected: `PASS load_lesmis_minplus`.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/load_lesmis_minplus
git commit -m "test(e2e): lesmis weighted min-plus (diagonal of W^2) on real graph"
```

---

## Task 9: Documentation

**Files:**
- Modify: `CLAUDE.md` (the "File input/output" section, after the `load_push.edge` reference ~line 158)

- [ ] **Step 1: Document the data directory and fetch script in CLAUDE.md**

In `CLAUDE.md`, at the end of the "File input/output" section (just before "### Visualization"), add:

```markdown
**Real test graphs.** `data/suitesparse/` holds a curated set of small
[SuiteSparse](https://sparse.tamu.edu) graphs. The three smallest
(`karate.mtx` 34, `lesmis.mtx` 77 weighted, `GD98_a.mtx` 38 directed) are
vendored; the rest are fetch-on-demand and gitignored. Download with
`python3 tools/fetch_suitesparse.py <name>` (or `--all`); the script (stdlib
only) reads `data/suitesparse/manifest.json`, pulls the Matrix Market tarball,
extracts the `.mtx`, and verifies its sha256. E2e tests source these via the
`{DATA}` token in a test's `args` file (resolved to repo-root `data/` and passed
as positional argv to the generated program). The current tests use the
diagonal-of-G^2 trick (`Z[i] = G[i,k].G[k,i]`) so output is a small,
hand-verifiable length-N vector and no seed vector is needed. See
`tests/e2e/load_karate_degree/`. Seed-vector loading (for BFS/SSSP on these
graphs) and a one-kernel-over-many-graphs data-sweep runner are future work.
```

- [ ] **Step 2: Run the full test suite**

Run: `cd build && cmake .. >/dev/null && make >/dev/null && ctest --output-on-failure`
Expected: all targets pass — `fetch_helper_tests`, `e2e_tests` (now including the three new tests), and all pre-existing tests.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document data/suitesparse + fetch script in CLAUDE.md"
```

---

## Self-Review Notes

- **Spec coverage:** A1 layout → Tasks 1,3,4; A2 manifest → Tasks 1,3; A3 fetch script → Task 2; A4 provenance → Task 4; argv-override path resolution → Task 5; three no-seed tests (boolean/directed/weighted) → Tasks 6,7,8; save/roundtrip on real graph → Task 6; future-work notes → Task 9 + spec. All covered.
- **Network dependency:** isolated to Task 3 (and the manual `fetch` of optional graphs). All committed tests run offline against vendored files + scipy.
- **Non-circular verification:** each test's expected output is checked against an independent scipy oracle (and karate against the externally-known degree sequence) before being committed as the regression baseline.
- **Type/name consistency:** `extract_mtx`, `sha256_bytes`, `load_manifest`, `fetch_one`, `download` used consistently across Task 2's script and test. `{DATA}` token and `args` filename consistent across Tasks 5–8.
