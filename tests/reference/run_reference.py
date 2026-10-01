#!/usr/bin/env python3
"""Independent-reference tests for edgecc.

Each kernel in kernels/ is compiled once, imported, and its run() called on
the vendored SuiteSparse graphs plus seeded random directed weighted graphs.
Results (final state or every generation) are compared against independent
implementations (scipy.sparse.csgraph / numpy) -- never against another edgecc form.

Usage: run_reference.py <edgecc> [--random N] [--seed S]
"""
import argparse
import heapq
import importlib.util
import math
import os
import random
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import scipy.io
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order, dijkstra, shortest_path
except ImportError:
    print("SKIP reference_tests: numpy/scipy not installed")
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VENDORED = ["karate", "lesmis", "GD98_a"]
INF = math.inf


# ---- inputs -----------------------------------------------------------------

def seed(n, value, empty):
    return [value if i == 0 else empty for i in range(n)]


def pattern(G):
    return [[1.0 if x != 0.0 else 0.0 for x in row] for row in G]


def csr(G):
    return sp.csr_matrix(np.array(G, dtype=float))


# ---- independent references ---------------------------------------------------

def ref_sssp(G):
    return list(dijkstra(csr(G), directed=True, indices=0))


def ref_widest(G, source_cap=1e9):
    """Max-bottleneck path values from node 0 (modified Dijkstra); 0.0 = unreachable."""
    n = len(G)
    best = [0.0] * n
    best[0] = source_cap
    heap = [(-source_cap, 0)]
    done = [False] * n
    while heap:
        negw, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for v in range(n):
            if G[u][v] != 0.0:
                w = min(-negw, G[u][v])
                if w > best[v]:
                    best[v] = w
                    heapq.heappush(heap, (-w, v))
    return best


def ref_diag_g2(G):
    A = np.array(G, dtype=float)
    return list(np.diag(A @ A))


def ref_closure(G):
    """T+[a][c] = 1 iff a path of length >= 1 leads from a to c."""
    n = len(G)
    D = shortest_path(csr(G), directed=True, unweighted=True)
    return [[1.0 if any(G[a][b] != 0.0 and D[b][c] < INF for b in range(n)) else 0.0
             for c in range(n)] for a in range(n)]


def hop_dist(G):
    return list(shortest_path(csr(G), directed=True, unweighted=True, indices=0))


def eccentricity(dist):
    return int(max(d for d in dist if d < INF))


def ref_reach_history(G):
    """P[g] support = vertices within hop distance g. A stable-stop BFS keeps
    the first repeated generation, so it has eccentricity + 2 generations."""
    dist = hop_dist(G)
    return [[1.0 if d <= g else 0.0 for d in dist] for g in range(eccentricity(dist) + 2)]


def ref_depth_history(G):
    """D[g][v] = hop distance if it is <= g, else inf (eccentricity + 2 generations)."""
    dist = hop_dist(G)
    return [[d if d <= g else INF for d in dist] for g in range(eccentricity(dist) + 2)]


def ref_frontier_history(G):
    """Paper BFS: F[g][v] = g for vertices at hop distance exactly g; the
    occupancy stop keeps the first empty generation (eccentricity + 2 in all)."""
    dist = hop_dist(G)
    return [[float(g) if d == g else INF for d in dist] for g in range(eccentricity(dist) + 2)]


# ---- comparisons --------------------------------------------------------------

def same_support(out, exp):
    return [x != 0.0 for x in out] == [x != 0.0 for x in exp]


def same_support_2d(out, exp):
    return all(same_support(o, e) for o, e in zip(out, exp)) and len(out) == len(exp)


def close(out, exp):
    return len(out) == len(exp) and all(
        (a == b) or (math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))
        for a, b in zip(out, exp))


def history(cmp):
    """Compare every generation (and the number of generations)."""
    def check(out, exp):
        return len(out) == len(exp) and all(cmp(o, e) for o, e in zip(out, exp))
    return check


# ---- cases --------------------------------------------------------------------
# graph: "weighted" uses the graph as loaded/generated; "pattern" its 0/1 support.
# inputs(G) -> {name: value} for every run() parameter.

def bfs_inputs(G):
    n = len(G)
    return {"G": G, "F": seed(n, 1.0, 0.0), "P": seed(n, 1.0, 0.0)}


def sssp_inputs(G):
    return {"W": G, "D": seed(len(G), 0.0, INF)}


def wave_inputs(G):
    n = len(G)
    return {"G": G, "F": seed(n, 0.0, INF), "D": seed(n, 0.0, INF)}


def widest_inputs(G):
    n = len(G)
    return {"G": G, "F": seed(n, 1e9, 0.0), "D": seed(n, 1e9, 0.0)}


def paper_bfs_inputs(G):
    n = len(G)
    return {"G": G, "F": seed(n, 0.0, INF), "P": seed(n, 1.0, 0.0)}


CASES = [
    dict(kernel="bfs_push_fused", graph="weighted", inputs=bfs_inputs, ref=ref_reach_history, cmp=history(same_support)),
    dict(kernel="bfs_pull_fused", graph="weighted", inputs=bfs_inputs, ref=ref_reach_history, cmp=history(same_support)),
    dict(kernel="bfs_push_fused_rhs", graph="weighted", inputs=bfs_inputs, ref=ref_reach_history, cmp=history(same_support)),
    dict(kernel="bfs_neg_cascade", graph="weighted", inputs=bfs_inputs, ref=ref_reach_history, cmp=history(same_support)),
    dict(kernel="bfs_naive", graph="weighted", inputs=bfs_inputs, ref=ref_reach_history, cmp=history(same_support)),
    dict(kernel="bfs_depth", graph="pattern", inputs=wave_inputs, ref=ref_depth_history, cmp=history(close)),
    dict(gen=True, kernel="sssp_bellman_ford", graph="weighted", inputs=sssp_inputs, ref=ref_sssp, cmp=close),
    dict(gen=True, kernel="sssp_fused", graph="weighted", inputs=sssp_inputs, ref=ref_sssp, cmp=close),
    dict(gen=True, kernel="sssp_fused_rhs", graph="weighted", inputs=sssp_inputs, ref=ref_sssp, cmp=close),
    dict(gen=True, kernel="sssp_wavefront", graph="weighted", inputs=wave_inputs, ref=ref_sssp, cmp=close),
    dict(gen=True, kernel="widest_path", graph="weighted", inputs=widest_inputs, ref=ref_widest, cmp=close),
    dict(kernel="bfs_paper", graph="pattern", inputs=paper_bfs_inputs,
         ref=ref_frontier_history, cmp=history(close)),
    dict(kernel="diag_g2", graph="weighted", inputs=lambda G: {"G": G}, ref=ref_diag_g2, cmp=close),
    dict(gen=True, kernel="transitive_closure", graph="pattern", inputs=lambda G: {"G": G, "T": [r[:] for r in G]},
         ref=ref_closure, cmp=same_support_2d, max_n=40),
]


# ---- driver -------------------------------------------------------------------

def graphs(n_random, seed_str):
    for name in VENDORED:
        path = os.path.join(REPO, "data", "suitesparse", name + ".mtx")
        yield name, scipy.io.mmread(path).toarray().astype(float).tolist()
    rng = random.Random(seed_str)
    for t in range(n_random):
        n = rng.randint(1, 25)
        p = rng.choice([0.05, 0.1, 0.2, 0.4])
        G = [[(round(rng.uniform(0.5, 9.5), 1) if (i != j and rng.random() < p) else 0.0)
              for j in range(n)] for i in range(n)]
        yield f"random{t}(n={n},p={p})", G


def compile_kernel(edgecc, kernel, workdir):
    src = os.path.join(HERE, "kernels", kernel + ".edge")
    out = os.path.join(workdir, kernel + ".py")
    p = subprocess.run([edgecc, src, "-o", out], capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"edgecc failed on {kernel}: {p.stderr.strip()}")
    spec = importlib.util.spec_from_file_location(kernel, out)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edgecc")
    ap.add_argument("--random", type=int, default=12, help="number of random graphs")
    ap.add_argument("--seed", default="edgecc-reference")
    args = ap.parse_args()

    all_graphs = list(graphs(args.random, args.seed))
    failures, checks = [], 0
    with tempfile.TemporaryDirectory() as workdir:
        for case in CASES:
            try:
                mod = compile_kernel(args.edgecc, case["kernel"], workdir)
            except RuntimeError as e:
                failures.append((case["kernel"], "-", str(e)))
                continue
            names = mod.__edgecc_meta__["inputs"]
            for gname, G in all_graphs:
                if len(G) > case.get("max_n", 10**9):
                    continue
                if case["graph"] == "pattern":
                    G = pattern(G)
                inputs = case["inputs"](G)
                checks += 1
                try:
                    out = mod.run(*[inputs[k] for k in names])
                except Exception as e:  # a crash is a failure, not a harness error
                    failures.append((case["kernel"], gname, f"run() raised {type(e).__name__}: {e}"))
                    continue
                if case.get("gen"):
                    # gen=True: compare only the final generation; otherwise run()'s output is compared as-is (history() cmps check every generation)
                    out = out[-1]
                exp = case["ref"](G)
                try:
                    ok = case["cmp"](out, exp)
                except Exception as e:  # e.g. wrong output shape
                    ok, out = False, f"{out!r} (comparison raised {type(e).__name__}: {e})"
                if not ok:
                    failures.append((case["kernel"], gname, f"got {out}\n       expected {exp}"))

    for kernel, gname, why in failures:
        print(f"  FAIL {kernel} on {gname}: {why}")
    print(f"{checks - len(failures)}/{checks} reference checks passed "
          f"({len(CASES)} kernels, {len(all_graphs)} graphs)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
