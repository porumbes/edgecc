#!/usr/bin/env python3
"""edgecc visualization: render a per-generation trace to PNG frames.

Usage:
    python tools/edgecc_viz.py TRACE.json --graph ADJ.mtx -o FRAMES_DIR \
        [--color TENSOR] [--layout spring|circular|shell|multipartite|kamada_kawai|spectral] [--theme dark]

matplotlib / networkx are imported lazily inside the render functions, so the
pure helpers below (load_trace, validate_trace, active_edges) are importable and
testable without those packages installed.
"""
import argparse
import json
import math
import os

SCHEMA = "edgecc-trace/1"

# --- Theme palette objects (only "dark" implemented; structured for more) ---
THEMES = {
    "dark": {
        "bg": "#000000",
        "node_cmap": "bone",        # cream -> dark ramp
        "edge": "#444444",
        "active_edge": "#22d3ee",   # cyan
        "panel_cmap": "bone",
        "text": "#cccccc",
    },
}


def validate_trace(tr):
    if tr.get("schema") != SCHEMA:
        raise ValueError(f"edgecc_viz: unsupported trace schema {tr.get('schema')!r} "
                         f"(expected {SCHEMA!r})")
    if "steps" not in tr or not isinstance(tr["steps"], list):
        raise ValueError("edgecc_viz: trace has no 'steps' list")
    return tr


def load_trace(path):
    with open(path) as f:
        tr = json.load(f)
    validate_trace(tr)
    return tr


def load_graph(path):
    """Return a dense adjacency matrix (list of lists) from a Matrix Market file."""
    try:
        from scipy.io import mmread
    except ImportError:
        raise ImportError("edgecc_viz requires scipy: pip install scipy")
    m = mmread(path)
    arr = m.toarray() if hasattr(m, "toarray") else m
    return arr.tolist()


def active_edges(adj, prev_tensors, curr_tensors, color_tensor, empty=0.0):
    """Edges (s, d) where s held a non-empty value last generation and d's value
    changed this generation. `empty` is the tensor's empty/identity value so this
    is correct across semirings (e.g. min-plus, where the source distance is 0.0
    but `empty` is +inf)."""
    prev = prev_tensors.get(color_tensor)
    curr = curr_tensors.get(color_tensor)
    n = len(adj)
    edges = set()
    if prev is None or curr is None:
        return edges
    # Node-indexed (1-D) tensors only; a 2-D tensor has no per-node activation.
    if (prev and isinstance(prev[0], list)) or (curr and isinstance(curr[0], list)):
        return edges
    for s in range(n):
        if prev[s] == empty:
            continue
        for d in range(n):
            if adj[s][d] != 0.0 and curr[d] != prev[d]:
                edges.add((s, d))
    return edges


def _node_colors(vals, empty):
    """Map a 1-D tensor to node brightness in [0, 1]: empty/non-finite -> 0 (dim);
    non-empty -> normalized by the finite non-empty range (larger = brighter;
    a single distinct value, e.g. boolean, -> full bright)."""
    reached = [v for v in vals if v != empty and math.isfinite(v)]
    lo = min(reached) if reached else 0.0
    hi = max(reached) if reached else 1.0
    span = (hi - lo) or 1.0
    out = []
    for v in vals:
        if v == empty or not math.isfinite(v):
            out.append(0.0)
        elif hi == lo:
            out.append(1.0)
        else:
            out.append(0.4 + 0.6 * (v - lo) / span)
    return out


def _finite_range(vals):
    """(vmin, vmax) over finite entries of a 1-D or 2-D tensor; defaults 0..1."""
    flat = []
    for row in (vals if (vals and isinstance(vals[0], list)) else [vals]):
        for x in row:
            if math.isfinite(x):
                flat.append(x)
    lo = min(flat) if flat else 0.0
    hi = max(flat) if flat else 1.0
    if hi == lo:
        hi = lo + 1.0
    return lo, hi


def _first_reached(steps, color, empty, n):
    """Per-node first-reached generation index, or None if never reached: the
    generation at which `color`'s value first left `empty`, walking `steps` in
    order. Pure (no networkx/matplotlib)."""
    first = [None] * n
    for gi, step in enumerate(steps):
        vals = step.get("tensors", {}).get(color)
        if not vals or isinstance(vals[0], list):   # tensor absent this gen, or 2-D
            continue
        for node in range(min(n, len(vals))):
            if first[node] is None and vals[node] != empty:
                first[node] = gi
    return first


def shell_nlist(steps, color, empty, n):
    """Ring assignment for the shell layout: shell k = node indices whose `color`
    tensor first became non-empty at generation index k; a trailing ring holds
    nodes never reached. Pure (no networkx/matplotlib)."""
    first = _first_reached(steps, color, empty, n)
    buckets, unreached = {}, []
    for node in range(n):
        if first[node] is None:
            unreached.append(node)
        else:
            buckets.setdefault(first[node], []).append(node)
    shells = [buckets[g] for g in sorted(buckets)]
    if unreached:
        shells.append(unreached)
    return shells


def _layout(adj, kind, seed=1, nlist=None, first=None):
    import sys
    import networkx as nx
    n = len(adj)
    directed = any(adj[i][j] != adj[j][i] for i in range(n) for j in range(n))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i][j] != 0.0:
                G.add_edge(i, j)
    if kind == "spring":
        return G, nx.spring_layout(G, seed=seed)
    try:
        if kind == "circular":
            pos = nx.circular_layout(G)
        elif kind == "shell":
            pos = nx.shell_layout(G, nlist=nlist)
        elif kind == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif kind == "spectral":
            pos = nx.spectral_layout(G)
        elif kind == "multipartite":
            reached = [g for g in (first or []) if g is not None]
            outer = (max(reached) + 1) if reached else 0
            for node in range(n):
                fr = first[node] if first else None
                G.nodes[node]["layer"] = fr if fr is not None else outer
            pos = nx.multipartite_layout(G, subset_key="layer")
        else:
            pos = nx.circular_layout(G)
        return G, pos
    except Exception as e:                       # disconnected graph, etc.
        print(f"edgecc_viz: {kind} layout failed ({e.__class__.__name__}: {e}), "
              f"falling back to spring", file=sys.stderr)
        return G, nx.spring_layout(G, seed=seed)


def render_all(tr, adj, color, out_dir, theme, layout_kind, empty=0.0):
    import matplotlib
    matplotlib.use("Agg")            # headless
    import matplotlib.pyplot as plt
    import networkx as nx

    os.makedirs(out_dir, exist_ok=True)
    nlist = (shell_nlist(tr["steps"], color, empty, len(adj))
             if layout_kind == "shell" else None)
    first = (_first_reached(tr["steps"], color, empty, len(adj))
             if layout_kind == "multipartite" else None)
    G, pos = _layout(adj, layout_kind, nlist=nlist, first=first)
    steps = tr["steps"]
    gen_tensors = tr.get("gen_tensors", [])
    empties = tr.get("empty", {})

    for idx, step in enumerate(steps):
        prev = steps[idx - 1]["tensors"] if idx > 0 else step["tensors"]
        cur = step["tensors"]
        act = active_edges(adj, prev, cur, color, empty)

        fig = plt.figure(figsize=(10, 5), facecolor=theme["bg"])
        # Left: graph
        axg = fig.add_axes([0.02, 0.05, 0.62, 0.9])
        axg.set_facecolor(theme["bg"])
        axg.axis("off")
        node_vals = cur.get(color, [empty] * len(adj))
        nx.draw_networkx_edges(G, pos, ax=axg, edge_color=theme["edge"],
                               width=0.6, alpha=0.5)
        if act:
            # cyan glow: a few strokes of increasing width / decreasing alpha
            for w, a in [(5.0, 0.15), (3.0, 0.3), (1.6, 0.9)]:
                nx.draw_networkx_edges(G, pos, ax=axg, edgelist=sorted(act),
                                       edge_color=theme["active_edge"], width=w, alpha=a)
        nx.draw_networkx_nodes(G, pos, ax=axg, node_color=_node_colors(node_vals, empty),
                               cmap=plt.get_cmap(theme["node_cmap"]),
                               vmin=0.0, vmax=1.0,
                               edgecolors=theme["edge"], node_size=300)
        axg.set_title(f"gen {step['gen']}  (color: {color})", color=theme["text"])

        # Right: tensor panels, stacked. Skip tensors absent this generation
        # (e.g. intermediates do not exist at the gen-0 seed).
        k = max(1, len(gen_tensors))
        for ti, tname in enumerate(gen_tensors):
            vals = cur.get(tname)
            if not vals:
                continue
            axp = fig.add_axes([0.68, 0.05 + ti * (0.9 / k), 0.3, (0.9 / k) * 0.8])
            row = vals if isinstance(vals[0], list) else [vals]
            pmin, pmax = _finite_range(vals)
            # Replace non-finite cells with the panel min so imshow can render them.
            disp = [[(x if math.isfinite(x) else pmin) for x in r] for r in row]
            axp.imshow(disp, cmap=plt.get_cmap(theme["panel_cmap"]),
                       vmin=pmin, vmax=pmax, aspect="auto")
            axp.set_yticks([])
            axp.set_xticks([])
            axp.set_ylabel(tname, color=theme["text"], rotation=0, labelpad=12)
            for sp in axp.spines.values():
                sp.set_color(theme["text"])

        fig.savefig(os.path.join(out_dir, f"gen_{idx:03d}.png"),
                    facecolor=theme["bg"], dpi=110)
        plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render an edgecc trace to PNG frames.")
    p.add_argument("trace")
    p.add_argument("--graph", required=True)
    p.add_argument("-o", "--out", default="frames")
    p.add_argument("--color", default=None, help="tensor to color nodes by (default: watch)")
    p.add_argument("--layout", default="spring",
                   choices=["spring", "circular", "shell",
                            "multipartite", "kamada_kawai", "spectral"])
    p.add_argument("--theme", default="dark")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.theme not in THEMES:
        raise ValueError(f"edgecc_viz: unknown theme {args.theme!r} "
                         f"(available: {', '.join(sorted(THEMES))})")
    tr = load_trace(args.trace)
    adj = load_graph(args.graph)
    n = len(adj)
    if n != tr.get("dims", {}).get("N", n):
        raise ValueError(f"edgecc_viz: graph has {n} nodes but trace dims.N="
                         f"{tr['dims']['N']}")
    color = args.color or tr.get("watch")
    if color not in tr.get("gen_tensors", []):
        raise ValueError(f"edgecc_viz: --color {color!r} not in gen_tensors "
                         f"{tr.get('gen_tensors')}")
    # Node coloring needs a 1-D (per-node) tensor; a 2-D watched tensor cannot
    # color nodes. Find a step where the color tensor is present and check shape.
    sample = next((s["tensors"][color] for s in tr["steps"]
                   if color in s["tensors"] and s["tensors"][color]), None)
    if sample is not None and isinstance(sample[0], list):
        raise ValueError(
            f"edgecc_viz: --color {color!r} is 2-D; node coloring needs a 1-D "
            f"(per-node) tensor. Pick a 1-D tensor via --color.")
    empty = tr.get("empty", {}).get(color, 0.0)
    render_all(tr, adj, color, args.out, THEMES[args.theme], args.layout, empty)


if __name__ == "__main__":
    main()
