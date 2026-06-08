#!/usr/bin/env python3
"""edgecc visualization: render a per-generation trace to PNG frames.

Usage:
    python tools/edgecc_viz.py TRACE.json --graph ADJ.mtx -o FRAMES_DIR \
        [--color TENSOR] [--layout spring] [--theme dark]

matplotlib / networkx are imported lazily inside the render functions, so the
pure helpers below (load_trace, validate_trace, active_edges) are importable and
testable without those packages installed.
"""
import argparse
import json
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


def _is_active(v):
    return v != 0.0


def active_edges(adj, prev_tensors, curr_tensors, color_tensor):
    """Edges (s, d) where s was active last generation and d's value changed."""
    prev = prev_tensors.get(color_tensor)
    curr = curr_tensors.get(color_tensor)
    n = len(adj)
    edges = set()
    if prev is None or curr is None:
        return edges
    for s in range(n):
        if not _is_active(prev[s]):
            continue
        for d in range(n):
            if adj[s][d] != 0.0 and curr[d] != prev[d]:
                edges.add((s, d))
    return edges


def _layout(adj, kind, seed=1):
    import networkx as nx
    n = len(adj)
    directed = any(adj[i][j] != adj[j][i] for i in range(n) for j in range(n))
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i][j] != 0.0:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, seed=seed) if kind == "spring" else nx.circular_layout(G)
    return G, pos


def render_all(tr, adj, color, out_dir, theme, layout_kind):
    import matplotlib
    matplotlib.use("Agg")            # headless
    import matplotlib.pyplot as plt
    import networkx as nx

    os.makedirs(out_dir, exist_ok=True)
    G, pos = _layout(adj, layout_kind)
    steps = tr["steps"]
    gen_tensors = tr.get("gen_tensors", [])

    for idx, step in enumerate(steps):
        prev = steps[idx - 1]["tensors"] if idx > 0 else step["tensors"]
        cur = step["tensors"]
        act = active_edges(adj, prev, cur, color)

        fig = plt.figure(figsize=(10, 5), facecolor=theme["bg"])
        # Left: graph
        axg = fig.add_axes([0.02, 0.05, 0.62, 0.9])
        axg.set_facecolor(theme["bg"])
        axg.axis("off")
        node_vals = cur.get(color, [0.0] * len(adj))
        nx.draw_networkx_edges(G, pos, ax=axg, edge_color=theme["edge"],
                               width=0.6, alpha=0.5)
        if act:
            # cyan glow: a few strokes of increasing width / decreasing alpha
            for w, a in [(5.0, 0.15), (3.0, 0.3), (1.6, 0.9)]:
                nx.draw_networkx_edges(G, pos, ax=axg, edgelist=sorted(act),
                                       edge_color=theme["active_edge"], width=w, alpha=a)
        vmax = max(1.0, max(node_vals) if node_vals else 1.0)
        nx.draw_networkx_nodes(G, pos, ax=axg, node_color=node_vals,
                               cmap=plt.get_cmap(theme["node_cmap"]),
                               vmin=0.0, vmax=vmax,
                               edgecolors=theme["edge"], node_size=300)
        axg.set_title(f"gen {step['gen']}  (color: {color})", color=theme["text"])

        # Right: tensor panels, stacked
        k = max(1, len(gen_tensors))
        for ti, tname in enumerate(gen_tensors):
            axp = fig.add_axes([0.68, 0.05 + ti * (0.9 / k), 0.3, (0.9 / k) * 0.8])
            vals = cur.get(tname, [])
            row = [vals] if (vals and not isinstance(vals[0], list)) else vals
            axp.imshow(row, cmap=plt.get_cmap(theme["panel_cmap"]),
                       vmin=0.0, vmax=1.0, aspect="auto")
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
    p.add_argument("--layout", default="spring")
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
    render_all(tr, adj, color, args.out, THEMES[args.theme], args.layout)


if __name__ == "__main__":
    main()
