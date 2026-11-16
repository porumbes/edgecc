#!/usr/bin/env python3
"""Render the edgecc feature-status matrix to a presentation PNG (dark theme).

A throwaway-friendly helper for slides/screenshots. The output PNG is not
committed (it is gitignored); regenerate it as needed:

    python3 tools/roadmap_png.py            # -> edgecc_status.png
    python3 tools/roadmap_png.py -o out.png

The ROWS list below mirrors the "Status at a glance" table in docs/roadmap.md —
keep the two in sync when features land. Uses font-safe marks (✓/◐/✗); matplotlib
(DejaVu Sans) does not render the ✅/🚧/❌ emoji used in the markdown.
Requires matplotlib (`pip install matplotlib`).
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# (kind, label, status)  kind: "head" | "item"   status: done | partial | planned
ROWS = [
    ("head", "Language", None),
    ("item", "Einsum expressions with Map / Reduce semirings", "done"),
    ("item", "Compute ops — add sub mul min max and or take_left", "done"),
    ("item", "Merge ops — 9 of 16", "partial"),
    ("item", "<< update op · compute labels · multi-op expressions", "planned"),
    ("item", "Anonymous tensors · unary funcs · user-defined ops", "planned"),
    ("head", "Iteration", None),
    ("item", "Generational ranks (gen, stop: stable)", "done"),
    ("item", "Cascades with ≥1 cross-generational tensor", "done"),
    ("item", "Nested cascades / multiple generational ranks", "planned"),
    ("head", "Graph algorithms", None),
    ("item", "SSSP / Bellman–Ford (min-plus)", "done"),
    ("item", "Reachability BFS", "done"),
    ("item", "Paper-faithful 3-step BFS (¬, ANY, occupancy)", "partial"),
    ("head", "Data & I/O", None),
    ("item", "Inline init literals", "done"),
    ("item", "Matrix Market load / save · .npy any-rank load / save", "done"),
    ("item", "Inferred dims (dim N = *)", "done"),
    ("item", "Multi-output run() · several save: directives", "done"),
    ("item", "seed(source, value) auto-sized source vectors", "done"),
    ("item", "Multi-source seed · +I preprocessing", "planned"),
    ("item", "Sparse storage + sparse compute (CSR/CSC/COO)", "planned"),
    ("item", "Formats beyond .mtx/.npy · general data loading · input CLI", "planned"),
    ("head", "Visualization", None),
    ("item", "--emit-trace → per-generation PNG frames", "done"),
    ("item", "6 layouts: spring circular shell multipartite kk spectral", "done"),
    ("item", "--active-edges-only edge filter", "planned"),
    ("item", "Per-scalar · scrubbing · animation · print themes", "planned"),
    ("head", "Test data", None),
    ("item", "SuiteSparse vendored graphs + fetch script", "done"),
    ("item", "Expanded test set · many-graphs sweep runner", "planned"),
    ("head", "Tooling", None),
    ("item", "CLI (--dump-ir, --emit-trace)", "done"),
    ("item", "VS Code: syntax + static autocomplete", "done"),
    ("item", "VS Code Tier-2 (context-aware) · Neovim port", "planned"),
    ("item", "Compiler optimizations", "planned"),
]

BG = "#0d1117"; HEAD = "#161b22"; TEXT = "#e6edf3"; SUB = "#8b949e"
MARK = {"done": ("✓", "#3fb950"), "partial": ("◐", "#d29922"),
        "planned": ("✗", "#f85149")}


def render(out_path):
    n = len(ROWS)
    rh = 0.42
    fig = plt.figure(figsize=(8.6, 1.5 + n * rh), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, n + 3.2)

    ax.text(3, n + 2.4, "edgecc — feature status", color=TEXT,
            fontsize=20, fontweight="bold", va="center")
    lx = 3
    for st, lbl in [("done", "implemented"), ("partial", "partial"), ("planned", "planned")]:
        sym, col = MARK[st]
        ax.text(lx, n + 1.4, sym, color=col, fontsize=13, fontweight="bold", va="center")
        ax.text(lx + 2.4, n + 1.4, lbl, color=SUB, fontsize=11, va="center")
        lx += 3.0 + len(lbl) * 1.05

    y = n + 0.3
    for kind, label, status in ROWS:
        y -= 1
        if kind == "head":
            ax.add_patch(FancyBboxPatch((2, y - 0.42), 96, 0.84,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         linewidth=0, facecolor=HEAD))
            ax.text(3.2, y, label.upper(), color=SUB, fontsize=11.5,
                    fontweight="bold", va="center")
        else:
            sym, col = MARK[status]
            ax.text(5.5, y, label, color=TEXT, fontsize=12, va="center")
            ax.text(94, y, sym, color=col, fontsize=15, fontweight="bold",
                    va="center", ha="center")

    fig.savefig(out_path, facecolor=BG, dpi=150, bbox_inches="tight")
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render edgecc feature-status PNG.")
    ap.add_argument("-o", "--out", default="edgecc_status.png")
    render(ap.parse_args().out)
