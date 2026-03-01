"""
visualize_cut.py -- Visualize the DC-QAOA Max-Cut result.

Usage:
  python visualize_cut.py ../dataset_A.parquet
  python visualize_cut.py ../dataset_B.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

try:
    from graph_loader import load_graph
    from pipeline import run_pipeline
except ImportError:
    from .graph_loader import load_graph
    from .pipeline import run_pipeline

def visualize(G: nx.Graph, assignment: dict, score: float, title: str = "DC-QAOA Max-Cut", out_file: str = "maxcut.png"):
    total_weight = sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True))
    ratio = score / total_weight

    # Separate cut vs uncut edges
    cut_edges, uncut_edges = [], []
    cut_weights, uncut_weights = [], []
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        if assignment.get(u, 1) != assignment.get(v, 1):
            cut_edges.append((u, v))
            cut_weights.append(w)
        else:
            uncut_edges.append((u, v))
            uncut_weights.append(w)

    # Partition nodes by spin
    plus_nodes = [n for n in G.nodes() if assignment.get(n, 1) == 1]
    minus_nodes = [n for n in G.nodes() if assignment.get(n, 1) == -1]

    # Kamada-Kawai for a clean layout
    pos = nx.kamada_kawai_layout(G, weight="weight")

    fig, ax = plt.subplots(figsize=(18, 14), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    # Uncut edges -- dashed, dim
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, edge_color="#ff4444",
                           width=2.0, alpha=0.45, style="dashed", ax=ax)

    # Cut edges -- solid, bright, width scaled by weight
    if cut_weights:
        max_w = max(cut_weights)
        cut_widths = [2.5 + 4.0 * (w / max_w) for w in cut_weights]
    else:
        cut_widths = []
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color="#00e676",
                           width=cut_widths, alpha=0.85, ax=ax)

    # Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=plus_nodes, node_color="#ff5252",
                           node_size=200, edgecolors="#ffffff", linewidths=1.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=minus_nodes, node_color="#448aff",
                           node_size=200, edgecolors="#ffffff", linewidths=1.0, ax=ax)

    # Legend + stats
    cut_weight = sum(cut_weights)
    uncut_weight = sum(uncut_weights)
    legend_handles = [
        mpatches.Patch(facecolor="#ff5252", edgecolor="white", label=f"Spin +1  ({len(plus_nodes)} nodes)"),
        mpatches.Patch(facecolor="#448aff", edgecolor="white", label=f"Spin -1  ({len(minus_nodes)} nodes)"),
        plt.Line2D([0], [0], color="#00e676", linewidth=2.5, label=f"Cut edges  ({len(cut_edges)})"),
        plt.Line2D([0], [0], color="#ff4444", linewidth=1, linestyle="dashed", label=f"Uncut edges  ({len(uncut_edges)})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10,
              frameon=True, facecolor="#2a2a4a", edgecolor="#555555", labelcolor="white")

    ax.set_title(
        f"{title}\n"
        f"Score: {score:.2f} / {total_weight:.2f}   |   "
        f"Ratio: {ratio:.4f}   |   "
        f"Cut: {cut_weight:.1f}  Uncut: {uncut_weight:.1f}",
        fontsize=14, color="white", pad=14,
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved to {out_file}")
    plt.show()


def main():
    if len(sys.argv) > 1:
        graph_path = sys.argv[1]
    else:
        graph_path = str(Path("..") / "dataset_B.parquet")

    dataset_name = Path(graph_path).stem

    print(f"Running DC-QAOA on {graph_path}...")
    assignment, score = run_pipeline(graph_path, max_size=8, top_t=10)

    G = load_graph(graph_path)
    print(f"Score: {score:.2f}")

    out_file = f"maxcut_{dataset_name}.png"
    visualize(G, assignment, score, title=f"DC-QAOA Max-Cut -- {dataset_name}", out_file=out_file)


if __name__ == "__main__":
    main()
