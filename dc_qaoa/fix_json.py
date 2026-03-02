import json

with open('dc_qaoa_visualization.ipynb', 'r') as f:
    text = f.read()

# find where it broke and just rewrite
try:
    # it's just missing the final closing brace for the main dict. Let's fix it by parsing valid string
    obj = json.loads(text.strip().rstrip('}') + '}')
except Exception:
    pass

import ast
cells = ast.literal_eval("""[
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DC-QAOA Visualization Notebook\\n",
    "This notebook runs the DC-QAOA Max-Cut pipeline step-by-step and visualizes the problem graph, subgraph partitions, and the final optimized solution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\n",
    "import ast\\n",
    "import networkx as nx\\n",
    "import matplotlib.pyplot as plt\\n",
    "import matplotlib.colors as mcolors\\n",
    "import numpy as np\\n",
    "\\n",
    "from pathlib import Path\\n",
    "from typing import Optional\\n",
    "\\n",
    "import solver as _solver_module\\n",
    "from graph_loader import load_graph\\n",
    "from partitioner import recursive_partition, PartitionNode\\n",
    "from solver import qaoa_solve, setup_qpu, USE_PYQUIL, _local_search\\n",
    "from merger import merge\\n",
    "from scorer import maxcut_score\\n",
    "\\n",
    "%matplotlib inline\\n",
    "plt.rcParams['figure.figsize'] = (10, 8)\\n",
    "plt.rcParams['figure.dpi'] = 100"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Helper Functions for Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def draw_graph(G: nx.Graph, title: str, node_colors: list = None, pos: dict = None):\\n",
    "    \\\"\\\"\\\"Helper to draw a NetworkX graph nicely.\\\"\\\"\\\"\\n",
    "    plt.figure(figsize=(10, 8))\\n",
    "    if pos is None:\\n",
    "        pos = nx.spring_layout(G, seed=42)\\n",
    "        \\n",
    "    if node_colors is None:\\n",
    "        node_colors = ['skyblue'] * G.number_of_nodes()\\n",
    "        \\n",
    "    edges = G.edges(data=True)\\n",
    "    weights = [d.get('weight', 1.0) for u, v, d in edges]\\n",
    "    max_weight = max(weights) if weights else 1.0\\n",
    "    edge_widths = [1 + 3 * (w / max_weight) for w in weights]\\n",
    "\\n",
    "    nx.draw(G, pos, \\n",
    "            node_color=node_colors, \\n",
    "            with_labels=True, \\n",
    "            node_size=600, \\n",
    "            font_size=10, \\n",
    "            font_color='black',\\n",
    "            font_weight='bold',\\n",
    "            edge_color='gray', \\n",
    "            width=edge_widths, \\n",
    "            alpha=0.9)\\n",
    "    \\n",
    "    plt.title(title, fontsize=16)\\n",
    "    plt.margins(x=0.1, y=0.1)\\n",
    "    plt.show()\\n",
    "    return pos"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Configuration & Load Graph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pipeline settings\\n",
    "GRAPH_PATH = \\\"../dataset_A.csv\\\"\\n",
    "MAX_SIZE = 8\\n",
    "TOP_T = 10\\n",
    "METHOD = \\\"separator\\\"\\n",
    "\\n",
    "# Solver settings\\n",
    "_solver_module.USE_PYQUIL = True    # Set to False to use stub local search\\n",
    "_solver_module.LAYER_COUNT = 1\\n",
    "_solver_module.SHOTS = 1024\\n",
    "_solver_module.SEED = 42\\n",
    "\\n",
    "# Quantum computer target (None = auto-detect QVM)\\n",
    "QC_NAME = \\\"9q-square-qvm\\\"\\n",
    "\\n",
    "if _solver_module.USE_PYQUIL and QC_NAME:\\n",
    "    print(f\\\"Setting up QPU: {QC_NAME}...\\\")\\n",
    "    setup_qpu(QC_NAME)\\n",
    "\\n",
    "# Load Graph\\n",
    "print(f\\\"\\\\nLoading graph from {GRAPH_PATH}...\\\")\\n",
    "G = load_graph(GRAPH_PATH)\\n",
    "print(f\\\"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\\\")\\n",
    "\\n",
    "# Pre-calculate a layout for consistent visualization\\n",
    "global_pos = nx.spring_layout(G, seed=42)\\n",
    "draw_graph(G, \\\"Original Problem Graph\\\", pos=global_pos)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Graph Partitioning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\\\"Partitioning graph (method={METHOD}, max_size={MAX_SIZE})...\\\")\\n",
    "partition_tree = recursive_partition(G, max_size=MAX_SIZE, method=METHOD)\\n",
    "leaves = partition_tree.leaves()\\n",
    "\\n",
    "print(f\\\"Created {len(leaves)} leaf subgraphs.\\\")\\n",
    "\\n",
    "# Assign a distinct color index to each leaf subgraph\\n",
    "node_to_leaf_idx = {}\\n",
    "for i, leaf in enumerate(leaves):\\n",
    "    for node in leaf.graph.nodes():\\n",
    "        if node not in node_to_leaf_idx:\\n",
    "            node_to_leaf_idx[node] = i\\n",
    "\\n",
    "# Map indices to unique colors\\n",
    "cmap = plt.cm.get_cmap('tab20', len(leaves))\\n",
    "colors = [cmap(node_to_leaf_idx.get(n, 0)) for n in G.nodes()]\\n",
    "\\n",
    "draw_graph(G, f\\\"Partitioned Graph ({len(leaves)} Subgraphs)\\\", node_colors=colors, pos=global_pos)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Solve Leaf Subgraphs with QAOA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "subgraph_solutions = {}\\n",
    "\\n",
    "for i, leaf in enumerate(leaves):\\n",
    "    n_nodes = leaf.graph.number_of_nodes()\\n",
    "    backend = \\\"pyQuil\\\" if _solver_module.USE_PYQUIL else \\\"stub\\\"\\n",
    "    print(f\\\"\\\\n--- Solving Leaf {i + 1}/{len(leaves)} ---\\\")\\n",
    "    print(f\\\"Nodes: {n_nodes}, Edges: {leaf.graph.number_of_edges()} [{backend}]\\\")\\n",
    "    \\n",
    "    # Solve (QAOA or stub)\\n",
    "    solutions = qaoa_solve(leaf.graph, top_t=TOP_T)\\n",
    "    subgraph_solutions[id(leaf)] = solutions\\n",
    "    \\n",
    "    best = maxcut_score(leaf.graph, solutions[0]) if solutions else 0.0\\n",
    "    print(f\\\"-> Found {len(solutions)} solution(s). Best subgraph cut = {best:.4f}\\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Merger & GR Policy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\\\"\\\\nMerging subgraphs via GR policy...\\\")\\n",
    "best_assignment = merge(G, partition_tree, subgraph_solutions, top_t=TOP_T)\\n",
    "\\n",
    "# Ensure all nodes have an assignment (fallback to +1)\\n",
    "for n in G.nodes():\\n",
    "    if n not in best_assignment:\\n",
    "        best_assignment[n] = 1\\n",
    "\\n",
    "pre_polish_score = maxcut_score(G, best_assignment)\\n",
    "print(f\\\"\\\\nPre-polish Max-Cut Score: {pre_polish_score:.4f}\\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Local Search Polish & Final Result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\\\"Running fast local search pass to polish border assignments...\\\")\\n",
    "best_assignment = _local_search(G, list(G.nodes()), best_assignment)\\n",
    "final_score = maxcut_score(G, best_assignment)\\n",
    "\\n",
    "total_weight = sum(d.get(\\\"weight\\\", 1.0) for _, _, d in G.edges(data=True))\\n",
    "\\n",
    "print(f\\\"\\\\n{'=' * 50}\\\")\\n",
    "print(f\\\"  FINAL RESULTS\\\")\\n",
    "print(f\\\"{'=' * 50}\\\")\\n",
    "print(f\\\"Pre-Polish Score : {pre_polish_score:.4f}\\\")\\n",
    "print(f\\\"Final Score      : {final_score:.4f} (+{final_score - pre_polish_score:.4f})\\\")\\n",
    "print(f\\\"Total Edge Weight: {total_weight:.4f}\\\")\\n",
    "print(f\\\"Approx Ratio     : {final_score / total_weight:.4f}\\\")\\n",
    "print(f\\\"{'=' * 50}\\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Visualize Final Cut Assignment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# +1 spin -> Lightgreen, -1 spin -> Tomato\\n",
    "color_map = {1: 'lightgreen', -1: 'tomato'}\\n",
    "final_colors = [color_map[best_assignment[n]] for n in G.nodes()]\\n",
    "\\n",
    "draw_graph(G, f\\\"Final Max-Cut (Score: {final_score:.2f})\\\\nGreen: +1, Red: -1\\\", \\n",
    "           node_colors=final_colors, pos=global_pos)"
   ]
  }
]""")

metadata = {
  "kernelspec": {
   "display_name": "Python (q-volution)",
   "language": "python",
   "name": "q-volution"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  }
}

full_dict = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 4}

with open('dc_qaoa_visualization.ipynb', 'w') as f:
    json.dump(full_dict, f, indent=2)

