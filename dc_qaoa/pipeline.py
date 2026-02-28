"""
pipeline.py -- Orchestrates the full DC-QAOA pipeline.

Steps:
  1. (Optional) Configure pyQuil quantum computer for QPU/QVM.
  2. Load graph from parquet.
  3. Recursively partition into subgraphs <= max_size nodes.
  4. Solve each leaf subgraph with QAOA (real or stub).
  5. Merge solutions via GR policy through the partition tree.
  6. Score the final assignment on the full graph.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx

from graph_loader import load_graph
from partitioner import recursive_partition, PartitionNode
from solver import qaoa_solve, setup_qpu, USE_PYQUIL, _local_search
from merger import merge
from scorer import maxcut_score


def run_pipeline(
    graph_path: str | Path,
    max_size: int = 8,
    top_t: int = 10,
    method: str = "separator",          # "separator" (NaiveLGP) | "community"
    qc_name: Optional[str] = None,      # pyQuil QC name, e.g. "Ankaa-3", "8q-qvm"
) -> tuple[dict, float]:
    """
    Run the full DC-QAOA pipeline.

    Args:
        graph_path: Path to the .parquet edge list file.
        max_size:   Max nodes per QAOA subgraph (qubit budget). Default 8.
        top_t:      Top-t solutions to keep per subgraph for merge. Default 10.
        method:     Partitioning strategy -- "separator" (NaiveLGP, minimum
                    vertex separator) or "community" (greedy modularity).
        qc_name:    pyQuil quantum computer name. None -> auto-sized QVM.
                    Examples: "8q-qvm", "Ankaa-3", "Ankaa-9Q-3".

    Returns:
        (best_assignment, score)
        best_assignment: {node_id: +1 | -1}
        score:           weighted Max-Cut value on the full graph.
    """
    # ------------------------------------------------------------------ 0 --
    if qc_name:
        print(f"\n=== Step 0: Configure quantum computer ({qc_name}) ===")
        setup_qpu(qc_name)
    elif USE_PYQUIL:
        print("\n[pipeline] USE_PYQUIL=True -- QC will be auto-configured per subgraph.")

    # ------------------------------------------------------------------ 1 --
    print("\n=== Step 1: Load graph ===")
    G = load_graph(graph_path)

    # ------------------------------------------------------------------ 2 --
    print(f"\n=== Step 2: Partition graph (method={method}) ===")
    partition_tree = recursive_partition(G, max_size=max_size, method=method)
    leaves = partition_tree.leaves()
    sizes = [leaf.graph.number_of_nodes() for leaf in leaves]
    print(
        f"[pipeline] {len(leaves)} leaf subgraph(s)  |  "
        f"sizes: {sizes}  |  max: {max(sizes)}  |  min: {min(sizes)}"
    )

    # ------------------------------------------------------------------ 3 --
    print("\n=== Step 3: Solve leaf subgraphs ===")
    subgraph_solutions: dict = {}
    for i, leaf in enumerate(leaves):
        n_nodes = leaf.graph.number_of_nodes()
        backend = "pyQuil" if USE_PYQUIL else "stub"
        print(
            f"[pipeline] Leaf {i + 1}/{len(leaves)}: "
            f"{n_nodes} nodes, {leaf.graph.number_of_edges()} edges  "
            f"[{backend}]"
        )
        solutions = qaoa_solve(leaf.graph, top_t=top_t)
        subgraph_solutions[id(leaf)] = solutions
        best = maxcut_score(leaf.graph, solutions[0]) if solutions else 0.0
        print(f"[pipeline]   -> {len(solutions)} solution(s), best subgraph cut = {best:.4f}")

    # ------------------------------------------------------------------ 4 --
    print("\n=== Step 4: Merge (GR policy) ===")
    best_assignment = merge(G, partition_tree, subgraph_solutions, top_t=top_t)

    # ------------------------------------------------------------------ 4b -
    print("\n=== Step 4b: Final local search polish ===")
    # Ensure all nodes have an assignment (default +1 for any missing)
    for n in G.nodes():
        if n not in best_assignment:
            best_assignment[n] = 1
    pre_polish = maxcut_score(G, best_assignment)
    best_assignment = _local_search(G, list(G.nodes()), best_assignment)
    post_polish = maxcut_score(G, best_assignment)
    print(f"[pipeline] Polish: {pre_polish:.4f} -> {post_polish:.4f} "
          f"(+{post_polish - pre_polish:.4f})")

    # ------------------------------------------------------------------ 5 --
    print("\n=== Step 5: Final score ===")
    score = maxcut_score(G, best_assignment)
    total_weight = sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True))
    print(f"[pipeline] Max-Cut score  : {score:.6f}")
    print(f"[pipeline] Total weight   : {total_weight:.6f}")
    print(f"[pipeline] Approx ratio   : {score / total_weight:.4f}")

    return best_assignment, score
