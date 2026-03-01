"""
merger.py — Merge subgraph solutions across the partition tree (GR policy).
"""
from __future__ import annotations

import itertools
import networkx as nx

try:
    from partitioner import PartitionNode
    from solver import maxcut_score
except ImportError:
    from .partitioner import PartitionNode
    from .solver import maxcut_score

Solution = dict  # {node_id: +1 | -1}


def merge(
    full_graph: nx.Graph,
    partition_tree: PartitionNode,
    subgraph_solutions: dict,
    top_t: int = 10,
) -> Solution:
    """Merge all leaf solutions through the partition tree. Returns the best global assignment."""
    results = _merge_node(full_graph, partition_tree, subgraph_solutions, top_t)
    best = results[0]
    print(f"[merger] Final score on full graph: {maxcut_score(full_graph, best):.4f}")
    return best


def _merge_node(
    full_graph: nx.Graph,
    node: PartitionNode,
    subgraph_solutions: dict,
    top_t: int,
) -> list[Solution]:
    """Recursively merge via GR policy. Returns up to top_t solutions, best first."""
    if node.is_leaf:
        return subgraph_solutions.get(id(node), [{}]) or [{}]

    left_sols  = _merge_node(full_graph, node.left,  subgraph_solutions, top_t)
    right_sols = _merge_node(full_graph, node.right, subgraph_solutions, top_t)
    sep_nodes  = list(node.separator)

    subtree_nodes = set(node.left.graph.nodes()) | set(node.right.graph.nodes())
    subtree_graph = full_graph.subgraph(subtree_nodes)

    ranked: list[tuple[float, Solution]] = []
    for left, right in itertools.product(left_sols, right_sols):
        base = {**left, **right}
        if not sep_nodes:
            ranked.append((maxcut_score(subtree_graph, base), base))
            continue
        for spins in itertools.product([-1, 1], repeat=len(sep_nodes)):
            trial = {**base, **dict(zip(sep_nodes, spins))}
            ranked.append((maxcut_score(subtree_graph, trial), trial))

    ranked.sort(key=lambda x: x[0], reverse=True)

    seen: set = set()
    results: list[Solution] = []
    subtree_nodes_sorted = sorted(subtree_nodes)
    for _, sol in ranked:
        key = tuple(sol.get(v, 1) for v in subtree_nodes_sorted)
        if key not in seen:
            seen.add(key)
            results.append(sol)
            if len(results) >= top_t:
                break

    print(
        f"[merger] Internal node: score={ranked[0][0] if ranked else 0:.4f}, "
        f"|S|={len(sep_nodes)}, pairs={len(left_sols)*len(right_sols)}, kept={len(results)}"
    )
    return results or [{}]
