"""
merger.py — Merge subgraph solutions across the partition tree.

Implements the GR (Greedy Reconstruction) policy from:
  Li et al. "Large-scale Quantum Approximate Optimization via
  Divide-and-Conquer" (2021) — arxiv.org/abs/2102.13288
  Also: PaddlePaddle Quantum DC-QAOA tutorial.

Algorithm per internal partition node:
  1. Represent each subgraph solution as a full-graph "sparse" string where
     nodes not in the subgraph are marked as None (placeholder).
  2. For each pair (left_sol, right_sol):
       a. Enumerate all 2^|S| separator spin assignments.
       b. Evaluate combined Max-Cut score on the full graph.
  3. Return the top-t best assignments (preserving diversity for parent merges).

Complexity: O(top_t² × 2^|S|) per internal node.
  With top_t=10, |S|≤5 → 3 200 evals/node — negligible.
"""
from __future__ import annotations

import itertools
from typing import Optional

import networkx as nx

from partitioner import PartitionNode
from scorer import maxcut_score

Solution = dict  # {node_id: +1 | -1}
_UNSET = None    # sentinel for "node not in this subgraph"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge(
    full_graph: nx.Graph,
    partition_tree: PartitionNode,
    subgraph_solutions: dict,  # id(leaf PartitionNode) → list[Solution]
    top_t: int = 10,
) -> Solution:
    """
    Merge all leaf solutions through the partition tree (GR policy).

    Returns the single best global spin assignment for full_graph.
    """
    all_nodes = list(full_graph.nodes())
    results = _merge_node(full_graph, all_nodes, partition_tree, subgraph_solutions, top_t)
    best = results[0]
    print(f"[merger] Final score on full graph: {maxcut_score(full_graph, best):.4f}")
    return best


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_node(
    full_graph: nx.Graph,
    all_nodes: list,
    node: PartitionNode,
    subgraph_solutions: dict,
    top_t: int,
) -> list[Solution]:
    """Recursively merge the partition tree via GR policy.
    Returns a list of up to top_t solutions (best first)."""
    if node.is_leaf:
        solutions = subgraph_solutions.get(id(node), [{}])
        return solutions if solutions else [{}]

    left_candidates  = _get_candidates(full_graph, all_nodes, node.left,  subgraph_solutions, top_t)
    right_candidates = _get_candidates(full_graph, all_nodes, node.right, subgraph_solutions, top_t)

    separator = node.separator
    sep_nodes = list(separator)

    # Build the subgraph induced by this subtree's nodes for correct scoring.
    # Scoring on full_graph would penalize partial assignments with default +1.
    subtree_node_set = _all_subtree_nodes(node)
    subtree_graph = full_graph.subgraph(subtree_node_set)

    ranked: list[tuple[float, Solution]] = []

    for l_sparse, r_sparse in itertools.product(left_candidates, right_candidates):
        # Merge both sides: real values take priority over None sentinels.
        # For separator nodes (present in both), right's value is used as
        # a default — the enumeration below overrides them anyway.
        combined = {}
        for k in l_sparse:
            lv = l_sparse[k]
            rv = r_sparse.get(k)
            if lv is not None:
                combined[k] = lv
            elif rv is not None:
                combined[k] = rv
            # else both None -> skip (node not in either subtree)

        if not sep_nodes:
            score = maxcut_score(subtree_graph, combined)
            ranked.append((score, combined))
            continue

        # Enumerate 2^|S| separator assignments (GR enumeration)
        for spins in itertools.product([-1, 1], repeat=len(sep_nodes)):
            trial = combined.copy()
            for nid, spin in zip(sep_nodes, spins):
                trial[nid] = spin
            score = maxcut_score(subtree_graph, trial)
            ranked.append((score, trial))

    # Deduplicate and keep top-t
    ranked.sort(key=lambda x: x[0], reverse=True)
    subtree_nodes_list = sorted(subtree_node_set)
    seen: set = set()
    results: list[Solution] = []
    for score, sol in ranked:
        key = tuple(sol.get(v, 1) for v in subtree_nodes_list)
        if key not in seen:
            seen.add(key)
            results.append(sol)
            if len(results) >= top_t:
                break

    if not results:
        results = [{}]

    print(
        f"[merger] Merged internal node: score={ranked[0][0] if ranked else 0:.4f}, "
        f"|S|={len(sep_nodes)}, pairs={len(left_candidates)*len(right_candidates)}, "
        f"candidates_kept={len(results)}"
    )
    return results


def _get_candidates(
    full_graph: nx.Graph,
    all_nodes: list,
    node: PartitionNode,
    subgraph_solutions: dict,
    top_t: int,
) -> list[dict]:
    """
    Return "sparse" solution dicts for this subtree.

    Sparse = {node_id: spin} for nodes IN the subtree,
             node absent (or None) for nodes NOT in the subtree.
    """
    if node.is_leaf:
        subgraph_node_set = set(node.graph.nodes())
        raw_solutions = subgraph_solutions.get(id(node), [{}])
        return [_to_sparse(sol, all_nodes, subgraph_node_set) for sol in raw_solutions]

    # Recurse for internal nodes — propagate top-t candidates.
    # Merged solutions cover the full union of left+right subtree nodes,
    # so use _all_subtree_nodes to get the correct coverage set.
    merged = _merge_node(full_graph, all_nodes, node, subgraph_solutions, top_t)
    subtree_nodes = _all_subtree_nodes(node)
    return [_to_sparse(sol, all_nodes, subtree_nodes) for sol in merged]


def _all_subtree_nodes(node: PartitionNode) -> set:
    """Collect the union of all node IDs across all leaves in this subtree."""
    if node.is_leaf:
        return set(node.graph.nodes())
    result = _all_subtree_nodes(node.left) | _all_subtree_nodes(node.right)
    result |= node.separator
    return result


def _to_sparse(sol: Solution, all_nodes: list, subgraph_nodes: set) -> dict:
    """
    Return a full-graph-sized dict where nodes outside the subgraph are None.
    """
    return {
        n: (sol.get(n) if n in subgraph_nodes else _UNSET)
        for n in all_nodes
    }
