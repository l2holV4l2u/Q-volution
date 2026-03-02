"""
partitioner.py — Recursive vertex-separator partitioning.

Uses NaiveLGP-style incremental minimum vertex separator
(PaddlePaddle / Junde Li et al. approach).
Finds the smallest possible separator set S by removing
increasing numbers of vertices until the graph disconnects.
Small |S| keeps the 2^|S| merge enumeration cheap.
"""
from __future__ import annotations

import networkx as nx
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional


@dataclass
class PartitionNode:
    graph: nx.Graph                      # subgraph at this node
    separator: set                       # node IDs shared with sibling
    left: Optional["PartitionNode"] = field(default=None, repr=False)
    right: Optional["PartitionNode"] = field(default=None, repr=False)
    is_leaf: bool = True

    def leaves(self) -> list["PartitionNode"]:
        """Return all leaf PartitionNodes in DFS order."""
        if self.is_leaf:
            return [self]
        result = []
        if self.left:
            result.extend(self.left.leaves())
        if self.right:
            result.extend(self.right.leaves())
        return result


# ---------------------------------------------------------------------------
# Strategy 1: NaiveLGP — minimum vertex separator (PaddlePaddle / Li et al.)
# ---------------------------------------------------------------------------

def naive_lgp(G: nx.Graph, max_sep_size: int = 8) -> tuple[set, set, set]:
    """
    Find the minimum vertex separator by incrementally removing vertices.

    Tries removing 1, 2, 3, … vertices until the remaining graph has exactly
    2 connected components. The removed nodes form separator S.

    Separator nodes are added back to *both* subgraphs so no edges are lost.

    Returns (A, S, B) where A ∪ S and B ∪ S are the two subgraphs.
    """
    V = list(G.nodes())

    for sep_size in range(1, min(max_sep_size + 1, len(V))):
        for candidate_sep in combinations(V, sep_size):
            sep = set(candidate_sep)
            H = G.copy()
            H.remove_nodes_from(sep)
            components = list(nx.connected_components(H))
            if len(components) >= 2:
                A = set(components[0])
                B = set().union(*components[1:])
                # Add separator edges back into both sides
                return A, sep, B

    # Could not split (complete graph or single node) — return as-is
    mid = len(V) // 2
    return set(V[:mid]), set(), set(V[mid:])


# ---------------------------------------------------------------------------
# Subgraph builder
# ---------------------------------------------------------------------------

def build_subgraphs(
    G: nx.Graph, A: set, S: set, B: set
) -> tuple[nx.Graph, nx.Graph]:
    """
    Return G[A ∪ S] and G[B ∪ S].

    Separator nodes appear in BOTH subgraphs — this preserves all
    cross-separator edges and is the key property that gives zero edge loss.
    """
    return G.subgraph(A | S).copy(), G.subgraph(B | S).copy()


# ---------------------------------------------------------------------------
# Recursive partitioner
# ---------------------------------------------------------------------------

def recursive_partition(
    G: nx.Graph,
    max_size: int = 84,
    verbose=False
) -> PartitionNode:
    """
    Recursively partition G until all leaves have ≤ max_size nodes.

    Args:
        G:        Input graph.
        max_size: Qubit budget per subgraph (default 84).

    Returns the root PartitionNode of the partition tree.
    """
    root = PartitionNode(graph=G, separator=set())
    _partition_recursive(root, max_size, verbose)
    return root


def _partition_recursive(
    node: PartitionNode, max_size: int, verbose=False
) -> None:
    G = node.graph

    if G.number_of_nodes() <= max_size:
        node.is_leaf = True
        return
    if verbose:
        print(
            f"[partitioner] Splitting: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )

    A, S, B = naive_lgp(G)

    if not A or not B:
        node.is_leaf = True
        if verbose:
            print(
                f"[partitioner] WARNING: could not split {G.number_of_nodes()}-node "
                f"graph; treating as oversized leaf."
            )
        return

    if verbose: 
        print(
        f"[partitioner]  -> |A|={len(A)}, |S|={len(S)}, |B|={len(B)}  "
        f"(2^|S|={2**len(S)} separator combos at merge)"
        )

    left_G, right_G = build_subgraphs(G, A, S, B)

    node.separator = S
    node.is_leaf = False
    node.left = PartitionNode(graph=left_G, separator=set())
    node.right = PartitionNode(graph=right_G, separator=set())

    _partition_recursive(node.left, max_size)
    _partition_recursive(node.right, max_size)
