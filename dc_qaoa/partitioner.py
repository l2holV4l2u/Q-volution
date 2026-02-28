"""
partitioner.py — Recursive vertex-separator partitioning.

Two strategies available (select via `method` param in recursive_partition):

  "community"  — greedy_modularity_communities (NVIDIA Lab 4 approach).
                 Fast, works well on weighted graphs. Does NOT guarantee a
                 strict minimum separator; border edges become the implicit
                 separator.

  "separator"  — NaiveLGP-style incremental minimum vertex separator
                 (PaddlePaddle / Junde Li et al. approach).
                 Finds the smallest possible separator set S by removing
                 increasing numbers of vertices until the graph disconnects.
                 Slow for large dense graphs but gives the tightest
                 2^|S| enumeration budget for the merger.

For the target graph (~180 nodes, ~2.5 avg degree) both strategies are fast.
"separator" is recommended because small |S| keeps the merge cheap.
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
# Strategy 2: greedy_modularity_communities (NVIDIA Lab 4 approach)
# ---------------------------------------------------------------------------

def community_partition(G: nx.Graph) -> tuple[set, set, set]:
    """
    Partition using greedy modularity community detection.

    Uses weight='weight' so edge weights influence community assignment.
    Returns (A, S, B) where S = border nodes (nodes with inter-community edges).
    """
    from networkx.algorithms import community as nx_community

    parts = list(
        nx_community.greedy_modularity_communities(
            G, weight="weight", resolution=1.1
        )
    )

    if len(parts) < 2:
        # Fallback: bisect by node order
        nodes = list(G.nodes())
        mid = len(nodes) // 2
        parts = [set(nodes[:mid]), set(nodes[mid:])]

    # Merge into two groups (take first vs rest)
    A = set(parts[0])
    B = set().union(*parts[1:])

    # Separator = border nodes (nodes in A that have a neighbour in B)
    S = set()
    for u in A:
        for v in G.neighbors(u):
            if v in B:
                S.add(u)
                S.add(v)

    # Remove separator from A and B (they'll be added back to both)
    A -= S
    B -= S

    return A, S, B


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
    method: str = "separator",   # "separator" | "community"
) -> PartitionNode:
    """
    Recursively partition G until all leaves have ≤ max_size nodes.

    Args:
        G:        Input graph.
        max_size: Qubit budget per subgraph (default 84).
        method:   Partitioning strategy — "separator" (NaiveLGP, recommended)
                  or "community" (greedy modularity, faster for large graphs).

    Returns the root PartitionNode of the partition tree.
    """
    root = PartitionNode(graph=G, separator=set())
    _partition_recursive(root, max_size, method)
    return root


def _partition_recursive(
    node: PartitionNode, max_size: int, method: str
) -> None:
    G = node.graph

    if G.number_of_nodes() <= max_size:
        node.is_leaf = True
        return

    print(
        f"[partitioner] Splitting: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges  (method={method})"
    )

    if method == "separator":
        A, S, B = naive_lgp(G)
    else:
        A, S, B = community_partition(G)

    if not A or not B:
        print(
            f"[partitioner] WARNING: could not split {G.number_of_nodes()}-node "
            f"graph; treating as oversized leaf."
        )
        node.is_leaf = True
        return

    print(
        f"[partitioner]  -> |A|={len(A)}, |S|={len(S)}, |B|={len(B)}  "
        f"(2^|S|={2**len(S)} separator combos at merge)"
    )

    left_G, right_G = build_subgraphs(G, A, S, B)

    node.separator = S
    node.is_leaf = False
    node.left = PartitionNode(graph=left_G, separator=set())
    node.right = PartitionNode(graph=right_G, separator=set())

    _partition_recursive(node.left, max_size, method)
    _partition_recursive(node.right, max_size, method)
