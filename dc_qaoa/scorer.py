"""
scorer.py — Max-Cut objective evaluation.
"""
import networkx as nx

Solution = dict  # {node_id: +1 | -1}


def maxcut_score(G: nx.Graph, assignment: Solution) -> float:
    """
    Evaluate the weighted Max-Cut objective C(z) for a spin assignment.

    C(z) = Σ_{(u,v) ∈ E} w_{uv} * (1 - z_u * z_v) / 2

    Nodes missing from `assignment` are treated as +1.
    """
    score = 0.0
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        zu = assignment.get(u, 1)
        zv = assignment.get(v, 1)
        score += w * (1 - zu * zv) / 2
    return score


def maxcut_score_bitstring(G: nx.Graph, bitstring: str) -> float:
    """
    Convenience wrapper: bitstring '01101...' → score.
    Nodes are indexed by their sorted order in G.nodes().
    '0' → +1 spin, '1' → -1 spin.
    """
    nodes = sorted(G.nodes())
    if len(bitstring) != len(nodes):
        raise ValueError(
            f"Bitstring length {len(bitstring)} != number of nodes {len(nodes)}"
        )
    assignment = {
        n: (1 if b == "0" else -1) for n, b in zip(nodes, bitstring)
    }
    return maxcut_score(G, assignment)
