"""
solver.py -- Public API for Max-Cut subgraph solving.

Routes between the quantum backend (pyQuil QAOA) and the classical fallback
(exact brute-force) based on config.USE_QUANTUM.

Public API:
    setup_qpu(qc_name)              -- call once before pipeline
    solve_subgraph(subgraph, top_t) -> list[Solution]
    maxcut_score(G, assignment)     -> float
"""
from __future__ import annotations

import networkx as nx

try:
    from . import config
    from .quantum_backend import setup_qpu, run_quantum, _PYQUIL_AVAILABLE
    from .classical_backend import run_classical
except ImportError:
    import config
    from quantum_backend import setup_qpu, run_quantum, _PYQUIL_AVAILABLE
    from classical_backend import run_classical

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


def solve_subgraph(subgraph: nx.Graph, top_t: int = 10) -> list[Solution]:
    """
    Solve Max-Cut on `subgraph`; return the top-t solutions sorted best-first.

    Uses quantum backend when config.USE_QUANTUM=True, classical otherwise.
    Solutions are deduplicated before ranking.
    """
    nodes = list(subgraph.nodes())
    if not nodes:
        return [{}]

    if config.USE_QUANTUM:
        if not _PYQUIL_AVAILABLE:
            raise RuntimeError(
                "config.USE_QUANTUM=True but pyquil is not installed. "
                "Run: pip install pyquil"
            )
        raw = run_quantum(subgraph, nodes)
    else:
        raw = run_classical(nodes)

    # Deduplicate and rank by score (best first)
    seen:   set = set()
    ranked: list[tuple[float, Solution]] = []
    for sol in raw:
        key = tuple(sol.get(v, 1) for v in nodes)
        if key not in seen:
            seen.add(key)
            ranked.append((maxcut_score(subgraph, sol), sol))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [sol for _, sol in ranked[:top_t]]
