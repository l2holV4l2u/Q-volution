"""
solver.py -- Public API for Max-Cut subgraph solving.

Routes between the quantum backend (pyQuil QAOA) and the classical fallback
(exact brute-force) based on _config.USE_QUANTUM.

Public API:
    setup_qpu(qc_name)              -- call once before pipeline
    solve_subgraph(subgraph, top_t) -> list[Solution]
    maxcut_score(G, assignment)     -> float
"""
from __future__ import annotations

from typing import Optional
import networkx as nx

from . import config as _config
from .quantum_backend import setup_qpu, run_quantum, _PYQUIL_AVAILABLE, Solution, Solutions
from .classical_backend import run_classical

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

def solve_subgraph(subgraph: nx.Graph, top_t: int = 10) -> Solutions:
    """
    Solve Max-Cut on `subgraph`; return the top-t solutions sorted best-first.

    Uses quantum backend when _config.USE_QUANTUM=True, classical otherwise.
    Solutions are deduplicated before ranking.
    """
    nodes = list(subgraph.nodes())
    if not nodes:
        return [{}]

    if _config.USE_QUANTUM:
        if not _PYQUIL_AVAILABLE:
            raise RuntimeError(
                "_config.USE_QUANTUM=True but pyquil is not installed. "
                "Run: pip install pyquil"
            )
        raw = run_quantum(subgraph, nodes, _config.OPTIMIZER)
    else:
        raw = run_classical(nodes)

    # Deduplicate and rank by score (best first)
    seen = set()
    ranked: list[tuple[float, Solution]] = []
    for sol in raw:
        key = tuple(sol.get(v, 1) for v in nodes)
        if key not in seen:
            seen.add(key)
            ranked.append((maxcut_score(subgraph, sol), sol))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [sol for _, sol in ranked[:top_t]]
    
def solve_maxcut(
    graph : nx.Graph,
    max_size: int = 8,
    top_t: int = 10,
    qc_name: Optional[str] = None,     # e.g. "Ankaa-3", "8q-qvm"
    use_quantum: Optional[bool] = None,
    verbose: bool = False,
) -> tuple[Solution, float]:
    # Decide backend
    if use_quantum is None:
        use_quantum = _config.USE_QUANTUM

    # Setup quantum computer
    if qc_name:
        setup_qpu(qc_name)

    # Partition
    from .partitioner import recursive_partition
    partition_tree = recursive_partition(graph, max_size=max_size)
    leaves = partition_tree.leaves()
    if verbose:
        sizes = [leaf.graph.number_of_nodes() for leaf in leaves]
        print(f"[solve] leaves={len(leaves)} sizes={sizes} max={max(sizes)} min={min(sizes)}")

    # Solve leaves
    subgraph_solutions: dict[int, list[Solution]] = {}
    for i, leaf in enumerate(leaves):
        if verbose:
            print(
                f"[solve] leaf {i+1}/{len(leaves)}: "
                f"n={leaf.graph.number_of_nodes()} m={leaf.graph.number_of_edges()} [{"quantum" if use_quantum else "classical"}]"
            )
        solutions = solve_subgraph(leaf.graph, top_t=top_t)
        subgraph_solutions[id(leaf)] = solutions
        if verbose:
            best_sub = maxcut_score(leaf.graph, solutions[0]) if solutions else 0.0
            print(f"[solve]   -> {len(solutions)} solution(s), best_sub={best_sub:.4f}")

    # Merge
    from .merger import merge
    best_assignment: Solution = merge(graph, partition_tree, subgraph_solutions, top_t=top_t)

    # Final score
    score = maxcut_score(graph, best_assignment)
    if verbose:
        total_weight = sum(d.get("weight", 1.0) for _, _, d in graph.edges(data=True))
        ratio = (score / total_weight) if total_weight else 0.0
        print(f"[solve] score={score:.6f} total_weight={total_weight:.6f} approx_ratio={ratio:.4f}")

    return best_assignment, score