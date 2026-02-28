"""
resource_estimation.py -- Resource estimation for standard QAOA vs DC-QAOA.

Demonstrates why standard QAOA is infeasible for Problem B (180 nodes)
on Rigetti Ankaa-3, and how DC-QAOA makes it tractable.

Usage:
    python resource_estimation.py ../dataset_B.parquet
    python resource_estimation.py ../dataset_A.parquet ../dataset_B.parquet
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import networkx as nx

from graph_loader import load_graph
from partitioner import recursive_partition


# -- Ankaa-3 hardware limits --------------------------------------------------
ANKAA3_QUBITS = 84          # total physical qubits on Ankaa-3
ANKAA3_MAX_PRACTICAL = 10   # practical qubit limit for meaningful results
ISWAP_PER_CNOT = 3          # CNOT decomposes to ~3 iSWAPs on square grid
MAX_TWO_QUBIT_GATES = 100   # target limit for acceptable fidelity
GATE_FIDELITY_ISWAP = 0.995 # approximate single iSWAP fidelity on Ankaa-3


def estimate_standard_qaoa(G: nx.Graph, layers: int = 2) -> dict:
    """Estimate resources for standard (non-partitioned) QAOA on graph G."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Gate counts per layer
    cnots_per_layer = m * 2          # CNOT-Rz-CNOT per edge = 2 CNOTs
    rz_per_layer = m                 # 1 Rz per edge
    rx_per_layer = n                 # 1 Rx per node (mixer)

    # iSWAP equivalents (Ankaa-3 native)
    iswaps_per_layer = cnots_per_layer * ISWAP_PER_CNOT
    total_iswaps = iswaps_per_layer * layers

    # SWAP routing overhead estimate (graph connectivity vs square grid)
    avg_degree = 2 * m / n if n > 0 else 0
    # Square grid has max degree 4; non-adjacent qubits need SWAP chains
    # Conservative estimate: 50% overhead for sparse graphs
    routing_overhead = 1.5
    total_iswaps_routed = int(total_iswaps * routing_overhead)

    # Circuit fidelity estimate
    fidelity = GATE_FIDELITY_ISWAP ** total_iswaps_routed

    return {
        "qubits": n,
        "edges": m,
        "layers": layers,
        "cnots_per_layer": cnots_per_layer,
        "iswaps_per_layer": iswaps_per_layer,
        "total_cnots": cnots_per_layer * layers,
        "total_iswaps": total_iswaps,
        "total_iswaps_routed": total_iswaps_routed,
        "circuit_fidelity": fidelity,
        "avg_degree": avg_degree,
        "feasible_qubits": n <= ANKAA3_MAX_PRACTICAL,
        "feasible_gates": total_iswaps_routed <= MAX_TWO_QUBIT_GATES,
    }


def estimate_dcqaoa(G: nx.Graph, max_size: int = 10, layers: int = 1) -> dict:
    """Estimate resources for DC-QAOA with partitioned subgraphs."""
    tree = recursive_partition(G, max_size=max_size)
    leaves = tree.leaves()

    subgraph_stats = []
    for leaf in leaves:
        sg = leaf.graph
        n = sg.number_of_nodes()
        m = sg.number_of_edges()
        cnots = m * 2 * layers
        iswaps = cnots * ISWAP_PER_CNOT
        iswaps_routed = int(iswaps * 1.5)
        fidelity = GATE_FIDELITY_ISWAP ** iswaps_routed
        subgraph_stats.append({
            "nodes": n,
            "edges": m,
            "iswaps": iswaps,
            "iswaps_routed": iswaps_routed,
            "fidelity": fidelity,
            "fits_qubit_limit": n <= ANKAA3_MAX_PRACTICAL,
            "fits_gate_limit": iswaps_routed <= MAX_TWO_QUBIT_GATES,
        })

    max_nodes = max(s["nodes"] for s in subgraph_stats)
    max_iswaps = max(s["iswaps_routed"] for s in subgraph_stats)
    min_fidelity = min(s["fidelity"] for s in subgraph_stats)
    all_fit_qubits = all(s["fits_qubit_limit"] for s in subgraph_stats)
    all_fit_gates = all(s["fits_gate_limit"] for s in subgraph_stats)

    return {
        "num_subgraphs": len(leaves),
        "max_size_param": max_size,
        "layers": layers,
        "subgraphs": subgraph_stats,
        "max_nodes": max_nodes,
        "max_iswaps_routed": max_iswaps,
        "min_fidelity": min_fidelity,
        "all_fit_qubits": all_fit_qubits,
        "all_fit_gates": all_fit_gates,
    }


def print_report(graph_path: str) -> None:
    """Print a full resource estimation report for a given graph."""
    G = load_graph(graph_path)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    name = Path(graph_path).stem

    print(f"\n{'='*72}")
    print(f"  RESOURCE ESTIMATION: {name}")
    print(f"  Graph: {n} nodes, {m} edges")
    print(f"{'='*72}")

    # -- Standard QAOA (infeasible for large graphs) --------------------------
    print(f"\n--- Standard QAOA (no partitioning) ---")
    for p in [1, 2]:
        est = estimate_standard_qaoa(G, layers=p)
        feasible = "YES" if (est["feasible_qubits"] and est["feasible_gates"]) else "NO"
        print(f"\n  p = {p} layers:")
        print(f"    Qubits required       : {est['qubits']:>8}   (Ankaa-3 practical limit: {ANKAA3_MAX_PRACTICAL})")
        print(f"    CNOTs per layer       : {est['cnots_per_layer']:>8}")
        print(f"    iSWAPs per layer      : {est['iswaps_per_layer']:>8}   (1 CNOT = {ISWAP_PER_CNOT} iSWAPs)")
        print(f"    Total iSWAPs          : {est['total_iswaps']:>8}")
        print(f"    + routing overhead    : {est['total_iswaps_routed']:>8}   (est. 1.5x for SWAP routing)")
        print(f"    Gate limit ({MAX_TWO_QUBIT_GATES:>3})       : {'PASS' if est['feasible_gates'] else 'FAIL -- exceeds limit'}")
        print(f"    Circuit fidelity      : {est['circuit_fidelity']:>8.6f}")
        print(f"    Feasible on Ankaa-3?  : {feasible}")

    # -- DC-QAOA (feasible) ---------------------------------------------------
    print(f"\n--- DC-QAOA (divide-and-conquer) ---")
    for max_size, p in [(8, 1), (6, 1), (10, 1)]:
        print(f"\n  max_size={max_size}, p={p}:")
        dc = estimate_dcqaoa(G, max_size=max_size, layers=p)
        feasible = "YES" if (dc["all_fit_qubits"] and dc["all_fit_gates"]) else "NO"
        print(f"    Subgraphs             : {dc['num_subgraphs']:>8}")
        print(f"    Largest subgraph      : {dc['max_nodes']:>8} nodes")
        print(f"    Max iSWAPs (routed)   : {dc['max_iswaps_routed']:>8}   (limit: {MAX_TWO_QUBIT_GATES})")
        print(f"    Gate limit ({MAX_TWO_QUBIT_GATES:>3})       : {'PASS' if dc['all_fit_gates'] else 'FAIL'}")
        print(f"    Worst-case fidelity   : {dc['min_fidelity']:>8.6f}")
        print(f"    All fit Ankaa-3?      : {feasible}")

        # Per-subgraph breakdown
        print(f"    Per-subgraph breakdown:")
        print(f"      {'#':>3}  {'Nodes':>5}  {'Edges':>5}  {'iSWAPs':>7}  {'Routed':>7}  {'Fidelity':>8}  {'OK?':>4}")
        for i, s in enumerate(dc["subgraphs"]):
            ok = "YES" if (s["fits_qubit_limit"] and s["fits_gate_limit"]) else "NO"
            print(f"      {i+1:>3}  {s['nodes']:>5}  {s['edges']:>5}  {s['iswaps']:>7}  {s['iswaps_routed']:>7}  {s['fidelity']:>8.4f}  {ok:>4}")

    # -- Summary --------------------------------------------------------------
    print(f"\n--- Summary ---")
    print(f"  Standard QAOA on {n}-node graph: INFEASIBLE")
    print(f"    - Requires {n} qubits (Ankaa-3 practical limit: ~{ANKAA3_MAX_PRACTICAL})")
    std_p1 = estimate_standard_qaoa(G, layers=1)
    print(f"    - Even p=1 needs {std_p1['total_iswaps_routed']} iSWAPs (limit: {MAX_TWO_QUBIT_GATES})")
    print(f"    - Circuit fidelity would be {std_p1['circuit_fidelity']:.6f} (unusable)")
    print(f"  DC-QAOA with max_size=8, p=1: FEASIBLE")
    dc_best = estimate_dcqaoa(G, max_size=8, layers=1)
    print(f"    - {dc_best['num_subgraphs']} subgraphs, max {dc_best['max_nodes']} qubits each")
    print(f"    - Max {dc_best['max_iswaps_routed']} iSWAPs per subgraph")
    print(f"    - Worst fidelity: {dc_best['min_fidelity']:.4f}")
    print(f"{'='*72}\n")


def main():
    if len(sys.argv) < 2:
        candidates = sorted(Path("..").glob("*.parquet"))
        if not candidates:
            print("Usage: python resource_estimation.py <graph.parquet> [graph2.parquet ...]")
            sys.exit(1)
        paths = [str(c) for c in candidates]
    else:
        paths = sys.argv[1:]

    for path in paths:
        print_report(path)


if __name__ == "__main__":
    main()
