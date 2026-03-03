"""
benchmark.py -- Compare DC-QAOA pipeline against standard Max-Cut baselines.

Baselines:
  1. Random assignment       -- expected C(z) = 0.5 * total_weight
  2. Greedy construction     -- assign each node to maximize marginal cut gain
  3. NetworkX Kernighan-Lin  -- greedy bisection heuristic (problem statement baseline)
  4. Random (500 trials)     -- best of many independent random starts
  5. DC-QAOA pipeline (classical backend) -- the full divide-and-conquer pipeline
  6. Graph-Decomposition+QAOA -- exact paper QUBO reduction + classical/QAOA solve
       Implements Algorithm 1 from Ponce et al. arXiv:2306.00494.
       QUBO fitting: Σ Ĵ_ij x_i x_j + Σ Ĵ_ii x_i + ĉ = C_max^s
       Score = MaxCut-edges(G_reduced, z') + Σ bias_v*(1-z_v)/2 + c_offset

For reference, the theoretical best classical guarantee is:
  - Goemans-Williamson SDP: 0.878 approximation ratio (requires cvxpy)
  - Exact brute-force optimal (exponential, only feasible for small graphs)
"""
from __future__ import annotations

import random
import time
import sys
from pathlib import Path

import networkx as nx

# Ensure benchmark uses the local workspace package implementation first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dc_qaoa import config as qconfig
from dc_qaoa.graph import load_graph
from dc_qaoa.solver import maxcut_score
from dc_qaoa.graph_decomposition_reducer import graph_decomposition_reduce, full_objective

# -- Pipeline config -----------------------------------------------------------
MAX_SIZE  = 8    # max subgraph size passed to DC-QAOA partitioner
TOP_T     = 10   # top-t solutions kept per subgraph for merge
CUTSET_M  = 4    # stop reduction when |min-vertex-cut| >= M  (2^M assignments)
# ------------------------------------------------------------------------------


def random_assignment(G: nx.Graph, trials: int = 1000) -> tuple[dict, float]:
    """Best of many random spin assignments."""
    best_sol = {}
    best_score = -1.0
    for _ in range(trials):
        sol = {v: random.choice([-1, 1]) for v in G.nodes()}
        s = maxcut_score(G, sol)
        if s > best_score:
            best_score = s
            best_sol = sol
    return best_sol, best_score


def greedy_construction(G: nx.Graph) -> tuple[dict, float]:
    """
    Greedy node-by-node assignment: place each node in the partition
    that maximizes the marginal cut increase.
    """
    sol = {}
    # Sort nodes by weighted degree (highest first)
    nodes_by_deg = sorted(G.nodes(), key=lambda v: sum(
        G[v][u].get("weight", 1.0) for u in G.neighbors(v)
    ), reverse=True)

    for v in nodes_by_deg:
        # Try both spins, pick the one that gives higher marginal cut
        score_pos = 0.0
        score_neg = 0.0
        for u in G.neighbors(v):
            if u in sol:
                w = G[v][u].get("weight", 1.0)
                zu = sol[u]
                score_pos += w * (1 - 1 * zu) / 2   # z_v = +1
                score_neg += w * (1 - (-1) * zu) / 2  # z_v = -1
        sol[v] = 1 if score_pos >= score_neg else -1

    return sol, maxcut_score(G, sol)


def random_best_of(G: nx.Graph, trials: int = 500) -> tuple[dict, float]:
    """Best of many independent random spin assignments."""
    nodes = list(G.nodes())
    best_sol = {}
    best_score = -1.0
    for _ in range(trials):
        sol = {v: random.choice([-1, 1]) for v in nodes}
        s = maxcut_score(G, sol)
        if s > best_score:
            best_score = s
            best_sol = sol
    return best_sol, best_score


def nx_one_exchange(G: nx.Graph) -> tuple[dict, float]:
    """
    NetworkX one_exchange greedy heuristic for Max-Cut.
    This is the baseline suggested by the problem statement.
    """
    partition = nx.algorithms.community.kernighan_lin_bisection(G, weight="weight")
    # Convert partition to spin assignment
    sol = {}
    for v in G.nodes():
        sol[v] = 1 if v in partition[0] else -1
    return sol, maxcut_score(G, sol)


def _qubo_eval(G_reduced: nx.Graph, assignment: dict) -> float:
    """
    Full QUBO objective on G_reduced via full_objective from the reducer.

    Handles edge types correctly:
      • qubo=False edges → MaxCut formula  w*(1-z_u*z_v)/2
      • qubo=True edges  → QUBO coupling   w*x_u*x_v, x=(1-z)/2
      • node 'bias' attr → QUBO diagonal   bias*(1-z)/2
    """
    return full_objective(G_reduced, assignment)


def graph_decomposition_qaoa(G: nx.Graph, M: int = CUTSET_M) -> tuple[dict, float, dict]:
    """
    Exact paper algorithm (Ponce et al. arXiv:2306.00494) + classical solve.

    Reduces G via QUBO-fitted vertex-cut decomposition, then finds the
    optimal assignment on G_reduced by evaluating the full QUBO objective:
        score = MaxCut-edges(G_reduced, z') + Σ bias_v*(1-z_v)/2 + c_offset

    For |G_reduced| ≤ 20: exact brute-force over all 2^n assignments.
    For |G_reduced| > 20: rank solve_subgraph candidates by QUBO objective.

    Returns:
        assignment : best spin dict over G_reduced nodes
        score      : qubo_score + c_offset  ≈  MaxCut(G, z*)
        info       : dict with qubo_score, c_offset, n_reduced,
                     reduction_pct, max_qubits
    """
    import itertools
    from dc_qaoa.solver import solve_subgraph

    n_original = G.number_of_nodes()
    G_reduced, c_offset = graph_decomposition_reduce(G, M=M)
    n_reduced = G_reduced.number_of_nodes()
    nodes = list(G_reduced.nodes())

    if n_reduced <= 20:
        # Exact brute-force over the full QUBO objective.
        best_score = -float("inf")
        best = {}
        for bits in itertools.product([-1, 1], repeat=n_reduced):
            assignment = dict(zip(nodes, bits))
            s = _qubo_eval(G_reduced, assignment)
            if s > best_score:
                best_score = s
                best = assignment
    else:
        # Heuristic: rank solve_subgraph candidates by full QUBO objective.
        candidates = solve_subgraph(G_reduced, top_t=20)
        best_score = -float("inf")
        best = {}
        for sol in candidates:
            s = _qubo_eval(G_reduced, sol)
            if s > best_score:
                best_score = s
                best = sol

    total_score = best_score + c_offset

    info = {
        "qaoa_score":    best_score,
        "c_offset":      c_offset,
        "n_reduced":     n_reduced,
        "max_qubits":    n_reduced,
        "reduction_pct": (1.0 - n_reduced / n_original) * 100.0,
    }
    return best, total_score, info


def main():
    random.seed(42)
    # Benchmark compares against classical baselines; keep backend deterministic and fast.
    qconfig.USE_QUANTUM = False

    # Find graph
    if len(sys.argv) > 1:
        graph_path = sys.argv[1]
    else:
        candidates = sorted(Path("..").glob("*.parquet"))
        if not candidates:
            print("Usage: python benchmark.py <graph.parquet>")
            sys.exit(1)
        graph_path = str(candidates[0])

    print(f"Graph: {graph_path}")
    print("=" * 70)

    G = load_graph(graph_path)
    total_weight = sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True))
    n = G.number_of_nodes()
    m = G.number_of_edges()

    print(f"Nodes: {n}  |  Edges: {m}  |  Total weight: {total_weight:.2f}")
    print(f"Avg degree: {2*m/n:.2f}  |  Density: {2*m/(n*(n-1)):.4f}")
    print("=" * 70)

    results = []

    # --- Baseline 1: Random ---
    print("\n[1/6] Random assignment (1000 trials)...")
    t0 = time.time()
    _, score_rand = random_assignment(G, trials=1000)
    t_rand = time.time() - t0
    results.append(("Random (best of 1000)", score_rand, t_rand, {}))

    # --- Baseline 2: Greedy ---
    print("[2/6] Greedy construction...")
    t0 = time.time()
    _, score_greedy = greedy_construction(G)
    t_greedy = time.time() - t0
    results.append(("Greedy construction", score_greedy, t_greedy, {}))

    # --- Baseline 3: NetworkX Kernighan-Lin (problem statement baseline) ---
    print("[3/6] NetworkX Kernighan-Lin bisection...")
    t0 = time.time()
    _, score_nx = nx_one_exchange(G)
    t_nx = time.time() - t0
    results.append(("NX Kernighan-Lin", score_nx, t_nx, {}))

    # --- Baseline 4: Random (500 trials) ---
    print("[4/6] Random assignment (500 trials)...")
    t0 = time.time()
    _, score_rls = random_best_of(G, trials=500)
    t_rls = time.time() - t0
    results.append(("Random (500 trials)", score_rls, t_rls, {}))

    # --- DC-QAOA pipeline ---
    from dc_qaoa.pipeline import run_pipeline
    print("[5/6] DC-QAOA pipeline (classical backend)...")
    t0 = time.time()
    _, score_dcqaoa = run_pipeline(graph_path, max_size=MAX_SIZE, top_t=TOP_T)
    t_dcqaoa = time.time() - t0
    results.append(("DC-QAOA (classical)", score_dcqaoa, t_dcqaoa,
                    {"max_qubits": MAX_SIZE,
                     "reduction_pct": (1.0 - MAX_SIZE / n) * 100.0}))

    # --- Graph decomposition + QAOA (exact paper algorithm) ---
    print(f"[6/6] Graph-Decomp+QAOA (Ponce et al., M={CUTSET_M}, QUBO fitting)...")
    t0 = time.time()
    _, score_cutset, cutset_info = graph_decomposition_qaoa(G, M=CUTSET_M)
    t_cutset = time.time() - t0
    results.append(("Graph-Decomp+QAOA (paper)", score_cutset, t_cutset, cutset_info))

    # --- Theoretical bounds ---
    expected_random = total_weight * 0.5
    gw_bound = total_weight * 0.878  # Goemans-Williamson guarantee

    # --- Print results ---
    print("\n" + "=" * 80)
    print(f"{'Method':<30} {'Score':>10} {'Ratio':>7} {'Qubits':>7} {'Reduc%':>7} {'Time':>7}")
    print("-" * 80)
    print(f"{'Theoretical random (E[C])':30} {expected_random:10.2f} {0.5:7.4f} {'—':>7} {'—':>7} {'—':>7}")
    print(f"{'GW SDP bound':30} {gw_bound:10.2f} {0.878:7.4f} {'—':>7} {'—':>7} {'—':>7}")
    print("-" * 80)
    for name, score, elapsed, info in results:
        ratio       = score / total_weight
        qubits_str  = str(info["max_qubits"])    if "max_qubits"    in info else "—"
        reduc_str   = f"{info['reduction_pct']:.1f}%" if "reduction_pct" in info else "—"
        print(f"{name:30} {score:10.2f} {ratio:7.4f} {qubits_str:>7} {reduc_str:>7} {elapsed:6.2f}s")

        # Extra detail line for reduced-graph run
        if "c_offset" in info:
            print(
                f"  {'':28} qubo_score={info['qaoa_score']:.2f}  "
                f"c_offset={info['c_offset']:.2f}  "
                f"n_reduced={info['n_reduced']}"
            )
    print("-" * 80)
    print(f"{'Total edge weight':30} {total_weight:10.2f} {1.0:7.4f}")
    print("=" * 80)

    # --- Verdict ---
    best_name, best_score, _, _ = max(results, key=lambda x: x[1])
    best_ratio = best_score / total_weight
    print(f"\nBest method: {best_name}")
    print(f"Best score:  {best_score:.2f} / {total_weight:.2f} = {best_ratio:.4f}")

    if best_ratio >= 0.95:
        print("Verdict: EXCELLENT -- near-optimal solution")
    elif best_ratio >= 0.90:
        print("Verdict: VERY GOOD -- strong approximation")
    elif best_ratio >= 0.878:
        print("Verdict: GOOD -- meets GW guarantee")
    elif best_ratio >= 0.80:
        print("Verdict: FAIR -- below GW guarantee, room for improvement")
    else:
        print("Verdict: POOR -- significantly below expectations")


if __name__ == "__main__":
    main()
