"""
benchmark.py -- Compare DC-QAOA pipeline against standard Max-Cut baselines.

Baselines:
  1. Random assignment       -- expected C(z) = 0.5 * total_weight
  2. Greedy construction     -- assign each node to maximize marginal cut gain
  3. NetworkX one_exchange   -- greedy heuristic from NetworkX (problem statement baseline)
  4. Random + local search   -- many random starts polished with hill-climbing
  5. DC-QAOA pipeline (classical backend) -- the full divide-and-conquer pipeline

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

from dc_qaoa.graph_loader import load_graph
from dc_qaoa.solver import maxcut_score
from dc_qaoa.solver import local_search

# -- Pipeline config -----------------------------------------------------------
MAX_SIZE = 84   # max subgraph size passed to DC-QAOA partitioner
TOP_T    = 10   # top-t solutions kept per subgraph for merge
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

    # Polish with local search
    sol = local_search(G, list(G.nodes()), dict(sol))
    return sol, maxcut_score(G, sol)


def random_local_search(G: nx.Graph, trials: int = 500) -> tuple[dict, float]:
    """Many random starts, each polished with greedy hill-climbing."""
    nodes = list(G.nodes())
    best_sol = {}
    best_score = -1.0
    for _ in range(trials):
        sol = {v: random.choice([-1, 1]) for v in nodes}
        sol = local_search(G, nodes, sol)
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
    # Polish with our local search for fair comparison
    sol = local_search(G, list(G.nodes()), sol)
    return sol, maxcut_score(G, sol)


def main():
    random.seed(42)

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
    print("\n[1/5] Random assignment (1000 trials)...")
    t0 = time.time()
    _, score_rand = random_assignment(G, trials=1000)
    t_rand = time.time() - t0
    results.append(("Random (best of 1000)", score_rand, t_rand))

    # --- Baseline 2: Greedy ---
    print("[2/5] Greedy construction + local search...")
    t0 = time.time()
    _, score_greedy = greedy_construction(G)
    t_greedy = time.time() - t0
    results.append(("Greedy + local search", score_greedy, t_greedy))

    # --- Baseline 3: NetworkX one_exchange (problem statement baseline) ---
    print("[3/5] NetworkX Kernighan-Lin bisection + local search...")
    t0 = time.time()
    _, score_nx = nx_one_exchange(G)
    t_nx = time.time() - t0
    results.append(("NX Kernighan-Lin + LS", score_nx, t_nx))

    # --- Baseline 4: Random + local search ---
    print("[4/5] Random + local search (500 trials)...")
    t0 = time.time()
    _, score_rls = random_local_search(G, trials=500)
    t_rls = time.time() - t0
    results.append(("Random+LS (500 trials)", score_rls, t_rls))

    # --- DC-QAOA pipeline ---
    from dc_qaoa.pipeline import run_pipeline
    print("[5/5] DC-QAOA pipeline (classical backend)...")
    t0 = time.time()
    _, score_dcqaoa = run_pipeline(graph_path, max_size=MAX_SIZE, top_t=TOP_T)
    t_dcqaoa = time.time() - t0
    results.append(("DC-QAOA pipeline (classical)", score_dcqaoa, t_dcqaoa))

    # --- Theoretical bounds ---
    expected_random = total_weight * 0.5
    gw_bound = total_weight * 0.878  # Goemans-Williamson guarantee

    # --- Print results ---
    print("\n" + "=" * 70)
    print(f"{'Method':<30} {'Score':>12} {'Ratio':>8} {'Time':>8}")
    print("-" * 70)
    print(f"{'Theoretical random (E[C])':30} {expected_random:12.2f} {0.5:8.4f} {'---':>8}")
    print(f"{'GW SDP lower bound':30} {gw_bound:12.2f} {0.878:8.4f} {'---':>8}")
    print("-" * 70)
    for name, score, elapsed in results:
        ratio = score / total_weight
        print(f"{name:30} {score:12.2f} {ratio:8.4f} {elapsed:7.2f}s")
    print("-" * 70)
    print(f"{'Total edge weight':30} {total_weight:12.2f} {1.0:8.4f}")
    print("=" * 70)

    # --- Verdict ---
    best_name, best_score, _ = max(results, key=lambda x: x[1])
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
