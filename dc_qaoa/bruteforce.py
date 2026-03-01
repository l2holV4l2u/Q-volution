"""
bruteforce.py -- Find optimal / upper-bound Max-Cut score.

  Dataset A (21 nodes): exact brute-force (2^21 ~ 2M, ~50s)
  Dataset B (180 nodes): SDP relaxation upper bound + simulated annealing best-known

Usage:
  python bruteforce.py ../dataset_A.parquet
  python bruteforce.py ../dataset_B.parquet
"""
import itertools
import math
import random
import time
import sys
from pathlib import Path

import numpy as np
import networkx as nx

try:
    from graph_loader import load_graph
    from scorer import maxcut_score
except ImportError:
    from .graph_loader import load_graph
    from .scorer import maxcut_score


# ---------------------------------------------------------------------------
# Exact brute-force (small graphs only)
# ---------------------------------------------------------------------------

def exact_bruteforce(G: nx.Graph) -> tuple[dict, float]:
    nodes = sorted(G.nodes())
    n = len(nodes)
    best_score = -1.0
    best_sol = None
    for bits in itertools.product([-1, 1], repeat=n):
        sol = dict(zip(nodes, bits))
        s = maxcut_score(G, sol)
        if s > best_score:
            best_score = s
            best_sol = sol
    return best_sol, best_score


# ---------------------------------------------------------------------------
# SDP relaxation upper bound (Goemans-Williamson)
# ---------------------------------------------------------------------------

def sdp_upper_bound(G: nx.Graph) -> float:
    """
    Solve the SDP relaxation of Max-Cut to get an upper bound.
    The SDP optimal value >= true Max-Cut optimal.
    """
    import cvxpy as cp

    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    # SDP variable: n x n positive semidefinite matrix with diag = 1
    X = cp.Variable((n, n), symmetric=True)

    # Objective: maximize (1/4) * sum w_ij * (1 - X_ij)  [equivalent to max-cut SDP]
    # Which equals (1/2) * sum w_ij * (1 - X_ij) / 2  ... wait
    # Standard: max (1/2) sum w_ij (1 - x_i . x_j) = max (1/2) sum w_ij (1 - X_ij)
    obj = 0
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        i, j = idx[u], idx[v]
        obj = obj + w * (1 - X[i, j]) / 2

    constraints = [X >> 0]  # positive semidefinite
    for i in range(n):
        constraints.append(X[i, i] == 1)  # diagonal = 1

    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)

    return prob.value


# ---------------------------------------------------------------------------
# Simulated annealing (best heuristic solution)
# ---------------------------------------------------------------------------

def simulated_annealing(
    G: nx.Graph,
    T_start: float = 100.0,
    T_end: float = 0.001,
    steps: int = 500_000,
    restarts: int = 10,
) -> tuple[dict, float]:
    """Simulated annealing for Max-Cut with multiple restarts."""
    nodes = list(G.nodes())
    n = len(nodes)
    adj = {v: [(u, G[v][u].get("weight", 1.0)) for u in G.neighbors(v)] for v in nodes}

    global_best_sol = None
    global_best_score = -1.0

    for _ in range(restarts):
        sol = {v: random.choice([-1, 1]) for v in nodes}
        score = maxcut_score(G, sol)
        best_sol = dict(sol)
        best_score = score

        for step in range(steps):
            T = T_start * (T_end / T_start) ** (step / steps)
            v = nodes[random.randint(0, n - 1)]

            # Delta score from flipping v
            delta = 0.0
            zv = sol[v]
            for u, w in adj[v]:
                delta += w * zv * sol[u]

            if delta > 0 or random.random() < math.exp(delta / T):
                sol[v] *= -1
                score += delta
                if score > best_score:
                    best_score = score
                    best_sol = dict(sol)

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_sol = best_sol

    return global_best_sol, global_best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)
    np.random.seed(42)

    if len(sys.argv) > 1:
        graph_path = sys.argv[1]
    else:
        graph_path = str(Path("..") / "dataset_A.parquet")

    G = load_graph(graph_path)
    nodes = sorted(G.nodes())
    n = len(nodes)
    total_weight = sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True))

    print(f"Nodes: {n}  |  Edges: {G.number_of_edges()}  |  Total weight: {total_weight:.2f}")
    print("=" * 60)

    if n <= 24:
        # Exact brute-force
        print(f"\nBrute-forcing 2^{n} = {2**n:,} assignments...")
        t0 = time.time()
        best_sol, best_score = exact_bruteforce(G)
        elapsed = time.time() - t0
        print(f"EXACT optimal: {best_score:.2f} / {total_weight:.2f} = {best_score/total_weight:.6f}")
        print(f"Time: {elapsed:.1f}s")
    else:
        # SDP upper bound
        print(f"\nSDP relaxation upper bound (cvxpy)...")
        t0 = time.time()
        sdp_val = sdp_upper_bound(G)
        elapsed = time.time() - t0
        print(f"SDP upper bound: {sdp_val:.2f} / {total_weight:.2f} = {sdp_val/total_weight:.6f}")
        print(f"Time: {elapsed:.1f}s")
        print("(True optimal <= SDP bound)")

        # Simulated annealing best-known
        print(f"\nSimulated annealing (10 restarts x 500k steps)...")
        t0 = time.time()
        best_sol, best_score = simulated_annealing(G)
        elapsed = time.time() - t0
        print(f"SA best score:  {best_score:.2f} / {total_weight:.2f} = {best_score/total_weight:.6f}")
        print(f"Time: {elapsed:.1f}s")

        # Summary
        print("\n" + "=" * 60)
        print(f"{'Method':<25} {'Score':>10} {'Ratio':>10}")
        print("-" * 60)
        print(f"{'SDP upper bound':<25} {sdp_val:>10.2f} {sdp_val/total_weight:>10.6f}")
        print(f"{'Simulated annealing':<25} {best_score:>10.2f} {best_score/total_weight:>10.6f}")
        print(f"{'Total weight':<25} {total_weight:>10.2f} {1.0:>10.6f}")
        print("=" * 60)
        gap = sdp_val - best_score
        print(f"Gap (SDP - SA): {gap:.2f}  ({gap/total_weight*100:.2f}%)")
        print("If gap is small, SA solution is near-optimal.")


if __name__ == "__main__":
    main()
