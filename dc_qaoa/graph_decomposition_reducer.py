"""
graph_decomposition_reducer.py -- Exact implementation of Algorithm 1 from
"Graph decomposition techniques for solving combinatorial optimization problems
 with variational quantum algorithms" (Ponce et al., arXiv:2306.00494).

Algorithm (one iteration):
  1. Find minimum vertex cut K = nx.minimum_node_cut(G).
  2. Remove K; identify V1 (larger component(s)) and V2 (smallest component).
  3. For every binary assignment s ∈ {0,1}^|K|, fix K to s and brute-force
     the exact objective over {-1,+1}^|V2| on G[V2 ∪ K].  Record b[s].
     The objective uses:
       • Standard MaxCut formula  w*(1-z_u*z_v)/2  for original edges.
       • QUBO coupling  w*x_u*x_v  (x=(1-z)/2) for K-K edges from prior iters
         (marked with edge attr 'qubo'=True).
       • QUBO diagonal  bias*(1-z)/2  for node biases from prior iters.
  4. Fit a QUBO over the K nodes (paper Eq. 7):
         Σ_{i<j} Ĵ_ij * x_i x_j  +  Σ_i Ĵ_ii * x_i  +  ĉ  ≈  b[s]
     with x_i = s_i ∈ {0,1}.
     • |K| ≤ 2 → square system (n_vars = 2^|K|) → exact solve via linalg.solve
     • |K| > 2 → LP   min Σ e_s   s.t.  Ax + e = b,  e ≥ 0  (paper Eq. 8)
  5. Build G[V1 ∪ K]:
       - Remove ALL existing K-K edges (original MaxCut or prior QUBO).
       - Insert new K-K QUBO edges: weight=Ĵ_ij, attr 'qubo'=True.
       - Accumulate diagonal biases Ĵ_ii onto K node attr 'bias'.
       - Add ĉ to c_offset.

Edge attribute 'qubo' (bool, default False):
    True  → QUBO coupling: contribution = w * x_i * x_j, x=(1-z)/2
    False → standard MaxCut: contribution = w * (1-z_i*z_j)/2

Node attribute 'bias' (float, default 0):
    Accumulated Ĵ_ii; contribution = bias * (1-z)/2.

Final score = full_objective(G_reduced, z') + c_offset
where full_objective handles both edge types and biases.

Public API:
    graph_decomposition_reduce(G, M=4) -> (G_reduced, c_offset)
    full_objective(G, assignment)      -> float   [exported for benchmark]
"""
from __future__ import annotations

import itertools
from itertools import combinations

import numpy as np
import networkx as nx
from scipy.optimize import linprog

Solution = dict  # {node_id: +1 | -1}


# ---------------------------------------------------------------------------
# Objective evaluation (exported for benchmark)
# ---------------------------------------------------------------------------

def full_objective(G: nx.Graph, assignment: Solution) -> float:
    """
    Evaluate the full objective on G, respecting edge types and node biases.

    For each edge (u, v):
      • qubo=False (default): w * (1 - z_u * z_v) / 2   [MaxCut]
      • qubo=True           : w * x_u * x_v              [QUBO coupling]
                              where x = (1 - z) / 2

    For each node v with 'bias' attribute:
      • bias * (1 - z_v) / 2   [QUBO diagonal: Ĵ_ii * x_i]
    """
    score = 0.0
    for u, v, data in G.edges(data=True):
        w  = data.get("weight", 1.0)
        zu = assignment.get(u, 1)
        zv = assignment.get(v, 1)
        if data.get("qubo", False):
            xu = (1.0 - zu) / 2.0
            xv = (1.0 - zv) / 2.0
            score += w * xu * xv
        else:
            score += w * (1.0 - zu * zv) / 2.0
    for v, data in G.nodes(data=True):
        bias = data.get("bias", 0.0)
        if abs(bias) > 1e-12:
            z_v = assignment.get(v, 1)
            score += bias * (1.0 - z_v) / 2.0
    return score


# ---------------------------------------------------------------------------
# Step 3: brute-force optimal on G[V2 ∪ K] with K pinned
# ---------------------------------------------------------------------------

def _exact_fixed_K(
    subgraph: nx.Graph,
    V2_nodes: list,
    K_assignment: Solution,
) -> float:
    """
    Optimal full_objective on `subgraph` with K nodes pinned to K_assignment.
    Enumerates all 2^|V2| assignments of V2 nodes.
    """
    if not V2_nodes:
        return full_objective(subgraph, K_assignment)

    best = -float("inf")
    for bits in itertools.product([-1, 1], repeat=len(V2_nodes)):
        assignment = {**K_assignment, **dict(zip(V2_nodes, bits))}
        score = full_objective(subgraph, assignment)
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Step 4: QUBO fitting — off-diagonal Ĵ_ij, diagonal Ĵ_ii, constant ĉ
# ---------------------------------------------------------------------------

def _solve_reweighting(
    K_list: list,
    b: dict,
) -> tuple[dict, dict, float]:
    """
    Fit the QUBO over K nodes (paper Eq. 7):

        Σ_{i<j} Ĵ_ij * x_i x_j  +  Σ_i Ĵ_ii * x_i  +  ĉ  =  b[s]

    for every s ∈ {0,1}^|K| (x_i = s_i).

    n_vars = C(|K|,2) + |K| + 1

    |K| ≤ 2  →  n_vars = 2^|K|  →  square, exactly solvable (Theorem 1).
    |K| > 2  →  n_vars < 2^|K|  →  LP  min Σ e_s  s.t.  Ax+e=b, e≥0.

    Returns:
        J_hat : {(n_i, n_j): Ĵ_ij}   off-diagonal QUBO coupling → edge weights
        D_hat : {n_i: Ĵ_ii}           diagonal QUBO bias → node 'bias' attr
        c_hat : float                  constant offset for this iteration
    """
    n_K      = len(K_list)
    pairs    = list(combinations(range(n_K), 2))
    n_off    = len(pairs)            # C(n_K, 2)
    n_vars   = n_off + n_K + 1      # off-diag + diag + constant

    assignments = sorted(b.keys())
    n_s = len(assignments)           # 2^n_K

    # Design matrix A  (n_s × n_vars)
    # cols: [x_i*x_j for i<j] | [x_k for k] | [1]
    A   = np.zeros((n_s, n_vars))
    rhs = np.array([b[s] for s in assignments], dtype=float)

    for row, s in enumerate(assignments):
        for col, (i, j) in enumerate(pairs):
            A[row, col] = float(s[i] * s[j])          # off-diagonal: x_i x_j
        for k in range(n_K):
            A[row, n_off + k] = float(s[k])            # diagonal:     x_k
        A[row, -1] = 1.0                               # constant

    if n_s == n_vars:
        # |K| ≤ 2: square, exactly invertible.
        try:
            solution = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            solution, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)

    else:
        # |K| > 2: LP  min Σ e_s  s.t.  Ax + e = b,  e ≥ 0.
        n_total = n_vars + n_s
        c_obj   = np.zeros(n_total)
        c_obj[n_vars:] = 1.0

        A_eq = np.zeros((n_s, n_total))
        A_eq[:, :n_vars] = A
        A_eq[:, n_vars:] = np.eye(n_s)

        bounds = [(None, None)] * n_vars + [(0.0, None)] * n_s

        lp = linprog(c_obj, A_eq=A_eq, b_eq=rhs, bounds=bounds, method="highs")

        if not lp.success:
            solution, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
        else:
            solution = lp.x[:n_vars]

    J_hat = {
        (K_list[i], K_list[j]): float(solution[col])
        for col, (i, j) in enumerate(pairs)
        if abs(solution[col]) > 1e-12
    }
    D_hat = {K_list[k]: float(solution[n_off + k]) for k in range(n_K)}
    c_hat = float(solution[-1])
    return J_hat, D_hat, c_hat


# ---------------------------------------------------------------------------
# Main reduction loop
# ---------------------------------------------------------------------------

def graph_decomposition_reduce(
    G: nx.Graph,
    M: int = 4,
) -> tuple[nx.Graph, float]:
    """
    Iteratively reduce G via the min vertex-cut decomposition (Algorithm 1).

    Returns:
        G_reduced : Reduced graph.
                    K-K edges carry attr 'qubo'=True (QUBO coupling formula).
                    K nodes may carry attr 'bias' (QUBO diagonal term).
        c_offset  : Accumulated scalar.
                    Full approximate score = full_objective(G_reduced, z') + c_offset.
    """
    G = G.copy()
    c_offset = 0.0

    for iteration in itertools.count():
        if G.number_of_nodes() <= 1:
            print(f"[cutset] Stopping: graph has ≤1 node after {iteration} iteration(s)")
            break

        if not nx.is_connected(G):
            print(f"[cutset] Stopping: graph is disconnected at iteration {iteration}")
            break

        K = nx.minimum_node_cut(G)

        if not K:
            print("[cutset] Stopping: minimum_node_cut returned ∅")
            break

        if len(K) >= M:
            print(
                f"[cutset] Stopping: |K|={len(K)} ≥ M={M} "
                f"(2^{len(K)}={2**len(K)} assignments)"
            )
            break

        H = G.copy()
        H.remove_nodes_from(K)
        components = sorted(nx.connected_components(H), key=len, reverse=True)

        if len(components) < 2:
            print("[cutset] Stopping: K does not disconnect the graph")
            break

        V2 = components[-1]
        V1 = set().union(*components[:-1])

        K_list  = list(K)
        V2_list = list(V2)
        subgraph_V2K = G.subgraph(V2 | K)

        print(
            f"[cutset] Iter {iteration}: |G|={G.number_of_nodes()}, "
            f"|K|={len(K)}, |V1|={len(V1)}, |V2|={len(V2)}, "
            f"2^|K|={2**len(K)}, 2^|V2|={2**len(V2)}"
        )

        # Step 3: b[s] = optimal full_objective on G[V2∪K] with K pinned to s.
        # Includes K-K edges (MaxCut or QUBO) and node biases from prior iters.
        b: dict = {}
        for bits in itertools.product([0, 1], repeat=len(K_list)):
            K_assign = {K_list[i]: 1 - 2 * bits[i] for i in range(len(K_list))}
            b[bits]  = _exact_fixed_K(subgraph_V2K, V2_list, K_assign)

        # Step 4: fit QUBO.
        J_hat, D_hat, c_hat = _solve_reweighting(K_list, b)

        print(
            f"[cutset]   c_hat={c_hat:.4f}, "
            f"J_hat={[f'{w:.4f}' for w in J_hat.values()]}, "
            f"D_hat={[f'{d:.4f}' for d in D_hat.values()]}"
        )

        # Step 5: build G[V1 ∪ K].
        G_new = G.subgraph(V1 | K).copy()

        # REPLACE all K-K edges (original MaxCut or prior QUBO) with fitted Ĵ.
        for ni, nj in list(G_new.edges()):
            if ni in K and nj in K:
                G_new.remove_edge(ni, nj)

        # Add new K-K edges as QUBO coupling (qubo=True).
        for (ni, nj), w in J_hat.items():
            G_new.add_edge(ni, nj, weight=w, qubo=True)

        # REPLACE diagonal biases on K nodes (mirrors J_hat REPLACE semantics).
        # D_hat is the total QUBO diagonal for K nodes in this iteration's fit;
        # b[s] already accounts for any old bias, so we must not accumulate.
        for node, bias_val in D_hat.items():
            G_new.nodes[node]["bias"] = bias_val

        c_offset += c_hat
        G = G_new

    print(
        f"[cutset] Done: {iteration} iter(s), "
        f"c_offset={c_offset:.4f}, "
        f"|G_reduced|={G.number_of_nodes()} nodes / {G.number_of_edges()} edges"
    )
    return G, c_offset
