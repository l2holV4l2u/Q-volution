import networkx as nx
import numpy as np

from . import config as _config
from pyquil import get_qc
from dc_qaoa.graph import edges
from solver import solve_maxcut
from .graph import graph_compressed
from .circuit import build_qaoa_circuit
from .quantum_backend import run_simulation, get_maxcut_params

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _matrix_to_graph(nodes: list, Z: np.ndarray, tol: float = 1e-2) -> nx.Graph:
    Gz = nx.Graph()
    n = len(nodes)
    for i in range(n):
        for j in range(i):
            w = float(Z[i, j])
            if abs(w) > tol:
                Gz.add_edge(nodes[i], nodes[j], weight=w)
    return Gz

# =============================================================================
# TWO-POINT CORRELATION
# =============================================================================

# semi-analytic (p = 1)
def zij_p1_analytical(
    graph: nx.Graph
) -> nx.Graph:
    """
    Semi-analytical <Z_i Z_j>_{p=1} from Appendix H, Eq. (H1).

    Works for any weighted N-variable QUBO adjacency matrix W.
    Complexity: O(N) per pair.

    Formula:
        <Z_i Z_j> =
          -sin(2b)cos(2b) sin(g W_ij) [prod_{k!=i,j} cos(g W_ik) + prod_{k!=i,j} cos(g W_jk)]
          - sin^2(2b)/2 [prod_{k!=i,j} cos(g(W_ik+W_jk)) - prod_{k!=i,j} cos(g(W_jk-W_ik))]
    """
    
    _, params = get_maxcut_params(graph, use_quantum=_config.USE_QUANTUM)
    
    nodes = list(graph.nodes())
    edges, n_node = graph_compressed(graph)
    prog = build_qaoa_circuit(n_node, edges, 1, mixer_mode=_config.MIXER_MODE)
    gamma = params[0]
    beta  = params[1]

    Wij = nx.to_numpy_array(graph, weight="weight")
    Zij = np.zeros((n_node, n_node), dtype=float)
    
    def zizj(i, j):
        for k in range(n_node):
            if k == i or k == j:
                continue
            p_ik  *= np.cos(gamma * Wij[i, k])
            p_jk  *= np.cos(gamma * Wij[j, k])
            p_sum *= np.cos(gamma * (Wij[i, k] + Wij[j, k]))
            p_diff *= np.cos(gamma * (Wij[j, k] - Wij[i, k]))

        s2b = np.sin(2.0 * beta)
        c2b = np.cos(2.0 * beta)
        t1  = -s2b * c2b * np.sin(gamma * Wij[i, j]) * (p_ik + p_jk)
        t2  = -(s2b ** 2 / 2.0) * (p_sum - p_diff)
        return t1 + t2
    
    for i in range(len(n_node)):
        for j in range(i):
            Zij[i,j] = Zij[j,i] = -zizj(i, j)
            
    new_graph = nx.Graph()
    for i in range(n_node):
        for j in range(i):
            new_graph.add_edge(nodes[i], nodes[j], weight=Zij)
            
    return new_graph

# semi-analytic (backpropagate)
def zij_p1_backpropagate(
    graph: nx.Graph
) -> nx.Graph:
    
    from numpy import e
    
    nodes = list(graph.nodes())
    _, n_node = graph_compressed(graph)

    Wij = nx.to_numpy_array(graph, weight="weight")
    Nij = -Wij @ Wij/2
    Zij = Wij/np.sqrt(e * n_node) + Nij/(e * n_node)
    
    new_graph = nx.Graph()
    for i in range(n_node):
        for j in range(i):
            new_graph.add_edge(nodes[i], nodes[j], weight=Zij)
            
    return new_graph
    

# measurement based
def zij_measurement(
    graph: nx.Graph,
) -> nx.Graph:
    """calculate <Z_i Z_j>_p by direct simulation (works for any p and mixer)."""
    _, params = get_maxcut_params(graph, use_quantum=_config.USE_QUANTUM)
    
    nodes = list(graph.nodes())
    edges, n_node = graph_compressed(graph)
    prog = build_qaoa_circuit(n_node, edges, _config.LAYER_COUNT, mixer_mode=_config.MIXER_MODE)
    gammas = params[:_config.LAYER_COUNT].tolist()
    betas  = params[_config.LAYER_COUNT:].tolist()
    measurement_results = run_simulation(prog, {"gammas": gammas, "betas": betas})
    
    assert n_node == len(bits)
    Zij = np.zeros((n_node, n_node), dtype=float)
    for bits in measurement_results:
        for i in range(len(bits)):
            zi = 1 if bits[i] == 0 else -1
            for j in range(i):
                zj = 1 if bits[j] == 0 else -1
                Zij[i,j] -= zi * zj
                Zij[j,i] -= zi * zj
    Zij /= n_node
    
    new_graph = nx.Graph()
    for i in range(n_node):
        for j in range(i):
            new_graph.add_edge(nodes[i], nodes[j], weight= Zij[i,j])
    return new_graph

# light cone
# def light_cone_subgraph(
#     G: nx.Graph, i, j, p: int
# ) -> nx.Graph:
#     """
#     Return the light-cone induced subgraph for pair (i, j) at depth p.

#     A p-layer QAOA only couples each qubit to its p-hop neighbourhood.
#     Therefore <Z_i Z_j>_p depends only on the union of those neighbourhoods
#     (Appendix C).  For the sparse MPES graph (avg degree ~2.5) this gives
#     subgraphs of at most 14 nodes at p=1 and 27 nodes at p=2 — well within
#     exact statevector capabilities.
#     """
#     cone_i = nx.ego_graph(G, i, radius=p)
#     cone_j = nx.ego_graph(G, j, radius=p)
#     return G.subgraph(set(cone_i.nodes()) | set(cone_j.nodes())).copy()

# def zizj_light_cone(
#     G: nx.Graph,
#     params: list[float],
#     p_layers: int = 1,
#     mixer: str = "X",
# ) -> nx.Graph:
#     """
#     Return nx.Graph with weights (δij-1)<ZiZj> = -<ZiZj> (off-diagonal).
#     """
#     nodes = sorted(G.nodes())
#     N     = len(nodes)
#     idx   = {v: i for i, v in enumerate(nodes)}

#     # We'll fill <ZiZj> into ZZ, then return Z_mat = -ZZ off-diagonal.
#     ZZ = np.zeros((N, N), dtype=float)

#     for a in range(N):
#         for b in range(a + 1, N):
#             ni, nj = nodes[a], nodes[b]
#             LC = light_cone_subgraph(G, ni, nj, p_layers)

#             lc_nodes = sorted(LC.nodes())
#             lc_idx   = {v: k for k, v in enumerate(lc_nodes)}
#             n_lc     = len(lc_nodes)

#             li = lc_idx.get(ni)
#             lj = lc_idx.get(nj)
#             if li is None or lj is None:
#                 continue  # shouldn't happen since LC includes both, but safe

#             lc_edges = [
#                 (lc_idx[u], lc_idx[v], float(d.get("weight", 1.0)))
#                 for u, v, d in LC.edges(data=True)
#             ]

#             if p_layers == 1 and mixer == "X":
#                 # expects val = <ZiZj> on that subgraph
#                 val = _zzij_p1_analytical_from_edges(lc_edges, n_lc, li, lj, params[0], params[1])
#             else:
#                 # expects val = <ZiZj> on that subgraph
#                 val = _zzij_statevector(lc_edges, n_lc, li, lj, params, p_layers, mixer)

#             ZZ[a, b] = val
#             ZZ[b, a] = val

#     Z_mat = -ZZ
#     np.fill_diagonal(Z_mat, 0.0)  # (δij-1) factor

#     return _matrix_to_graph(nodes, Z_mat)