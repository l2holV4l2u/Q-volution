import networkx as nx
import numpy as np

from . import config as _config
from .graph import graph_compressed
from .circuit import build_qaoa_circuit

# NOTE: run_simulation and get_maxcut_params are imported lazily inside each
# function to avoid a circular import with quantum_backend.py.

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
    from .quantum_backend import get_maxcut_params  # lazy import to avoid circular dep

    _, params = get_maxcut_params(graph, method=_config.OPTIMIZER,
                                  label=f"precond:analytic-p1 [{_config.OPTIMIZER}]")

    nodes = list(graph.nodes())
    edges, n_node = graph_compressed(graph)
    gamma = params[0]
    beta  = params[1]

    Wij = nx.to_numpy_array(graph, weight="weight")
    Zij = np.zeros((n_node, n_node), dtype=float)

    def zizj(i, j):
        p_ik  = 1.0
        p_jk  = 1.0
        p_sum = 1.0
        p_diff = 1.0
        for k in range(n_node):
            if k == i or k == j:
                continue
            p_ik   *= np.cos(gamma * Wij[i, k])
            p_jk   *= np.cos(gamma * Wij[j, k])
            p_sum  *= np.cos(gamma * (Wij[i, k] + Wij[j, k]))
            p_diff *= np.cos(gamma * (Wij[j, k] - Wij[i, k]))

        s2b = np.sin(2.0 * beta)
        c2b = np.cos(2.0 * beta)
        t1  = -s2b * c2b * np.sin(gamma * Wij[i, j]) * (p_ik + p_jk)
        t2  = -(s2b ** 2 / 2.0) * (p_sum - p_diff)
        return t1 + t2

    for i in range(n_node):
        for j in range(i):
            Zij[i, j] = Zij[j, i] = -zizj(i, j)

    new_graph = nx.Graph()
    for i in range(n_node):
        for j in range(i):
            new_graph.add_edge(nodes[i], nodes[j], weight=Zij[i, j])

    return new_graph


# semi-analytic (backpropagate)
def zij_p1_backpropagate(
    graph: nx.Graph
) -> nx.Graph:

    from numpy import e

    nodes = list(graph.nodes())
    _, n_node = graph_compressed(graph)

    Wij = nx.to_numpy_array(graph, weight="weight")
    Nij = -Wij @ Wij / 2
    Zij = Wij / np.sqrt(e * n_node) + Nij / (e * n_node)

    new_graph = nx.Graph()
    for i in range(n_node):
        for j in range(i):
            new_graph.add_edge(nodes[i], nodes[j], weight=Zij[i, j])

    return new_graph


# measurement based
def zij_measurement(
    graph: nx.Graph,
) -> nx.Graph:
    """calculate <Z_i Z_j>_p by direct simulation (works for any p and mixer)."""
    from .quantum_backend import get_maxcut_params, run_simulation  # lazy import

    _, params = get_maxcut_params(graph, method=_config.OPTIMIZER,
                                  label=f"precond:measurement [{_config.OPTIMIZER}]")

    nodes = list(graph.nodes())
    edges, n_node = graph_compressed(graph)
    prog = build_qaoa_circuit(n_node, edges, _config.LAYER_COUNT, mixer_mode=_config.MIXER_MODE)
    gammas = params[:_config.LAYER_COUNT].tolist()
    betas  = params[_config.LAYER_COUNT:].tolist()
    raw = run_simulation(prog, {"gammas": gammas, "betas": betas})
    measurement_results = np.array(raw.get_register_map().get("ro"))

    Zij = np.zeros((n_node, n_node), dtype=float)
    for bits in measurement_results:
        assert n_node == len(bits)
        for i in range(len(bits)):
            zi = 1 if bits[i] == 0 else -1
            for j in range(i):
                zj = 1 if bits[j] == 0 else -1
                Zij[i, j] -= zi * zj
                Zij[j, i] -= zi * zj
    Zij /= len(measurement_results)

    new_graph = nx.Graph()
    for i in range(n_node):
        for j in range(i):
            new_graph.add_edge(nodes[i], nodes[j], weight=Zij[i, j])
    return new_graph
