from __future__ import annotations

import random
import itertools
import os
os.environ["QCS_SETTINGS_APPLICATIONS_QVM_URL"] = "http://127.0.0.1:5001"
import numpy as np
import networkx as nx

from scorer import maxcut_score

Solution = dict

# -- User-facing knobs --------------------------------------------------------
USE_PYQUIL   = False   # set True when pyquil is installed & QVM/QPU ready
MIXER_MODE   = "X"     # "X" (standard), "XX" (graph-coupled), "XY" (XY-mixer)
LAYER_COUNT  = 1       # QAOA depth p  (p=1 for noisy hardware; increase for sim)
SHOTS        = 1024    # measurement shots per circuit run
SEED         = 42
NUM_STARTS   = 5       # multi-start COBYLA initializations
# ------------------------------------------------------------------------------
try:
    from pyquil import Program, get_qc
    from pyquil.gates import H, RZ, RX, RY, CNOT, MEASURE
    _PYQUIL_AVAILABLE = True
except ImportError:
    _PYQUIL_AVAILABLE = False

#set by setup_qpu
_QC = None

def setup_qpu(qc_name: str) -> None:
    """Set the global quantum computer reference."""
    global _QC
    if not _PYQUIL_AVAILABLE:
        raise RuntimeError("pyquil is not installed. Cannot set up QPU.")
    _QC = get_qc(qc_name)
    print(f"[solver] Using quantum computer: {qc_name}")

def qaoa_solve(subgraph: nx.Graph, top_t: int = 10) -> list[Solution]:
    """Solve Max-Cut on `subgraph`; return the top-t solutions by cut value."""
    nodes = list(subgraph.nodes())
    if not nodes:
        return [{}]
    
    if USE_PYQUIL:
        if not _PYQUIL_AVAILABLE:
            raise RuntimeError("USE_PYQUIL=True but pyquil is not installed.")
        raw = _pyquil_backend(subgraph, nodes) # solving with qaoa
    else:
        raw = _stub_backend(subgraph, nodes, top_t)
        print("no qaoa")
    
    # Deduplicate and rank by score (best first)
    seen: set = set()
    ranked: list[tuple[float, Solution]] = []
    for sol in raw:
        key = tuple(sol.get(v, 1) for v in nodes)
        if key not in seen:
            seen.add(key)
            ranked.append((maxcut_score(subgraph, sol), sol))
    
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [sol for _, sol in ranked[:top_t]]

GATES_PER_CNOT = 3       # CNOT -> ~3 native two-qubit gates (CZ/iSWAP)
ROUTING_OVERHEAD = 1.5
MAX_TWO_QUBIT_GATES = 100  # target limit for acceptable fidelity on Ankaa-3

def estimate_native_2q_count(num_edges: int, layer_count: int) -> int:
    """Estimate native two-qubit gate count for Ankaa-3."""
    cnots = num_edges * 2 * layer_count  # CNOT-Rz-CNOT per edge per layer
    native_2q = cnots * GATES_PER_CNOT
    return int(native_2q * ROUTING_OVERHEAD)



def _ring_edges(n_qubits: int) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []
    return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

def _prepare_initial_X(prog: "Program", n_qubits: int) -> None:
    for q in range(n_qubits):
        prog += H(q)

def _prepare_initial_XX(prog: "Program", n_qubits: int) -> None:
    for q in range(n_qubits):
        prog += H(q)

def _prepare_initial_XY(prog: "Program", n_qubits: int) -> None:
    k = n_qubits // 2
    if k == 0:
        # Edge case: 0 or 1 qubit — leave as |0>
        return
    for q in range(k):
        prog += XGATE(q)

_INITIAL_STATE_PREP = {
    "X":  _prepare_initial_X,
    "XX": _prepare_initial_XX,
    "XY": _prepare_initial_XY,
}


def _build_qaoa_program(
    n_qubits: int,
    edges: list[tuple[int, int, float]],
    p_layers: int,
    mixer_mode: str = "X",
) -> "Program":
    """
    Parametric QAOA circuit for weighted Max-Cut.

    Cost layer:   CNOT–RZ(−γw)–CNOT per problem edge  →  exp(+iγw/2 · ZZ)
    Mixer layer:  ring topology (n edges) for XX and XY modes.
    Initial state: mode-dependent (see _prepare_initial_* functions).
    """
    if mixer_mode not in _INITIAL_STATE_PREP:
        raise ValueError(
            f"Unknown mixer_mode: {mixer_mode!r}. "
            f"Choose from {list(_INITIAL_STATE_PREP)}"
        )

    prog = Program()
    gammas = prog.declare("gammas", "REAL", p_layers)
    betas  = prog.declare("betas",  "REAL", p_layers)

    # Mode-specific initial state
    _INITIAL_STATE_PREP[mixer_mode](prog, n_qubits)

    # Ring edges shared by XX and XY mixers
    ring = _ring_edges(n_qubits)

    for layer in range(p_layers):

        # ------------------------------------------------------------------
        # Cost layer — problem graph edges (unchanged)
        # ------------------------------------------------------------------
        for (u, v, w) in edges:
            prog += CNOT(u, v)
            prog += RZ(gammas[layer] * (-w), v)
            prog += CNOT(u, v)

        # ------------------------------------------------------------------
        # Mixer layer — ring topology
        # ------------------------------------------------------------------
        if mixer_mode == "X":
            # Standard transverse-field: exp(-i β Σ_j X_j) = Π_j RX(2β, j)
            # Single-qubit only — ring topology not needed.
            for q in range(n_qubits):
                prog += RX(betas[layer] * 2.0, q)

        elif mixer_mode == "XX":
            # exp(-i β X_i X_j) per ring edge
            # H⊗H diagonalises XX as ZZ, then CNOT–RZ–CNOT implements ZZ:
            #   H–H–CNOT–RZ(2β)–CNOT–H–H  →  exp(-i β X_i X_j)
            for (u, v) in ring:
                prog += H(u)
                prog += H(v)
                prog += CNOT(u, v)
                prog += RZ(betas[layer] * 2.0, v)
                prog += CNOT(u, v)
                prog += H(u)
                prog += H(v)

        elif mixer_mode == "XY":
            # exp(-i β (X_iX_j + Y_iY_j) / 2) per ring edge.
            #
            # [X⊗X, Y⊗Y] = 0  ✓  →  exact Trotter split:
            #   exp(-i β/2 XX) · exp(-i β/2 YY)
            #
            # XX term:  H⊗H  conjugates ZZ → XX
            #   H–H–CNOT–RZ(β)–CNOT–H–H
            #
            # YY term:  RX(π/2)⊗RX(π/2)  conjugates ZZ → YY
            #   RX(π/2)–RX(π/2)–CNOT–RZ(β)–CNOT–RX(-π/2)–RX(-π/2)
            for (u, v) in ring:
                # exp(-i β/2 · X_u X_v)
                prog += H(u)
                prog += H(v)
                prog += CNOT(u, v)
                prog += RZ(betas[layer], v)
                prog += CNOT(u, v)
                prog += H(u)
                prog += H(v)
                # exp(-i β/2 · Y_u Y_v)
                prog += RX( np.pi / 2, u)
                prog += RX( np.pi / 2, v)
                prog += CNOT(u, v)
                prog += RZ(betas[layer], v)
                prog += CNOT(u, v)
                prog += RX(-np.pi / 2, u)
                prog += RX(-np.pi / 2, v)

    # Measurement
    ro = prog.declare("ro", "BIT", n_qubits)
    for q in range(n_qubits):
        prog += MEASURE(q, ro[q])

    return prog

def _pyquil_backend(subgraph: nx.Graph, nodes: list) -> list[Solution]:
    """
    Run weighted QAOA via pyQuil and return sampled solutions.

    Requires `setup_qpu()` to have been called, or uses default QVM.
    """
    global _QC

    n = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}

    # Build 0-based qubit-indexed edge list
    edges: list[tuple[int, int, float]] = []
    for u, v, data in subgraph.edges(data=True):
        edges.append((
            node_index[u],
            node_index[v],
            float(data.get("weight", 1.0)),
        ))

    # Gate count check
    est_2q = estimate_native_2q_count(len(edges), LAYER_COUNT)
    status = "OK" if est_2q <= MAX_TWO_QUBIT_GATES else f"WARNING: exceeds {MAX_TWO_QUBIT_GATES} limit"
    print(f"[solver] Gate estimate: {n} qubits, {len(edges)} edges, "
          f"p={LAYER_COUNT} -> ~{est_2q} native 2Q gates ({status})")

    # Set up quantum computer if not already configured
    if _QC is None:
        qc_name = f"{n}q-qvm"
        _QC = get_qc(qc_name)
        print(f"[solver] Auto-configured QVM: {qc_name}")

    # Build and compile parametric circuit (compile once)
    prog = _build_qaoa_program(n, edges, LAYER_COUNT, mixer_mode=MIXER_MODE)
    prog_shots = prog.wrap_in_numshots_loop(SHOTS)
    executable = _QC.compile(prog_shots)

    # Multi-start COBYLA optimization
    parameter_count = 2 * LAYER_COUNT
    rng = np.random.default_rng(SEED)

    best_cut = -float("inf")
    best_params = None

    for trial in range(NUM_STARTS):
        x0 = rng.uniform(-np.pi, np.pi, parameter_count)

        def objective(params, _edges=edges, _exe=executable):
            gammas_val = params[:LAYER_COUNT].tolist()
            betas_val = params[LAYER_COUNT:].tolist()

            result = _QC.run(
                _exe,
                memory_map={
                    "gammas": gammas_val,
                    "betas": betas_val,
                },
            )
            bitstrings = np.array(result.get_register_map().get("ro"))

            # Compute average cut value over all shots
            total_cut = 0.0
            for shot in range(len(bitstrings)):
                for (u, v, w) in _edges:
                    if bitstrings[shot, u] != bitstrings[shot, v]:
                        total_cut += w
            avg_cut = total_cut / len(bitstrings)
            return -avg_cut  # minimize negative cut = maximize cut

        from scipy.optimize import minimize as scipy_minimize
        result = scipy_minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": 200, "rhobeg": 0.5},
        )

        trial_cut = -result.fun
        if trial_cut > best_cut:
            best_cut = trial_cut
            best_params = result.x

    print(f"[solver] QAOA best E[cut]: {best_cut:.4f} (from {NUM_STARTS} starts)")

    # Sample the optimal circuit
    gammas_opt = best_params[:LAYER_COUNT].tolist()
    betas_opt = best_params[LAYER_COUNT:].tolist()

    result = _QC.run(
        executable,
        memory_map={
            "gammas": gammas_opt,
            "betas": betas_opt,
        },
    )
    bitstrings = np.array(result.get_register_map().get("ro"))

    # Convert bitstrings to Solution dicts (0 -> +1 spin, 1 -> -1 spin)
    solutions: list[Solution] = []
    for shot in range(len(bitstrings)):
        sol: Solution = {
            nodes[i]: (1 if bitstrings[shot, i] == 0 else -1)
            for i in range(n)
        }
        solutions.append(sol)

    return solutions


# ---------------------------------------------------------------------------
# Stub backend -- for local testing without a QPU or QVM
# ---------------------------------------------------------------------------

def _stub_backend(
    subgraph: nx.Graph, nodes: list, top_t: int
) -> list[Solution]:
    """Exact brute-force for <=24 nodes; random + local search otherwise."""
    if len(nodes) <= 24:
        return _brute_force(nodes)
    return _random_local_search(subgraph, nodes, samples=max(200, top_t * 20))


def _brute_force(nodes: list) -> list[Solution]:
    return [
        dict(zip(nodes, bits))
        for bits in itertools.product([-1, 1], repeat=len(nodes))
    ]


def _random_local_search(
    subgraph: nx.Graph, nodes: list, samples: int
) -> list[Solution]:
    results = []
    for _ in range(samples):
        sol = {v: random.choice([-1, 1]) for v in nodes}
        sol = _local_search(subgraph, nodes, sol)
        results.append(sol)
    return results


def _local_search(subgraph: nx.Graph, nodes: list, sol: Solution) -> Solution:
    """Greedy single-flip hill-climb using O(deg(v)) delta scoring."""
    improved = True
    while improved:
        improved = False
        for v in nodes:
            # Delta score: how much the cut changes if we flip v
            delta = 0.0
            zv = sol[v]
            for u in subgraph.neighbors(v):
                w = subgraph[v][u].get("weight", 1.0)
                zu = sol.get(u, 1)
                delta += w * zv * zu
            if delta > 1e-12:  # flipping v improves the cut
                sol[v] *= -1
                improved = True
    return sol

