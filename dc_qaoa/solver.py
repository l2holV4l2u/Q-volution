"""
solver.py -- QAOA solver for weighted Max-Cut on a subgraph.

Public API:
    setup_qpu(qc_name)                    # call once before pipeline
    qaoa_solve(subgraph, top_t) -> list[Solution]

Two backends -- switch via USE_PYQUIL flag:

  False (default) -- stub: exact brute-force <=24 nodes, random+local-search
                    otherwise. No hardware needed. Use this to verify the
                    full pipeline locally before submitting to QPU.

  True            -- pyQuil + QCS SDK (Rigetti). Requires:
                       pip install pyquil
                    Uses weighted QAOA circuit with CNOT-Rz-CNOT cost layers,
                    Rx mixer, multi-start COBYLA (5 inits).
                    The Quil compiler (quilc) handles decomposition to native
                    gates (CZ/iSWAP + RZ + RX(pi/2)) automatically.

                    For local simulation: run `quilc -S` and `qvm -S` servers.
                    For QPU: configure QCS credentials via `qcs auth login`.

pyQuil circuit design notes:
  - Weighted cost layer: CNOT - Rz(gamma*w) - CNOT per edge (u,v,w).
    Implements exp(-i*gamma*w*ZZ/2), which maximises the weighted cut.
  - Mixer: Rx(2*beta) per node -- standard transverse-field mixer.
  - The Quil compiler (quilc) decomposes CNOT to native CZ/iSWAP gates.
  - Parametric circuit: angles declared as REAL memory, compiled once,
    run many times with different memory_map values.
"""
from __future__ import annotations

import random
import itertools

import numpy as np
import networkx as nx

from .scorer import maxcut_score

Solution = dict  # {node_id: +1 | -1}

# -- User-facing knobs --------------------------------------------------------
USE_PYQUIL   = False   # set True when pyquil is installed & QVM/QPU ready
MIXER_MODE   = "X"     # "X" (standard), "XX" (graph-coupled), "XY" (XY-mixer)
LAYER_COUNT  = 1       # QAOA depth p  (p=1 for noisy hardware; increase for sim)
SHOTS        = 1024    # measurement shots per circuit run
SEED         = 42
NUM_STARTS   = 5       # multi-start COBYLA initializations
# ------------------------------------------------------------------------------

# -- Result stores populated by _pyquil_backend after each subgraph solve -----
# Keyed by id(subgraph) so every DC-QAOA leaf gets its own record.
#
# OPTIMIZATION_HISTORY[id(subgraph)] -> list[list[float]]
#   Outer list : one entry per COBYLA trial  (len = NUM_STARTS)
#   Inner list : E[cut] at every objective function evaluation within that trial
#   → used by the loss-curve notebook cell to plot convergence
#
# FINAL_PARAMETERS[id(subgraph)] -> dict
#   "gammas"     list[float]  optimal cost angles  γ_1 … γ_p
#   "betas"      list[float]  optimal mixer angles β_1 … β_p
#   "best_e_cut" float        E[cut] at optimal params
#   "best_trial" int          which multi-start trial won  (0-indexed)
#   "mixer_mode" str
#   "n_layers"   int          LAYER_COUNT (p)
#   "n_qubits"   int
OPTIMIZATION_HISTORY: dict = {}
FINAL_PARAMETERS:     dict = {}

# -- Try importing pyQuil ------------------------------------------------------
try:
    from pyquil import Program, get_qc
    from pyquil.gates import H, RZ, RX, RY, CNOT, MEASURE
    _PYQUIL_AVAILABLE = True
except ImportError:
    _PYQUIL_AVAILABLE = False

# Global quantum computer reference (set by setup_qpu)
_QC = None


# ---------------------------------------------------------------------------
# QPU / QVM configuration (call once before running the pipeline)
# ---------------------------------------------------------------------------

def setup_qpu(qc_name: str = "8q-qvm") -> None:
    """
    Configure the pyQuil quantum computer target.

    Args:
        qc_name: Quantum computer name. Examples:
            "8q-qvm"                  -- generic 8-qubit local QVM simulator
            "Ankaa-3"                 -- real Rigetti Ankaa-3 QPU
            "Ankaa-9Q-3"              -- 9-qubit Ankaa QPU
            "Ankaa-3-qvm"             -- QVM simulating Ankaa-3 topology

    For QVM simulation: run `quilc -S` and `qvm -S` in separate terminals.
    For QPU access: configure QCS credentials via `qcs auth login`.
    """
    global _QC
    if not _PYQUIL_AVAILABLE:
        raise RuntimeError(
            "pyquil is not installed. Run: pip install pyquil"
        )
    _QC = get_qc(qc_name)
    print(f"[solver] Quantum computer set -> {qc_name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def qaoa_solve(subgraph: nx.Graph, top_t: int = 10) -> list[Solution]:
    """
    Solve Max-Cut on `subgraph`; return the top-t solutions by cut value.

    Each solution is {node_id: +1 | -1}.
    Solutions are deduplicated and sorted best-first.
    """
    nodes = list(subgraph.nodes())
    if not nodes:
        return [{}]

    if USE_PYQUIL:
        if not _PYQUIL_AVAILABLE:
            raise RuntimeError(
                "USE_PYQUIL=True but pyquil is not installed. "
                "Run: pip install pyquil"
            )
        raw = _pyquil_backend(subgraph, nodes)
    else:
        raw = _stub_backend(subgraph, nodes, top_t)

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


# ---------------------------------------------------------------------------
# Gate count estimation (Ankaa-3: CZ/iSWAP native gate set)
# ---------------------------------------------------------------------------

# On Rigetti Ankaa-3, each CNOT decomposes to ~2-3 native two-qubit gates
# (CZ or iSWAP family). The Quil compiler (quilc) handles this automatically.
# The QAOA cost layer uses CNOT-Rz-CNOT per edge = 2 CNOTs.
# With SWAP routing overhead (~1.5x for non-adjacent qubits on square grid),
# total native 2Q gates ~ num_edges * 2 * GATES_PER_CNOT * routing * layers.

GATES_PER_CNOT = 3       # CNOT -> ~3 native two-qubit gates (CZ/iSWAP)
ROUTING_OVERHEAD = 1.5
MAX_TWO_QUBIT_GATES = 100  # target limit for acceptable fidelity on Ankaa-3


def estimate_native_2q_count(num_edges: int, layer_count: int) -> int:
    """Estimate native two-qubit gate count for Ankaa-3."""
    cnots = num_edges * 2 * layer_count  # CNOT-Rz-CNOT per edge per layer
    native_2q = cnots * GATES_PER_CNOT
    return int(native_2q * ROUTING_OVERHEAD)


# ---------------------------------------------------------------------------
# Real QAOA backend -- pyQuil + QCS SDK
# ---------------------------------------------------------------------------

def _build_qaoa_program(
    n_qubits: int,
    edges: list[tuple[int, int, float]],
    p_layers: int,
    mixer_mode: str = "X",
    preconditioned = False,
) -> "Program":
    """
    Build a parametric QAOA circuit for weighted Max-Cut.

    Uses DECLARE'd parameters (gammas, betas) so the circuit can be compiled
    once and run many times with different parameter values via memory_map.

    Cost layer per edge (u,v,w):  CNOT(u,v) - RZ(gamma*w, v) - CNOT(u,v)

    Mixer modes:
      "X"  -- standard transverse-field: RX(2*beta) per qubit
      "XX" -- graph-coupled XX: exp(-i*beta*X_iX_j) for each edge (i,j)
      "XY" -- XY-mixer: exp(-i*beta*(X_iX_j + Y_iY_j)/2) for each edge (i,j)
    """
    prog = Program()

    # Declare parameter memory regions
    gammas = prog.declare("gammas", "REAL", p_layers)
    betas = prog.declare("betas", "REAL", p_layers)

    # Initial state: |+>^n
    for q in range(n_qubits):
        prog += H(q)

    # QAOA layers
    for layer in range(p_layers):
        for (u, v, w) in edges:
            prog += CNOT(u, v)
            prog += RZ(gammas[layer] * (-w), v)
            prog += CNOT(u, v)

        # Mixing layer
        if mixer_mode == "X":
            for q in range(n_qubits):
                prog += RX(betas[layer] * 2.0, q)

        elif mixer_mode == "XX":
            for (u, v, _w) in edges:
                prog += H(u)
                prog += H(v)
                prog += CNOT(u, v)
                prog += RZ(betas[layer] * 2.0, v)
                prog += CNOT(u, v)
                prog += H(u)
                prog += H(v)

        elif mixer_mode == "XY":
            #   Rx(π/2)-Rx(π/2)-CNOT-RZ(beta)-CNOT-Rx(-π/2)-Rx(-π/2)
            for (u, v, _w) in edges:
                # exp(-i * beta/2 * XX)
                prog += H(u)
                prog += H(v)
                prog += CNOT(u, v)
                prog += RZ(betas[layer], v)
                prog += CNOT(u, v)
                prog += H(u)
                prog += H(v)
                # exp(-i * beta/2 * YY)
                prog += RX(np.pi / 2, u)
                prog += RX(np.pi / 2, v)
                prog += CNOT(u, v)
                prog += RZ(betas[layer], v)
                prog += CNOT(u, v)
                prog += RX(-np.pi / 2, u)
                prog += RX(-np.pi / 2, v)

        else:
            raise ValueError(f"mixer mode: {mixer_mode!r} is not supported")

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
    best_trial_idx = -1
    all_trials_history: list[list[float]] = []

    for trial in range(NUM_STARTS):
        trial_history: list[float] = []
        x0 = rng.uniform(-np.pi, np.pi, parameter_count)

        def objective(params, _edges=edges, _exe=executable, _hist=trial_history):
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
            _hist.append(avg_cut)        # record E[cut] at this function eval
            return -avg_cut  # minimize negative cut = maximize cut

        from scipy.optimize import minimize as scipy_minimize
        result = scipy_minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": 200, "rhobeg": 0.5},
        )

        all_trials_history.append(trial_history)
        trial_cut = -result.fun
        if trial_cut > best_cut:
            best_cut = trial_cut
            best_params = result.x
            best_trial_idx = trial

    # Store per-trial E[cut] traces for loss-curve plotting
    OPTIMIZATION_HISTORY[id(subgraph)] = all_trials_history

    print(f"[solver] QAOA best E[cut]: {best_cut:.4f} (from {NUM_STARTS} starts)")

    # Decompose flat parameter vector into named per-layer lists
    gammas_opt = best_params[:LAYER_COUNT].tolist()
    betas_opt  = best_params[LAYER_COUNT:].tolist()

    # Store final optimised parameters
    FINAL_PARAMETERS[id(subgraph)] = {
        "gammas":      gammas_opt,
        "betas":       betas_opt,
        "best_e_cut":  best_cut,
        "best_trial":  best_trial_idx,
        "mixer_mode":  MIXER_MODE,
        "n_layers":    LAYER_COUNT,
        "n_qubits":    n,
    }
    print(
        f"[solver] Optimal params | "
        f"gammas={[f'{g:.4f}' for g in gammas_opt]} | "
        f"betas={[f'{b:.4f}' for b in betas_opt]} | "
        f"trial {best_trial_idx + 1}/{NUM_STARTS}"
    )

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