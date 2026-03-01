"""
quantum_backend.py -- pyQuil QAOA backend for weighted Max-Cut.

Requires:
    pip install pyquil
    quilc -S        (Quil compiler server)
    qvm -S          (QVM simulator, for local testing)
    qcs auth login  (for real QPU access)

Public:
    setup_qpu(qc_name)
    run_quantum(subgraph, nodes) -> list[Solution]

Result store (populated after each solve, keyed by id(subgraph)):
    FINAL_PARAMETERS  -- optimal angles and run metadata
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.optimize import dual_annealing

try:
    from . import config
except ImportError:
    import config

# ── pyQuil availability ────────────────────────────────────────────────────────
try:
    from pyquil import Program, get_qc
    from pyquil.gates import H, RZ, RX, CNOT, MEASURE
    _PYQUIL_AVAILABLE = True
except ImportError:
    _PYQUIL_AVAILABLE = False

# Global quantum computer handle (set by setup_qpu)
_QC = None

# ── Result store ───────────────────────────────────────────────────────────────
FINAL_PARAMETERS: dict = {}  # id(subgraph) -> dict


def setup_qpu(qc_name: str = "8q-qvm") -> None:
    """
    Configure the pyQuil quantum computer target.

    Args:
        qc_name: Quantum computer name. Examples:
            "8q-qvm"      -- generic 8-qubit local QVM simulator
            "Ankaa-3"     -- real Rigetti Ankaa-3 QPU
            "Ankaa-9Q-3"  -- 9-qubit Ankaa QPU
            "Ankaa-3-qvm" -- QVM simulating Ankaa-3 topology

    For QVM: run `quilc -S` and `qvm -S` in separate terminals.
    For QPU: configure QCS credentials via `qcs auth login`.
    """
    global _QC
    if not _PYQUIL_AVAILABLE:
        raise RuntimeError("pyquil is not installed. Run: pip install pyquil")
    _QC = get_qc(qc_name)
    print(f"[solver] Quantum computer set -> {qc_name}")


def _build_qaoa_circuit(
    n_qubits: int,
    edges: list[tuple[int, int, float]],
    p_layers: int,
    mixer_mode: str = "X",
) -> "Program":
    """
    Build a parametric QAOA circuit for weighted Max-Cut.

    Compiled once, run many times via memory_map (parametric execution).

    Cost layer per edge (u, v, w):  CNOT(u,v) - RZ(-gamma*w, v) - CNOT(u,v)
      => implements exp(-i*gamma*w*ZZ/2), maximising the weighted cut.

    Mixer modes:
      "X"  -- standard transverse-field: RX(2*beta) per qubit
      "XX" -- graph-coupled XX mixer: exp(-i*beta*X_iX_j) per edge
      "XY" -- XY-mixer: exp(-i*beta*(X_iX_j + Y_iY_j)/2) per edge
    """
    prog   = Program()
    gammas = prog.declare("gammas", "REAL", p_layers)
    betas  = prog.declare("betas",  "REAL", p_layers)

    # Initial state: |+>^n
    for q in range(n_qubits):
        prog += H(q)

    for layer in range(p_layers):
        # Cost layer
        for (u, v, w) in edges:
            prog += CNOT(u, v)
            prog += RZ(gammas[layer] * (-w), v)
            prog += CNOT(u, v)

        # Mixer layer
        if mixer_mode == "X":
            for q in range(n_qubits):
                prog += RX(betas[layer] * 2.0, q)

        elif mixer_mode == "XX":
            for (u, v, _w) in edges:
                prog += H(u);  prog += H(v)
                prog += CNOT(u, v)
                prog += RZ(betas[layer] * 2.0, v)
                prog += CNOT(u, v)
                prog += H(u);  prog += H(v)

        elif mixer_mode == "XY":
            for (u, v, _w) in edges:
                # exp(-i * beta/2 * XX)
                prog += H(u);  prog += H(v)
                prog += CNOT(u, v)
                prog += RZ(betas[layer], v)
                prog += CNOT(u, v)
                prog += H(u);  prog += H(v)
                # exp(-i * beta/2 * YY)
                prog += RX(np.pi / 2, u);  prog += RX(np.pi / 2, v)
                prog += CNOT(u, v)
                prog += RZ(betas[layer], v)
                prog += CNOT(u, v)
                prog += RX(-np.pi / 2, u); prog += RX(-np.pi / 2, v)

        else:
            raise ValueError(f"mixer mode: {mixer_mode!r} is not supported")

    ro = prog.declare("ro", "BIT", n_qubits)
    for q in range(n_qubits):
        prog += MEASURE(q, ro[q])

    return prog


def run_quantum(subgraph: nx.Graph, nodes: list) -> list:
    """
    Run weighted QAOA via pyQuil and return sampled solutions.

    Reads runtime settings from the config module, so CLI patches apply.
    Auto-configures a QVM if setup_qpu() was not called.
    """
    global _QC

    n          = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}
    edges: list[tuple[int, int, float]] = [
        (node_index[u], node_index[v], float(data.get("weight", 1.0)))
        for u, v, data in subgraph.edges(data=True)
    ]

    if _QC is None:
        qc_name = f"{n}q-qvm"
        _QC = get_qc(qc_name)
        print(f"[solver] Auto-configured QVM: {qc_name}")

    # Compile parametric circuit once
    prog       = _build_qaoa_circuit(n, edges, config.LAYER_COUNT, mixer_mode=config.MIXER_MODE)
    executable = _QC.compile(prog.wrap_in_numshots_loop(config.SHOTS))

    # Simulated annealing optimisation
    def objective(params):
        gammas_val = params[:config.LAYER_COUNT].tolist()
        betas_val  = params[config.LAYER_COUNT:].tolist()
        result     = _QC.run(executable, memory_map={"gammas": gammas_val, "betas": betas_val})
        bitstrings = np.array(result.get_register_map().get("ro"))
        total_cut  = sum(
            w
            for shot in range(len(bitstrings))
            for (u, v, w) in edges
            if bitstrings[shot, u] != bitstrings[shot, v]
        )
        return -(total_cut / len(bitstrings))

    bounds     = [(-np.pi, np.pi)] * (2 * config.LAYER_COUNT)
    result_opt = dual_annealing(
        objective,
        bounds=bounds,
        maxiter=config.SA_MAXITER,
        seed=config.SEED,
    )
    best_cut    = -result_opt.fun
    best_params = result_opt.x

    print(f"[solver] QAOA best E[cut]: {best_cut:.4f}")

    gammas_opt = best_params[:config.LAYER_COUNT].tolist()
    betas_opt  = best_params[config.LAYER_COUNT:].tolist()

    FINAL_PARAMETERS[id(subgraph)] = {
        "gammas":      gammas_opt,
        "betas":       betas_opt,
        "best_e_cut":  best_cut,
        "mixer_mode":  config.MIXER_MODE,
        "n_layers":    config.LAYER_COUNT,
        "n_qubits":    n,
    }
    print(
        f"[solver] Optimal params | "
        f"gammas={[f'{g:.4f}' for g in gammas_opt]} | "
        f"betas={[f'{b:.4f}' for b in betas_opt]}"
    )

    # Final sample at optimal parameters
    result     = _QC.run(executable, memory_map={"gammas": gammas_opt, "betas": betas_opt})
    bitstrings = np.array(result.get_register_map().get("ro"))

    return [
        {nodes[i]: (1 if bitstrings[shot, i] == 0 else -1) for i in range(n)}
        for shot in range(len(bitstrings))
    ]
