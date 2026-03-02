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

import time
import numpy as np
import networkx as nx
from scipy.optimize import minimize, dual_annealing, differential_evolution

# ── pyQuil availability ────────────────────────────────────────────────────────
try:
    from pyquil import Program, get_qc
    _PYQUIL_AVAILABLE = True
except ImportError:
    _PYQUIL_AVAILABLE = False
    
from . import config as _config
from .cost_function import *
from .circuit import _build_qaoa_circuit

# Global quantum computer handle (set by setup_qpu)
_QC = None

# Data types
from typing import Literal
type Solution = dict[int, Literal[1, -1]]  # {node_id: +1 | -1}
type Solutions = list[Solution]

# ── Result store ───────────────────────────────────────────────────────────────
FINAL_PARAMETERS: dict = {}  # id(subgraph) -> dict
PARAMS_PATHS: dict = {}       # id(subgraph) -> parameter space's trajectory (list of dicts)
LOSS_HISTORY: dict = {}      # id(subgraph) -> list[float]

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
    

def run_quantum(subgraph: nx.Graph, nodes: list, method="SA", precondition=False, pcond_layer=1) -> list[dict]:
    """
    Run weighted QAOA via pyQuil and return sampled solutions. Retun list of dictionary

    Reads runtime settings from the config module, so CLI patches apply.
    Auto-configures a QVM if setup_qpu() was not called.
    """
    global _QC

    n          = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}
    
    if precondition:
        # pcond_edges = 
        pass
    else:
        edges = [
            (node_index[u], node_index[v], float(data.get("weight", 1.0)))
            for u, v, data in subgraph.edges(data=True)
        ]

    if _QC is None:
        qc_name = f"{n}q-qvm"
        _QC = get_qc(qc_name)
        print(f"[solver] Auto-configured QVM: {qc_name}")
        
    subgraph_id = id(subgraph)
    LOSS_HISTORY[subgraph_id] = []
    PARAMS_PATHS[subgraph_id] = []

    # Compile parametric circuit once
    prog = _build_qaoa_circuit(n, edges, config.LAYER_COUNT, mixer_mode=config.MIXER_MODE)
    executable = _QC.compile(prog.wrap_in_numshots_loop(config.SHOTS))

    # calculate objective for optimization
    iter_count = {"k": 0}
    last_eval = {"x": None, "loss": None, "nfev": 0}

    def cost_func_estimator(params):
        gammas_val = params[:config.LAYER_COUNT].tolist()
        betas_val  = params[config.LAYER_COUNT:].tolist()
        result     = _QC.run(executable, memory_map={"gammas": gammas_val, "betas": betas_val})
        bitstrings = np.array(result.get_register_map().get("ro"))
        scores = [qaoa_cut_score(edges, bitstring) for bitstring in bitstrings]
        loss = -np.average(scores)
        last_eval["x"] = np.array(params, dtype=float, copy=True)
        last_eval["loss"] = float(loss)
        last_eval["nfev"] += 1

        LOSS_HISTORY[id(subgraph)].append(float(loss))
        PARAMS_PATHS[id(subgraph)].append(last_eval["x"].tolist())

        return loss
    
    # callback functions
    def cb_dual_annealing(xk, f, context):
        iter_count["k"] += 1
        PARAMS_PATHS[id(subgraph)].append({
            "iter": iter_count["k"],
            "nfev": last_eval["nfev"],
            "loss": last_eval["loss"],
            "params": np.array(xk, dtype=float, copy=True),
            "context": int(context),
        })
        return False
    
    def cb_minimize(xk):
        iter_count["k"] += 1
        PARAMS_PATHS[id(subgraph)].append({
            "iter": iter_count["k"],
            "nfev": last_eval["nfev"],
            "loss": last_eval["loss"],
            "params": np.array(xk, dtype=float, copy=True),
        })

    def cb_differential_evolution(xk, convergence):
        iter_count["k"] += 1
        PARAMS_PATHS[id(subgraph)].append({
            "iter": iter_count["k"],
            "nfev": last_eval["nfev"],
            "loss": last_eval["loss"],
            "params": np.array(xk, dtype=float, copy=True),
            "convergence": float(convergence),
        })
        return False

    # arguments for scipy.optimize functions
    bounds = [(-np.pi/2, np.pi/2)] * (2 * config.LAYER_COUNT)
    rng = np.random.default_rng(config.SEED)
    z0 = rng.uniform(-np.pi/2, np.pi/2, 2 * config.LAYER_COUNT)
    
    t_start = time.perf_counter()
    print(f"using {method}...")
    match method:
        case "SA":
            result = dual_annealing(
                cost_func_estimator,
                bounds=bounds,
                maxiter=config.MAXITER,
                callback=cb_dual_annealing,
            )

        case "DE":
            result = differential_evolution(
                cost_func_estimator,
                bounds=bounds,
                maxiter=config.MAXITER,
                callback=cb_differential_evolution,
            )

        case "SLSQP":
            result = minimize(
                cost_func_estimator,
                z0,
                method="SLSQP",
                options={"maxiter": config.MAXITER},
                callback=cb_minimize,
                tol=1e-1,
            )
        case "COBYLA": 
            result = minimize(
                cost_func_estimator,
                z0,
                method="COBYLA",
                options={"maxiter": _config.MAXITER},
                callback=cb_minimize,
                tol=1e-1,
            )
        case "COBYQA": 
            result = minimize(
                cost_func_estimator,
                z0,
                method="COBYQA",
                options={"maxiter": _config.MAXITER},
                callback=cb_minimize,
                tol=1e-1,
            )
        case _:
            raise ValueError(f"optimizer \"f{method}\" is not supported")
    t_end = time.perf_counter()
    print(f"[optimizer] time elapsed with optimizer {method}: {t_end-t_start:.2f}")
    
    cut_opt    = -result.fun
    params_opt = result.x

    print(f"[solver] QAOA best E[cut]: {cut_opt:.4f}")

    gammas_opt = params_opt[:config.LAYER_COUNT].tolist()
    betas_opt  = params_opt[config.LAYER_COUNT:].tolist()

    FINAL_PARAMETERS[id(subgraph)] = {
        "gammas":      gammas_opt,
        "betas":       betas_opt,
        "best_e_cut":  cut_opt,
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
    result_final = _QC.run(executable, memory_map={"gammas": gammas_opt, "betas": betas_opt})
    bitstrings = np.array(result_final.get_register_map().get("ro"))

    # encode bit 0 to 1 and 1 to -1
    return [
        {nodes[i]: (1 if bitstrings[shot, i] == 0 else -1) for i in range(n)}
        for shot in range(len(bitstrings))
    ]
