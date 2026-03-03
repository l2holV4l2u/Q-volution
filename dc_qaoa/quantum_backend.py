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

from typing import Any, TypeAlias
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
from .precondition import *
from .graph import graph_compressed
from .circuit import build_qaoa_circuit
from .cost_function import *

# Global quantum computer handle (set by setup_qpu)
_QC = None

# Data types
from typing import Literal
Solution: TypeAlias = dict[int, Literal[1, -1]]  # {node_id: +1 | -1}
Solutions: TypeAlias = list[Solution]

# ── Result store ───────────────────────────────────────────────────────────────
FINAL_PARAMETERS: dict = {}  # id(subgraph) -> dict
PARAMS_PATHS: dict = {}       # id(subgraph) -> parameter space's trajectory (list of dicts)
LOSS_HISTORY: dict = {}       # id(subgraph) -> list[float]
ITER_LOSS_HISTORY: dict = {}  # id(subgraph) -> list[float] (callback/iteration-level loss)
LOSS_LABELS: dict = {}        # id(subgraph) -> human-readable label for plotting

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

def run_simulation(prog: Program, memory_map, n: int = None) -> Any:
    global _QC
    if _QC is None:
        if n is None:
            raise RuntimeError("QVM not configured. Call setup_qpu() first.")
        qc_name = f"{n}q-qvm"
        _QC = get_qc(qc_name)
        print(f"[solver] Auto-configured QVM: {qc_name}")

    executable = _QC.compile(prog.wrap_in_numshots_loop(_config.SHOTS))
    return _QC.run(executable, memory_map)

def get_maxcut_params(subgraph: nx.Graph, method="SA", precondition=None, label: str = None) -> list[dict]:
    """
    Run weighted QAOA via pyQuil and return sampled solutions. Retun list of dictionary

    Reads runtime settings from the config module, so CLI patches apply.
    Auto-configures a QVM if setup_qpu() was not called.
    """
    global _QC
    edges, n = graph_compressed(subgraph)
    if _QC is None:
        qc_name = f"{n}q-qvm"
        _QC = get_qc(qc_name)
        print(f"[solver] Auto-configured QVM: {qc_name}")

    z0 = None
    p  = _config.LAYER_COUNT

    if precondition is not None:
        match precondition:
            case "measurement":     subgraph = zij_measurement(subgraph)
            case "analytic-p1":     subgraph = zij_p1_analytical(subgraph)
            case "back-propagate":  subgraph = zij_p1_backpropagate(subgraph)
            # case "light-cone":  subgraph = zizj_light_cone(subgraph)

    subgraph_id = id(subgraph)
    LOSS_HISTORY[subgraph_id] = []
    ITER_LOSS_HISTORY[subgraph_id] = []
    PARAMS_PATHS[subgraph_id] = []
    LOSS_LABELS[subgraph_id] = label or method

    # Compile parametric circuit once
    prog = build_qaoa_circuit(n, edges, _config.LAYER_COUNT, mixer_mode=_config.MIXER_MODE)

    # calculate objective for optimization
    iter_count = {"k": 0}
    last_eval = {"x": None, "loss": None, "nfev": 0}

    def cost_func_estimator(params):
        gammas_val = params[:_config.LAYER_COUNT].tolist()
        betas_val  = params[_config.LAYER_COUNT:].tolist()
        result     = run_simulation(prog, memory_map={"gammas": gammas_val, "betas": betas_val}, n=n)
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
        ITER_LOSS_HISTORY[id(subgraph)].append(float(f))
        PARAMS_PATHS[id(subgraph)].append({
            "iter": iter_count["k"],
            "nfev": last_eval["nfev"],
            "loss": float(f),
            "params": np.array(xk, dtype=float, copy=True),
            "context": int(context),
        })
        return False
    
    def cb_minimize(xk):
        iter_count["k"] += 1
        if last_eval["loss"] is not None:
            ITER_LOSS_HISTORY[id(subgraph)].append(float(last_eval["loss"]))
        PARAMS_PATHS[id(subgraph)].append({
            "iter": iter_count["k"],
            "nfev": last_eval["nfev"],
            "loss": last_eval["loss"],
            "params": np.array(xk, dtype=float, copy=True),
        })

    def cb_differential_evolution(xk, convergence):
        iter_count["k"] += 1
        if last_eval["loss"] is not None:
            ITER_LOSS_HISTORY[id(subgraph)].append(float(last_eval["loss"]))
        PARAMS_PATHS[id(subgraph)].append({
            "iter": iter_count["k"],
            "nfev": last_eval["nfev"],
            "loss": last_eval["loss"],
            "params": np.array(xk, dtype=float, copy=True),
            "convergence": float(convergence),
        })
        return False

    # arguments for scipy.optimize functions
    bounds = [(-np.pi/2, np.pi/2)] * (2 * _config.LAYER_COUNT)
    if z0 is None:
        rng = np.random.default_rng(_config.SEED)
        z0 = rng.uniform(-np.pi/2, np.pi/2, 2 * _config.LAYER_COUNT)
    
    t_start = time.perf_counter()
    match method:
        case "SA":
            result = dual_annealing(
                cost_func_estimator,
                bounds=bounds,
                maxiter=_config.MAXITER,
                callback=cb_dual_annealing,
            )

        case "DE":
            result = differential_evolution(
                cost_func_estimator,
                bounds=bounds,
                maxiter=_config.MAXITER,
                callback=cb_differential_evolution,
            )

        case "SLSQP":
            result = minimize(
                cost_func_estimator,
                z0,
                method="SLSQP",
                options={"maxiter": _config.MAXITER},
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

    gammas_opt = params_opt[:_config.LAYER_COUNT].tolist()
    betas_opt  = params_opt[_config.LAYER_COUNT:].tolist()

    FINAL_PARAMETERS[id(subgraph)] = {
        "gammas":      gammas_opt,
        "betas":       betas_opt,
        "best_e_cut":  cut_opt,
        "mixer_mode":  _config.MIXER_MODE,
        "n_layers":    _config.LAYER_COUNT,
        "n_qubits":    n,
    }
    print(
        f"[solver] Optimal params | "
        f"gammas={[f'{g:.4f}' for g in gammas_opt]} | "
        f"betas={[f'{b:.4f}' for b in betas_opt]}"
    )
    
    return cut_opt, params_opt

def run_quantum(subgraph: nx.Graph, nodes: list, precondition):
    _lbl = f"DC-QAOA [{_config.OPTIMIZER}]"
    if precondition:
        _lbl += f" + {precondition}"
    cut_opt, params_opt = get_maxcut_params(subgraph, method=_config.OPTIMIZER, precondition=precondition, label=_lbl)

    edges, n = graph_compressed(subgraph)
    gammas_opt = params_opt[:_config.LAYER_COUNT].tolist()
    betas_opt  = params_opt[_config.LAYER_COUNT:].tolist()

    # Final sample at optimal parameters
    prog = build_qaoa_circuit(n, edges, _config.LAYER_COUNT, mixer_mode=_config.MIXER_MODE)
    executable = _QC.compile(prog.wrap_in_numshots_loop(_config.SHOTS))
    result_final = _QC.run(executable, memory_map={"gammas": gammas_opt, "betas": betas_opt})
    bitstrings = np.array(result_final.get_register_map().get("ro"))

    print(f"[result] cut_opt    = {cut_opt:.6f}")
    print(f"[result] params_opt = {params_opt.tolist()}")

    # encode bit 0 to 1 and 1 to -1
    return [
        {nodes[i]: (1 if bitstrings[shot, i] == 0 else -1) for i in range(n)}
        for shot in range(len(bitstrings))
    ]
