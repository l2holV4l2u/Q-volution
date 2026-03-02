import numpy as np
import networkx as nx

from pyquil import Program, get_qc
from pyquil.gates import Gate, H, RZ, RX, CNOT, MEASURE
from pyquil.quil import MemoryReference

from .graph_loader import edges

# QUANTUM GATES
def MEASURE_ZZ(counts, i, j):
    total_shots = sum(counts.values())
    expectation = 0.0
    for bitstring, count in counts.items():
        xi = int(bitstring[-(i+1)])
        xj = int(bitstring[-(j+1)])
        eigenvalue = (-1) ** (xi + xj)
        expectation += (count / total_shots) * eigenvalue
    return expectation

def U_C(edges: edges, param: MemoryReference) -> tuple[Gate]:
    gate = ()
    for (u, v, w) in edges:
        gate += CNOT(u, v)
        gate += RZ(param * (-w), v)
        gate += CNOT(u, v)
    return gate

def U_X(n_qubits : int, param: MemoryReference) -> tuple[Gate]:
    gate = ()
    for q in range(n_qubits):
        gate += RX(param * 2.0, q)
    return gate

def U_XX(edges: edges, param: MemoryReference) -> tuple[Gate]:
    gate = ()
    for (u, v, _w) in edges:
        gate += H(u)
        gate += H(v)
        gate += CNOT(u, v)
        gate += RZ(param * 2.0, v)
        gate += CNOT(u, v)
        gate += H(u)
        prog += H(v)
    return gate

def U_XY(edges: edges, param: MemoryReference) -> tuple[Gate]:
    gate = ()
    for (u, v, _) in edges:
        # exp(-i * beta/2 * XX)
        gate += H(u);  gate += H(v)
        gate += CNOT(u, v)
        gate += RZ(param, v)
        gate += CNOT(u, v)
        gate += H(u);  gate += H(v)
        
        # exp(-i * beta/2 * YY)
        gate += RX(np.pi / 2, u);  gate += RX(np.pi / 2, v)
        gate += CNOT(u, v)
        gate += RZ(param, v)
        gate += CNOT(u, v)
        gate += RX(-np.pi / 2, u); gate += RX(-np.pi / 2, v)
    
    return gate

def _build_qaoa_circuit(
    n_qubits: int,
    edges: list[tuple[int, int, float]],
    p_layers: int,
    mixer_mode: str = "X",
) -> Program:
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

    # QAOA gate layer
    for layer in range(p_layers):
        # Cost layer
        prog += U_C(edges, gammas[layer])

        # Mixer layer
        if mixer_mode == "X": prog += U_X(n_qubits, betas[layer])
        elif mixer_mode == "XX": prog += U_XX(edges, betas)
        elif mixer_mode == "XY": prog += U_XY(edges, betas)
        else:
            raise ValueError(f"mixer mode: {mixer_mode!r} is not supported")

    # measurement
    ro = prog.declare("ro", "BIT", n_qubits)
    for q in range(n_qubits):
        prog += MEASURE(q, ro[q])

    return prog