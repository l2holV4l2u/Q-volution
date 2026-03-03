from __future__ import annotations

import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    from dc_qaoa import config as _config
    from dc_qaoa import solver as _solver_module
    from dc_qaoa.pipeline import run_pipeline
    from dc_qaoa.quantum_backend import setup_qpu
    from dc_qaoa.graph import load_graph
    from dc_qaoa.visualization import draw_graph

except ImportError:
    sys.path.insert(0, "./")
    from dc_qaoa import config as _config
    from dc_qaoa.pipeline import run_pipeline
    from dc_qaoa import solver as _solver_module

# ── Pipeline settings ────────────────────────────────────────────────────────
GRAPH_PATH = Path("../dataset_A.csv")
MAX_SIZE   = 8
TOP_T      = 10
METHOD     = "separator"   # "separator" (NaiveLGP) | "community"

# ── QVM / QPU connection ─────────────────────────────────────────────────────
import os
os.environ["QCS_SETTINGS_APPLICATIONS_QVM_URL"] = "http://127.0.0.1:5001"

# ── Solver knobs ─────────────────────────────────────────────────────────────
_config.USE_PYQUIL   = True    # False → stub (brute-force / local search)
_config.LAYER_COUNT  = 1       # QAOA depth p
_config.SHOTS        = 1024
_config.SEED         = 42

# ── Quantum computer target ───────────────────────────────────────────────────
QC_NAME = "9q-square-qvm"

if _solver_module.USE_PYQUIL and QC_NAME:
    print(f"Setting up QPU: {QC_NAME}...")
    setup_qpu(QC_NAME)

# ── Load graph ────────────────────────────────────────────────────────────────
print(f"\nLoading graph from {GRAPH_PATH}...")
G = load_graph(GRAPH_PATH)
print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

global_pos = nx.spring_layout(G, seed=42)
draw_graph(G, "Original Problem Graph", pos=global_pos)


# ──────────────────────────────────────────────────────────────────────────────
mixers = ["X", "XX", "XY"]
_MIXER_DESCRIPTIONS = {
    "X":  ("Standard transverse-field",
           "exp(-i β Σ X_j)  — single-qubit RX rotations, lowest gate cost"),
    "XX": ("Graph-coupled XX",
           "exp(-i β Σ_{ring} X_iX_j)  — two-qubit XX on ring, richer correlations"),
    "XY": ("XY conserving mixer",
           "exp(-i β Σ_{ring} (X_iX_j+Y_iY_j)/2)  — preserves Hamming weight"),
}

for MIXER_MODE in mixers:
    print("\n" + "=" * 60)
    print(f"Running DC-QAOA with mixer: {MIXER_MODE}")
    print("=" * 60)
    print(f"Mixer description: {_MIXER_DESCRIPTIONS[MIXER_MODE][0]}")
    print(f"{_MIXER_DESCRIPTIONS[MIXER_MODE][1]}")
    _config.MIXER_MODE = MIXER_MODE

    assignment, score = run_pipeline(
        GRAPH_PATH,
        max_size=MAX_SIZE,
        top_t=TOP_T,
        qc_name=QC_NAME,
    )
    print(f"\nFinal cut value: {score:.4f}")




