import sys
import ast
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import dc_qaoa.config

from pathlib import Path
from typing import Optional

from dc_qaoa import solver as _solver_module
from dc_qaoa.graph_loader import load_graph
from dc_qaoa.partitioner import recursive_partition, PartitionNode
from dc_qaoa.merger import merge
from dc_qaoa.quantum_backend import FINAL_PARAMETERS, LOSS_HISTORY, PARAMS_PATHS, setup_qpu
from dc_qaoa.circuit import _build_qaoa_circuit
#from dc_qaoa.scorer import maxcut_score

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 100


def draw_graph(G: nx.Graph, title: str, node_colors: list = None, pos: dict = None):
    """Helper to draw a NetworkX graph nicely."""
    plt.figure(figsize=(10, 8))
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
        
    if node_colors is None:
        node_colors = ['skyblue'] * G.number_of_nodes()
        
    edges = G.edges(data=True)
    weights = [d.get('weight', 1.0) for u, v, d in edges]
    max_weight = max(weights) if weights else 1.0
    edge_widths = [1 + 3 * (w / max_weight) for w in weights]

    nx.draw(G, pos, 
            node_color=node_colors, 
            with_labels=True, 
            node_size=600, 
            font_size=10, 
            font_color='black',
            font_weight='bold',
            edge_color='gray', 
            width=edge_widths, 
            alpha=0.9)
    
    plt.title(title, fontsize=16)
    plt.margins(x=0.1, y=0.1)
    plt.show()
    return pos


def QAOA_training_loss_history(subgraph_id: int) -> Optional[list[float]]:
    """Return the training loss history for a given subgraph, if available."""
    loss_trace = LOSS_HISTORY.get(subgraph_id)
    return loss_trace

def QAOA_parameter_trajectory(subgraph_id: int) -> Optional[list[dict]]:
    """Return the parameter trajectory (gammas, betas) for a given subgraph, if available."""
    params_trace = PARAMS_PATHS.get(subgraph_id)
    return params_trace

def plot_loss_history(loss_history: list[float], title: str = "QAOA Training Loss History"):
    """Plot the QAOA training loss history."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o', linestyle='-', color='blue')
    plt.title(title, fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss (Negative Expected Cut)", fontsize=12)
    plt.grid(True)
    plt.show()



def plot_QAOA_landscape(prog, subgraph, GRID, title: str = "QAOA Parameter Landscape"):
    """Plot the QAOA parameter landscape for 2 parameters (gamma_1, beta_1)."""
    gamma_vals = np.linspace(-np.pi, np.pi, GRID)
    beta_vals  = np.linspace(-np.pi, np.pi, GRID)
    
    landscape = np.zeros((GRID, GRID))
    for i, gamma in enumerate(gamma_vals):
        for j, beta in enumerate(beta_vals):
            params = {"gammas": [gamma], "betas": [beta]}
            result = _solver_module._QC.run(prog, memory_map=params)
            bitstrings = np.array(result.get_register_map().get("ro"))
            total_cut = sum(w for shots in bitstrings for (u, v, w) in prog.edges if shots[u] != shots[v])
            landscape[i, j] = total_cut

    plt.figure(figsize=(8, 6))
    plt.contourf(beta_vals, gamma_vals, landscape, levels=50, cmap='viridis')

    plt.scatter(PARAMS_PATHS.get(id(subgraph), []), 
                c='red', marker='x', label='Optimization Path')
    plt.colorbar(label='Expected Cut Value')
    plt.title(title, fontsize=14)
    plt.xlabel("Beta (Mixer Angle)", fontsize=12)
    plt.ylabel("Gamma (Cost Angle)", fontsize=12)
    plt.grid(True)
    plt.show()
    

