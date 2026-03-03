

from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.linalg import expm
from typing import Optional, Union

from dc_qaoa.quantum_backend import FINAL_PARAMETERS, LOSS_HISTORY, PARAMS_PATHS

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.dpi"] = 100

SavePath = Optional[Union[str, Path]]


# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
# Internal helper: show interactively or save to file
# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

def _show_or_save(fig: plt.Figure, save_path: SavePath) -> None:
    """
    Display *fig* in an interactive window, or save it to *save_path*.

    Calling plt.show() on a non-interactive (Agg) backend emits:
        "FigureCanvasAgg is non-interactive, and thus cannot be shown"
    Passing a save_path avoids that entirely and lets callers (tests,
    batch scripts) control where the image lands.
    """
    if save_path is not None:
        fig.savefig(save_path, dpi=fig.dpi, bbox_inches="tight")
        print(f"[vis] Saved โ’ {save_path}")
    else:
        plt.show()
    plt.close(fig)


# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
# Graph drawing
# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

def draw_graph(
    G: nx.Graph,
    title: str,
    node_colors: list = None,
    pos: dict = None,
    save_path: SavePath = None,
) -> dict:
    """
    Draw a NetworkX graph; return the layout positions used.

    Parameters
    ----------
    save_path : If given, save the figure to this path instead of opening
                an interactive window (avoids Agg backend warning).
    """
    fig = plt.figure(figsize=(10, 8))
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    if node_colors is None:
        node_colors = ["skyblue"] * G.number_of_nodes()

    edges = G.edges(data=True)
    weights = [d.get("weight", 1.0) for _, _, d in edges]
    max_w = max(weights) if weights else 1.0
    edge_widths = [1 + 3 * (w / max_w) for w in weights]

    nx.draw(
        G, pos,
        node_color=node_colors,
        with_labels=True,
        node_size=600,
        font_size=10,
        font_color="black",
        font_weight="bold",
        edge_color="gray",
        width=edge_widths,
        alpha=0.9,
    )
    plt.title(title, fontsize=16)
    plt.margins(x=0.1, y=0.1)
    _show_or_save(fig, save_path)
    return pos


# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
# Training-loss curve
# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

def QAOA_training_loss_history(
    subgraph_id: int,
    best_so_far: bool = True,
) -> Optional[list[float]]:
    """
    Return the training loss history for *subgraph_id*.

    By default this returns cumulative best-so-far loss so the series is
    monotonic and easier to interpret as convergence.
    """
    history = LOSS_HISTORY.get(subgraph_id)
    if history is None:
        return None
    if not best_so_far:
        return list(history)
    arr = np.asarray(history, dtype=float)
    return np.minimum.accumulate(arr).tolist()


def QAOA_parameter_trajectory(subgraph_id: int) -> Optional[list]:
    """Return the raw parameter-trajectory entries stored for *subgraph_id*."""
    return PARAMS_PATHS.get(subgraph_id)


def plot_loss_history(
    loss_history: list[float],
    title: str = "QAOA Training Loss History",
    best_so_far: bool = True,
    relative_to_final: bool = False,
    x_axis_label: str = "Iteration",
    save_path: SavePath = None,
) -> None:
    """
    Plot the QAOA training loss curve.

    Parameters
    ----------
    save_path : If given, save the figure to this path instead of opening
                an interactive window.
    """
    y = np.asarray(loss_history, dtype=float)
    if best_so_far:
        y = np.minimum.accumulate(y)
    if relative_to_final and len(y) > 0:
        final_val = y[-1]
        if final_val != 0:
            y = y / final_val
    x = np.arange(1, len(y) + 1, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linestyle="-", color="royalblue")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_axis_label, fontsize=12)
    ylabel = "Relative Loss (Loss / Final Loss)" if relative_to_final else "Loss (Negative Expected Cut)"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    fig.tight_layout()
    _show_or_save(fig, save_path)


# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
# Numpy statevector QAOA simulator  (p = 1, exact)
# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

def plot_multi_loss_history(
    histories: dict[str, list[float]],
    title: str = "QAOA Training Loss History (All Subgraphs)",
    best_so_far: bool = True,
    x_axis_label: str = "Iteration",
    save_path: SavePath = None,
) -> None:
    """
    Plot multiple QAOA loss curves in one figure (one line per subgraph).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0
    for label, loss_history in histories.items():
        if not loss_history:
            continue
        y = np.asarray(loss_history, dtype=float)
        if best_so_far:
            y = np.minimum.accumulate(y)
        x = np.arange(1, len(y) + 1, dtype=int)
        ax.plot(x, y, marker="o", linestyle="-", linewidth=1.4, markersize=3.5, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel("Loss (Negative Expected Cut)", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _show_or_save(fig, save_path)

_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_single(op: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """n-qubit operator with *op* on *qubit* and I on all others."""
    ops = [_I2] * n
    ops[qubit] = op
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def _zz_op(u: int, v: int, n: int) -> np.ndarray:
    """Z_u โ— Z_v embedded in n-qubit space."""
    ops = [_I2] * n
    ops[u] = _Z2
    ops[v] = _Z2
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def _build_cost_operator(
    edges: list[tuple[int, int, float]], n: int
) -> np.ndarray:
    """
    C = ฮฃ_{(u,v,w) โ E}  w ยท (I โ’ Z_u Z_v) / 2

    Standard weighted Max-Cut cost operator whose expectation equals
    the expected cut weight.
    """
    dim = 2 ** n
    C = np.zeros((dim, dim), dtype=complex)
    for u, v, w in edges:
        C += w * (np.eye(dim, dtype=complex) - _zz_op(u, v, n)) / 2
    return C


def _statevector_qaoa_expectation(
    edges: list[tuple[int, int, float]],
    n: int,
    gamma: float,
    beta: float,
    mixer_mode: str = "X",
) -> float:
    """
    Exact p = 1 QAOA expected cut value via statevector simulation.

    |ฯโฉ = U_M(ฮฒ) ยท U_C(ฮณ) ยท |+โฉ^n
    โจCโฉ = โจฯ|C|ฯโฉ

    Mixer modes
    -----------
    "X"  : U_M = โ—_q exp(โ’iฮฒ X_q)                    (standard transverse field)
    "XX" : U_M = ฮ _{(u,v)} exp(โ’iฮฒ X_u X_v)           (graph-coupled XX)
    "XY" : U_M = ฮ _{(u,v)} exp(โ’iฮฒ (X_u X_v + Y_u Y_v)/2)  (XY mixer)
    """
    dim = 2 ** n
    C = _build_cost_operator(edges, n)

    # |+โฉ^n
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    state = plus.copy()
    for _ in range(n - 1):
        state = np.kron(state, plus)

    # Cost unitary
    state = expm(-1j * gamma * C) @ state

    # Mixer unitary
    if mixer_mode == "X":
        UM = np.eye(dim, dtype=complex)
        for q in range(n):
            Xq = _kron_single(_X2, q, n)
            UM = expm(-1j * beta * Xq) @ UM
    elif mixer_mode == "XX":
        UM = np.eye(dim, dtype=complex)
        for u, v, _ in edges:
            XXuv = _kron_single(_X2, u, n) @ _kron_single(_X2, v, n)
            UM = expm(-1j * beta * XXuv) @ UM
    elif mixer_mode == "XY":
        UM = np.eye(dim, dtype=complex)
        for u, v, _ in edges:
            XXuv = _kron_single(_X2, u, n) @ _kron_single(_X2, v, n)
            YYuv = _kron_single(_Y2, u, n) @ _kron_single(_Y2, v, n)
            UM = expm(-1j * beta * (XXuv + YYuv) / 2) @ UM
    else:
        raise ValueError(f"Unknown mixer_mode: {mixer_mode!r}. Use 'X', 'XX', or 'XY'.")

    state = UM @ state
    return float(np.real(state.conj() @ C @ state))


def _compute_landscape(
    edges: list[tuple[int, int, float]],
    n: int,
    grid: int,
    mixer_mode: str,
    gamma_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    beta_range: tuple[float, float]  = (-np.pi / 2, np.pi / 2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 2-D QAOA landscape โจCโฉ(ฮณ, ฮฒ) for p = 1.

    Returns (gamma_vals, beta_vals, landscape) where
    landscape[i, j] = โจCโฉ at (gamma_vals[i], beta_vals[j]).
    """
    gamma_vals = np.linspace(*gamma_range, grid)
    beta_vals  = np.linspace(*beta_range,  grid)
    landscape  = np.zeros((grid, grid))

    for i, gamma in enumerate(gamma_vals):
        for j, beta in enumerate(beta_vals):
            landscape[i, j] = _statevector_qaoa_expectation(
                edges, n, gamma, beta, mixer_mode
            )
    return gamma_vals, beta_vals, -landscape


# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
# Trajectory extraction helper
# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

def _extract_gamma_beta(entry, layer_count: int = 1):
    """
    Extract (gamma_0, beta_0) from a PARAMS_PATHS entry.

    Handles two storage formats written by quantum_backend.py:
      - plain list / ndarray  [gamma_0, ..., beta_0, ...]  (from objective())
      - dict with "params" key                             (from optimizer callbacks)
    """
    if isinstance(entry, dict):
        params = entry.get("params", [])
    else:
        params = entry

    params = np.asarray(params, dtype=float).ravel()
    if len(params) < 2 * layer_count:
        return None, None

    gamma = float(params[0])            # first gamma
    beta  = float(params[layer_count])  # first beta (offset by layer_count)
    return gamma, beta


# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
# Primary landscape + trajectory plot
# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

def plot_QAOA_landscape(
    subgraph: nx.Graph,
    subgraph_id: int = None,
    *,
    grid: int = 30,
    mixer_mode: str = "X",
    layer_count: int = 1,
    title: str = "QAOA Parameter Landscape",
    gamma_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    beta_range:  tuple[float, float] = (-np.pi / 2, np.pi / 2),
    save_path: SavePath = None,
) -> None:

    if subgraph_id is None:
        subgraph_id = id(subgraph)

    nodes = list(subgraph.nodes())
    n     = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}
    edges = [
        (node_index[u], node_index[v], float(d.get("weight", 1.0)))
        for u, v, d in subgraph.edges(data=True)
    ]

    print(f"[vis] Computing landscape ({grid}ร—{grid}) with mixer='{mixer_mode}' โ€ฆ")
    gamma_vals, beta_vals, landscape = _compute_landscape(
        edges, n, grid, mixer_mode, gamma_range, beta_range
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    cf = ax.contourf(beta_vals, gamma_vals, landscape, levels=50, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="โจCโฉ (Expected Cut Value)")

    # โ”€โ”€ Trajectory overlay โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    traj_entries = PARAMS_PATHS.get(subgraph_id, [])
    gammas_traj, betas_traj = [], []
    for entry in traj_entries:
        g, b = _extract_gamma_beta(entry, layer_count)
        if g is not None:
            gammas_traj.append(g)
            betas_traj.append(b)

    if gammas_traj:
        # Connecting line (white, semi-transparent)
        ax.plot(
            betas_traj, gammas_traj,
            color="white", linewidth=0.8, alpha=0.5, zorder=3,
        )
        # Intermediate steps โ€” black dots
        ax.scatter(
            betas_traj[:-1], gammas_traj[:-1],
            c="black", s=25, marker="o", zorder=4, label="Trajectory",
        )
        # Final (optimal) point โ€” red star
        ax.scatter(
            [betas_traj[-1]], [gammas_traj[-1]],
            c="red", s=200, marker="o", zorder=5, label="Optimal",
        )
        ax.legend(fontsize=10)

    elif subgraph_id in FINAL_PARAMETERS:
        # Fall back to the stored optimal parameters if no trajectory recorded
        fp    = FINAL_PARAMETERS[subgraph_id]
        g_opt = fp["gammas"][0] if fp.get("gammas") else None
        b_opt = fp["betas"][0]  if fp.get("betas")  else None
        if g_opt is not None and b_opt is not None:
            ax.scatter(
                [b_opt], [g_opt],
                c="red", s=200, marker="*", zorder=5, label="Optimal (stored)",
            )
            ax.legend(fontsize=10)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("ฮฒ (Mixer Angle)", fontsize=12)
    ax.set_ylabel("ฮณ (Cost Angle)", fontsize=12)
    ax.grid(False)
    fig.tight_layout()
    _show_or_save(fig, save_path)


# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
# Multi-mixer landscape comparison
# โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€

def draw_qaoa_landscape(
    G: nx.Graph,
    *,
    grid: int = 30,
    mixer_modes: tuple[str, ...] = ("X", "XX", "XY"),
    gamma_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
    beta_range:  tuple[float, float] = (-np.pi / 2, np.pi / 2),
    title: str = "QAOA Landscape โ€” p=1",
    share_colorscale: bool = True,
    save_path: SavePath = None,
) -> None:

    nodes = list(G.nodes())
    n     = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}
    edges = [
        (node_index[u], node_index[v], float(d.get("weight", 1.0)))
        for u, v, d in G.edges(data=True)
    ]

    n_mixers   = len(mixer_modes)
    landscapes = {}
    gamma_vals = beta_vals = None

    print(f"[vis] Computing {n_mixers} landscape(s) ({grid}ร—{grid} each) โ€ฆ")
    for mode in mixer_modes:
        print(f"  mixer = '{mode}' โ€ฆ")
        gv, bv, land = _compute_landscape(
            edges, n, grid, mode, gamma_range, beta_range
        )
        landscapes[mode] = land
        gamma_vals = gv
        beta_vals  = bv

    # Shared colour limits
    if share_colorscale:
        all_vals = np.concatenate([l.ravel() for l in landscapes.values()])
        vmin, vmax = all_vals.min(), all_vals.max()
    else:
        vmin = vmax = None

    # Figure
    fig, axes = plt.subplots(1, n_mixers, figsize=(6 * n_mixers, 5), sharey=True)
    if n_mixers == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=15, fontweight="bold")

    contour_sets = []
    for ax, mode in zip(axes, mixer_modes):
        land = landscapes[mode]
        kw = dict(levels=50, cmap="viridis")
        if share_colorscale:
            kw["vmin"] = vmin
            kw["vmax"] = vmax

        cf = ax.contourf(beta_vals, gamma_vals, land, **kw)
        contour_sets.append(cf)

        # Mark the landscape maximum
        idx    = np.unravel_index(np.argmin(land), land.shape)
        g_best = gamma_vals[idx[0]]
        b_best = beta_vals[idx[1]]
        ax.scatter(
            [b_best], [g_best],
            c="red", s=160, marker="o", zorder=5,
            label=f"max โจCโฉ={land[idx]:.3f}",
        )
        ax.legend(fontsize=9)
        ax.set_title(f"Mixer: {mode}", fontsize=12)
        ax.set_xlabel("ฮฒ (Mixer Angle)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("ฮณ (Cost Angle)", fontsize=11)

    # Shared colourbar on the right
    fig.colorbar(contour_sets[-1], ax=axes, label="โจCโฉ (Expected Cut Value)")
    #fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.12, wspace=0.15)
    _show_or_save(fig, save_path)

