"""
testing_vis.py — Tests for visualization.py landscape functions.

Every visual test saves its output to ./test_output_pngs/ and verifies:
  1. The PNG file was created.
  2. The file is a valid, non-empty image (checked via PNG magic bytes).
  3. The file is large enough to contain real content (> 10 KB).

Test graph
----------
    0 ─(3.0)─ 1
    │           │
  (1.5)       (2.5)
    │           │
    2 ─(2.0)─ 3

4 nodes, 4 edges, total weight 9.0.  Optimal Max-Cut ≈ 5.5.

No running QVM needed — all landscape computations use the numpy
statevector simulator built into visualization.py.

Notes on physics / degenerate grid points
------------------------------------------
The 5-point linspace {−π/2, −π/4, 0, π/4, π/2} are degenerate period points
for the X/XX mixers on this graph: every combination evaluates to total_weight/2.
Tests use offset interior grids to sample non-degenerate landscape regions.

The XY mixer at γ=0 departs from total_weight/2 as β increases because |+⟩^n
is not an eigenstate of (X_u X_v + Y_u Y_v)/2 — this is correct physics.
"""

from __future__ import annotations

import sys
import importlib.util
import pathlib
import types
import struct
import zlib
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")   # headless — always save to file, never open a window
import matplotlib.pyplot as plt

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = pathlib.Path(__file__).parent / "test_output_pngs"
OUT_DIR.mkdir(exist_ok=True)


# ── Import visualization (package or standalone) ─────────────────────────────

def _load_visualization():
    try:
        from dc_qaoa import visualization as vis
        return vis
    except ImportError:
        pass
    here = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "visualization", here / "visualization.py"
    )
    vis = importlib.util.module_from_spec(spec)

    # Minimal stubs so visualization.py can import dc_qaoa.quantum_backend
    stub_pkg = types.ModuleType("dc_qaoa")
    stub_qb  = types.ModuleType("dc_qaoa.quantum_backend")
    stub_qb.FINAL_PARAMETERS = {}
    stub_qb.LOSS_HISTORY      = {}
    stub_qb.PARAMS_PATHS      = {}
    sys.modules.setdefault("dc_qaoa", stub_pkg)
    sys.modules["dc_qaoa.quantum_backend"] = stub_qb

    spec.loader.exec_module(vis)
    return vis


vis = _load_visualization()

_statevector_qaoa_expectation = vis._statevector_qaoa_expectation
_extract_gamma_beta           = vis._extract_gamma_beta
draw_graph                    = vis.draw_graph
plot_loss_history             = vis.plot_loss_history
plot_QAOA_landscape           = vis.plot_QAOA_landscape
draw_qaoa_landscape           = vis.draw_qaoa_landscape
PARAMS_PATHS                  = vis.PARAMS_PATHS
LOSS_HISTORY                  = vis.LOSS_HISTORY


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _graph_edges(G):
    nodes = list(G.nodes())
    idx   = {v: i for i, v in enumerate(nodes)}
    return (
        nodes,
        [(idx[u], idx[v], float(d.get("weight", 1.0))) for u, v, d in G.edges(data=True)],
        len(nodes),
    )


def _is_valid_png(path: pathlib.Path) -> tuple[bool, str]:
    """Return (True, '') if path is a valid PNG, else (False, reason)."""
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
    if not path.exists():
        return False, "file does not exist"
    data = path.read_bytes()
    if len(data) < 8:
        return False, f"file too small ({len(data)} bytes)"
    if data[:8] != PNG_MAGIC:
        return False, f"bad magic bytes: {data[:8]!r}"
    return True, ""


def check(cond: bool, msg: str):
    status = "PASS" if cond else "FAIL"
    print(f"    {status}  {msg}")
    if not cond:
        raise AssertionError(msg)


def check_png(path: pathlib.Path, min_bytes: int = 10_000):
    """Assert path is a valid, non-trivial PNG."""
    ok, reason = _is_valid_png(path)
    check(ok, f"valid PNG: {path.name}  ({reason or 'ok'})")
    size = path.stat().st_size
    check(size >= min_bytes,
          f"PNG size {size:,} bytes >= {min_bytes:,} (non-trivial image): {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Test graph
# ─────────────────────────────────────────────────────────────────────────────

def build_test_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edge(0, 1, weight=3.0)
    G.add_edge(0, 2, weight=1.5)
    G.add_edge(1, 3, weight=2.5)
    G.add_edge(2, 3, weight=2.0)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — draw_graph → PNG
# ─────────────────────────────────────────────────────────────────────────────

def test_draw_graph(G: nx.Graph) -> None:
    print("\n[test 1] draw_graph → PNG")
    out = OUT_DIR / "01_graph.png"
    pos = draw_graph(G, title="Test Graph — 4 Nodes, Weighted", save_path=out)

    check(isinstance(pos, dict) and len(pos) == G.number_of_nodes(),
          "draw_graph returns position dict with one entry per node")
    check_png(out)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — numpy statevector sanity (no file output, pure maths)
# ─────────────────────────────────────────────────────────────────────────────

def test_statevector_sanity(G: nx.Graph) -> None:
    print("\n[test 2] numpy statevector sanity checks")
    _, edges, n = _graph_edges(G)
    total_w     = sum(w for _, _, w in edges)

    # X and XX at gamma=0 are eigenstates of the mixer → <C> = total_w/2 always
    for mixer in ("X", "XX"):
        for beta in (0.0, 0.3, 0.7, 1.2):
            val = _statevector_qaoa_expectation(edges, n, 0.0, beta, mixer)
            check(abs(val - total_w / 2) < 1e-8,
                  f"{mixer} gamma=0 beta={beta:.1f}: <C>={val:.8f} = {total_w/2}")

    # XY departs from total_w/2 as beta increases (|+>^n not an XY eigenstate)
    print("    [XY at gamma=0 — expected departure as beta increases]")
    xy_vals = []
    for beta in (0.0, 0.3, 0.5, 1.0):
        v = _statevector_qaoa_expectation(edges, n, 0.0, beta, "XY")
        xy_vals.append(v)
        print(f"      beta={beta:.1f}  <C>={v:.6f}  delta={abs(v - total_w/2):.4f}")
    check(abs(xy_vals[0] - total_w / 2) < 1e-8, "XY beta=0: <C>=total_w/2 (UM=I)")
    check(max(abs(v - total_w/2) for v in xy_vals[1:]) > 1e-4,
          "XY moves away from total_w/2 for beta>0")

    # Physical bounds for all mixers
    for mixer in ("X", "XX", "XY"):
        for g, b in ((0.31, 0.41), (0.82, -0.62), (-0.51, 1.01)):
            val = _statevector_qaoa_expectation(edges, n, g, b, mixer)
            check(-1e-8 <= val <= total_w + 1e-8,
                  f"{mixer} <C>({g:.2f},{b:.2f})={val:.5f} in [0, {total_w}]")

    # Landscape is non-constant over interior grid (avoids degenerate multiples)
    gv = np.linspace(-np.pi/2, np.pi/2, 7, endpoint=False) + 0.12
    bv = np.linspace(-np.pi/2, np.pi/2, 7, endpoint=False) + 0.12
    for mixer in ("X", "XX", "XY"):
        vals   = np.array([_statevector_qaoa_expectation(edges, n, g, b, mixer)
                           for g in gv for b in bv])
        spread = vals.max() - vals.min()
        check(spread > 0.5,
              f"{mixer} landscape spread={spread:.4f} over interior grid (>0.5)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — plot_loss_history → PNG
# ─────────────────────────────────────────────────────────────────────────────

def test_plot_loss_history() -> None:
    print("\n[test 3] plot_loss_history → PNG")
    rng  = np.random.default_rng(0)
    hist = (np.cumsum(-rng.exponential(0.1, 30)) + 2.0).tolist()
    out  = OUT_DIR / "03_loss_history.png"
    plot_loss_history(hist, title="Synthetic QAOA Loss History (test)", save_path=out)
    check_png(out)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — _extract_gamma_beta
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_gamma_beta() -> None:
    print("\n[test 4] _extract_gamma_beta")
    cases = [
        ([0.3, 0.7],                           1, 0.3, 0.7),
        (np.array([0.4, 0.8]),                 1, 0.4, 0.8),
        ({"params": np.array([0.5, 0.9])},     1, 0.5, 0.9),
        ({"iter": 3, "params": [0.6, 1.0]},    1, 0.6, 1.0),
        ([0.2, 0.3, 0.8, 0.9],                 2, 0.2, 0.8),   # p=2
    ]
    for entry, lc, exp_g, exp_b in cases:
        g, b = _extract_gamma_beta(entry, layer_count=lc)
        check(g is not None and abs(g - exp_g) < 1e-12 and abs(b - exp_b) < 1e-12,
              f"type={type(entry).__name__} p={lc}: gamma={g} beta={b}")

    g, b = _extract_gamma_beta([0.1], layer_count=1)
    check(g is None and b is None, "too-short entry → (None, None)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — plot_QAOA_landscape + trajectory → PNG
# ─────────────────────────────────────────────────────────────────────────────

def test_plot_landscape_with_trajectory(G: nx.Graph) -> None:
    print("\n[test 5] plot_QAOA_landscape + trajectory overlay → PNG")
    _, edges, n = _graph_edges(G)

    # Grid-search approximate optimum (25×25 over non-degenerate interior)
    gv = np.linspace(-np.pi/2, np.pi/2, 25)
    bv = np.linspace(-np.pi/2, np.pi/2, 25)
    best_val, best_g, best_b = -np.inf, 0.0, 0.0
    for g in gv:
        for b in bv:
            v = _statevector_qaoa_expectation(edges, n, g, b, "X")
            if v > best_val:
                best_val, best_g, best_b = v, g, b
    print(f"    Grid-search: gamma={best_g:.3f} beta={best_b:.3f} <C>={best_val:.4f}")
    check(best_val > 5.0, f"best <C>={best_val:.4f} > 5.0 (near classical opt 5.5)")

    # Build synthetic trajectory approaching the optimum
    rng    = np.random.default_rng(1)
    K      = 30
    g_path = np.linspace(rng.uniform(-np.pi/2, 0), best_g, K) + rng.normal(0, 0.04, K)
    b_path = np.linspace(rng.uniform(-np.pi/2, 0), best_b, K) + rng.normal(0, 0.04, K)

    sid = id(G)
    PARAMS_PATHS[sid] = []
    LOSS_HISTORY[sid] = []

    for k, (g, b) in enumerate(zip(g_path, b_path)):
        # Alternate between the two storage formats used by quantum_backend.py
        if k % 3 == 0:
            PARAMS_PATHS[sid].append([g, b])           # list format (from objective)
        else:
            PARAMS_PATHS[sid].append({                  # dict format (from callbacks)
                "iter":   k,
                "params": np.array([g, b]),
                "loss":   -_statevector_qaoa_expectation(edges, n, g, b, "X"),
            })
        LOSS_HISTORY[sid].append(
            -_statevector_qaoa_expectation(edges, n, g, b, "X")
        )

    extracted = [_extract_gamma_beta(e, 1) for e in PARAMS_PATHS[sid]]
    check(all(g is not None for g, _ in extracted),
          f"All {len(extracted)} trajectory entries extracted OK")

    # ── Landscape with trajectory ──────────────────────────────────────────────
    out = OUT_DIR / "05_landscape_X_with_trajectory.png"
    plot_QAOA_landscape(
        G,
        subgraph_id=sid,
        grid=30,
        mixer_mode="X",
        title="QAOA Landscape (X mixer) + Optimisation Trajectory",
        save_path=out,
    )
    check_png(out)

    # ── Corresponding loss curve ───────────────────────────────────────────────
    out_loss = OUT_DIR / "05_loss_history_from_stored.png"
    plot_loss_history(
        LOSS_HISTORY[sid],
        title="Loss History from Stored Trajectory",
        save_path=out_loss,
    )
    check_png(out_loss)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — landscape with NO stored trajectory (tests FINAL_PARAMETERS fallback)
# ─────────────────────────────────────────────────────────────────────────────

def test_plot_landscape_no_trajectory(G: nx.Graph) -> None:
    print("\n[test 6] plot_QAOA_landscape without trajectory (empty PARAMS_PATHS)")
    G2 = G.copy()                   # fresh object → id(G2) not in PARAMS_PATHS
    out = OUT_DIR / "06_landscape_X_no_trajectory.png"
    plot_QAOA_landscape(
        G2,
        grid=25,
        mixer_mode="X",
        title="QAOA Landscape — No Trajectory",
        save_path=out,
    )
    check_png(out)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — draw_qaoa_landscape: X / XX / XY side-by-side → PNG
# ─────────────────────────────────────────────────────────────────────────────

def test_draw_qaoa_landscape_all_mixers(G: nx.Graph) -> None:
    print("\n[test 7] draw_qaoa_landscape — X / XX / XY side-by-side → PNG")
    out = OUT_DIR / "07_landscape_all_mixers.png"
    draw_qaoa_landscape(
        G,
        grid=25,
        mixer_modes=("X", "XX", "XY"),
        title="QAOA Landscape Comparison — 4-Node Test Graph",
        share_colorscale=True,
        save_path=out,
    )
    check_png(out, min_bytes=20_000)   # three-panel figure — expect > 20 KB


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — draw_qaoa_landscape: individual mixers saved separately
# ─────────────────────────────────────────────────────────────────────────────

def test_draw_qaoa_landscape_individual(G: nx.Graph) -> None:
    print("\n[test 8] draw_qaoa_landscape — each mixer saved individually → PNG")
    for mixer in ("X", "XX", "XY"):
        out = OUT_DIR / f"08_landscape_{mixer}.png"
        draw_qaoa_landscape(
            G,
            grid=25,
            mixer_modes=(mixer,),
            title=f"QAOA Landscape — {mixer} mixer",
            share_colorscale=False,
            save_path=out,
        )
        check_png(out)
        print(f"    PASS  {mixer} landscape saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — landscape over XX mixer with trajectory → PNG
# ─────────────────────────────────────────────────────────────────────────────

def test_plot_landscape_XX_trajectory(G: nx.Graph) -> None:
    print("\n[test 9] plot_QAOA_landscape — XX mixer + trajectory → PNG")
    _, edges, n = _graph_edges(G)

    # Coarse grid search for XX mixer optimum
    gv = np.linspace(-np.pi/2, np.pi/2, 20)
    bv = np.linspace(-np.pi/2, np.pi/2, 20)
    best_val, best_g, best_b = -np.inf, 0.0, 0.0
    for g in gv:
        for b in bv:
            v = _statevector_qaoa_expectation(edges, n, g, b, "XX")
            if v > best_val:
                best_val, best_g, best_b = v, g, b

    G_xx = G.copy()
    sid  = id(G_xx)
    rng  = np.random.default_rng(7)
    K    = 20
    g_path = np.linspace(rng.uniform(-1, 0), best_g, K) + rng.normal(0, 0.05, K)
    b_path = np.linspace(rng.uniform(-1, 0), best_b, K) + rng.normal(0, 0.05, K)

    PARAMS_PATHS[sid] = [{"params": np.array([g, b])} for g, b in zip(g_path, b_path)]
    LOSS_HISTORY[sid] = [-_statevector_qaoa_expectation(edges, n, g, b, "XX")
                          for g, b in zip(g_path, b_path)]

    out = OUT_DIR / "09_landscape_XX_with_trajectory.png"
    plot_QAOA_landscape(
        G_xx,
        subgraph_id=sid,
        grid=28,
        mixer_mode="XX",
        title="QAOA Landscape (XX mixer) + Optimisation Trajectory",
        save_path=out,
    )
    check_png(out)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("  DC-QAOA visualisation test suite")
    print(f"  Output PNGs → {OUT_DIR}")
    print("=" * 64)

    G = build_test_graph()
    print(
        f"\nTest graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n"
        f"  Edges: {[(u, v, d['weight']) for u, v, d in G.edges(data=True)]}"
    )

    test_draw_graph(G)
    test_statevector_sanity(G)
    test_plot_loss_history()
    test_extract_gamma_beta()
    test_plot_landscape_with_trajectory(G)
    test_plot_landscape_no_trajectory(G)
    test_draw_qaoa_landscape_all_mixers(G)
    test_draw_qaoa_landscape_individual(G)
    test_plot_landscape_XX_trajectory(G)

    print("\n" + "=" * 64)
    print("  All tests passed.")
    print(f"\n  PNG outputs in: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"    {p.name:50s}  {p.stat().st_size:>9,} bytes")
    print("=" * 64)


if __name__ == "__main__":
    main()