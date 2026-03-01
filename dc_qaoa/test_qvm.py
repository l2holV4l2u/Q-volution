"""
test_qvm.py -- Run the QAOA solver against a local QVM via Docker.

Before running:
  1. cd "Q-volution 2025"
  2. docker compose up -d
  3. pip install pyquil scipy
  4. python dc_qaoa/test_qvm.py [A|B]   (default: B)

The full DC-QAOA pipeline is used -- large graphs are partitioned into
subgraphs <= max_size nodes, each solved as a separate QVM circuit.
"""
import sys
import time
from pathlib import Path

import solver
from pipeline import run_pipeline
from graph_loader import load_graph

# ── Config ────────────────────────────────────────────────────────────────────
MAX_SIZE      = 8     # qubits per QVM subgraph circuit (increase for deeper sim)
TOP_T         = 10    # top-t solutions kept per subgraph for merge
SA_MAXITER    = 100   # SA iterations per subgraph (bump for better quality)
QC_NAME       = None  # None = auto-size QVM per subgraph; or e.g. "8q-qvm"
# ──────────────────────────────────────────────────────────────────────────────

DATASET_DIR = Path(__file__).parent.parent   # Q-volution 2025/

DATASETS = {
    "A": DATASET_DIR / "dataset_A.parquet",
    "B": DATASET_DIR / "dataset_B.parquet",
}


def run_stub_pipeline(graph_path: Path) -> tuple[float, float]:
    """Run pipeline with stub backend (no QVM). Returns (score, elapsed)."""
    solver.USE_PYQUIL = False
    t0 = time.time()
    _, score = run_pipeline(str(graph_path), max_size=MAX_SIZE, top_t=TOP_T)
    return score, time.time() - t0


def run_qvm_pipeline(graph_path: Path) -> tuple[float, float]:
    """Run pipeline with pyQuil QVM backend. Returns (score, elapsed)."""
    solver.SA_MAXITER = SA_MAXITER
    t0 = time.time()
    _, score = run_pipeline(
        str(graph_path),
        max_size=MAX_SIZE,
        top_t=TOP_T,
        qc_name=QC_NAME,
    )
    return score, time.time() - t0


def main():
    dataset_key = sys.argv[1].upper() if len(sys.argv) > 1 else "B"
    if dataset_key not in DATASETS:
        print(f"Unknown dataset '{dataset_key}'. Choose A or B.")
        sys.exit(1)

    graph_path = DATASETS[dataset_key]
    if not graph_path.exists():
        print(f"Dataset not found: {graph_path}")
        sys.exit(1)

    G = load_graph(str(graph_path))
    total_w = sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True))
    n, m = G.number_of_nodes(), G.number_of_edges()

    print(f"\nDataset {dataset_key}: {n} nodes, {m} edges, total weight = {total_w:.2f}")
    print(f"Subgraph size cap: {MAX_SIZE} qubits  |  SA iters: {SA_MAXITER}")
    print("=" * 65)

    # -- Stub baseline --
    print("\n[1/2] Stub backend (brute-force/local-search, no QVM)...")
    stub_score, stub_time = run_stub_pipeline(graph_path)
    stub_ratio = stub_score / total_w
    print(f"\n      score = {stub_score:.4f}  ratio = {stub_ratio:.4f}  time = {stub_time:.1f}s")

    # -- QVM backend --
    print("\n[2/2] QVM backend (QAOA + Simulated Annealing)...")
    print("      Make sure Docker containers are running:")
    print("        docker compose up -d   (from Q-volution 2025/)")
    print()

    try:
        qvm_score, qvm_time = run_qvm_pipeline(graph_path)
        qvm_ratio = qvm_score / total_w

        print()
        print("=" * 65)
        print(f"{'Method':<30} {'Score':>10} {'Ratio':>8} {'Time':>8}")
        print("-" * 65)
        print(f"{'Stub (baseline)':<30} {stub_score:>10.4f} {stub_ratio:>8.4f} {stub_time:>7.1f}s")
        print(f"{'QAOA + SA (QVM)':<30} {qvm_score:>10.4f} {qvm_ratio:>8.4f} {qvm_time:>7.1f}s")
        print("=" * 65)

        if qvm_score >= stub_score:
            print("Verdict: QAOA matched or beat the stub baseline")
        else:
            diff = stub_score - qvm_score
            print(f"Verdict: Stub wins by {diff:.4f} -- expected at p=1, try LAYER_COUNT=3")

    except Exception as e:
        print(f"\n[ERROR] QVM run failed: {e}")
        print("\nTroubleshooting:")
        print("  docker compose ps          -- check both services show 'running'")
        print("  docker compose logs quilc  -- check quilc server output")
        print("  docker compose logs qvm    -- check qvm server output")


if __name__ == "__main__":
    main()
