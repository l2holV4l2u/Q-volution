"""
main.py -- Entry point for the DC-QAOA Max-Cut pipeline.

------------------------------------------------------------
Local stub test (no QPU needed):
    python main.py graph.parquet

Local QVM simulation (requires pyquil + quilc/qvm servers):
    python main.py --qaoa --qc 8q-qvm graph.parquet

Rigetti QPU (requires pyquil + QCS credentials):
    python main.py --qaoa --qc Ankaa-3 graph.parquet

Hardware preset (max_size=8, p=1, pyQuil enabled):
    python main.py --hardware graph.parquet
------------------------------------------------------------
"""
from __future__ import annotations


import sys
import json
import random
import argparse
from pathlib import Path

import solver as _solver_module   # for patching USE_PYQUIL / LAYER_COUNT / SHOTS
from pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DC-QAOA Weighted Max-Cut Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Positional ------------------------------------------------------------
    p.add_argument(
        "graph",
        nargs="?",
        help="Path to graph .parquet file (auto-detected if omitted).",
    )

    # -- QAOA activation -------------------------------------------------------
    p.add_argument(
        "--qaoa",
        action="store_true",
        default=False,
        help="Use pyQuil QAOA backend (requires pyquil + QVM or QPU). "
             "Without this flag the stub backend runs locally.",
    )

    # -- Quantum computer target -----------------------------------------------
    p.add_argument(
        "--qc",
        type=str,
        default=None,
        help=(
            "pyQuil quantum computer name. "
            "Examples: 8q-qvm | Ankaa-3 | Ankaa-9Q-3. "
            "Only used when --qaoa is set. Default: auto-sized QVM."
        ),
    )

    # -- Pipeline hyper-parameters ---------------------------------------------
    p.add_argument(
        "--max-size",
        type=int,
        default=8,
        help="Max nodes per QAOA subgraph (qubit budget). Default 8 for Ankaa-3 gate limit.",
    )
    p.add_argument(
        "--top-t",
        type=int,
        default=10,
        help="Number of top-t solutions to keep per subgraph for merging.",
    )
    p.add_argument(
        "--method",
        choices=["separator", "community"],
        default="separator",
        help=(
            "Partitioning strategy. "
            "'separator' = NaiveLGP (minimum vertex separator, recommended). "
            "'community' = greedy modularity communities (faster for large graphs)."
        ),
    )

    # -- QAOA circuit parameters -----------------------------------------------
    p.add_argument(
        "--layers",
        type=int,
        default=1,
        help="QAOA circuit depth p (default 1 for noisy hardware).",
    )
    p.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of measurement shots per QAOA circuit evaluation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for COBYLA initial parameters and stub backend.",
    )

    # -- Hardware convenience preset -------------------------------------------
    p.add_argument(
        "--hardware",
        action="store_true",
        default=False,
        help="Ankaa-3 hardware preset: enables --qaoa --max-size 8 --layers 1 --shots 1024.",
    )

    # -- Output ----------------------------------------------------------------
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save best assignment + score as JSON to this path.",
    )

    return p.parse_args()


def find_parquet() -> Path:
    """Auto-discover a .parquet file in the current or parent directory."""
    for search_dir in [Path("."), Path("..")]:
        candidates = sorted(search_dir.glob("*.parquet"))
        if candidates:
            if len(candidates) > 1:
                print(f"[main] Multiple .parquet files found; using: {candidates[0]}")
            return candidates[0]
    print("ERROR: No .parquet file found. Pass the path as the first argument.")
    sys.exit(1)


def main() -> None:
    args = parse_args()

    # -- Apply --hardware preset -----------------------------------------------
    if args.hardware:
        args.qaoa = True
        args.max_size = 8
        args.layers = 1
        args.shots = 1024

    # -- Locate graph file -----------------------------------------------------
    graph_path = Path(args.graph) if args.graph else find_parquet()
    if not graph_path.exists():
        print(f"ERROR: File not found: {graph_path}")
        sys.exit(1)

    # -- Seed RNG for reproducibility ------------------------------------------
    random.seed(args.seed)

    # -- Patch solver module globals -------------------------------------------
    _solver_module.USE_PYQUIL   = args.qaoa
    _solver_module.LAYER_COUNT  = args.layers
    _solver_module.SHOTS        = args.shots
    _solver_module.SEED         = args.seed

    # -- Set up quantum computer if using QAOA ---------------------------------
    qc_name = args.qc
    if args.qaoa and qc_name:
        _solver_module.setup_qpu(qc_name)

    # -- Print run config ------------------------------------------------------
    print("=" * 60)
    print("  DC-QAOA Weighted Max-Cut Pipeline")
    print("=" * 60)
    print(f"  Graph      : {graph_path.resolve()}")
    print(f"  Backend    : {'pyQuil QAOA' if args.qaoa else 'stub (local)'}")
    if args.qaoa:
        print(f"  QC target  : {qc_name or 'auto (Nq-qvm)'}")
        print(f"  QAOA depth : p = {args.layers}")
        print(f"  Shots      : {args.shots}")
    print(f"  Max size   : {args.max_size} nodes/subgraph")
    print(f"  Top-t      : {args.top_t}")
    print(f"  Method     : {args.method}")
    print("=" * 60)

    # -- Run -------------------------------------------------------------------
    assignment, score = run_pipeline(
        graph_path,
        max_size=args.max_size,
        top_t=args.top_t,
        method=args.method,
        qc_name=qc_name if args.qaoa else None,
    )

    # -- Result ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  FINAL Max-Cut score = {score:.6f}")
    print(f"{'=' * 60}")

    if args.output:
        out = Path(args.output)
        out.write_text(
            json.dumps(
                {
                    "score": score,
                    "assignment": {str(k): v for k, v in assignment.items()},
                    "config": {
                        "graph": str(graph_path),
                        "backend": "pyquil" if args.qaoa else "stub",
                        "qc": qc_name,
                        "max_size": args.max_size,
                        "top_t": args.top_t,
                        "method": args.method,
                        "layers": args.layers,
                        "shots": args.shots,
                    },
                },
                indent=2,
            )
        )
        print(f"[main] Assignment saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
