"""
main.py -- Entry point for the DC-QAOA Max-Cut pipeline.

Classical (no hardware needed):
    python main.py graph.parquet

Quantum via QVM (requires quilc -S and qvm -S running):
    python main.py --quantum graph.parquet

Quantum on Rigetti QPU (requires qcs auth login):
    python main.py --quantum --qc Ankaa-3 graph.parquet
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

try:
    from dc_qaoa import config as _config
    from dc_qaoa import solver as _solver_module
    from dc_qaoa.pipeline import run_pipeline
except ImportError:
    sys.path.insert(0, "./")
    from dc_qaoa import config as _config
    from dc_qaoa import solver as _solver_module
    from dc_qaoa.pipeline import run_pipeline

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DC-QAOA Weighted Max-Cut Pipeline")
    p.add_argument("input", help="Path to graph .parquet file.")
    p.add_argument("--quantum", action="store_true", default=False,
                   help="Use QAOA quantum backend via pyQuil.")
    p.add_argument("--qc", type=str, default="8q-qvm",
                   help="pyQuil quantum computer name (only used with --quantum). "
                        "Examples: 8q-qvm | Ankaa-3")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    graph_path = Path(args.input)
    if not graph_path.exists():
        print(f"ERROR: File not found: {graph_path}")
        sys.exit(1)

    _config.USE_QUANTUM = args.quantum

    if args.quantum:
        _solver_module.setup_qpu(args.qc)

    print("=" * 60)
    print("  DC-QAOA Weighted Max-Cut Pipeline")
    print("=" * 60)
    print(f"  Graph   : {graph_path}")
    print(f"  Backend : {'quantum (pyQuil QAOA) -- ' + args.qc if args.quantum else 'classical'}")
    print("=" * 60)

    assignment, score = run_pipeline(
        graph_path,
        qc_name=args.qc if args.quantum else None,
    )

    print(f"\n{'=' * 60}")
    print(f"  FINAL Max-Cut score = {score:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted')
        sys.exit(0)
