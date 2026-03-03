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
import time
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
    p.add_argument("--optimizer",
                   help="classical optimizer method")
    p.add_argument("--precondition", default=None,
                   choices=["analytic-p1", "measurement", "back-propagate"],
                   help="initial parameter strategy for QAOA (only with --quantum)")
    p.add_argument("--plot-loss", action="store_true", default=False,
                   help="Plot loss curve(s) after optimization (only with --quantum).")
    p.add_argument("--save-plots", type=str, default=None, metavar="DIR",
                   help="Save loss plots as PNGs to DIR instead of displaying interactively. "
                        "Useful for headless server runs.")
    return p.parse_args()


def main() -> None:
    t_start = time.perf_counter()
    args = parse_args()

    graph_path = Path(args.input)
    if not graph_path.exists():
        print(f"ERROR: File not found: {graph_path}")
        sys.exit(1)

    _config.USE_QUANTUM  = args.quantum
    if args.optimizer:    _config.OPTIMIZER    = args.optimizer.upper()
    if args.precondition: _config.PRECONDITION = args.precondition

    if args.quantum:
        _solver_module.setup_qpu(args.qc)

    print("=" * 60)
    print("  DC-QAOA Weighted Max-Cut Pipeline")
    print("=" * 60)
    print(f"  Graph   : {graph_path}")
    print(f"  Backend : {'quantum (pyQuil QAOA) -- ' + args.qc if args.quantum else 'classical'}")
    if args.quantum:
        print(f"  Optimizer   : {_config.OPTIMIZER}")
        print(f"  Precondition: {_config.PRECONDITION or 'none (random init)'}")
    print("=" * 60)

    assignment, score = run_pipeline(
        graph_path,
        qc_name=args.qc if args.quantum else None,
    )
    t_end = time.perf_counter()

    print(f"\n{'=' * 60}")
    print(f"  FINAL Max-Cut score = {score:.6f}")
    print(f"  Total time elapsed: {t_end-t_start:.2f} seconds")
    print(f"{'=' * 60}")

    # โ”€โ”€ Loss plots โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    if args.quantum and args.plot_loss:
        try:
            from dc_qaoa.quantum_backend import ITER_LOSS_HISTORY, LOSS_LABELS
            from dc_qaoa.visualization import plot_loss_history
        except ImportError:
            sys.path.insert(0, "./")
            from dc_qaoa.quantum_backend import ITER_LOSS_HISTORY, LOSS_LABELS
            from dc_qaoa.visualization import plot_loss_history

        save_dir = Path(args.save_plots) if args.save_plots else Path("output")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot one curve only: choose the longest iteration history.
        iter_candidates = [
            (sg_id, losses)
            for sg_id, losses in ITER_LOSS_HISTORY.items()
            if losses
        ]

        if not iter_candidates:
            print("[plot] No iteration loss history recorded.")
        else:
            sg_id, losses = max(iter_candidates, key=lambda item: len(item[1]))
            dataset_name = graph_path.stem
            precond_name = _config.PRECONDITION or "none"
            title = (
                f"Loss History (iteration) - {_config.OPTIMIZER} | "
                f"precondition={precond_name} | dataset={dataset_name}"
            )
            save_path = save_dir / (
                f"loss_{_config.OPTIMIZER}_pre-{precond_name}_data-{dataset_name}.png"
            )
            plot_loss_history(
                losses,
                title=title,
                best_so_far=True,
                relative_to_final=False,
                x_axis_label="Iteration",
                save_path=save_path,
            )
            print(
                f"[plot] 1 loss curve ({len(losses)} iterations) saved to {save_path}. "
                f"Selected subgraph id={sg_id}."
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted')
        sys.exit(0)


