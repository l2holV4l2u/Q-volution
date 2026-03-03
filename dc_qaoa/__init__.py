"""Package initializer for the DC-QAOA Max-Cut solver.

This turns the ``dc_qaoa`` directory into an importable Python package.  It
re-exports the core solver submodules so that users can ``import dc_qaoa``
and access the pipeline programmatically.

Example::

    >>> import dc_qaoa
    >>> from dc_qaoa import pipeline
    >>> result, score = pipeline.run_pipeline("graph.parquet")

Tooling scripts (benchmark, resource estimation, visualization, QVM test)
live in the ``tools/`` directory alongside this package.
"""

from __future__ import annotations

# package metadata
__version__ = "0.1.0"

# public API -- core solver modules only
__all__ = [
    "config",
    "graph",
    "partitioner",
    "classical_backend",
    "quantum_backend",
    "solver",
    "merger",
    "pipeline",
    "graph_decomposition_reducer",
]

from . import (
    config,
    graph,
    partitioner,
    classical_backend,
    quantum_backend,
    solver,
    merger,
    pipeline,
    graph_decomposition_reducer,
)
