"""Package initializer for the DC-QAOA Max-Cut project.

This turns the ``dc_qaoa`` directory into an importable Python package.  It
re-exports the most important submodules so that users can ``import
dc_qaoa`` and access the core functionality programmatically.

Example::

    >>> import dc_qaoa
    >>> from dc_qaoa import pipeline
    >>> result, score = pipeline.run_pipeline("graph.parquet")

"""

from __future__ import annotations

# ``main.py`` lives alongside the package and is not part of the
# dc_qaoa package itself, so we don’t import it here.  Consumers can run the
# CLI via ``python -m dc_qaoa.main`` or use the installed console script.

# package metadata
__version__ = "0.1.0"

# public API
__all__ = [
    "graph_loader",
    "partitioner",
    "solver",
    "merger",
    "scorer",
    "pipeline",
    "visualize_cut",
    "resource_estimation",
    "benchmark",
    "bruteforce",
]

from . import graph_loader, partitioner, solver, merger, scorer
from . import pipeline, visualize_cut, resource_estimation
from . import benchmark, bruteforce
