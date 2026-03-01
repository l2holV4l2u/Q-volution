"""
classical_backend.py -- Classical fallback solver for weighted Max-Cut.

Public:
    run_classical(nodes) -> list[Solution]  (all 2^n assignments)
"""
from __future__ import annotations

import itertools

Solution = dict  # {node_id: +1 | -1}


def run_classical(nodes: list) -> list[Solution]:
    """Return all 2^n spin assignments (exact brute-force)."""
    return [
        dict(zip(nodes, bits))
        for bits in itertools.product([-1, 1], repeat=len(nodes))
    ]
