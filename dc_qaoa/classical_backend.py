"""
classical_backend.py -- Classical fallback solver for weighted Max-Cut.

Public:
    run_classical(nodes) -> list[Solution]
"""
from __future__ import annotations

import itertools
import random

Solution = dict  # {node_id: +1 | -1}

# Exact enumeration grows as 2^n and becomes impractical quickly.
_EXACT_ENUMERATION_LIMIT = 20
_HEURISTIC_TRIALS = 256


def run_classical(nodes: list) -> list[Solution]:
    """
    Return candidate spin assignments for classical solving.

    For small subgraphs (n <= 20), this remains exact brute force.
    For larger subgraphs, return a bounded set of random candidates to avoid
    non-terminating benchmark runs.
    """
    n = len(nodes)
    if n <= _EXACT_ENUMERATION_LIMIT:
        return [
            dict(zip(nodes, bits))
            for bits in itertools.product([-1, 1], repeat=n)
        ]

    rng = random.Random(42 + n)
    return [
        {v: rng.choice([-1, 1]) for v in nodes}
        for _ in range(_HEURISTIC_TRIALS)
    ]
