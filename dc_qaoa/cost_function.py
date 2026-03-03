from . import config
from pyquil import Program
from .graph import edges

import numpy as np

def qaoa_cut_score(edges: edges, bitstring: list[0 | 1]) -> float:
    total_cut  = np.sum(
        w for (u, v, w) in edges if bitstring[u] != bitstring[v]
    )
    return total_cut