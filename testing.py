from __future__ import annotations

import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    from dc_qaoa import config as _config
    from dc_qaoa import solver as _solver_module
    from dc_qaoa.pipeline import run_pipeline
    from dc_qaoa.quantum_backend import setup_qpu
    from dc_qaoa.graph import load_graph
    from dc_qaoa.visualization import draw_graph

except ImportError:
    sys.path.insert(0, "./")
    from dc_qaoa import config as _config
    from dc_qaoa import solver as _solver_module
    from dc_qaoa.pipeline import run_pipeline

print("This is a testing file for quick experiments and debugging. Not meant for production use.")