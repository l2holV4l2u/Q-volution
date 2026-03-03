"""
config.py -- Runtime-configurable parameters for the DC-QAOA solver.

Patched at startup by main.py based on CLI flags.
All backend modules read these as `config.VAR` (not `from config import VAR`)
so that patches applied after import still take effect.
"""

# ── Backend selection ──────────────────────────────────────────────────────────
USE_QUANTUM  = False   # set True when pyquil is installed & QVM/QPU ready
QC_NAME = None
OPTIMIZER = "SA"
# ── QAOA circuit ───────────────────────────────────────────────────────────────
MIXER_MODE   = "X"     # "X" (standard), "XX" (graph-coupled), "XY" (XY-mixer)
LAYER_COUNT  = 1       # QAOA depth p  (p=1 for noisy hardware; increase for sim)
SHOTS        = 1024    # measurement shots per circuit run
SEED         = 42
MAXITER   = 100    # simulated annealing iterations
