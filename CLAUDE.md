# DC-QAOA Pipeline -- CLAUDE.md

## Competition Context
**Q-volution 2025 -- Energy Grid Optimization** (Aqora.io, hosted by Rigetti Computing)

**Problem:** Maximize the Maximum Power Energy Section (MPES) of an electrical grid,
which is mathematically a **weighted Max-Cut** problem:

```
MPES = max C(z)  where  C(z) = (1/2) * sum_{(i,j) in E} w_ij * (1 - z_i * z_j)
       z in {+1,-1}^|V|
```

**Datasets:**
- Problem A: 21 nodes, 28 edges (South Carolina grid subset)
- Problem B: 180 nodes, 226 edges (larger grid section)
- Edge weights = line admittances (higher = stronger electrical link)

**Competition deliverables:**
1. Resource estimation -- why standard QAOA is infeasible for Problem B
2. Optimization challenge -- strategy to solve despite qubit limitations (DC-QAOA)
3. Benchmarking -- compare quantum solution vs classical baselines
4. Hardware execution -- run on Rigetti Ankaa-3 QPU via QCS SDK + pyQuil

**Hardware constraints (Rigetti Ankaa-3):**
- SDK: QCS SDK + pyQuil (as specified by competition)
- Native gates: CZ/iSWAP (2-qubit), RZ(theta), RX(k*pi/2) (1-qubit)
- Target: ~10-qubit circuits with <= 100 two-qubit gates
- Square-grid topology -- non-adjacent qubits need SWAP routing
- quilc compiler handles CNOT -> native gate decomposition automatically

## Project Overview
Divide-and-Conquer QAOA for weighted Max-Cut. The pipeline partitions the graph into
<=8-node subgraphs (hardware default), solves each with QAOA, then merges solutions
via GR policy with 2^|S| separator enumeration.

## File Structure
```
main.py                       # CLI entry point (--quantum, --qc flags)
dc_qaoa/
├── config.py               # Runtime constants (SHOTS, LAYER_COUNT, SEED, …)
├── graph_loader.py         # parquet/csv -> nx.Graph (auto-detects column names)
├── partitioner.py          # recursive vertex-separator partitioning -> PartitionNode tree
├── solver.py               # backend dispatch + maxcut_score
├── quantum_backend.py      # pyQuil QAOA circuit + dual_annealing optimisation
├── classical_backend.py    # exact brute-force over all 2^n assignments
├── merger.py               # top-t merge with 2^|S| separator enumeration (GR policy)
└── pipeline.py             # 5-step orchestration (load -> partition -> solve -> merge -> score)
tools/
├── benchmark.py            # Compare DC-QAOA vs classical baselines
├── bruteforce.py           # Exact brute-force optimal (Dataset A only, 2^21)
├── visualize_cut.py        # Visualize Max-Cut result on graph
├── resource_estimation.py  # Resource analysis: standard QAOA vs DC-QAOA
└── test_qvm.py             # Smoke test against local QVM (Docker)
```

## Setup
```bash
pip install .
# Optional extras for tools/
pip install matplotlib cvxpy pyquil
```

## Running
```bash
# Classical backend (no hardware needed)
python main.py datasets/dataset_A.parquet

# QVM simulation (requires quilc -S and qvm -S running)
python main.py --quantum --qc 8q-qvm datasets/dataset_A.parquet

# Rigetti QPU (requires QCS credentials via `qcs auth login`)
python main.py --quantum --qc Ankaa-3 datasets/dataset_B.parquet

# Resource estimation report
python tools/resource_estimation.py datasets/dataset_B.parquet

# Benchmark vs classical baselines
python tools/benchmark.py datasets/dataset_B.parquet
```

## Resource Estimation
**Why standard QAOA is infeasible for Problem B (180 nodes):**
- Requires 180 qubits (Ankaa-3 practical limit: ~10)
- p=1: 226 edges x 2 CNOTs x ~3 native 2Q gates x 1.5 routing = ~2034 gates (limit: 100)
- Circuit fidelity at 0.995/gate: 0.995^2034 ~ 0.00004 (unusable)

**DC-QAOA with max_size=8, p=1:**
- Partitions into ~60+ subgraphs of <=8 nodes each
- Max ~90 native 2Q gates per subgraph (within 100 limit)
- Worst-case fidelity ~0.64 (workable with error mitigation)

**Gate count formula:** `native_2Q = num_edges * 2 * 3 * 1.5 * layers`

## Key Design Decisions

### Separator duplication
Separator nodes are included in **both** subgraphs (`G[A | S]` and `G[B | S]`).
This preserves all cross-separator edges so no cut weight is lost.

### Merge (GR policy)
For each pair of (left, right) top-t solutions, all `2^|S|` separator spin assignments
are tried. Scoring uses the subtree-induced subgraph (not full graph) to avoid
penalizing partial assignments. Top-t diversity is propagated through the entire
merge tree. Sparse dicts use None sentinels; merge combines them with real-value
priority (never let None overwrite a real spin).

### Post-merge polish
After merging, a final greedy local search pass runs on the full graph to squeeze
out any remaining improvement from cross-partition interactions.

### Solver backends
- **Classical (default):** exact brute-force over all 2^n spin assignments.
- **pyQuil QAOA (`--quantum`):** weighted QAOA circuit with CNOT-RZ-CNOT cost layers
  (quilc compiles to native CZ/iSWAP), configurable mixer (X/XX/XY).
  Parametric circuit compiled once, run many times via `memory_map`.
  Angles optimised with `scipy.dual_annealing` (SA_MAXITER iterations).

## Data Structures
```python
@dataclass
class PartitionNode:
    graph: nx.Graph       # subgraph at this node
    separator: set        # node IDs shared with sibling
    left: PartitionNode
    right: PartitionNode
    is_leaf: bool         # True -> send to QAOA

Solution = dict[int, int]  # {node_id: +1 | -1}
```

## Partitioning Strategy
1. NaiveLGP -- minimum vertex separator (tries sep_size=1,2,...,8)
2. Simple half-split fallback for degenerate cases (complete graph or single node)
3. Recursion stops when subgraph has <= `max_size` nodes (default 8)

## Known Results (stub backend, max_size=8)
| Dataset | Nodes | Score | Total Weight | Approx Ratio |
|---------|-------|-------|-------------|-------------|
| A | 21 | 3728.41 | 4215.67 | 0.8844 |
| B | 180 | 7099.57 | 7465.71 | 0.9510 |

## Exact Optimal (brute-force, Dataset A only)
| Dataset | Optimal Score | Total Weight | Optimal Ratio |
|---------|--------------|-------------|---------------|
| A | 3728.41 | 4215.67 | 0.884419 |

DC-QAOA with max_size=8 matches the exact brute-force optimum on Dataset A.
Dataset B (180 nodes, 2^180 assignments) is not brute-forceable.

Run: `python tools/bruteforce.py datasets/dataset_A.parquet`

## Tuning Parameters
Edit `dc_qaoa/config.py` directly to change defaults.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `LAYER_COUNT` | 1 | QAOA circuit depth p (p=1 for noisy hardware) |
| `SHOTS` | 1024 | Measurement shots per QAOA circuit |
| `SEED` | 42 | Random seed for reproducibility |
| `MIXER_MODE` | `"X"` | Mixer type: `"X"` (standard), `"XX"` (graph-coupled), `"XY"` |
| `MAXITER` | 1000 | maximum optimization iterations (QAOA only) |

Pipeline parameters (`max_size`, `top_t`) are hardcoded in `run_pipeline()` defaults (`max_size=8`, `top_t=10`).
