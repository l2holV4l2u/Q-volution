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
dc_qaoa/
├── graph_loader.py         # parquet -> nx.Graph (auto-detects column names)
├── partitioner.py          # recursive vertex-separator partitioning -> PartitionNode tree
├── solver.py               # QAOA backend (pyQuil) + stub for local testing
├── merger.py               # top-t merge with 2^|S| separator enumeration (GR policy)
├── scorer.py               # maxcut_score(G, assignment) objective
├── pipeline.py             # 6-step orchestration (partition -> solve -> merge -> polish)
├── main.py                 # CLI entry point
├── benchmark.py            # Compare pipeline vs classical baselines
└── resource_estimation.py  # Resource analysis: standard QAOA vs DC-QAOA
```

## Running
```bash
pip install pandas pyarrow networkx numpy scipy pyquil

# Local testing (stub backend, no QPU needed)
python main.py ../dataset_A.parquet
python main.py --max-size 8 --top-t 10 --output result.json ../dataset_B.parquet

# QVM simulation (requires quilc -S and qvm -S running)
python main.py --qaoa --qc 8q-qvm ../dataset_A.parquet

# Rigetti QPU (requires QCS credentials via `qcs auth login`)
python main.py --qaoa --qc Ankaa-3 ../dataset_B.parquet

# Hardware preset (enables QAOA, max_size=8, p=1, 1024 shots)
python main.py --hardware ../dataset_B.parquet

# Resource estimation report
python resource_estimation.py ../dataset_B.parquet

# Benchmark vs classical baselines
python benchmark.py ../dataset_B.parquet
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
- **Stub (default):** brute-force <=24 nodes, random+local-search otherwise.
  Local search uses O(deg(v)) delta scoring. For local pipeline testing.
- **pyQuil QAOA (`--qaoa`):** weighted QAOA circuit with CNOT-Rz-CNOT cost layers
  (quilc compiles to native CZ/iSWAP), Rx mixer, multi-start COBYLA (5 inits).
  Parametric circuit compiled once, run many times via memory_map.
  Logs native 2Q gate count estimate per subgraph and warns if >100.

### QAOA multi-start
The pyQuil backend runs COBYLA from 5 random initial parameter vectors and keeps
the best result, avoiding poor local minima in the VQE landscape.

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
2. Community-based fallback (greedy modularity)
3. Simple half-split last resort for degenerate cases
4. Recursion stops when subgraph has <= `max_size` nodes (default 8)

## Known Results (stub backend, max_size=8)
| Dataset | Nodes | Score | Total Weight | Approx Ratio |
|---------|-------|-------|-------------|-------------|
| A | 21 | 3728.41 | 4215.67 | 0.8844 |
| B | 180 | 7099.57 | 7465.71 | 0.9510 |

## Tuning Parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_size` | 8 | Max qubits per subgraph (hardware-compatible, all <=100 native 2Q gates) |
| `top_t` | 10 | Solutions kept per subgraph -- higher = better merge quality, slower |
| `--layers` | 1 | QAOA circuit depth p (p=1 for noisy hardware) |
| `--shots` | 1024 | Measurement shots per QAOA circuit |
| `--seed` | 42 | Random seed for reproducibility |
| `--hardware` | off | Convenience preset: --qaoa --max-size 8 --layers 1 --shots 1024 |
| `--qc` | auto | pyQuil quantum computer name (e.g. "8q-qvm", "Ankaa-3") |
