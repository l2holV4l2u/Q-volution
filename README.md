# DC-QAOA — Divide-and-Conquer QAOA for Weighted Max-Cut

Q-volution 2025 competition entry (Aqora.io / Rigetti). Solves the Maximum Power
Energy Section (MPES) problem on electrical grids, which is a **weighted Max-Cut**:

```
C(z) = 1/2 * Σ_(i,j)∈E  w_ij (1 - z_i z_j),   z ∈ {+1,-1}^|V|
```

Standard QAOA needs one qubit per node (180 for Dataset B), far beyond what
Rigetti Ankaa-3 can run usefully. DC-QAOA partitions the graph into ≤8-node
subgraphs, solves each with QAOA (or exact brute-force), and merges the pieces
back up a partition tree.

---

## Project Structure

```
.
├── main.py                       # CLI entry point
├── setup.py                      # pip install .
├── docker-compose.yml            # quilc + qvm containers
│
├── dc_qaoa/                      # Core package
│   ├── config.py                 # Runtime constants, patched by CLI flags
│   ├── graph.py                  # .parquet / .csv -> nx.Graph
│   ├── partitioner.py            # Recursive NaiveLGP partitioning -> PartitionNode tree
│   ├── solver.py                 # Backend dispatch + maxcut_score
│   ├── classical_backend.py      # Exact brute-force over all 2^n assignments
│   ├── quantum_backend.py        # pyQuil QAOA + angle optimisation (SA/SLSQP/COBYLA/COBYQA)
│   ├── circuit.py                # QAOA circuit construction
│   ├── cost_function.py          # Loss evaluation from bitstring samples
│   ├── precondition.py           # Initial-angle strategies (analytic-p1, back-propagate, ...)
│   ├── merger.py                 # GR-policy merge through the partition tree
│   ├── pipeline.py               # Orchestrates load -> partition -> solve -> merge -> score
│   ├── visualization.py          # Loss-curve / graph plotting helpers
│   └── graph_decomposition_reducer.py  # QUBO reduction (Ponce et al. arXiv:2306.00494)
│
├── tools/
│   ├── benchmark.py              # Optimizer x precondition benchmark (9 combos)
│   ├── bruteforce.py             # Exact optimum (A) / SDP bound + SA (B)
│   ├── resource_estimation.py    # Why standard QAOA is infeasible on Ankaa-3
│   ├── visualize_cut.py          # Draw the Max-Cut result
│   └── test_qvm.py               # Smoke test against local QVM
│
├── datasets/
│   ├── dataset_A.parquet / .csv  # 21 nodes, 28 edges (South Carolina grid subset)
│   ├── dataset_B.parquet / .csv  # 180 nodes, 226 edges
│   └── datagenerator.py
│
└── output/                       # Benchmark spreadsheets / plots
```

---

## Pipeline

1. **Load** (`graph.py`) — read edge list, edge weight = line admittance.
2. **Partition** (`partitioner.py`) — recursively split until every leaf has ≤ `max_size` nodes.
   NaiveLGP finds the smallest vertex separator S; S is kept in **both** halves so no
   cross-separator edge is lost. Result is a binary `PartitionNode` tree.
3. **Solve leaves** (`solver.py`) — `config.USE_QUANTUM` picks the backend:
   - classical: enumerate all 2^n spin assignments (default).
   - quantum: parametric QAOA circuit (|+>^n, p × [cost layer + mixer], measure),
     compiled once via quilc, angles optimised with the chosen optimizer.
   Each leaf returns up to `top_t` solutions `{node_id: ±1}`.
4. **Merge** (`merger.py`) — bottom-up. At each internal node, for every (left, right)
   solution pair enumerate all 2^|S| separator assignments, score on the subtree
   subgraph, keep the top-t. A greedy local-search polish runs on the full graph at the end.
5. **Score** — `maxcut_score(G, assignment)`; prints score, total weight, approximation ratio.

---

## Setup

Python ≥ 3.11.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install .                     # pandas, pyarrow, networkx, numpy, scipy

# Optional, for tools/ and the quantum backend
pip install matplotlib openpyxl cvxpy pyquil
```

> **Windows note:** `pip install -e .` needs an Administrator shell. The no-install
> alternative is setting `PYTHONPATH` to the repo root (the `.env` file does this for
> VSCode). In a plain terminal:
>
> ```powershell
> $env:PYTHONPATH = $PWD   # PowerShell
> ```
> ```bash
> export PYTHONPATH=.      # bash / Git Bash
> ```

**Quantum backend** needs quilc and a QVM running:

```bash
docker compose up -d      # or: quilc -S  and  qvm -S  in two terminals
qcs auth login            # only for real QPU runs
```

---

## Running

```bash
# Classical (no hardware)
python main.py datasets/dataset_A.parquet

# Quantum via local QVM
python main.py --quantum datasets/dataset_A.parquet

# Quantum on Rigetti QPU
python main.py --quantum --qc Ankaa-3 datasets/dataset_B.parquet

# Pick optimizer / initial angles, save loss plot
python main.py --quantum --optimizer COBYLA --precondition analytic-p1 --plot-loss datasets/dataset_A.parquet
```

| Flag             | Default  | Description                                                    |
| ---------------- | -------- | -------------------------------------------------------------- |
| `--quantum`      | off      | Use the pyQuil QAOA backend                                    |
| `--qc`           | `8q-qvm` | pyQuil quantum computer name (with `--quantum`)                |
| `--optimizer`    | `SA`     | `SA` (dual annealing), `SLSQP`, `COBYLA`, `COBYQA`             |
| `--precondition` | none     | `analytic-p1`, `measurement`, `back-propagate`                 |
| `--plot-loss`    | off      | Plot the loss curve after optimisation                         |
| `--save-plots`   | `output` | Directory for loss PNGs (headless runs)                        |

`main_mac.py` is the same entry point pinned to a QVM on port 6000.

---

## Resource Estimation

Ankaa-3 practical target: ~10 qubits and ≤100 two-qubit gates per circuit.
`tools/resource_estimation.py` with repo defaults:

| Approach                   | Dataset A                       | Dataset B                         |
| -------------------------- | ------------------------------- | --------------------------------- |
| Standard QAOA, p=1         | 21 qubits, ~252 iSWAPs, F≈0.28  | 180 qubits, ~2034 iSWAPs, F≈4e-5  |
| DC-QAOA, max_size=8, p=1   | ≤8 qubits, ≤99 iSWAPs/subgraph  | ≤8 qubits, ≤90 iSWAPs/subgraph    |

```bash
python tools/resource_estimation.py datasets/dataset_A.parquet datasets/dataset_B.parquet
```

---

## Tools

Run every tool from the repo root.

### `benchmark.py`

Runs every optimizer × precondition combination (3 × 3 = 9), `--runs` times each
(default 5). Records best-so-far loss per iteration, `gamma`/`beta` trajectories,
and the final bitstring distribution. Saves to `output/`:

- `avg_loss_params_<dataset>_all_combinations.png`
- `avg_final_probability_<dataset>_all_combinations.png`
- `benchmark_data_<dataset>_all_combinations.xlsx`

```bash
python tools/benchmark.py datasets/dataset_A.parquet
python tools/benchmark.py datasets/dataset_B.parquet \
  --methods SLSQP COBYLA COBYQA \
  --preconditions none back-propagate analytic-p1 \
  --runs 5 --qc 8q-qvm --output-dir output
```

### `bruteforce.py`

- Dataset A: exact brute-force over 2^21 assignments (~50 s).
- Dataset B: Goemans-Williamson SDP upper bound (needs `cvxpy`) + simulated annealing.

```bash
python tools/bruteforce.py datasets/dataset_A.parquet
python tools/bruteforce.py datasets/dataset_B.parquet
```

### `visualize_cut.py`

Runs the pipeline and saves `maxcut_<dataset>.png` (cut edges green, uncut red
dashed, nodes coloured by spin). `MAX_SIZE` / `TOP_T` are set at the top of the file.

```bash
python tools/visualize_cut.py datasets/dataset_A.parquet
```

### `test_qvm.py`

Runs the pipeline twice — classical, then QVM — and compares scores.

```bash
docker compose up -d
python tools/test_qvm.py A     # or B (default)
```

---

## Config

Defaults live in `dc_qaoa/config.py`; CLI flags override them at startup.

```python
USE_QUANTUM  = False   # True -> quantum backend
OPTIMIZER    = "SA"    # "SA" | "SLSQP" | "COBYLA" | "COBYQA"
PRECONDITION = None    # None | "analytic-p1" | "measurement" | "back-propagate"
MIXER_MODE   = "X"     # "X" (standard) | "XX" (graph-coupled) | "XY"
LAYER_COUNT  = 1       # QAOA depth p
SHOTS        = 1024    # Measurement shots per circuit run
SEED         = 42
MAXITER      = 100     # Optimizer iterations
```

`max_size=8` and `top_t=10` are defaults of `run_pipeline()` in `dc_qaoa/pipeline.py`.

---

## Known Results (classical backend, max_size=8)

| Dataset | Nodes | Score   | Total Weight | Ratio  |
| ------- | ----- | ------- | ------------ | ------ |
| A       | 21    | 3728.41 | 4215.67      | 0.8844 |
| B       | 180   | 7099.57 | 7465.71      | 0.9510 |

Dataset A matches the exact brute-force optimum. Dataset B (2^180) is not brute-forceable.
