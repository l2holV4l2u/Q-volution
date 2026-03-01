# DC-QAOA — Divide-and-Conquer QAOA for Weighted Max-Cut

Q-volution 2025 competition entry. Solves the Maximum Power Energy Section (MPES)
problem on electrical grids, which is a **weighted Max-Cut** problem.

---

## Project Structure

```
.
├── main.py                   # CLI entry point
│
├── dc_qaoa/                  # Core package
│   ├── config.py             # All runtime-tunable constants (patched by CLI)
│   ├── graph_loader.py       # .parquet → nx.Graph
│   ├── partitioner.py        # Recursive NaiveLGP graph partitioning → PartitionNode tree
│   ├── solver.py             # Public API: backend dispatch + maxcut_score
│   ├── quantum_backend.py    # pyQuil QAOA circuit + simulated annealing optimisation
│   ├── classical_backend.py  # Exact brute-force over all 2^n assignments
│   ├── merger.py             # GR policy merge through the partition tree
│   └── pipeline.py           # Orchestrates all 5 steps end-to-end
│
├── tools/
│   ├── benchmark.py          # DC-QAOA vs classical baselines comparison
│   ├── bruteforce.py         # Exact 2^n brute-force (Dataset A only)
│   ├── resource_estimation.py # Why standard QAOA is infeasible for Dataset B
│   ├── visualize_cut.py      # Draw the Max-Cut result on the graph
│   └── test_qvm.py           # Smoke test against local QVM
│
└── datasets/
    ├── dataset_A.parquet     # 21 nodes, 28 edges (South Carolina grid subset)
    └── dataset_B.parquet     # 180 nodes, 226 edges (larger grid section)
```

---

## Data Flow & Pipeline

```
.parquet file
     │
     ▼
[graph_loader]  ──────────────────────────────────  Step 1
  load_graph()
  Reads edge list (node_a, node_b, weight columns).
  Returns nx.Graph with edge weights = line admittances.
     │
     ▼
[partitioner]  ────────────────────────────────────  Step 2
  recursive_partition(G, max_size)
  Recursively splits G until every leaf has ≤ max_size nodes.

  NaiveLGP: finds the smallest vertex separator S that disconnects
  G into A and B. Separator nodes S are included in BOTH subgraphs
  (A∪S and B∪S) so no cross-separator edges are lost.

  Builds a binary PartitionNode tree.
  Leaves are the subgraphs sent to the solver.
     │
     ▼
[solver → backend]  ───────────────────────────────  Step 3
  solve_subgraph(leaf.graph, top_t)
  Reads config.USE_QUANTUM to pick the backend:

    classical_backend  (USE_QUANTUM=False, default)
      Exact brute-force: enumerates all 2^n spin assignments.

    quantum_backend  (USE_QUANTUM=True, requires pyQuil)
      Builds a parametric QAOA circuit:
        - Initial state: |+>^n (Hadamard on all qubits)
        - p layers of: cost layer (CNOT-RZ-CNOT per edge)
                     + mixer layer (RX per qubit, default "X" mixer)
        - Measure all qubits
      Compiled once via quilc, then optimised with simulated annealing
      (dual_annealing) over the angle parameters (gammas, betas).
      Returns the SHOTS bitstring samples at the optimal angles.

  Each leaf gets a list of up to top_t Solution dicts: {node_id: +1|-1}
     │
     ▼
[merger]  ─────────────────────────────────────────  Step 4
  merge(G, partition_tree, subgraph_solutions, top_t)
  Walks the partition tree bottom-up (GR policy):

  At each internal node with separator S, left solutions, right solutions:
    1. For every pair (left_sol, right_sol):
         enumerate all 2^|S| spin assignments for separator nodes
         score each combined assignment on the subtree subgraph
    2. Keep the top-t highest-scoring unique assignments.

  Propagates diversity (top-t) up the tree until the root.
  Returns the single best global assignment.
     │
     ▼
[solver]  ─────────────────────────────────────────  Step 5
  maxcut_score(G, assignment)
  C(z) = Σ w_uv * (1 - z_u * z_v) / 2  for all edges (u,v)
  Prints final score, total weight, and approximation ratio.
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install .

# Optional extras for tools/
pip install matplotlib cvxpy pyquil
```

For QPU/QVM access (optional):
```bash
# Start Quil compiler and QVM servers in separate terminals
quilc -S
qvm -S

# Authenticate with Rigetti QCS for real QPU runs
qcs auth login
```

---

## Running

**Classical (no hardware needed):**
```bash
python main.py datasets/dataset_A.parquet
```

**Quantum via QVM (requires `quilc -S` and `qvm -S` running):**
```bash
python main.py --quantum datasets/dataset_A.parquet
```

**Quantum on Rigetti QPU (requires `qcs auth login`):**
```bash
python main.py --quantum --qc Ankaa-3 datasets/dataset_B.parquet
```

| Flag | Default | Description |
|------|---------|-------------|
| `--quantum` | off | Use QAOA quantum backend |
| `--qc` | `8q-qvm` | pyQuil quantum computer (only with `--quantum`) |

---

## Tools

All tools are run from the **project root** (`Q-volution 2025/`), not from inside `tools/`.

---

### `benchmark.py` — Compare DC-QAOA against classical baselines

Runs 5 methods side-by-side and prints scores, approximation ratios, timing, and a verdict.

```bash
python tools/benchmark.py datasets/dataset_A.parquet
python tools/benchmark.py datasets/dataset_B.parquet
```

Methods compared: random (1000 trials), greedy+LS, NetworkX Kernighan-Lin+LS, random+LS (500 trials), DC-QAOA pipeline (classical backend).

Tune `MAX_SIZE` and `TOP_T` at the top of the file to adjust the pipeline.

---

### `bruteforce.py` — Find the optimal / upper-bound Max-Cut score

- **Dataset A (21 nodes):** exact brute-force over all 2²¹ assignments (~50 s)
- **Dataset B (180 nodes):** SDP relaxation upper bound (`cvxpy` required) + simulated annealing best-known

```bash
python tools/bruteforce.py datasets/dataset_A.parquet   # exact optimal
python tools/bruteforce.py datasets/dataset_B.parquet   # SDP bound + SA
```

SDP requires `cvxpy`: `pip install cvxpy`

---

### `resource_estimation.py` — Hardware feasibility analysis

Shows why standard QAOA is infeasible for large graphs on Rigetti Ankaa-3, and proves DC-QAOA fits within qubit and gate limits.

```bash
python tools/resource_estimation.py datasets/dataset_B.parquet
python tools/resource_estimation.py datasets/dataset_A.parquet datasets/dataset_B.parquet  # both at once
```

No extra dependencies. Outputs qubit counts, iSWAP counts, routing overhead, circuit fidelity, and a per-subgraph breakdown.

---

### `visualize_cut.py` — Draw the Max-Cut result

Runs the DC-QAOA pipeline, then saves a dark-themed PNG of the graph with cut edges (green) and uncut edges (red dashed), nodes coloured by spin.

```bash
python tools/visualize_cut.py datasets/dataset_A.parquet   # → maxcut_dataset_A.png
python tools/visualize_cut.py datasets/dataset_B.parquet   # → maxcut_dataset_B.png
```

Output file is saved in the current directory. Tune `MAX_SIZE` and `TOP_T` at the top of the file.

---

### `test_qvm.py` — Smoke test against a local QVM (Docker)

Runs the full DC-QAOA pipeline twice — once with the classical backend, once with the pyQuil QVM — and compares scores.

**Prerequisites:**
```bash
# 1. Start the QVM and Quil compiler via Docker
docker compose up -d

# 2. Confirm both containers are running
docker compose ps
```

**Run:**
```bash
python tools/test_qvm.py A   # Dataset A (21 nodes, faster)
python tools/test_qvm.py B   # Dataset B (180 nodes, default)
```

If the QVM run fails, the script prints Docker troubleshooting commands. Tune `MAX_SIZE`, `TOP_T`, and `QC_NAME` at the top of the file.

---

## Config

All tunable parameters live in `dc_qaoa/config.py` and are patched at startup by
`main.py`. To change defaults permanently, edit that file directly.

```python
USE_QUANTUM  = False   # True → quantum backend
MIXER_MODE   = "X"     # "X" (standard) | "XX" (graph-coupled) | "XY"
LAYER_COUNT  = 1       # QAOA depth p
SHOTS        = 1024    # Measurement shots per circuit run
SEED         = 42
SA_MAXITER   = 1000    # Simulated annealing iterations (quantum backend only)
```
