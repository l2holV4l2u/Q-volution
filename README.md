# DC-QAOA โ€” Divide-and-Conquer QAOA for Weighted Max-Cut

Q-volution 2025 competition entry. Solves the Maximum Power Energy Section (MPES)
problem on electrical grids, which is a **weighted Max-Cut** problem.

---

## Project Structure

```
.
โ”โ”€โ”€ main.py                   # CLI entry point
โ”
โ”โ”€โ”€ dc_qaoa/                  # Core package
โ”   โ”โ”€โ”€ config.py             # All runtime-tunable constants (patched by CLI)
โ”   โ”โ”€โ”€ graph_loader.py       # .parquet โ’ nx.Graph
โ”   โ”โ”€โ”€ partitioner.py        # Recursive NaiveLGP graph partitioning โ’ PartitionNode tree
โ”   โ”โ”€โ”€ solver.py             # Public API: backend dispatch + maxcut_score
โ”   โ”โ”€โ”€ quantum_backend.py    # pyQuil QAOA circuit + simulated annealing optimisation
โ”   โ”โ”€โ”€ classical_backend.py  # Exact brute-force over all 2^n assignments
โ”   โ”โ”€โ”€ merger.py             # GR policy merge through the partition tree
โ”   โ”โ”€โ”€ pipeline.py           # Orchestrates all 5 steps end-to-end
โ”   โ””โ”€โ”€ graph_decomposition_reducer.py  # graph decomposition reduction
โ”
โ”โ”€โ”€ tools/
โ”   โ”โ”€โ”€ benchmark.py          # DC-QAOA vs classical baselines comparison
โ”   โ”โ”€โ”€ bruteforce.py         # Exact 2^n brute-force (Dataset A only)
โ”   โ”โ”€โ”€ resource_estimation.py # Why standard QAOA is infeasible for Dataset B
โ”   โ”โ”€โ”€ visualize_cut.py      # Draw the Max-Cut result on the graph
โ”   โ””โ”€โ”€ test_qvm.py           # Smoke test against local QVM
โ”
โ””โ”€โ”€ datasets/
    โ”โ”€โ”€ dataset_A.parquet     # 21 nodes, 28 edges (South Carolina grid subset)
    โ””โ”€โ”€ dataset_B.parquet     # 180 nodes, 226 edges (larger grid section)
```

---

## Data Flow & Pipeline

```
.parquet file
     โ”
     โ–ผ
[graph_loader]  โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€  Step 1
  load_graph()
  Reads edge list (node_a, node_b, weight columns).
  Returns nx.Graph with edge weights = line admittances.
     โ”
     โ–ผ
[partitioner]  โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€  Step 2
  recursive_partition(G, max_size)
  Recursively splits G until every leaf has โค max_size nodes.

  NaiveLGP: finds the smallest vertex separator S that disconnects
  G into A and B. Separator nodes S are included in BOTH subgraphs
  (AโชS and BโชS) so no cross-separator edges are lost.

  Builds a binary PartitionNode tree.
  Leaves are the subgraphs sent to the solver.
     โ”
     โ–ผ
[solver โ’ backend]  โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€  Step 3
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
     โ”
     โ–ผ
[merger]  โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€  Step 4
  merge(G, partition_tree, subgraph_solutions, top_t)
  Walks the partition tree bottom-up (GR policy):

  At each internal node with separator S, left solutions, right solutions:
    1. For every pair (left_sol, right_sol):
         enumerate all 2^|S| spin assignments for separator nodes
         score each combined assignment on the subtree subgraph
    2. Keep the top-t highest-scoring unique assignments.

  Propagates diversity (top-t) up the tree until the root.
  Returns the single best global assignment.
     โ”
     โ–ผ
[solver]  โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€  Step 5
  maxcut_score(G, assignment)
  C(z) = ฮฃ w_uv * (1 - z_u * z_v) / 2  for all edges (u,v)
  Prints final score, total weight, and approximation ratio.
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt 2>/dev/null || pip install pandas pyarrow networkx numpy scipy

# Optional extras for tools/
pip install matplotlib cvxpy pyquil
```

> **Windows note:** `pip install -e .` requires an Administrator shell (to write
> `dc_qaoa.exe` into `Scripts/`). The simpler alternative โ€” no install needed โ€”
> is the `.env` file already in the repo root. VSCode picks it up automatically.
> For a plain terminal, set it once per session:
>
> ```powershell
> $env:PYTHONPATH = $PWD   # PowerShell
> ```
>
> ```bash
> export PYTHONPATH=.      # bash / Git Bash
> ```

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

| Flag        | Default  | Description                                     |
| ----------- | -------- | ----------------------------------------------- |
| `--quantum` | off      | Use QAOA quantum backend                        |
| `--qc`      | `8q-qvm` | pyQuil quantum computer (only with `--quantum`) |

---

## Problem Constraints and Resource Estimation

From the competition statement (`Q-volution problem statement.html`):

- Objective: weighted Max-Cut / MPES  
  `C(z) = 1/2 * sum_(i,j in E) w_ij (1 - z_i z_j)`
- Problem A: 21 nodes, 28 edges
- Problem B: 180 nodes, 226 edges
- Hardware guidance (Rigetti Ankaa-3): practical target around ~10 qubits and <=100 two-qubit gates per circuit

Using `tools/resource_estimation.py` with current repo defaults:

- Standard QAOA (one qubit per node), `p=1`:
  - Dataset A: 21 qubits, ~252 routed iSWAPs, estimated fidelity ~0.2828 -> infeasible
  - Dataset B: 180 qubits, ~2034 routed iSWAPs, estimated fidelity ~0.000037 -> infeasible
- DC-QAOA (`max_size=8`, `p=1`):
  - Dataset A: max 8 qubits/subgraph, max 99 routed iSWAPs -> feasible
  - Dataset B: max 8 qubits/subgraph, max 90 routed iSWAPs -> feasible

Reproduce:

```bash
python tools/resource_estimation.py datasets/dataset_A.parquet datasets/dataset_B.parquet
```

---

## Tools

All tools are run from the **project root** (`Q-volution 2025/`), not from inside `tools/`.

---

### `benchmark.py` - Quantum benchmark over 9 method/precondition combinations

By default it runs all combinations:

- Methods: `SLSQP`, `COBYLA`, `COBYQA`
- Preconditions: `none`, `back-propagate`, `analytic-p1`
- Total: 9 combinations, each repeated `--runs` times (default `5`)

For each combination, benchmark collects:

- best-so-far loss per iteration (averaged over runs)
- parameter trajectories (`gamma`, `beta`) per iteration
- final bitstring probability distribution (averaged over runs)

Outputs (saved to `output/`):

- `avg_loss_params_<dataset>_all_combinations.png`
- `avg_final_probability_<dataset>_all_combinations.png`
- `benchmark_data_<dataset>_all_combinations.xlsx` (all plotted data)

```bash
python tools/benchmark.py datasets/dataset_A.parquet
python tools/benchmark.py datasets/dataset_B.parquet
```

Optional overrides:

```bash
python tools/benchmark.py datasets/dataset_B.parquet \
  --methods SLSQP COBYLA COBYQA \
  --preconditions none back-propagate analytic-p1 \
  --runs 5 \
  --qc 8q-qvm \
  --output-dir output
```

---

### `bruteforce.py` โ€” Find the optimal / upper-bound Max-Cut score

- **Dataset A (21 nodes):** exact brute-force over all 2ยฒยน assignments (~50 s)
- **Dataset B (180 nodes):** SDP relaxation upper bound (`cvxpy` required) + simulated annealing best-known

```bash
python tools/bruteforce.py datasets/dataset_A.parquet   # exact optimal
python tools/bruteforce.py datasets/dataset_B.parquet   # SDP bound + SA
```

SDP requires `cvxpy`: `pip install cvxpy`

---

### `resource_estimation.py` โ€” Hardware feasibility analysis

Shows why standard QAOA is infeasible for large graphs on Rigetti Ankaa-3, and proves DC-QAOA fits within qubit and gate limits.

```bash
python tools/resource_estimation.py datasets/dataset_B.parquet
python tools/resource_estimation.py datasets/dataset_A.parquet datasets/dataset_B.parquet  # both at once
```

No extra dependencies. Outputs qubit counts, iSWAP counts, routing overhead, circuit fidelity, and a per-subgraph breakdown.

---

### `visualize_cut.py` โ€” Draw the Max-Cut result

Runs the DC-QAOA pipeline, then saves a dark-themed PNG of the graph with cut edges (green) and uncut edges (red dashed), nodes coloured by spin.

```bash
python tools/visualize_cut.py datasets/dataset_A.parquet   # โ’ maxcut_dataset_A.png
python tools/visualize_cut.py datasets/dataset_B.parquet   # โ’ maxcut_dataset_B.png
```

Output file is saved in the current directory. Tune `MAX_SIZE` and `TOP_T` at the top of the file.

---

### `test_qvm.py` โ€” Smoke test against a local QVM (Docker)

Runs the full DC-QAOA pipeline twice โ€” once with the classical backend, once with the pyQuil QVM โ€” and compares scores.

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
USE_QUANTUM  = False   # True โ’ quantum backend
MIXER_MODE   = "X"     # "X" (standard) | "XX" (graph-coupled) | "XY"
LAYER_COUNT  = 1       # QAOA depth p
SHOTS        = 1024    # Measurement shots per circuit run
SEED         = 42
MAXITER   = 100     # Optimizer iterations
```
