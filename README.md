# README.md

This repository contains an implementation of the Divide-and-Conquer QAOA (DC-QAOA) pipeline developed for the Q-volution 2025 energy grid optimization challenge. The project uses Python and the Rigetti QCS SDK to partition Max-Cut instances, solve subproblems using QAOA, and merge solutions.

## Prerequisites

- Python 3.9+ (Linux recommended)
- `pip` for Python package installation
- (Optional) Rigetti QCS credentials for running on QPU
- `quilc` and `qvm` if you plan to simulate circuits locally

## Setup

1. Clone or open this workspace in VS Code.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r dc_qaoa/requirements.txt
   ```
   The main dependencies include `pandas`, `pyarrow`, `networkx`, `numpy`, `scipy`, and `pyquil`.

4. (Optional) Install development extras if modifying the library:
   ```bash
   pip install -e .
   ```
   This installs `dc_qaoa` as an editable package.

## Usage

The CLI entry point is `main.py`. Datasets are stored as Parquet or CSV files.

Basic local run (stub solver):
```bash
python main.py ../dataset_A.parquet
```

Control the maximum subgraph size, top‑t diversity, and output file:
```bash
python main.py --max-size 8 --top-t 10 --output result.json ../dataset_B.parquet
```

Simulate with Rigetti QVM (requires `quilc` and `qvm` running):
```bash
python main.py --qaoa --qc 8q-qvm ../dataset_A.parquet
```

Execute on Rigetti QPU (requires `qcs auth login`):
```bash
python main.py --qaoa --qc Ankaa-3 ../dataset_B.parquet
```

Hardware preset with sensible defaults:
```bash
python main.py --hardware ../dataset_B.parquet
```

Other utilities:

- `dc_qaoa/resource_estimation.py` – analyze resource usage of standard QAOA vs DC-QAOA
- `dc_qaoa/benchmark.py` – compare pipeline performance to classical baselines

## Project Structure

```
dc_qaoa/
├── graph_loader.py         # parquet -> nx.Graph
├── partitioner.py          # recursive graph partitioning logic
├── solver.py               # QAOA and brute-force backends
├── merger.py               # top-t merge logic
├── scorer.py               # objective evaluation
├── pipeline.py             # orchestration of the pipeline
├── benchmark.py            # benchmarking scripts
├── resource_estimation.py  # resource analysis scripts
└── requirements.txt        # Python dependencies
```

## Contributing

Feel free to explore and extend the code. The pipeline is structured for easy testing and modification. For detailed competition context and design notes, refer to `dc_qaoa/CLAUDE.md`.

## License

[MIT License](LICENSE)
