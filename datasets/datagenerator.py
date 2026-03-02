from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

def generate_edges(
    num_nodes: int,
    num_edges: int,
    w_min: float = 1.0,
    w_max: float = 400.0,
    seed: int | None = 0,
) -> list[tuple[int, int, float]]:
    """
    Generate an undirected weighted edge list:
    - nodes are integers [0, num_nodes-1]
    - no self-loops
    - no duplicate edges (u,v) == (v,u)
    """
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")

    max_edges = num_nodes * (num_nodes - 1) // 2
    if num_edges > max_edges:
        raise ValueError(f"num_edges too large: max for {num_nodes} nodes is {max_edges}")

    rng = random.Random(seed)

    chosen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int, float]] = []

    while len(edges) < num_edges:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in chosen:
            continue
        chosen.add((a, b))

        w = rng.uniform(w_min, w_max)
        edges.append((a, b, w))

    return edges


def write_csv(edges: Iterable[tuple[int, int, float]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_1", "node_2", "weight"])
        for u, v, w in edges:
            writer.writerow([u, v, f"{w:.12f}"])  # similar style to your example

if __name__ == "__main__":
    # Example: generate something like your sample
    edges = generate_edges(num_nodes=10, num_edges=28, w_min=40.0, w_max=300.0, seed=42)
    write_csv(edges, "graph_edges.csv")
    print("Wrote graph_edges.csv")