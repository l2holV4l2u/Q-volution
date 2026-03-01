"""
graph_loader.py — Load parquet edge list into a NetworkX graph.
"""
import pandas as pd
import networkx as nx
from pathlib import Path


def load_graph(path: str | Path) -> nx.Graph:
    """
    Load a parquet file representing a weighted graph.

    Expected columns (auto-detected):
      - source / target  (or 'src'/'dst', 'u'/'v', 'node1'/'node2')
      - weight           (optional, defaults to 1)

    Returns a nx.Graph with integer node IDs and 'weight' edge attributes.
    """

    G = nx.Graph()
    suffix = path.suffix
    if suffix == ".csv": df = pd.read_csv(path)
    elif suffix == ".parquet": df = pd.read_parquet(path)
    else:
        raise ValueError(f"can't read file in {suffix} format")
        
    
    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Detect source/target columns
    col_aliases = {
        "source": ["source", "src", "u", "node1", "node_1", "from"],
        "target": ["target", "dst", "v", "node2", "node_2", "to"],
        "weight": ["weight", "w", "cost", "distance"],
    }

    def find_col(aliases):
        for a in aliases:
            if a in df.columns:
                return a
        return None

    src_col = find_col(col_aliases["source"])
    tgt_col = find_col(col_aliases["target"])
    wgt_col = find_col(col_aliases["weight"])

    if src_col is None or tgt_col is None:
        raise ValueError(
            f"Cannot detect source/target columns. Found: {list(df.columns)}"
        )
        
    for _, row in df.iterrows():
        u = int(row[src_col])
        v = int(row[tgt_col])
        w = float(row[wgt_col]) if wgt_col else 1.0
        G.add_edge(u, v, weight=w)

    print(
        f"[graph_loader] Loaded graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges  |  "
        f"avg degree = {sum(d for _, d in G.degree()) / G.number_of_nodes():.2f}"
    )
    return G
