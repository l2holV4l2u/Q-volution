import json

with open('dc_qaoa_visualization.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'GRAPH_PATH = "../dataset_A.csv"' in "".join(cell['source']):
        source = "".join(cell['source'])
        new_source = source.replace('GRAPH_PATH = "../dataset_A.csv"', 'GRAPH_PATH = Path("../dataset_A.csv")')
        cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in new_source.splitlines()]

with open('dc_qaoa_visualization.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

