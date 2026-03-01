import json

with open('dc_qaoa_visualization.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '_solver_module.USE_PYQUIL = True    # Set to False to use stub local search' in "".join(cell['source']):
        source = "".join(cell['source'])
        new_source = source.replace('_solver_module.USE_PYQUIL = True    # Set to False to use stub local search', 'import os\nos.environ["QCS_SETTINGS_APPLICATIONS_QVM_URL"] = "http://127.0.0.1:5001"\n_solver_module.USE_PYQUIL = True    # Set to False to use stub local search')
        cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in new_source.splitlines()]

with open('dc_qaoa_visualization.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
