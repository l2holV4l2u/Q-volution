import json

with open('dc_qaoa_visualization.ipynb', 'r') as f:
    nb = json.load(f)

cells = nb['cells']

# test

# 1. Update MIXER_MODE in setup cell (cell 5)
setup_cell = cells[5]
source = "".join(setup_cell['source'])
if '_solver_module.MIXER_MODE' not in source:
    new_source = source.replace('_solver_module.LAYER_COUNT', '_solver_module.MIXER_MODE = "X"      # "X" (standard), "XX" (graph-coupled), "XY" (XY-mixer)\n_solver_module.LAYER_COUNT')
    setup_cell['source'] = [line + ("\n" if not line.endswith("\n") else "") for line in new_source.splitlines()]

# 2. Add plotting cell after Step 4 (cell 9)
# Find the cell index that contains "## 5. Merger & GR Policy"
idx = next(i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and '## 5. Merger' in "".join(c.get('source', [])))

plot_md = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## 4b. Plot Optimization History\n",
  "Visualize the COBYLA optimization convergence over all multi-start initializations for each subgraph."
 ]
}

plot_code = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "histories = _solver_module.OPTIMIZATION_HISTORY\n",
  "if histories:\n",
  "    num_plots = len(histories)\n",
  "    cols = min(3, num_plots)\n",
  "    rows = (num_plots + cols - 1) // cols\n",
  "    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)\n",
  "    axes = axes.flatten()\n",
  "    \n",
  "    for i, (subgraph_id, trials) in enumerate(histories.items()):\n",
  "        ax = axes[i]\n",
  "        for trial_idx, trace in enumerate(trials):\n",
  "            ax.plot(trace, label=f\"Init {trial_idx+1}\", alpha=0.7)\n",
  "        ax.set_title(f\"Subgraph {i+1} Optimization\")\n",
  "        ax.set_xlabel(\"COBYLA Iteration\")\n",
  "        ax.set_ylabel(\"Average Cut (Loss)\")\n",
  "        ax.grid(True, linestyle=\"--\", alpha=0.6)\n",
  "        if len(trials) <= 5:  # only show legend if not too many\n",
  "            ax.legend()\n",
  "    \n",
  "    # Hide any unused subplots\n",
  "    for j in range(i + 1, len(axes)):\n",
  "        fig.delaxes(axes[j])\n",
  "    \n",
  "    plt.tight_layout()\n",
  "    plt.show()\n",
  "else:\n",
  "    print(\"No optimization history captured (Make sure USE_PYQUIL=True)\")"
 ]
}

# Only insert if they are not already there
if "4b. Plot" not in "".join(cells[idx-1].get('source', [])):
    cells.insert(idx, plot_md)
    cells.insert(idx+1, plot_code)

# 3. Add Landscape suggestion at the end
landscape_md = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## 8. Plotting the QAOA Cost Landscape (Suggestion)\n",
  "\n",
  "To visualize the full objective function landscape for $p=1$ (one `gamma` and one `beta` parameter), you can use a grid search approach:\n",
  "1. Define a 2D grid over $\\gamma \\in [-\\pi, \\pi]$ and $\\beta \\in [-\\pi, \\pi]$.\n",
  "2. For each point $(\\gamma, \\beta)$ in the grid, set the circuit parameters and evaluate the objective function (average cut).\n",
  "3. Store the results in a 2D numpy array.\n",
  "4. Use `matplotlib.pyplot.contourf` or `plot_surface` from `mpl_toolkits.mplot3d` to plot the 2D array.\n",
  "\n",
  "**Example pseudo-code:**\n",
  "```python\n",
  "gamma_vals = np.linspace(-np.pi, np.pi, 20)\n",
  "beta_vals = np.linspace(-np.pi, np.pi, 20)\n",
  "Z = np.zeros((len(gamma_vals), len(beta_vals)))\n",
  "\n",
  "for i, g in enumerate(gamma_vals):\n",
  "    for j, b in enumerate(beta_vals):\n",
  "        Z[i, j] = evaluate_cut_at(g, b)  # Run parameterized circuit\n",
  "\n",
  "X, Y = np.meshgrid(gamma_vals, beta_vals)\n",
  "plt.contourf(X, Y, Z.T, levels=30, cmap='viridis')\n",
  "plt.colorbar(label='Average Cut')\n",
  "plt.xlabel('Gamma')\n",
  "plt.ylabel('Beta')\n",
  "plt.show()\n",
  "```\n",
  "This visualizes the peaks (max-cut) and valleys of the optimization space, letting you see the non-convex landscape the COBYLA optimizer is navigating!"
 ]
}

if "Cost Landscape" not in "".join(cells[-1].get('source', [])):
    cells.append(landscape_md)

with open('dc_qaoa_visualization.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated.")
