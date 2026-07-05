r"""Plot activation patching summary."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = "results/microground_patch_summary.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tasks = ["attr", "counterfactual"]
    conditions = ["state_grounded"]  # focus on grounded condition
    attrs = ["color", "shape", "position", "size"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, tasks):
        x = np.arange(len(attrs))
        means = [data[f"{task}_state_grounded"][attr]["mean"] * 100 for attr in attrs]
        mins = [data[f"{task}_state_grounded"][attr]["min"] * 100 for attr in attrs]
        maxs = [data[f"{task}_state_grounded"][attr]["max"] * 100 for attr in attrs]
        ax.bar(x, means, color="#4c78a8")
        ax.errorbar(x, means, yerr=[np.array(means)-np.array(mins), np.array(maxs)-np.array(means)], fmt="none", color="black", capsize=4)
        ax.set_ylabel("Patch flip rate (%)")
        ax.set_title(f"{task} (state_grounded)")
        ax.set_xticks(x)
        ax.set_xticklabels(attrs)
        ax.set_ylim(0, 105)
    plt.tight_layout()
    out = "figures/microground_patch_summary.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
