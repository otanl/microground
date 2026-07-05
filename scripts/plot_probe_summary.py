r"""Plot aggregated probe results."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = "results/microground_probe_summary.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tasks = ["attr", "counterfactual"]
    conditions = ["text_only", "text_minimal", "state_grounded"]
    attrs = ["color", "shape", "position", "size"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, task in zip(axes, tasks):
        x = np.arange(len(attrs))
        width = 0.25
        for i, cond in enumerate(conditions):
            key = f"{task}_{cond}"
            means = [data[key]["probe"][attr]["mean"] * 100 for attr in attrs]
            ax.bar(x + i * width, means, width, label=cond)
        ax.set_ylabel("Linear probe accuracy (%)")
        ax.set_title(f"{task}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(attrs)
        ax.set_ylim(0, 105)
        ax.axhline(25, color="gray", linestyle="--")
        ax.legend()
    plt.tight_layout()
    out = "figures/microground_probe_summary.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
