r"""Plot multi-object summary."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = "results/multiobject_summary.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    conditions = ["text_only", "text_minimal", "state_grounded"]
    tasks = ["loc_color", "loc_shape", "relation", "counterfactual"]
    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cond in enumerate(conditions):
        means = [data[cond]["per_task"][task]["mean"] * 100 for task in tasks]
        ax.bar(x + i * width, means, width, label=cond)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 105)
    ax.axhline(25, color="gray", linestyle="--", label="chance (4-class)")
    ax.legend()
    ax.set_title("Two-object MicroGround: per-task accuracy (seed 10 mean)")
    plt.tight_layout()
    out = "figures/multiobject_summary.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
