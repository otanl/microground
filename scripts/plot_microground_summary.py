r"""Plot MicroGround sweep summary."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = "results/microground_summary.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tasks = ["attr", "counterfactual"]
    conditions = ["text_only", "text_minimal", "state_grounded"]
    labels = []
    means = []
    mins = []
    maxs = []
    for task in tasks:
        for cond in conditions:
            key = f"{task}_{cond}"
            labels.append(f"{task}\n{cond}")
            means.append(data[key]["test_mean"] * 100)
            mins.append(data[key]["test_min"] * 100)
            maxs.append(data[key]["test_max"] * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, color=["#4c78a8", "#54a24b", "#e45756", "#4c78a8", "#54a24b", "#e45756"])
    ax.errorbar(x, means, yerr=[np.array(means)-np.array(mins), np.array(maxs)-np.array(means)], fmt="none", color="black", capsize=4)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.axhline(25, color="gray", linestyle="--", label="chance (4-class)")
    ax.set_title("MicroGround: 10-seed test accuracy (mean, min, max)")
    ax.legend()
    plt.tight_layout()
    out = "figures/microground_summary.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
