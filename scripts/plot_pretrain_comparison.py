r"""Plot from-scratch vs pre-trained MicroGround comparison."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("results/microground_summary.json", encoding="utf-8") as f:
        scratch = json.load(f)
    with open("results/microground_pretrained_summary.json", encoding="utf-8") as f:
        pretrain = json.load(f)

    keys = ["attr_text_only", "attr_text_minimal", "attr_state_grounded",
            "counterfactual_text_only", "counterfactual_text_minimal", "counterfactual_state_grounded"]
    labels = ["attr\ntext_only", "attr\ntext_minimal", "attr\nstate_grounded",
              "cf\ntext_only", "cf\ntext_minimal", "cf\nstate_grounded"]
    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    means_scratch = [scratch[k]["test_mean"] * 100 for k in keys]
    means_pretrain = [pretrain[k]["mean"] * 100 for k in keys]
    ax.bar(x - width/2, means_scratch, width, label="from-scratch")
    ax.bar(x + width/2, means_pretrain, width, label="pre-trained")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.axhline(25, color="gray", linestyle="--")
    ax.legend()
    ax.set_title("MicroGround: from-scratch vs synthetic-pre-trained (seed 10 mean)")
    plt.tight_layout()
    out = "figures/microground_pretrain_comparison.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
