r"""Render the failure-anatomy figure from results/mg/bind_probe.json.

Left: probe selectivity (probe - control) across checkpoints. Right: held-out behavioral
accuracy. The symbolic route's selectivity transient co-occurs with its behavioral transient;
index routes keep selectivity high while behavior stays below chance; the entangled route's
selectivity is pinned near its input's raw readability.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

agg = json.load(open("results/mg/bind_probe.json", encoding="utf-8"))
TAGS = ["ep5", "ep10", "ep15", "ep20", "ep50", "converged"]
X = [5, 10, 15, 20, 50, 500]
ROUTES = ["text_only", "state_factored", "state_onehot_shared", "state_perceptual", "state_perceptual_hard"]
COLORS = {"text_only": "tab:blue", "state_factored": "tab:orange", "state_onehot_shared": "tab:purple",
          "state_perceptual": "tab:green", "state_perceptual_hard": "tab:red"}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for rt in ROUTES:
    sel = [agg[f"{rt}/{t}"]["mean"]["selectivity"] for t in TAGS if f"{rt}/{t}" in agg]
    beh = [agg[f"{rt}/{t}"]["mean"]["acc"] for t in TAGS if f"{rt}/{t}" in agg]
    xs = X[:len(sel)]
    axes[0].plot(xs, sel, "o-", color=COLORS[rt], label=rt, lw=1.8)
    axes[1].plot(xs, beh, "o-", color=COLORS[rt], lw=1.8)
axes[0].axhline(0, color="k", ls=":", lw=1)
axes[0].set_xscale("log"); axes[1].set_xscale("log")
axes[0].set_title("probe selectivity (probe $-$ control)")
axes[1].set_title("held-out behavioral accuracy")
axes[1].axhline(0.25, color="k", ls=":", lw=1, label="chance")
for ax in axes: ax.set_xlabel("epoch (log)")
axes[0].set_ylabel("selectivity"); axes[1].set_ylabel("balanced accuracy")
axes[0].legend(fontsize=7, loc="center right")
fig.tight_layout()
for p in ("figures/mg/bind_probe_anatomy.png", "paper/figs/bind_probe_anatomy.png"):
    fig.savefig(p, dpi=160)
print("saved bind_probe_anatomy.png")
