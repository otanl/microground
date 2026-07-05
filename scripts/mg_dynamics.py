r"""
Transient-generalisation ("generalise-then-collapse") dynamics analysis from a manifest.

Because evaluation is exhaustive, every point of the test trajectory is an exact property of
the network at that epoch -- a transient peak is a real behavioural state, not sampling noise.
This script quantifies, per route: the peak held-out accuracy, when it occurs, the converged
accuracy, and the collapse magnitude (peak - converged); tests route differences on the
collapse; and renders a paper-ready trajectory figure with bootstrap CI bands.

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\mg_dynamics.py --name bind_holdout_route
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from mg import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=None, help="figure path (default figures/mg/<name>_dynamics.png)")
    return p.parse_args()


def per_seed_stats(history):
    test = [h["test"] for h in history]
    eps = [h["ep"] for h in history]
    k = max(1, len(test) // 10)
    converged = float(np.mean(test[-k:]))
    peak_i = int(np.argmax(test))
    return {
        "peak": test[peak_i],
        "peak_ep": eps[peak_i],
        "converged": converged,
        "collapse": test[peak_i] - converged,
        "all_at_peak": history[peak_i]["all"],
    }


def main():
    args = parse_args()
    path = f"results/mg/{args.name}.jsonl"
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    conds, order = {}, []
    for r in records:
        c = r["condition"]
        if c not in conds:
            conds[c] = []; order.append(c)
        conds[c].append(r)

    chance = records[0]["baseline_test"]["balanced_chance"]
    print(f"manifest: {path} ({len(records)} runs)  chance={chance:.3f}")
    print("=" * 96)
    print(f"{'condition':20s} {'peak':>14s} {'peak_ep(med)':>12s} {'converged':>14s} {'collapse':>14s}")
    print("-" * 96)

    per_cond = {}
    for c in order:
        ss = [per_seed_stats(r["history"]) for r in conds[c] if r.get("history")]
        per_cond[c] = ss
        pk = stats.bootstrap_ci([s["peak"] for s in ss])
        cv = stats.bootstrap_ci([s["converged"] for s in ss])
        cl = stats.bootstrap_ci([s["collapse"] for s in ss])
        med_ep = int(np.median([s["peak_ep"] for s in ss]))
        print(f"{c:20s} {pk['mean']:.3f} [{pk['lo']:.2f},{pk['hi']:.2f}] {med_ep:>12d} "
              f"{cv['mean']:.3f} [{cv['lo']:.2f},{cv['hi']:.2f}] {cl['mean']:.3f} [{cl['lo']:.2f},{cl['hi']:.2f}]")

    # Route comparisons on collapse magnitude (paired by seed), Holm-corrected.
    routes = [c for c in ("text_only", "state_factored", "state_perceptual") if c in per_cond]
    pairs = [(a, b) for i, a in enumerate(routes) for b in routes[i + 1:]]
    if pairs:
        pvals = []
        for a, b in pairs:
            va = [s["collapse"] for s in per_cond[a]]
            vb = [s["collapse"] for s in per_cond[b]]
            n = min(len(va), len(vb))
            pvals.append(stats.wilcoxon_signed_rank(va[:n], vb[:n])["p"])
        holm = stats.holm_bonferroni(pvals)
        print("-" * 96)
        print("collapse-magnitude route comparisons (paired Wilcoxon, Holm-corrected):")
        for (a, b), h in zip(pairs, holm):
            va = [s["collapse"] for s in per_cond[a]]
            vb = [s["collapse"] for s in per_cond[b]]
            n = min(len(va), len(vb))
            d = stats.cliffs_delta(va[:n], vb[:n])
            sig = "*" if h["reject"] else " "
            print(f"  {a} vs {b}: p_adj={h['p_adj']:.3g}{sig}  Cliff's d={d:+.2f}")

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    colors = {"text_only": "tab:blue", "state_factored": "tab:orange",
              "state_perceptual": "tab:green", "state_perceptual_hard": "tab:red",
              "state_onehot_shared": "tab:purple", "text_minimal": "gray",
              "uninformative_state": "lightgray", "scrambled_state": "tab:olive"}
    for c in order:
        hists = [r["history"] for r in conds[c] if r.get("history")]
        if not hists:
            continue
        L = min(len(h) for h in hists)
        eps = [h["ep"] for h in hists[0][:L]]
        for ax, key in zip(axes, ("test", "all")):
            mat = np.array([[h[i][key] for i in range(L)] for h in hists])
            mean = mat.mean(axis=0)
            boots = np.random.default_rng(0).choice(mat.shape[0], size=(2000, mat.shape[0]))
            bm = mat[boots].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5], axis=0)
            ax.plot(eps, mean, label=c, color=colors.get(c), lw=1.8)
            ax.fill_between(eps, lo, hi, color=colors.get(c), alpha=0.18)
    axes[0].axhline(chance, color="k", ls=":", lw=1, label="chance")
    axes[0].set_title("held-out (exhaustive) balanced accuracy")
    axes[1].set_title("all-space balanced accuracy (train proxy)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("balanced accuracy")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    out = args.out or f"figures/mg/{args.name}_dynamics.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
