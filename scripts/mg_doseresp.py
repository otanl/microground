r"""
Dose-response analysis of the k-shot binding sweep (E9), with a pooled route comparison that
has more power than per-fraction tests.

For each route we fit, per seed, the half-max dose D50: the leaked fraction at which converged
held-out accuracy first crosses the midpoint between chance and 1.0 (0.625), by linear
interpolation across the measured fractions. A more sample-efficient (more compositional) route
crosses at a SMALLER dose. We report D50 with bootstrap CIs and compare routes with a paired
Wilcoxon (Holm-corrected) across seeds -- pooling the whole curve rather than one fraction.
We also render the dose-response figure with CI bands.

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\mg_doseresp.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from mg import stats

FRACS = [0.02, 0.05, 0.1, 0.2]
ROUTES = ["text_only", "state_factored", "state_perceptual",
          "state_perceptual_hard", "state_onehot_shared"]
CHANCE = 0.25
MID = (CHANCE + 1.0) / 2.0  # 0.625 half-max threshold


def converged(r):
    h = r["history"]; k = max(1, len(h) // 10)
    return float(np.mean([x["test"] for x in h[-k:]]))


def load():
    """route -> seed -> {frac: converged acc}."""
    data = {rt: {} for rt in ROUTES}
    for fr in FRACS:
        path = f"results/mg/bind_kshot_f{fr}.jsonl"
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["condition"] not in ROUTES:
                continue
            data[r["condition"]].setdefault(r["init_seed"], {})[fr] = converged(r)
    return data


def d50(curve):
    """Interpolated leaked-fraction where the curve crosses MID; None if never reached."""
    xs = FRACS
    ys = [curve.get(f, np.nan) for f in xs]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if np.isnan(y0) or np.isnan(y1):
            continue
        if (y0 < MID <= y1) or (y0 >= MID > y1):
            t = (MID - y0) / (y1 - y0) if y1 != y0 else 0.0
            return xs[i] + t * (xs[i + 1] - xs[i])
    if ys[-1] >= MID:      # already above at smallest measured dose
        return xs[0]
    return None            # never reaches half-max within measured range


def main():
    data = load()
    print(f"held-out binding dose-response (chance={CHANCE}, half-max threshold={MID})")
    print("=" * 74)
    # per-fraction table
    print(f"{'route':18s}" + "".join(f"  f={f:>4}" for f in FRACS) + f"  {'D50':>18s}")
    d50s = {}
    for rt in ROUTES:
        seeds = sorted(data[rt])
        row = []
        for fr in FRACS:
            vals = [data[rt][s][fr] for s in seeds if fr in data[rt][s]]
            row.append(f"{np.mean(vals):.2f}")
        ds = [d50(data[rt][s]) for s in seeds]
        ds_valid = [d for d in ds if d is not None]
        d50s[rt] = {"per_seed": ds, "valid": ds_valid, "n": len(seeds),
                    "reached": len(ds_valid)}
        ci = stats.bootstrap_ci(ds_valid) if ds_valid else {"mean": float("nan"), "lo": 0, "hi": 0}
        print(f"{rt:18s}" + "".join(f"  {c:>6}" for c in row) +
              f"  {ci['mean']:.3f}[{ci['lo']:.3f},{ci['hi']:.3f}] ({len(ds_valid)}/{len(seeds)})")

    # pooled route comparison on D50 (smaller = more sample-efficient)
    print("-" * 74)
    print("D50 route comparison (paired Wilcoxon across seeds, Holm; smaller D50 = more efficient):")
    pairs = [("text_only", "state_factored"), ("state_perceptual", "state_factored"),
             ("text_only", "state_perceptual")]
    pvals, kept = [], []
    for a, b in pairs:
        sa, sb = data[a], data[b]
        common = sorted(set(sa) & set(sb))
        va, vb = [], []
        for s in common:
            da, db = d50(sa[s]), d50(sb[s])
            if da is not None and db is not None:
                va.append(da); vb.append(db)
        if len(va) >= 2:
            pvals.append(stats.wilcoxon_signed_rank(va, vb)["p"]); kept.append((a, b, va, vb))
    holm = stats.holm_bonferroni(pvals) if pvals else []
    for (a, b, va, vb), h in zip(kept, holm):
        d = stats.cliffs_delta(va, vb)
        sig = "*" if h["reject"] else " "
        print(f"  {a} vs {b}: D50 {np.mean(va):.3f} vs {np.mean(vb):.3f}  "
              f"p_adj={h['p_adj']:.3g}{sig} d={d:+.2f} (n={len(va)})")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"text_only": "tab:blue", "state_factored": "tab:orange", "state_perceptual": "tab:green",
              "state_perceptual_hard": "tab:red", "state_onehot_shared": "tab:purple"}
    plt.figure(figsize=(6, 4.2))
    for rt in ROUTES:
        seeds = sorted(data[rt])
        means, los, his = [], [], []
        for fr in FRACS:
            vals = [data[rt][s][fr] for s in seeds if fr in data[rt][s]]
            ci = stats.bootstrap_ci(vals)
            means.append(ci["mean"]); los.append(ci["lo"]); his.append(ci["hi"])
        plt.plot(FRACS, means, "o-", color=colors[rt], label=rt, lw=1.8)
        plt.fill_between(FRACS, los, his, color=colors[rt], alpha=0.18)
    plt.axhline(CHANCE, color="k", ls=":", lw=1, label="chance")
    plt.xlabel("fraction of held-out query-type leaked into training")
    plt.ylabel("converged held-out balanced accuracy")
    plt.title("Sample efficiency of compositional binding, by route")
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs("figures/mg", exist_ok=True)
    for p in ("figures/mg/bind_doseresp.png", "paper/figs/bind_doseresp.png"):
        plt.savefig(p, dpi=160)
    print("\nfigure: figures/mg/bind_doseresp.png (+ paper/figs/)")


if __name__ == "__main__":
    main()
