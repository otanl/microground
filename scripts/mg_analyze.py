r"""
Re-aggregate a mg run manifest (NO retraining) with bootstrap CIs, paired Wilcoxon+Holm
vs a reference condition, and a grokking / delayed-generalisation analysis from the saved
trajectories. The manifest (results/mg/<name>.jsonl) is the single source of truth.

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\mg_analyze.py --name cf_route_trans --reference text_minimal
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mg import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--reference", default="text_minimal")
    p.add_argument("--mem_thresh", type=float, default=0.9,
                   help="all-space balanced acc above which we call train 'memorised'")
    p.add_argument("--grok_delta", type=float, default=0.2,
                   help="test-balanced rise after memorisation that counts as grokking")
    return p.parse_args()


def load(name):
    path = f"results/mg/{name}.jsonl"
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return records, path


def grok_stats(history, mem_thresh, grok_delta):
    """Detect delayed generalisation: test rises >=grok_delta AFTER all-space is memorised."""
    if not history:
        return {"grokked": False, "mem_ep": None, "final_test": None, "peak_test": None}
    mem_ep = next((h["ep"] for h in history if h["all"] >= mem_thresh), None)
    final_test = history[-1]["test"]
    peak_test = max(h["test"] for h in history)
    grokked = False
    if mem_ep is not None:
        test_at_mem = next(h["test"] for h in history if h["ep"] == mem_ep)
        later = [h["test"] for h in history if h["ep"] > mem_ep]
        if later and (max(later) - test_at_mem) >= grok_delta:
            grokked = True
    return {"grokked": grokked, "mem_ep": mem_ep, "final_test": final_test, "peak_test": peak_test}


def main():
    args = parse_args()
    records, path = load(args.name)
    conds, order = {}, []
    for r in records:
        c = r["condition"]
        if c not in conds:
            conds[c] = []; order.append(c)
        conds[c].append(r)

    base = records[0]["baseline_test"]
    task = records[0]["task"]; split = records[0]["split"]
    print(f"manifest: {path}  ({len(records)} runs)")
    print(f"task={task} split={split}  baseline(test): chance={base['balanced_chance']:.3f} "
          f"majority={base['balanced_majority']:.3f}")
    print("=" * 92)
    print(f"{'condition':20s} {'converged':>9s} {'95% CI':>16s} {'vs '+args.reference:>16s} {'peak':>7s}")
    print("-" * 92)

    # PRIMARY metric = CONVERGED test (mean of last 10% of evals). We do NOT headline the peak
    # over training: the peak inflates transient/noisy generalisation spikes that decay by
    # convergence (observed in bind_holdout_route). Peak is shown only as a secondary column.
    def cval(r):
        h = r.get("history", [])
        if not h:
            return r["best_test_balanced"]
        k = max(1, len(h) // 10)
        return sum(x["test"] for x in h[-k:]) / k

    # Pair by init_seed (a paired test requires aligned samples; also robust to partial runs).
    ref_by_seed = {r["init_seed"]: cval(r) for r in conds.get(args.reference, [])}
    comps, pvals, paired = [], [], {}
    for c in order:
        if c == args.reference or not ref_by_seed:
            continue
        vbs = {r["init_seed"]: cval(r) for r in conds[c]}
        common = sorted(set(ref_by_seed) & set(vbs))
        if len(common) < 2:
            continue
        a = [vbs[s] for s in common]; b = [ref_by_seed[s] for s in common]
        comps.append(c); pvals.append(stats.wilcoxon_signed_rank(a, b)["p"]); paired[c] = (a, b)
    holm = {c: h for c, h in zip(comps, stats.holm_bonferroni(pvals))} if pvals else {}

    for c in order:
        vals = [cval(r) for r in conds[c]]
        peak_mean = sum(r["best_test_balanced"] for r in conds[c]) / len(conds[c])
        ci = stats.bootstrap_ci(vals)
        tail = "(reference)"
        if c in holm:
            a, b = paired[c]
            d = stats.cliffs_delta(a, b)
            sig = "*" if holm[c]["reject"] else " "
            tail = f"p={holm[c]['p_adj']:.3g}{sig} d={d:+.2f}"
        bc = stats.bimodality_coefficient(vals)
        bc_flag = " <bimodal?>" if (bc == bc and bc > 0.555) else ""
        print(f"{c:20s} {ci['mean']:9.3f} [{ci['lo']:.3f},{ci['hi']:.3f}] {tail:>16s} {peak_mean:7.3f}{bc_flag}")

    print("=" * 92)
    print("converged = mean test-balanced over last 10% of evals (PRIMARY); peak = max over training (secondary).")
    print("* = significant after Holm-Bonferroni (alpha=0.05); d = Cliff's delta vs reference")


if __name__ == "__main__":
    main()
