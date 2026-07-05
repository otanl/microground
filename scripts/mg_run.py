r"""
Config-driven sweep over the redesigned MicroGround core, with a JSONL run manifest and
rigorous aggregation (bootstrap 95% CI, paired Wilcoxon vs a reference condition,
Holm-Bonferroni correction, Cliff's delta, bimodality coefficient).

Every run appends one self-describing line to results/mg/<name>.jsonl, so the manifest is
the single source of truth and re-aggregation never re-trains.

Parallelism: --workers N runs (condition, seed) jobs in a process pool (each worker pinned
to 1 torch thread). Resume: by default, (condition, init_seed) pairs already present in the
manifest are skipped; pass --fresh to wipe and redo.

Example:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\mg_run.py `
        --name cf_validate --task counterfactual `
        --conditions text_minimal uninformative_state text_only state_factored `
        --seeds 5 --epochs 120 --workers 8
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mg.train import train_one
from mg import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="manifest name -> results/mg/<name>.jsonl")
    p.add_argument("--task", default="counterfactual", choices=["attr", "counterfactual", "bind", "bind3"])
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--split", default="random")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_size", type=int, default=24)
    p.add_argument("--mlp_dim", type=int, default=48)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--save_dir", default=None, help="save converged models for mechanistic follow-up")
    p.add_argument("--checkpoints", default=None,
                   help="comma-separated epochs for mid-training checkpoints, e.g. '5,10,15,20'")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    p.add_argument("--fresh", action="store_true", help="wipe the manifest instead of resuming")
    p.add_argument("--reference", default="text_minimal",
                   help="condition each other is compared against (paired Wilcoxon)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _init_worker():
    import torch
    torch.set_num_threads(1)


def build_cfg(args, cond, seed):
    return {
        "task": args.task, "condition": cond, "split": args.split,
        "init_seed": seed, "split_seed": seed,  # separable; kept equal here
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "hidden_size": args.hidden_size, "mlp_dim": args.mlp_dim,
        "num_layers": args.num_layers, "device": args.device,
        "wd": args.wd, "save_dir": args.save_dir, "eval_every": args.eval_every,
        "checkpoint_epochs": [int(e) for e in args.checkpoints.split(",")] if args.checkpoints else None,
    }


def main():
    args = parse_args()
    os.makedirs("results/mg", exist_ok=True)
    manifest = f"results/mg/{args.name}.jsonl"

    done = set()
    if args.fresh or not os.path.exists(manifest):
        open(manifest, "w").close()
    else:
        for line in open(manifest, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["init_seed"]))

    jobs = [(c, s) for c in args.conditions for s in range(args.seeds) if (c, s) not in done]
    print(f"sweep: task={args.task} conditions={args.conditions} seeds={args.seeds} "
          f"epochs={args.epochs} split={args.split} wd={args.wd}")
    print(f"jobs: {len(jobs)} to run ({len(done)} resumed from manifest), workers={args.workers}\n")

    if args.workers <= 1:
        for cond, seed in jobs:
            res = train_one(build_cfg(args, cond, seed))
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            print(f"  {cond:20s} seed={seed} test_balanced={res['best_test_balanced']:.3f} "
                  f"(all={res['best_all_balanced']:.3f}, best_ep={res['best_epoch']})")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
            futures = {ex.submit(train_one, build_cfg(args, c, s)): (c, s) for c, s in jobs}
            for fut in as_completed(futures):
                c, s = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  FAILED {c} seed={s}: {e}")
                    continue
                with open(manifest, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"  {c:20s} seed={s} test_balanced={res['best_test_balanced']:.3f} "
                      f"(all={res['best_all_balanced']:.3f}, best_ep={res['best_epoch']})")

    # ---- aggregation (from the manifest, so resumed runs are included) -----
    records = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    if not records:
        print("no records; nothing to aggregate")
        return
    by_cond = {c: [r["best_test_balanced"] for r in records if r["condition"] == c]
               for c in args.conditions}
    chance = records[0]["baseline_test"]["balanced_chance"]
    majority = records[0]["baseline_test"]["balanced_majority"]

    print("\n" + "=" * 78)
    print(f"RESULT  task={args.task}  split={args.split}  n_seeds={args.seeds}  wd={args.wd}")
    print(f"baseline (test): chance={chance:.3f}  majority={majority:.3f}")
    print("-" * 78)
    print(f"{'condition':20s} {'mean':>6s} {'95% CI':>16s} {'vs '+args.reference:>22s}")

    ref = by_cond.get(args.reference)
    pvals, comps = [], []
    for cond in args.conditions:
        if cond == args.reference or not ref or len(by_cond[cond]) != len(ref):
            continue
        rep = stats.paired_report(by_cond[cond], ref)
        comps.append(cond)
        pvals.append(rep["wilcoxon"]["p"])
    holm = stats.holm_bonferroni(pvals) if pvals else []
    holm_by_cond = {c: h for c, h in zip(comps, holm)}

    for cond in args.conditions:
        if not by_cond[cond]:
            continue
        ci = stats.bootstrap_ci(by_cond[cond])
        bc = stats.bimodality_coefficient(by_cond[cond])
        tail = ""
        if cond in holm_by_cond:
            h = holm_by_cond[cond]
            d = stats.cliffs_delta(by_cond[cond], ref)
            sig = "*" if h["reject"] else " "
            tail = f"p_adj={h['p_adj']:.3g}{sig} d={d:+.2f}"
        elif cond == args.reference:
            tail = "(reference)"
        bc_flag = "  <bimodal?>" if (bc == bc and bc > 0.555) else ""
        print(f"{cond:20s} {ci['mean']:6.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]  {tail}{bc_flag}")

    print("=" * 78)
    print(f"manifest: {manifest}")
    print("* = significant after Holm-Bonferroni (alpha=0.05); d = Cliff's delta")
    print("NOTE: this table uses peak test_balanced; use scripts/mg_analyze.py for the "
          "converged (primary) metric.")


if __name__ == "__main__":
    main()
