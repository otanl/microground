r"""Multi-seed sweep for two-object MicroGround."""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=["text_only", "text_minimal", "state_grounded"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--mlp_dim", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def run_train(condition, seed, args):
    cmd = [
        sys.executable, "scripts/train_multiobject.py",
        "--condition", condition,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--hidden_size", str(args.hidden_size),
        "--mlp_dim", str(args.mlp_dim),
        "--device", args.device,
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run(cmd, check=True, env=env)
    meta_path = f"models/microground/multi_{condition}_s{seed}.json"
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    workers = args.workers or os.cpu_count()
    jobs = [(c, s) for c in args.conditions for s in range(args.seeds)]
    print(f"Running multi-object training for {len(jobs)} configs on {workers} workers")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_train, c, s, args): (c, s) for c, s in jobs}
        for fut in as_completed(futures):
            c, s = futures[fut]
            try:
                data = fut.result()
                results.setdefault(c, []).append(data)
                print(f"{c} seed={s}: best_test={data['best_test']:.1%}")
            except Exception as e:
                print(f"FAILED {c} seed={s}: {e}")

    summary = {}
    for c, runs in results.items():
        best_tests = [r["best_test"] for r in runs]
        summary[c] = {
            "best_test_mean": sum(best_tests) / len(best_tests),
            "best_test_min": min(best_tests),
            "best_test_max": max(best_tests),
            "per_task": {
                task: {
                    "mean": sum(r["history"][-1]["per_task"][task] for r in runs) / len(runs),
                    "min": min(r["history"][-1]["per_task"][task] for r in runs),
                    "max": max(r["history"][-1]["per_task"][task] for r in runs),
                }
                for task in runs[0]["history"][-1]["per_task"]
            },
        }
        print(f"\n{c}")
        print(f"  best_test: {summary[c]['best_test_mean']:.1%} [{summary[c]['best_test_min']:.1%}-{summary[c]['best_test_max']:.1%}]")
        for task, stats in summary[c]["per_task"].items():
            print(f"  {task}: {stats['mean']:.1%} [{stats['min']:.1%}-{stats['max']:.1%}]")

    os.makedirs("results", exist_ok=True)
    with open("results/multiobject_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/multiobject_summary.json")


if __name__ == "__main__":
    main()
