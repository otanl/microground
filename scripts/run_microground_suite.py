r"""
Run a grid of MicroGround task/condition experiments and aggregate results.

Example:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\run_microground_suite.py --seeds 5 --epochs 200
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["attr", "id", "counterfactual"])
    p.add_argument("--conditions", nargs="+", default=["text_only", "state_grounded", "counterfactual"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_size", type=int, default=24)
    p.add_argument("--mlp_dim", type=int, default=48)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def run_once(task, condition, seed, args):
    cmd = [
        sys.executable, "scripts/train_microground.py",
        "--task", task,
        "--condition", condition,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--hidden_size", str(args.hidden_size),
        "--mlp_dim", str(args.mlp_dim),
        "--seed", str(seed),
        "--device", args.device,
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run(cmd, check=True, env=env)


def run_eval(task, condition, seed, args):
    cmd = [
        sys.executable, "scripts/eval_microground.py",
        "--task", task,
        "--condition", condition,
        "--seed", str(seed),
        "--hidden_size", str(args.hidden_size),
        "--mlp_dim", str(args.mlp_dim),
        "--device", args.device,
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    subprocess.run(cmd, check=True, env=env)


def main():
    args = parse_args()
    workers = args.workers or os.cpu_count()
    print(f"Running grid: tasks={args.tasks} conditions={args.conditions} seeds={args.seeds} on {workers} workers")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(t, c, s) for t in args.tasks for c in args.conditions for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_once, t, c, s, args): (t, c, s) for t, c, s in jobs}
        for fut in as_completed(futures):
            t, c, s = futures[fut]
            try:
                fut.result()
                print(f"trained {t}/{c} seed={s}")
            except Exception as e:
                print(f"FAILED {t}/{c} seed={s}: {e}")

    # Evaluate each trained model
    for t, c, s in jobs:
        run_eval(t, c, s, args)

    # Aggregate
    summary = {}
    for t in args.tasks:
        for c in args.conditions:
            key = f"{t}_{c}"
            best_tests = []
            all_states = []
            for s in range(args.seeds):
                path = f"models/microground/{t}_{c}_s{s}_eval.json"
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                best_tests.append(data["test"])
                all_states.append(data["all_states"])
            if best_tests:
                summary[key] = {
                    "test_mean": sum(best_tests) / len(best_tests),
                    "test_min": min(best_tests),
                    "test_max": max(best_tests),
                    "all_states_mean": sum(all_states) / len(all_states),
                }
                print(f"{key}: test={summary[key]['test_mean']:.1%} [{min(best_tests):.1%}-{max(best_tests):.1%}]  all_states={summary[key]['all_states_mean']:.1%}")

    with open("results/microground_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved results/microground_summary.json")


if __name__ == "__main__":
    main()
