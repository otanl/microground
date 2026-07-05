r"""Multi-seed sweep for pre-trained MicroGround models.

Assumes a pre-trained model already exists at --pretrained_path. Fine-tunes on MicroGround
tasks for multiple seeds and conditions, then compares to the from-scratch summary.

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts/run_pretrained_microground_suite.py \
        --pretrained_path models/microground_pretrain_synthetic/model.pt \
        --tasks attr counterfactual --conditions text_only text_minimal state_grounded --seeds 10
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--tasks", nargs="+", default=["attr", "counterfactual"])
    p.add_argument("--conditions", nargs="+", default=["text_only", "text_minimal", "state_grounded"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden_size", type=int, default=24)
    p.add_argument("--mlp_dim", type=int, default=48)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def run_train(task, condition, seed, args):
    save_dir = f"models/microground_pretrained"
    model_path = f"{save_dir}/{task}_{condition}_s{seed}.pt"
    cmd = [
        sys.executable, "scripts/train_microground.py",
        "--task", task,
        "--condition", condition,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--hidden_size", str(args.hidden_size),
        "--mlp_dim", str(args.mlp_dim),
        "--device", args.device,
        "--pretrained_path", args.pretrained_path,
        "--save_dir", save_dir,
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run(cmd, check=True, env=env)
    meta_path = model_path.replace(".pt", ".json")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    workers = args.workers or os.cpu_count()
    jobs = [(t, c, s) for t in args.tasks for c in args.conditions for s in range(args.seeds)]
    print(f"Fine-tuning pre-trained model for {len(jobs)} configs on {workers} workers")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_train, t, c, s, args): (t, c, s) for t, c, s in jobs}
        for fut in as_completed(futures):
            t, c, s = futures[fut]
            try:
                data = fut.result()
                results.setdefault(f"{t}_{c}", []).append(data)
                print(f"{t}/{c} seed={s}: best_test={data['best_test']:.1%}")
            except Exception as e:
                print(f"FAILED {t}/{c} seed={s}: {e}")

    summary = {}
    for key, runs in results.items():
        best_tests = [r["best_test"] for r in runs]
        summary[key] = {
            "mean": sum(best_tests) / len(best_tests),
            "min": min(best_tests),
            "max": max(best_tests),
        }
        print(f"\n{key}")
        print(f"  best_test: {summary[key]['mean']:.1%} [{summary[key]['min']:.1%}-{summary[key]['max']:.1%}]")

    with open("results/microground_pretrained_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/microground_pretrained_summary.json")


if __name__ == "__main__":
    main()
