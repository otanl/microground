r"""Multi-seed fine-tune sweep for pre-trained SLM on MicroGround text-only tasks."""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=["attr", "counterfactual"])
    p.add_argument("--conditions", nargs="+", default=["text_only", "text_minimal"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def run_finetune(task, condition, seed, args):
    cmd = [
        sys.executable, "scripts/finetune_slm_microground.py",
        "--model_dir", args.model_dir,
        "--task", task,
        "--condition", condition,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--device", args.device,
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run(cmd, check=True, env=env)
    meta_path = f"models/microground_slm/{task}_{condition}_s{seed}.json"
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    workers = args.workers or os.cpu_count()
    jobs = [(t, c, s) for t in args.tasks for c in args.conditions for s in range(args.seeds)]
    print(f"Fine-tuning SLM for {len(jobs)} configs on {workers} workers")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_finetune, t, c, s, args): (t, c, s) for t, c, s in jobs}
        for fut in as_completed(futures):
            t, c, s = futures[fut]
            try:
                data = fut.result()
                results.setdefault(f"{t}_{c}", []).append(data)
                print(f"{t}/{c} seed={s}: best_balanced={data['best_balanced']:.1%}")
            except Exception as e:
                print(f"FAILED {t}/{c} seed={s}: {e}")

    summary = {}
    for key, runs in results.items():
        best_balanced = [r["best_balanced"] for r in runs]
        best_tests = [r["best_test"] for r in runs]
        summary[key] = {
            "best_balanced": {
                "mean": sum(best_balanced) / len(best_balanced),
                "min": min(best_balanced),
                "max": max(best_balanced),
            },
            "best_test": {
                "mean": sum(best_tests) / len(best_tests),
                "min": min(best_tests),
                "max": max(best_tests),
            },
        }
        print(f"\n{key}")
        print(f"  best_balanced: {summary[key]['best_balanced']['mean']:.1%} [{summary[key]['best_balanced']['min']:.1%}-{summary[key]['best_balanced']['max']:.1%}]")
        print(f"  best_test:     {summary[key]['best_test']['mean']:.1%} [{summary[key]['best_test']['min']:.1%}-{summary[key]['best_test']['max']:.1%}]")

    with open("results/microground_slm_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/microground_slm_summary.json")


if __name__ == "__main__":
    main()
