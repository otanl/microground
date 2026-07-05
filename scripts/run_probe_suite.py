r"""
Run linear probing / ablation for all saved MicroGround models.

Example:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\run_probe_suite.py --tasks attr counterfactual --conditions text_only text_minimal state_grounded --seeds 10
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["attr", "counterfactual"])
    p.add_argument("--conditions", nargs="+", default=["text_only", "text_minimal", "state_grounded"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--hidden_size", type=int, default=24)
    p.add_argument("--mlp_dim", type=int, default=48)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def run_probe(task, condition, seed, args):
    model_path = f"models/microground/{task}_{condition}_s{seed}.pt"
    if not os.path.exists(model_path):
        print(f"SKIP missing {model_path}")
        return None
    cmd = [
        sys.executable, "scripts/probe_microground.py",
        "--task", task,
        "--condition", condition,
        "--seed", str(seed),
        "--hidden_size", str(args.hidden_size),
        "--mlp_dim", str(args.mlp_dim),
        "--device", args.device,
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run(cmd, check=True, env=env)
    return f"models/microground/{task}_{condition}_s{seed}_probe.json"


def main():
    args = parse_args()
    workers = args.workers or os.cpu_count()
    jobs = [(t, c, s) for t in args.tasks for c in args.conditions for s in range(args.seeds)]
    print(f"Running probes for {len(jobs)} models on {workers} workers")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_probe, t, c, s, args): (t, c, s) for t, c, s in jobs}
        for fut in as_completed(futures):
            t, c, s = futures[fut]
            try:
                path = fut.result()
                if path:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    results.setdefault(f"{t}_{c}", []).append(data)
                    print(f"probed {t}/{c} seed={s}: color={data['probe']['color']:.1%}")
            except Exception as e:
                print(f"FAILED {t}/{c} seed={s}: {e}")

    # Aggregate
    summary = {}
    for key, runs in results.items():
        probe = {attr: [r["probe"][attr] for r in runs] for attr in ["color", "shape", "position", "size"]}
        baseline = {attr: [r["baseline"][attr] for r in runs] for attr in ["color", "shape", "position", "size"]}
        summary[key] = {
            "probe": {attr: {"mean": sum(v)/len(v), "min": min(v), "max": max(v)} for attr, v in probe.items()},
            "baseline": {attr: {"mean": sum(v)/len(v), "min": min(v), "max": max(v)} for attr, v in baseline.items()},
        }
        print(f"\n{key}:")
        for attr in ["color", "shape", "position", "size"]:
            p = summary[key]["probe"][attr]
            print(f"  probe {attr}: {p['mean']:.1%} [{p['min']:.1%}-{p['max']:.1%}]")

    with open("results/microground_probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/microground_probe_summary.json")


if __name__ == "__main__":
    main()
