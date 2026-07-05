r"""Run activation patching over all state-grounded and counterfactual models."""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["attr", "counterfactual"])
    p.add_argument("--conditions", nargs="+", default=["state_grounded", "counterfactual"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--hidden_size", type=int, default=24)
    p.add_argument("--mlp_dim", type=int, default=48)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def run_patch(task, condition, seed, args):
    model_path = f"models/microground/{task}_{condition}_s{seed}.pt"
    if not os.path.exists(model_path):
        return None
    cmd = [
        sys.executable, "scripts/patch_microground.py",
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
    return f"models/microground/{task}_{condition}_s{seed}_patch.json"


def main():
    args = parse_args()
    workers = args.workers or os.cpu_count()
    jobs = [(t, c, s) for t in args.tasks for c in args.conditions for s in range(args.seeds)]
    print(f"Running activation patching for {len(jobs)} models on {workers} workers")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_patch, t, c, s, args): (t, c, s) for t, c, s in jobs}
        for fut in as_completed(futures):
            t, c, s = futures[fut]
            try:
                path = fut.result()
                if path:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    results.setdefault(f"{t}_{c}", []).append(data)
                    print(f"patched {t}/{c} seed={s}: color={data['summary']['color']:.1%}")
            except Exception as e:
                print(f"FAILED {t}/{c} seed={s}: {e}")

    summary = {}
    for key, runs in results.items():
        attrs = ["color", "shape", "position", "size"]
        summary[key] = {
            attr: {"mean": sum(r["summary"][attr] for r in runs) / len(runs),
                  "min": min(r["summary"][attr] for r in runs),
                  "max": max(r["summary"][attr] for r in runs)}
            for attr in attrs
        }
        print(f"\n{key}")
        for attr in attrs:
            print(f"  {attr}: {summary[key][attr]['mean']:.1%} [{summary[key][attr]['min']:.1%}-{summary[key][attr]['max']:.1%}]")

    with open("results/microground_patch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/microground_patch_summary.json")


if __name__ == "__main__":
    main()
