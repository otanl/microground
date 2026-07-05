r"""
Linear probing and simple neuron ablation for MicroGround models.

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\probe_microground.py --task attr --condition state_grounded --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from micro_ground import ObjectWorld, Task, Condition, Vocab, MicroGroundDataset
from micro_ground.model import MicroGroundTransformer
from micro_ground.data import collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["attr", "id", "counterfactual"], required=True)
    p.add_argument("--condition", choices=["text_only", "text_minimal", "state_grounded", "counterfactual"], required=True)
    p.add_argument("--hidden_size", type=int, default=24)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model_dir", default="models/microground")
    p.add_argument("--max_len", type=int, default=32)
    return p.parse_args()


def extract_representations(model, loader, condition, device):
    """Return a dict with last-token hidden states, labels, and state vectors."""
    model.eval()
    reps = []
    labels = {attr: [] for attr in ["color", "shape", "position", "size"]}
    states = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            state = None if condition in ("text_only", "text_minimal") else batch["state"].to(device)
            # Hook to capture layer outputs
            hidden_states = []
            def hook(mod, inp, out):
                hidden_states.append(out.cpu())
            handles = []
            for layer in model.layers:
                handles.append(layer.register_forward_hook(hook))
            model(input_ids, state)
            for h in handles:
                h.remove()
            # last-token hidden for each layer
            last = [h[:, -1, :].cpu().numpy() for h in hidden_states]
            reps.append(last)
            for meta in batch["meta"]:
                state_vec = meta.get("state", meta.get("old_state", [0, 0, 0, 0]))
                for i, attr in enumerate(["color", "shape", "position", "size"]):
                    labels[attr].append(state_vec[i])
                states.append(state_vec)
    # concatenate across batches
    reps = [np.concatenate([r[i] for r in reps], axis=0) for i in range(len(reps[0]))]
    labels = {k: np.array(v) for k, v in labels.items()}
    states = np.array(states)
    return reps, labels, states


def probe(reps, labels, attr_sizes):
    """Train linear probes and return accuracies."""
    results = {}
    for attr, size in attr_sizes.items():
        y = labels[attr]
        X = reps
        # simple train/test split using first 80%
        n = len(y)
        n_train = int(0.8 * n)
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        results[attr] = float(acc)
    return results


def ablation_sensitivity(model, loader, task, condition, device, attr_sizes):
    """Zero-out each hidden dimension and measure accuracy drop on each attribute."""
    model.eval()
    # baseline accuracy
    def acc_on_attr(attr_idx):
        correct = {attr: 0 for attr in attr_sizes}
        total = {attr: 0 for attr in attr_sizes}
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                state = None if condition in ("text_only", "text_minimal") else batch["state"].to(device)
                token_logits, id_logits = model(input_ids, state)
                preds = id_logits.argmax(-1) if task == "id" else token_logits.argmax(-1)
                ok = (preds == labels)
                for i, meta in enumerate(batch["meta"]):
                    a = meta.get("attr")
                    if a:
                        correct[a] += ok[i].item()
                        total[a] += 1
        return {a: correct[a] / total[a] if total[a] else 0 for a in attr_sizes}

    baseline = acc_on_attr(attr_sizes)
    sensitivity = {}
    for dim in range(model.hidden_size):
        # Hook to zero out this dimension in the last token
        def make_hook(d):
            def hook(mod, inp, out):
                out[:, -1, d] = 0
                return out
            return hook
        handles = [layer.register_forward_hook(make_hook(dim)) for layer in model.layers]
        acc = acc_on_attr(attr_sizes)
        for h in handles:
            h.remove()
        sensitivity[dim] = {a: baseline[a] - acc[a] for a in attr_sizes}
    return baseline, sensitivity


def main():
    args = parse_args()
    world = ObjectWorld()
    vocab = Vocab()
    task = Task(args.task)
    condition = Condition(args.condition)

    model = MicroGroundTransformer(
        vocab_size=len(vocab),
        num_states=world.num_states,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
    ).to(args.device)
    model_path = os.path.join(args.model_dir, f"{args.task}_{args.condition}_s{args.seed}.pt")
    model.load_state_dict(torch.load(model_path, map_location=args.device))

    ds = MicroGroundDataset(world, task, condition, vocab, split="all", seed=args.seed)
    loader = DataLoader(ds, batch_size=32, collate_fn=lambda b: collate_fn(b, vocab.pad_id, args.max_len))

    reps, labels, states = extract_representations(model, loader, args.condition, args.device)
    attr_sizes = {"color": 4, "shape": 4, "position": 4, "size": 2}

    # Probe from the last layer
    probe_results = probe(reps[-1], labels, attr_sizes)
    print("Linear probe accuracy (last layer):")
    for attr, acc in probe_results.items():
        print(f"  {attr}: {acc:.1%}")

    # Ablation sensitivity
    baseline, sensitivity = ablation_sensitivity(model, loader, args.task, args.condition, args.device, attr_sizes)
    print("\nBaseline accuracy:")
    for attr, acc in baseline.items():
        print(f"  {attr}: {acc:.1%}")

    out = {
        "task": args.task,
        "condition": args.condition,
        "seed": args.seed,
        "probe": probe_results,
        "baseline": baseline,
    }
    out_path = model_path.replace(".pt", "_probe.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
