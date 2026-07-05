r"""
Evaluate a trained MicroGround model exhaustively.

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\eval_microground.py \
        --task attr --condition state_grounded --seed 0
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from micro_ground import ObjectWorld, Task, Condition, Vocab, MicroGroundDataset, COLORS, SHAPES
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


def evaluate(model, loader, task, condition, device):
    model.eval()
    correct = 0
    total = 0
    per_attr = {attr: [0, 0] for attr in ["color", "shape", "position", "size"]}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            state = None if condition in ("text_only", "text_minimal") else batch["state"].to(device)
            token_logits, id_logits = model(input_ids, state)
            if task == "id":
                preds = id_logits.argmax(dim=-1)
            else:
                preds = token_logits.argmax(dim=-1)
            ok = (preds == labels)
            correct += ok.sum().item()
            total += labels.numel()
            for i, meta in enumerate(batch["meta"]):
                attr = meta.get("attr")
                if attr:
                    per_attr[attr][0] += ok[i].item()
                    per_attr[attr][1] += 1
    return correct / total, {k: (v[0] / v[1] if v[1] else 0.0) for k, v in per_attr.items()}


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
    print(f"Loaded {model_path} ({model.count_params():,} params)")

    # Standard split
    train_ds = MicroGroundDataset(world, task, condition, vocab, split="train", seed=args.seed)
    test_ds = MicroGroundDataset(world, task, condition, vocab, split="test", seed=args.seed)

    def make_loader(ds):
        return DataLoader(ds, batch_size=32, collate_fn=lambda b: collate_fn(b, vocab.pad_id, args.max_len))

    train_acc, _ = evaluate(model, make_loader(train_ds), args.task, args.condition, args.device)
    test_acc, _ = evaluate(model, make_loader(test_ds), args.task, args.condition, args.device)
    print(f"train={train_acc:.1%} test={test_acc:.1%}")

    # All-states evaluation
    all_examples = world.generate_examples(task, condition, "all", args.seed)
    all_ds = MicroGroundDataset(world, task, condition, vocab, split="all", seed=args.seed)
    all_acc, per_attr = evaluate(model, make_loader(all_ds), args.task, args.condition, args.device)
    print(f"all_states={all_acc:.1%}")
    for attr, acc in per_attr.items():
        print(f"  {attr}={acc:.1%}")

    # Holdout evaluation per attribute
    holdout_results = {}
    for attr in ["color", "shape", "position", "size"]:
        ho_ds = MicroGroundDataset(world, task, condition, vocab, split="test", seed=args.seed, holdout_attr=attr)
        if len(ho_ds) == 0:
            continue
        ho_acc, _ = evaluate(model, make_loader(ho_ds), args.task, args.condition, args.device)
        holdout_results[attr] = ho_acc
        print(f"holdout {attr}={ho_acc:.1%}")

    # Composition holdout: e.g., hold out red square combinations
    comp_results = {}
    for color in range(len(COLORS)):
        for shape in range(len(SHAPES)):
            comp = {"color": color, "shape": shape}
            ho_ds = MicroGroundDataset(world, task, condition, vocab, split="test", seed=args.seed, holdout_composition=comp)
            if len(ho_ds) == 0:
                continue
            ho_acc, _ = evaluate(model, make_loader(ho_ds), args.task, args.condition, args.device)
            comp_results[f"color{color}_shape{shape}"] = ho_acc
    print(f"composition holdouts: mean={sum(comp_results.values())/len(comp_results):.1%}")

    out = {
        "task": args.task,
        "condition": args.condition,
        "seed": args.seed,
        "train": train_acc,
        "test": test_acc,
        "all_states": all_acc,
        "per_attr": per_attr,
        "holdout": holdout_results,
        "composition_holdout": comp_results,
    }
    out_path = model_path.replace(".pt", "_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
