r"""
Train a MicroGround model on the two-object world.

Example:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\train_multiobject.py --condition state_grounded --epochs 200 --seed 0
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from micro_ground import Condition, Vocab, MicroGroundDataset
from micro_ground.multi_world import MultiObjectWorld
from micro_ground.model import MicroGroundTransformer
from micro_ground.data import collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["text_only", "text_minimal", "state_grounded", "counterfactual"], required=True)
    p.add_argument("--hidden_size", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save_dir", default="models/microground")
    p.add_argument("--max_len", type=int, default=32)
    return p.parse_args()


def evaluate(model, loader, device, condition):
    model.eval()
    correct = 0
    total = 0
    per_task = {}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            state = None if condition in ("text_only", "text_minimal") else batch["state"].to(device)
            token_logits, id_logits = model(input_ids, state)
            preds = token_logits.argmax(dim=-1)
            ok = (preds == labels)
            correct += ok.sum().item()
            total += labels.numel()
            for i, meta in enumerate(batch["meta"]):
                task = meta["task"]
                per_task.setdefault(task, [0, 0])
                per_task[task][0] += ok[i].item()
                per_task[task][1] += 1
    return correct / total, {k: v[0]/v[1] for k, v in per_task.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    world = MultiObjectWorld()
    vocab = Vocab()
    condition = Condition(args.condition)

    train_ds = MicroGroundDataset(world, None, condition, vocab, split="train", seed=args.seed)
    test_ds = MicroGroundDataset(world, None, condition, vocab, split="test", seed=args.seed)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, vocab.pad_id, args.max_len),
        )

    train_loader = make_loader(train_ds, True)
    test_loader = make_loader(test_ds, False)

    # 6 state attributes for two objects
    model = MicroGroundTransformer(
        vocab_size=len(vocab),
        num_states=world.num_states,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        attr_sizes=[4, 4, 2, 4, 4, 2],
    ).to(args.device)
    print(f"Model params: {model.count_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_acc, _ = evaluate(model, train_loader, args.device, args.condition)
    test_acc, _ = evaluate(model, test_loader, args.device, args.condition)
    print(f"BASE train={train_acc:.1%} test={test_acc:.1%}")

    best_test = 0.0
    history = []
    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            state = None if args.condition in ("text_only", "text_minimal") else batch["state"].to(args.device)
            opt.zero_grad()
            token_logits, id_logits = model(input_ids, state)
            loss = torch.nn.functional.cross_entropy(token_logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        train_acc, _ = evaluate(model, train_loader, args.device, args.condition)
        test_acc, per_task = evaluate(model, test_loader, args.device, args.condition)
        best_test = max(best_test, test_acc)
        history.append({"epoch": ep + 1, "loss": total_loss / len(train_loader), "train": train_acc, "test": test_acc, "per_task": per_task})
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"ep {ep+1:3d} loss={total_loss/len(train_loader):.4f} train={train_acc:.1%} test={test_acc:.1%} best={best_test:.1%}")

    print(f"FINAL best_test={best_test:.1%}")
    for task, acc in per_task.items():
        print(f"  {task}: {acc:.1%}")

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"multi_{args.condition}_s{args.seed}.pt")
    torch.save(model.state_dict(), model_path)
    meta = {
        "task": "multi",
        "condition": args.condition,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "mlp_dim": args.mlp_dim,
        "params": model.count_params(),
        "best_test": best_test,
        "history": history,
    }
    with open(model_path.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
