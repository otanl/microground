r"""
Activation patching for MicroGround models.

For a state-grounded model, we compare a clean input and a corrupted input (one attribute changed).
Then we patch the hidden state at each layer from corrupted into clean to see if the output flips.

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\patch_microground.py --task attr --condition state_grounded --seed 0
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from micro_ground import ObjectWorld, Task, Condition, Vocab, ATTRIBUTE_NAMES, ATTRIBUTE_LISTS
from micro_ground.model import MicroGroundTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["attr", "counterfactual"], required=True)
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


def get_hidden_states(model, input_ids, state):
    """Return list of hidden states (one per layer) for the last token."""
    states = []
    def hook(mod, inp, out):
        states.append(out[:, -1, :].detach().clone())
    handles = [layer.register_forward_hook(hook) for layer in model.layers]
    with torch.no_grad():
        model(input_ids, state)
    for h in handles:
        h.remove()
    return states


def run_with_patch(model, input_ids, state, layer_idx, patch_vector):
    """Run clean input but replace the last-token hidden at layer_idx with patch_vector."""
    def hook(mod, inp, out):
        out = out.clone()
        out[:, -1, :] = patch_vector
        return out
    handle = model.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        token_logits, id_logits = model(input_ids, state)
    handle.remove()
    return token_logits, id_logits


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
    model.eval()

    all_examples = world.generate_examples(task, condition, "all", args.seed)
    attr_flip_rate = {attr: [] for attr in ATTRIBUTE_NAMES}

    for text, target, meta in all_examples:
        state = torch.tensor(meta.get("state", meta.get("old_state", [0, 0, 0, 0])), dtype=torch.long).unsqueeze(0).to(args.device)
        input_ids = torch.tensor(vocab.encode(text), dtype=torch.long).unsqueeze(0).to(args.device)
        target_id = vocab.token_to_id.get(target)
        if target_id is None:
            target_id = int(target)

        clean_states = get_hidden_states(model, input_ids, state)
        clean_logits = model(input_ids, state)
        clean_pred = clean_logits[1].argmax(-1).item() if args.task == "id" else clean_logits[0].argmax(-1).item()

        for attr in ATTRIBUTE_NAMES:
            attr_idx = ATTRIBUTE_NAMES.index(attr)
            cur_val = state[0, attr_idx].item()
            new_val = (cur_val + 1) % len(ATTRIBUTE_LISTS[attr_idx])
            corrupted_state = state.clone()
            corrupted_state[0, attr_idx] = new_val
            corrupted_states = get_hidden_states(model, input_ids, corrupted_state)
            corrupted_pred = model(input_ids, corrupted_state)[1 if args.task == "id" else 0].argmax(-1).item()

            # Patch each layer
            patched_preds = []
            for layer_idx in range(args.num_layers):
                p_logits = run_with_patch(model, input_ids, state, layer_idx, corrupted_states[layer_idx])
                p_pred = p_logits[1].argmax(-1).item() if args.task == "id" else p_logits[0].argmax(-1).item()
                patched_preds.append(p_pred)

            # Record if any patch causes the model to output the corrupted answer
            if clean_pred == target_id and corrupted_pred != target_id:
                attr_flip_rate[attr].append(any(p == corrupted_pred for p in patched_preds))

    summary = {}
    for attr, vals in attr_flip_rate.items():
        summary[attr] = sum(vals) / len(vals) if vals else 0.0

    print("Activation patching: fraction of clean-correct cases where patching layer flips to corrupted output")
    for attr, rate in summary.items():
        print(f"  {attr}: {rate:.1%}")

    out = {
        "task": args.task,
        "condition": args.condition,
        "seed": args.seed,
        "summary": summary,
    }
    out_path = model_path.replace(".pt", "_patch.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
