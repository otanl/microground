"""Single-run training for the redesigned core, driven by a plain-dict config.

``train_one(cfg)`` is pure and importable (picklable for future parallel runners). It uses
seed separation (``init_seed`` for weights, ``split_seed`` for the data partition) and
evaluates *exhaustively* -- so the reported generalisation numbers carry zero sampling
variance. The tracked headline metric is the best test-set balanced accuracy over epochs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .world import FactoredWorld
from .data import (
    CONDITIONS, STATE_MODE, Vocab, Example,
    build_examples_from_queries, baseline_metrics, accuracy_metrics,
)
from .model import MGConfig, MGTransformer, save_mg


def parse_holdout(spec: str):
    """'random' | 'attr:<f>:<v>' | 'comp:<f>=<v>,..' | 'transition:<f>:<v>' -> tuple/None."""
    if spec == "random":
        return None
    if spec.startswith("attr:"):
        _, f, v = spec.split(":")
        return ("attr", int(f), int(v))
    if spec.startswith("transition:"):
        _, f, v = spec.split(":")
        return ("transition", int(f), int(v))
    if spec.startswith("transition_frac:"):
        return ("transition_frac", float(spec.split(":")[1]))
    if spec.startswith("bind:"):
        return ("bind", int(spec.split(":")[1]))
    if spec.startswith("bind_kshot:"):
        _, f, frac = spec.split(":")
        return ("bind_kshot", int(f), float(frac))
    if spec.startswith("comp:"):
        combo = {}
        for pair in spec[len("comp:"):].split(","):
            f, v = pair.split("=")
            combo[int(f)] = int(v)
        return ("comp", combo)
    raise ValueError(spec)


def _collate(batch: List[Example], pad_id: int, max_len: int, state_mode: str, device: str):
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, e in enumerate(batch):
        l = min(len(e.input_ids), max_len)
        ids[i, :l] = torch.tensor(e.input_ids[:l], dtype=torch.long)
    targets = torch.tensor([e.target_id for e in batch], dtype=torch.long)
    if state_mode == "none":
        state = None
    elif state_mode == "index":
        state = torch.tensor([e.state for e in batch], dtype=torch.long).to(device)
    else:  # perceptual / real
        state = torch.tensor([e.state for e in batch], dtype=torch.float32).to(device)
    return ids.to(device), targets.to(device), state


def _batches(examples, batch_size, shuffle, gen):
    if shuffle:
        idx = torch.randperm(len(examples), generator=gen).tolist()
    else:
        idx = list(range(len(examples)))
    for i in range(0, len(examples), batch_size):
        yield [examples[j] for j in idx[i:i + batch_size]]


def _evaluate(model, examples, pad_id, max_len, state_mode, device) -> Dict[str, float]:
    model.eval()
    flags: List[bool] = []
    with torch.no_grad():
        for batch in _batches(examples, 256, False, None):
            ids, targets, state = _collate(batch, pad_id, max_len, state_mode, device)
            logits = model(ids, state)
            preds = logits.argmax(dim=-1)
            flags.extend((preds == targets).cpu().tolist())
    return accuracy_metrics(examples, flags)


def train_one(cfg: Dict) -> Dict:
    """Train one model and return a fully self-describing result record."""
    device = cfg.get("device", "cpu")
    init_seed = cfg.get("init_seed", 0)
    split_seed = cfg.get("split_seed", 0)
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", 1e-3)
    batch_size = cfg.get("batch_size", 32)
    max_len = cfg.get("max_len", 32)
    eval_every = cfg.get("eval_every", 5)

    torch.manual_seed(init_seed)
    gen = torch.Generator().manual_seed(init_seed)

    if cfg["task"] == "bind":
        from .multiworld import TwoObjectWorld
        world = TwoObjectWorld()
    elif cfg["task"] == "bind3":
        from .multiworld import NObjectWorld
        world = NObjectWorld(num_objects=3, ncol=3, nshape=3)
    else:
        world = FactoredWorld()
    vocab = Vocab(world)
    cond = CONDITIONS[cfg["condition"]]
    state_mode = STATE_MODE[cond.encoder_name]
    holdout = parse_holdout(cfg.get("split", "random"))
    splits = world.split_queries(cfg["task"], split_seed=split_seed, holdout=holdout)

    train_ex = build_examples_from_queries(world, cond, splits["train"], vocab)
    test_ex = build_examples_from_queries(world, cond, splits["test"], vocab)
    all_ex = build_examples_from_queries(world, cond, splits["all"], vocab)

    model_cfg = MGConfig(
        vocab_size=len(vocab),
        hidden_size=cfg.get("hidden_size", 24),
        num_layers=cfg.get("num_layers", 1),
        num_heads=cfg.get("num_heads", 4),
        mlp_dim=cfg.get("mlp_dim", 48),
        max_len=max_len,
        state_mode=state_mode,
        attr_sizes=world.sizes if state_mode == "index" else None,
        perceptual_dim=(len(train_ex[0].state) if state_mode == "perceptual" else None),
    )
    model = MGTransformer(model_cfg).to(device)
    wd = cfg.get("wd", 0.01)  # AdamW default; sweepable (weight decay is a known grokking lever)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_test_balanced = -1.0  # so the first eval always records best_all (test may be 0.0)
    best_epoch = -1
    best_all = {}
    history = []
    # Mid-training checkpoints (for mechanistic analysis of transient states, e.g. the
    # route-dependent compositional phase found in E4). cfg["checkpoint_epochs"] = [5, 12, ...]
    ckpt_eps = set(cfg.get("checkpoint_epochs") or [])

    def _save_at(tag: str):
        import os
        safe_split = cfg.get("split", "random").replace(":", "-")
        path = os.path.join(cfg["save_dir"],
                            f"{cfg['task']}_{cfg['condition']}_{safe_split}_s{init_seed}_{tag}.pt")
        save_mg(model, path)
    for ep in range(epochs):
        model.train()
        for batch in _batches(train_ex, batch_size, True, gen):
            ids, targets, state = _collate(batch, vocab.pad_id, max_len, state_mode, device)
            opt.zero_grad()
            logits = model(ids, state)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            opt.step()
        if (ep + 1) in ckpt_eps and cfg.get("save_dir"):
            _save_at(f"ep{ep + 1}")
        if (ep + 1) % eval_every == 0 or ep == 0 or ep == epochs - 1:
            test_m = _evaluate(model, test_ex, vocab.pad_id, max_len, state_mode, device)
            all_m = _evaluate(model, all_ex, vocab.pad_id, max_len, state_mode, device)
            history.append({"ep": ep + 1, "test": test_m["balanced"], "all": all_m["balanced"]})
            if test_m["balanced"] > best_test_balanced:
                best_test_balanced = test_m["balanced"]
                best_epoch = ep + 1
                best_all = all_m

    best_test_balanced = max(best_test_balanced, 0.0)  # clamp the -1.0 sentinel

    # Save the CONVERGED model for mechanistic follow-up (probing / error analysis).
    if cfg.get("save_dir"):
        import os
        safe_split = cfg.get("split", "random").replace(":", "-")
        path = os.path.join(cfg["save_dir"],
                            f"{cfg['task']}_{cfg['condition']}_{safe_split}_s{init_seed}.pt")
        save_mg(model, path)

    base_test = baseline_metrics(world, test_ex)
    base_all = baseline_metrics(world, all_ex)
    return {
        "task": cfg["task"],
        "condition": cfg["condition"],
        "split": cfg.get("split", "random"),
        "init_seed": init_seed,
        "split_seed": split_seed,
        "state_mode": state_mode,
        "params": model.count_params(),
        "n_train": len(train_ex),
        "n_test": len(test_ex),
        "best_test_balanced": best_test_balanced,
        "best_epoch": best_epoch,
        "best_all_balanced": best_all.get("balanced"),
        "baseline_test": base_test,
        "baseline_all": base_all,
        "epochs": epochs,
        "lr": lr,
        "wd": wd,
        "hidden_size": model_cfg.hidden_size,
        "mlp_dim": model_cfg.mlp_dim,
        "history": history,
    }
