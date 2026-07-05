"""Dataset for MicroGround tasks."""
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .world import ObjectWorld, Condition, Task, ATTRIBUTE_NAMES, ATTRIBUTE_LISTS
from .multi_world import MultiObjectWorld


class Vocab:
    """Vocabulary for text tokens."""

    def __init__(self):
        self.tokens = ["<pad>", "<s>", "</s>", "<unk>"] + ATTRIBUTE_NAMES + [
            tok for lst in ATTRIBUTE_LISTS for tok in lst
        ] + ["what", "is", "the", "if", "change", "to", "of", "yes", "no", "first", "second", "move", "object", "where", "it"]
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<s>"]
        self.eos_id = self.token_to_id["</s>"]
        self.unk_id = self.token_to_id["<unk>"]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        toks = text.split()
        if add_special_tokens:
            toks = ["<s>"] + toks + ["</s>"]
        return [self.token_to_id.get(t, self.unk_id) for t in toks]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.tokens[i] for i in ids)

    def __len__(self):
        return len(self.tokens)


class MicroGroundDataset(Dataset):
    """PyTorch dataset for MicroGround."""

    def __init__(
        self,
        world,
        task: Task,
        condition: Condition,
        vocab: Vocab,
        split: str = "train",
        seed: int = 0,
        holdout_attr: str = None,
        holdout_composition: dict = None,
        include_state: bool = True,
    ):
        self.world = world
        self.task = task
        self.condition = condition
        self.vocab = vocab
        self.include_state = include_state
        if isinstance(world, MultiObjectWorld):
            self.examples = world.generate_examples(condition, split, seed)
        else:
            self.examples = world.generate_examples(task, condition, split, seed, holdout_attr, holdout_composition)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, target, meta = self.examples[idx]
        input_ids = self.vocab.encode(text)
        target_id = self.vocab.token_to_id.get(target)
        if target_id is None:
            # target may be an object id string
            target_id = int(target)
        state = meta.get("state", meta.get("old_state", []))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target": torch.tensor(target_id, dtype=torch.long),
            "state": torch.tensor(state, dtype=torch.long),
            "meta": meta,
        }


def collate_fn(batch, pad_id: int, max_len: int):
    """Pad a batch of variable-length token sequences."""
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        l = len(b["input_ids"])
        input_ids[i, :l] = b["input_ids"]
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.long)
    states = torch.stack([b["state"] for b in batch])
    return {
        "input_ids": input_ids,
        "labels": targets,
        "state": states,
        "meta": [b["meta"] for b in batch],
    }
