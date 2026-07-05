r"""Pre-train a MicroGround-sized transformer as a causal language model on text.

This connects the SLM pre-training pipeline to the MicroGround testbed.
The model is the same MicroGroundTransformer architecture, trained on general text
using the word-level MicroGround vocabulary.

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts/pretrain_microground.py --corpus data/sample_corpus.txt --epochs 200 --save_dir models/microground_pretrain
"""
import argparse
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from micro_ground import Vocab
from micro_ground.model import MicroGroundTransformer


class TextDataset(Dataset):
    def __init__(self, text_file: str, vocab: Vocab, seq_len: int = 16):
        self.vocab = vocab
        self.seq_len = seq_len
        # Read all text, tokenize, and create sliding windows
        with open(text_file, encoding="utf-8") as f:
            text = f.read()
        tokens = vocab.encode(text, add_special_tokens=False)
        self.tokens = tokens
        print(f"Loaded {len(self.tokens)} tokens from {text_file}")

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [self.vocab.pad_id] * (self.seq_len + 1 - len(chunk))
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="Path to text file for pre-training")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=24)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=48)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save_dir", default="models/microground_pretrain")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def collate(batch, pad_id):
    input_ids = torch.nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=pad_id)
    return input_ids, labels


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    vocab = Vocab()
    ds = TextDataset(args.corpus, vocab, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate(batch, vocab.pad_id))

    model = MicroGroundTransformer(
        vocab_size=len(vocab),
        num_states=128,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        attr_sizes=[4, 4, 4, 2],
    ).to(args.device)
    print(f"Model params: {model.count_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for input_ids, labels in dl:
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            opt.zero_grad()
            token_logits, _ = model(input_ids, return_all_logits=True)
            loss = torch.nn.functional.cross_entropy(token_logits.view(-1, len(vocab)), labels.view(-1), ignore_index=vocab.pad_id)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            steps += 1
        print(f"ep {ep+1:3d} loss={total_loss/steps:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
