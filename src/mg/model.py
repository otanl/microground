"""Tiny transformer for the redesigned MicroGround core.

Supports three state channels (RESEARCH_PLAN §4/§5.1):

* ``state_mode="none"``       -- no side channel (text_only / text_minimal).
* ``state_mode="index"``      -- per-factor embedding tables (factored / control codes).
* ``state_mode="perceptual"`` -- a linear projection of a real-valued perceptual vector,
  which entangles factors so that recovering them is a genuine learning problem (this is
  what turns "grounding" from a lookup into a computation).

The whole model is deliberately small and inspectable. Save/load round-trips the config
and uses ``strict=True`` so a size mismatch fails loudly (fixes the legacy silent
``strict=False`` load).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class MGConfig:
    vocab_size: int
    hidden_size: int = 24
    num_layers: int = 1
    num_heads: int = 4
    mlp_dim: int = 48
    max_len: int = 32
    dropout: float = 0.0
    state_mode: str = "none"                 # "none" | "index" | "perceptual"
    attr_sizes: Optional[List[int]] = None   # required for state_mode="index"
    perceptual_dim: Optional[int] = None     # required for state_mode="perceptual"


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len: int):
        return self.pe[:, :seq_len]


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, hidden_size))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class MGTransformer(nn.Module):
    def __init__(self, cfg: MGConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = SinusoidalPositionalEmbedding(cfg.hidden_size, cfg.max_len)

        if cfg.state_mode == "index":
            assert cfg.attr_sizes, "state_mode=index requires attr_sizes"
            self.attr_embs = nn.ModuleList([nn.Embedding(s, cfg.hidden_size) for s in cfg.attr_sizes])
        elif cfg.state_mode == "perceptual":
            assert cfg.perceptual_dim, "state_mode=perceptual requires perceptual_dim"
            self.state_proj = nn.Linear(cfg.perceptual_dim, cfg.hidden_size)
        elif cfg.state_mode != "none":
            raise ValueError(cfg.state_mode)

        self.layers = nn.ModuleList([
            TransformerBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_dim, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        self.token_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def _state_embedding(self, state):
        if self.cfg.state_mode == "index":
            return sum(emb(state[:, i]) for i, emb in enumerate(self.attr_embs))
        return self.state_proj(state)  # perceptual

    def forward(self, input_ids: torch.Tensor, state: Optional[torch.Tensor] = None,
                return_hidden: bool = False):
        b, l = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb(l)
        if state is not None and self.cfg.state_mode != "none":
            x = x + self._state_embedding(state).unsqueeze(1)  # broadcast to all positions
        mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        x = self.ln_f(x)
        last = x[:, -1, :]                    # final-token representation (probe target)
        logits = self.token_head(last)
        if return_hidden:
            return logits, last
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_mg(model: MGTransformer, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    with open(path.replace(".pt", ".config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(model.cfg), f, indent=2)


def load_mg(path: str, device: str = "cpu") -> MGTransformer:
    with open(path.replace(".pt", ".config.json"), encoding="utf-8") as f:
        cfg = MGConfig(**json.load(f))
    model = MGTransformer(cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))  # strict=True by default
    return model
