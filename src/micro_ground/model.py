"""MicroGround model: small transformer with optional state input.

Designed for mechanistic interpretability: weights and activations are inspectable.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embeddings (small, inspectable)."""

    def __init__(self, hidden_size: int, max_len: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        pe = torch.zeros(max_len, hidden_size)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len: int):
        return self.pe[:, :seq_len]


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size),
        )

    def forward(self, x, attn_mask=None):
        # Pre-norm style
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class MicroGroundTransformer(nn.Module):
    """Tiny transformer for MicroGround tasks.

    Inputs:
        input_ids: token IDs (batch, seq_len)
        state: optional attribute vector (batch, num_attrs) for grounded conditions
    Outputs:
        token_logits: (batch, vocab_size)
        id_logits: (batch, num_states)
    """

    def __init__(
        self,
        vocab_size: int,
        num_states: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_dim: int = 128,
        num_attrs: int = 4,
        attr_sizes: Optional[list] = None,
        dropout: float = 0.0,
        max_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_states = num_states
        self.hidden_size = hidden_size
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = SinusoidalPositionalEmbedding(hidden_size, max_len)

        # State embedding: one embedding table per attribute, project to hidden_size
        self.attr_sizes = attr_sizes or [4, 4, 4, 2]
        self.attr_embs = nn.ModuleList([
            nn.Embedding(size, hidden_size) for size in self.attr_sizes
        ])

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_size)
        self.token_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.id_head = nn.Linear(hidden_size, num_states, bias=False)

    def forward(self, input_ids: torch.Tensor, state: Optional[torch.Tensor] = None, return_all_logits: bool = False):
        b, l = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb(l)

        if state is not None:
            # Add state attribute embeddings to every token position
            state_emb = sum(emb(state[:, i]) for i, emb in enumerate(self.attr_embs))
            x = x + state_emb.unsqueeze(1)

        mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        x = self.ln_f(x)

        if return_all_logits:
            token_logits = self.token_head(x)
            return token_logits, None
        # Use the last token representation for classification
        last = x[:, -1, :]
        token_logits = self.token_head(last)
        id_logits = self.id_head(last)
        return token_logits, id_logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
