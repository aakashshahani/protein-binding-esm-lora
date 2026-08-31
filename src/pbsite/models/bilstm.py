"""(a) BiLSTM baseline over per-residue ESM-2 embeddings.

Consumes cached embeddings (optionally concatenated with physicochemical
features) and emits a per-residue binding logit. This is the honest classical
baseline the modern models must beat.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, D) -> logits (B, L). Packing keeps padding from leaking."""
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        else:
            out, _ = self.lstm(x)
        return self.classifier(self.dropout(out)).squeeze(-1)
