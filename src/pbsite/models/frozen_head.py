"""(b) Frozen ESM-2 + token-classification head.

The ESM-2 backbone is not trained; we only learn a small MLP over the cached
per-residue embeddings. This mirrors the CLAPE-SMB setup (frozen encoder + head)
and is our fair, cheap point of comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidueMLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, D) -> logits (B, L)."""
        return self.net(x).squeeze(-1)
