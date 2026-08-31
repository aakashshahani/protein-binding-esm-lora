"""Masked losses for heavily imbalanced per-residue binding labels.

Binding residues are rare (single-digit % of positions), so we support
weighted BCE and focal loss, both masked to ignore padding positions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    pw = torch.tensor(pos_weight, device=logits.device) if pos_weight else None
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def masked_focal(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
    gamma: float = 2.0, alpha: float = 0.25,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal = ce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal = alpha_t * focal
    focal = focal * mask
    return focal.sum() / mask.sum().clamp_min(1.0)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, mask):
        return masked_focal(logits, targets, mask, self.gamma, self.alpha)
