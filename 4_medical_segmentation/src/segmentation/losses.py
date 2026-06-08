from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation logits."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = DiceLoss()(logits, targets)
    return bce_weight * bce + dice_weight * dice

