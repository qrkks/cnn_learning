from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_prediction_grid(
    image: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
    output_path: str | Path,
    threshold: float = 0.5,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().squeeze(0).numpy()
    pred_np = (torch.sigmoid(logits.detach().cpu()).squeeze(0) >= threshold).float().numpy()

    overlay = image_np.copy()
    overlay[..., 1] = overlay[..., 1] * 0.55 + pred_np * 0.45

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    panels = [
        (image_np, "Image", None),
        (target_np, "Ground truth", "gray"),
        (pred_np, "Prediction", "gray"),
        (overlay, "Overlay", None),
    ]
    for axis, (content, title, cmap) in zip(axes, panels, strict=True):
        axis.imshow(content, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

