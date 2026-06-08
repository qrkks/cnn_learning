import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _write_sample(root: Path, name: str, mask_value: int = 255) -> None:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((16, 20, 3), dtype=np.uint8)
    image[..., 0] = 128
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[4:12, 5:15] = mask_value

    Image.fromarray(image).save(image_dir / f"{name}.jpg")
    Image.fromarray(mask).save(mask_dir / f"{name}.jpg")


def test_kvasir_dataset_pairs_images_and_masks(tmp_path: Path):
    from segmentation.datasets import KvasirSegDataset

    _write_sample(tmp_path, "case_001")

    dataset = KvasirSegDataset(tmp_path, image_size=32)
    image, mask = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert image.min() >= 0
    assert image.max() <= 1
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


def test_dice_loss_is_low_for_correct_prediction():
    from segmentation.losses import DiceLoss

    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits = torch.tensor([[[[8.0, -8.0], [8.0, -8.0]]]])

    loss = DiceLoss()(logits, target)

    assert loss.item() < 0.01


def test_segmentation_metrics_for_perfect_mask():
    from segmentation.metrics import segmentation_metrics

    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits = torch.tensor([[[[8.0, -8.0], [8.0, -8.0]]]])

    metrics = segmentation_metrics(logits, target)

    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_attention_unet_forward_shape():
    from segmentation.models.attention_unet import AttentionUNet

    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=8)
    output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 1, 64, 64)


def test_project_path_accepts_workspace_relative_paths():
    from evaluate import project_path

    resolved = project_path("4_medical_segmentation/configs/smoke.yaml")

    assert resolved == PROJECT_ROOT / "configs" / "smoke.yaml"
