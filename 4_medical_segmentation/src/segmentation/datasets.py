from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _stem_map(paths: Iterable[Path]) -> dict[str, Path]:
    return {path.stem: path for path in paths}


class KvasirSegDataset(Dataset):
    """Load Kvasir-SEG style `images/` and `masks/` folders."""

    def __init__(self, root: str | Path, image_size: int = 256, limit: int | None = None):
        self.root = Path(root)
        self.image_size = int(image_size)
        image_dir = self.root / "images"
        mask_dir = self.root / "masks"

        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Expected dataset folders at {image_dir} and {mask_dir}."
            )

        images = _stem_map(_list_images(image_dir))
        masks = _stem_map(_list_images(mask_dir))
        paired_stems = sorted(set(images) & set(masks))
        if not paired_stems:
            raise ValueError(f"No image/mask pairs found under {self.root}.")

        if limit is not None:
            paired_stems = paired_stems[: int(limit)]

        self.samples = [(images[stem], masks[stem]) for stem in paired_stems]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).contiguous()
        return image_tensor, mask_tensor

