from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from segmentation.datasets import KvasirSegDataset
from segmentation.losses import bce_dice_loss
from segmentation.metrics import segmentation_metrics
from segmentation.models.attention_unet import AttentionUNet


PROJECT_DIR = Path(__file__).resolve().parents[1]


def read_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if path.parts and path.parts[0] == PROJECT_DIR.name:
        return cwd_candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return PROJECT_DIR / path


def create_sample_dataset(root: Path, count: int = 8, size: int = 128) -> None:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(2026)
    for index in range(count):
        image = Image.new("RGB", (size, size), color=(22, 20, 18))
        mask = Image.new("L", (size, size), color=0)
        draw_image = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)

        for _ in range(18):
            x = rng.randint(0, size - 8)
            y = rng.randint(0, size - 8)
            shade = rng.randint(35, 95)
            draw_image.ellipse((x, y, x + rng.randint(4, 18), y + rng.randint(4, 18)), fill=(shade, 35, 30))

        cx = rng.randint(size // 3, 2 * size // 3)
        cy = rng.randint(size // 3, 2 * size // 3)
        rx = rng.randint(size // 8, size // 4)
        ry = rng.randint(size // 10, size // 5)
        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
        draw_image.ellipse(bbox, fill=(185, 95, 82))
        draw_mask.ellipse(bbox, fill=255)

        name = f"sample_{index:03d}.jpg"
        image.save(image_dir / name, quality=95)
        mask.save(mask_dir / name, quality=95)


def split_dataset(dataset: KvasirSegDataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    indices = list(range(len(dataset)))
    generator = random.Random(seed)
    generator.shuffle(indices)
    val_size = max(1, int(len(indices) * val_fraction))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    if not train_indices:
        raise ValueError("Training split is empty; provide at least two samples.")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    seen = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        batch_metrics = segmentation_metrics(logits.detach(), masks.detach())
        totals["loss"] += loss.item() * batch_size
        for key, value in batch_metrics.items():
            totals[key] += value * batch_size
        seen += batch_size

    return {key: value / seen for key, value in totals.items()}


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "train_loss",
        "train_dice",
        "train_iou",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Attention U-Net on Kvasir-SEG.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--make-sample-data", action="store_true", help="Create toy data for smoke runs.")
    args = parser.parse_args()

    config = read_config(args.config)
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    torch.manual_seed(seed)

    data_root = project_path(args.data_root or config["data_root"])
    output_dir = project_path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.make_sample_data:
        create_sample_dataset(data_root, count=int(config.get("sample_count", 8)))

    dataset = KvasirSegDataset(
        data_root,
        image_size=int(config["image_size"]),
        limit=config.get("limit_samples"),
    )
    train_set, val_set = split_dataset(dataset, float(config["val_fraction"]), seed)
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )

    device_name = "cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu"
    device = torch.device(device_name)
    model = AttentionUNet(base_channels=int(config["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    best_dice = -1.0
    history = []
    checkpoint_path = output_dir / "best_model.pth"
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f}"
        )

        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_dice": best_dice,
                },
                checkpoint_path,
            )

    write_history(output_dir / "history.csv", history)
    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
