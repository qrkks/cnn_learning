from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from segmentation.datasets import KvasirSegDataset
from segmentation.metrics import segmentation_metrics
from segmentation.models.attention_unet import AttentionUNet
from segmentation.visualize import save_prediction_grid


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


def validation_subset(dataset: KvasirSegDataset, val_fraction: float, seed: int) -> Subset:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_fraction))
    return Subset(dataset, indices[:val_size])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Attention U-Net checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the YAML config used for training.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved .pth checkpoint.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--split", choices=["val", "full"], default="val")
    parser.add_argument("--num-visuals", type=int, default=8)
    args = parser.parse_args()

    config = read_config(args.config)
    seed = int(config.get("seed", 2026))
    data_root = project_path(args.data_root or config["data_root"])
    output_dir = project_path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = KvasirSegDataset(
        data_root,
        image_size=int(config["image_size"]),
        limit=config.get("limit_samples"),
    )
    eval_set = (
        validation_subset(dataset, float(config["val_fraction"]), seed)
        if args.split == "val"
        else dataset
    )
    loader = DataLoader(eval_set, batch_size=int(config["batch_size"]), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    model = AttentionUNet(base_channels=int(config["base_channels"])).to(device)
    checkpoint = torch.load(project_path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    seen = 0
    visual_count = 0
    prediction_dir = output_dir / "predictions"

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            batch_size = images.size(0)
            batch_metrics = segmentation_metrics(logits, masks)
            for key, value in batch_metrics.items():
                totals[key] += value * batch_size
            seen += batch_size

            for batch_index in range(batch_size):
                if visual_count >= args.num_visuals:
                    break
                save_prediction_grid(
                    images[batch_index],
                    masks[batch_index],
                    logits[batch_index],
                    prediction_dir / f"prediction_{visual_count:03d}.png",
                )
                visual_count += 1

    metrics = {key: value / seen for key, value in totals.items()}
    metrics_path = output_dir / f"metrics_{args.split}.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved visualizations to {prediction_dir}")


if __name__ == "__main__":
    main()
