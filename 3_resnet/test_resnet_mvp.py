from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

import torch


def load_resnet_mvp_module() -> ModuleType:
    """Load resnet_mvp.py directly because this folder name starts with a digit."""
    module_path: Path = Path(__file__).with_name("resnet_mvp.py")
    spec = importlib.util.spec_from_file_location("resnet_mvp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ResNetMvpTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.resnet_mvp: ModuleType = load_resnet_mvp_module()

    def test_basic_block_preserves_shape_when_stride_is_one(self) -> None:
        block = self.resnet_mvp.BasicBlock(in_channels=16, out_channels=16, stride=1)
        x: torch.Tensor = torch.randn(2, 16, 32, 32)

        y: torch.Tensor = block(x)

        self.assertEqual(tuple(y.shape), (2, 16, 32, 32))

    def test_basic_block_projects_shortcut_when_shape_changes(self) -> None:
        block = self.resnet_mvp.BasicBlock(in_channels=16, out_channels=32, stride=2)
        x: torch.Tensor = torch.randn(2, 16, 32, 32)

        y: torch.Tensor = block(x)

        self.assertEqual(tuple(y.shape), (2, 32, 16, 16))

    def test_mini_resnet_outputs_one_logit_vector_per_image(self) -> None:
        model = self.resnet_mvp.MiniResNet(num_classes=10)
        x: torch.Tensor = torch.randn(4, 3, 32, 32)

        logits: torch.Tensor = model(x)

        self.assertEqual(tuple(logits.shape), (4, 10))


if __name__ == "__main__":
    unittest.main()
