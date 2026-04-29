from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


# ============================================================
# 0. 这个文件在做什么？
# ============================================================
# ResNet 的核心思想只有一句话：
#
#     不要让一个网络层直接学习 H(x)，而是让它学习 F(x) = H(x) - x。
#
# 于是输出就变成：
#
#     y = F(x) + x
#
# 这里的 x 叫 identity / shortcut / skip connection。
# 这条“捷径”让梯度可以更顺畅地往回传，也让深层网络更容易训练。
#
# 这个 MVP 用 CIFAR-10 做演示，所以模型比论文里的 ResNet-18 小很多：
# - 输入图片：3 x 32 x 32
# - 输出类别：10 类
# - 残差块：BasicBlock
# - 网络：MiniResNet
# ============================================================


CIFAR10_CLASSES: Final[tuple[str, ...]] = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class TrainConfig:
    """训练配置。

    dataclass 的好处是：训练参数集中放在一起，函数签名不会变得很长。
    frozen=True 表示创建后不再修改，避免训练中途悄悄改配置。
    """

    data_dir: Path
    batch_size: int = 128
    epochs: int = 1
    learning_rate: float = 1e-3
    train_limit: int | None = 4096
    test_limit: int | None = 1024
    verbose_shapes: bool = False


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """创建 ResNet BasicBlock 里最常用的 3x3 卷积。

    bias=False 是因为后面紧跟 BatchNorm2d。
    BatchNorm 自己有可学习的平移参数 beta，所以卷积层的 bias 通常可以省掉。
    """

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """ResNet 的基础残差块。

    一个 BasicBlock 有两条路：

    1. main path:
       x -> Conv -> BN -> ReLU -> Conv -> BN

    2. shortcut path:
       x -> 原样返回
       或者
       x -> 1x1 Conv -> BN

    最后：
       output = ReLU(main_path + shortcut_path)

    什么时候 shortcut 不能原样返回？
    - 通道数变了：例如 16 channels -> 32 channels
    - 宽高变了：例如 stride=2 时 32x32 -> 16x16

    Tensor 相加要求 shape 完全一致，所以 shape 变化时必须用 projection shortcut。
    """

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1: nn.Conv2d = conv3x3(in_channels, out_channels, stride)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

        self.conv2: nn.Conv2d = conv3x3(out_channels, out_channels)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)

        needs_projection: bool = stride != 1 or in_channels != out_channels
        if needs_projection:
            # 1x1 卷积只调整 shape：改变通道数，也可以用 stride 改变宽高。
            # 它不是为了扩大感受野，而是为了让 shortcut 能和 main path 相加。
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # shape 不变时，shortcut 就是真正的 identity：什么也不做，直接把 x 传过去。
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先保存 shortcut 分支。
        # 这一步非常关键：后面 main path 会改变 x，但残差连接需要原始输入。
        identity: torch.Tensor = self.shortcut(x)

        # main path：学习残差 F(x)。
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # residual add：H(x) = F(x) + x
        # 如果网络暂时学不到好东西，至少可以让 F(x) 接近 0，
        # 这样整个 block 接近 identity，不会把信息严重破坏。
        out = out + identity
        out = self.relu(out)
        return out


class MiniResNet(nn.Module):
    """一个适合 CIFAR-10 教学的迷你 ResNet。

    真实 ResNet-18 面向 ImageNet，输入通常是 224x224。
    CIFAR-10 只有 32x32，如果一开始就用 7x7 stride=2 + maxpool，
    图片会太快被压小，很多细节还没来得及学习就没了。

    所以这里使用 CIFAR 风格 stem：
    - 3x3 convolution
    - stride=1
    - 不用开头 maxpool
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels: int = 16

        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 三个 stage：
        # stage1: 保持 32x32，学习低层纹理
        # stage2: 变成 16x16，通道变多，学习中层形状
        # stage3: 变成 8x8，通道继续变多，学习更抽象语义
        self.stage1: nn.Sequential = self._make_stage(out_channels=16, blocks=2, stride=1)
        self.stage2: nn.Sequential = self._make_stage(out_channels=32, blocks=2, stride=2)
        self.stage3: nn.Sequential = self._make_stage(out_channels=64, blocks=2, stride=2)

        # AdaptiveAvgPool2d((1, 1)) 会把任意 HxW 压成 1x1。
        # 对 CIFAR-10 来说，最后 64 x 8 x 8 -> 64 x 1 x 1。
        self.avg_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier: nn.Linear = nn.Linear(64, num_classes)

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        """堆叠多个 BasicBlock。

        每个 stage 的第一个 block 可能负责降采样，所以它使用传入的 stride。
        后面的 block 都保持 shape，所以 stride=1。
        """

        layers: list[nn.Module] = [
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        ]
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits: torch.Tensor = self.classifier(x)
        return logits


def log_shape(name: str, x: torch.Tensor, verbose: bool) -> None:
    """按需打印 tensor shape，帮助你观察图片如何穿过网络。"""

    if verbose:
        print(f"{name:<18} shape = {tuple(x.shape)}")


def explain_forward_shapes(model: MiniResNet, images: torch.Tensor, verbose: bool) -> None:
    """手动走一遍 forward，用来展示每个 stage 的 shape。

    这不是训练必须的函数，只是教学辅助。
    如果你想理解 ResNet，可以先打开 --verbose-shapes 跑一次。
    """

    if not verbose:
        return

    print("\nShape trace for one batch:")
    log_shape("input", images, verbose)
    x: torch.Tensor = model.stem(images)
    log_shape("stem", x, verbose)
    x = model.stage1(x)
    log_shape("stage1", x, verbose)
    x = model.stage2(x)
    log_shape("stage2", x, verbose)
    x = model.stage3(x)
    log_shape("stage3", x, verbose)
    x = model.avg_pool(x)
    log_shape("avg_pool", x, verbose)
    x = torch.flatten(x, start_dim=1)
    log_shape("flatten", x, verbose)
    logits: torch.Tensor = model.classifier(x)
    log_shape("classifier", logits, verbose)
    print()


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """创建训练和测试 transform。

    训练集多做 RandomCrop / RandomHorizontalFlip，是轻量数据增强：
    - RandomCrop: 让模型不要过度依赖物体出现在图片正中央
    - RandomHorizontalFlip: 水平翻转后类别通常不变，例如车、狗、船

    测试集不能随机增强，否则每次评估都不稳定。
    """

    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    train_transform: transforms.Compose = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform: transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform


def maybe_limit_dataset(dataset: Dataset[tuple[torch.Tensor, int]], limit: int | None) -> Dataset[tuple[torch.Tensor, int]]:
    """为了 MVP 快速跑通，可以只取一小部分数据。

    limit=None 表示使用完整 CIFAR-10。
    默认只训练 4096 张、测试 1024 张，是为了让 CPU 上也能快速看到完整流程。
    """

    if limit is None:
        return dataset

    safe_limit: int = min(limit, len(dataset))
    indices: range = range(safe_limit)
    return Subset(dataset, indices)


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """下载/读取 CIFAR-10，并包装成 DataLoader。"""

    train_transform, test_transform = build_transforms()
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    limited_train_dataset = maybe_limit_dataset(train_dataset, config.train_limit)
    limited_test_dataset = maybe_limit_dataset(test_dataset, config.test_limit)

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        limited_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        limited_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def choose_device() -> torch.device:
    """优先使用 CUDA；没有 GPU 时自动回到 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """根据 logits 计算一个 batch 的准确率。"""

    predictions: torch.Tensor = torch.argmax(logits, dim=1)
    correct: int = int((predictions == labels).sum().item())
    return correct / labels.numel()


def train_one_epoch(
    model: MiniResNet,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    verbose_shapes: bool,
) -> tuple[float, float]:
    """训练一个 epoch，返回平均 loss 和平均 accuracy。"""

    model.train()
    total_loss: float = 0.0
    total_correct: int = 0
    total_examples: int = 0

    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        if epoch == 1 and batch_index == 0:
            explain_forward_shapes(model, images, verbose_shapes)

        optimizer.zero_grad()
        logits: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size: int = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_examples += batch_size

    average_loss: float = total_loss / total_examples
    average_accuracy: float = total_correct / total_examples
    return average_loss, average_accuracy


@torch.no_grad()
def evaluate(
    model: MiniResNet,
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在测试集上评估模型。

    @torch.no_grad() 表示这里不需要保存梯度，速度更快，也更省内存。
    """

    model.eval()
    total_loss: float = 0.0
    total_correct: int = 0
    total_examples: int = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(logits, labels)

        batch_size: int = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_examples += batch_size

    average_loss: float = total_loss / total_examples
    average_accuracy: float = total_correct / total_examples
    return average_loss, average_accuracy


def train(config: TrainConfig) -> MiniResNet:
    """完整训练入口：准备数据、创建模型、训练、评估。"""

    device: torch.device = choose_device()
    print(f"device = {device}")
    print(f"data_dir = {config.data_dir}")

    train_loader, test_loader = build_dataloaders(config)
    model = MiniResNet(num_classes=len(CIFAR10_CLASSES)).to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            verbose_shapes=config.verbose_shapes,
        )
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(
            f"epoch {epoch:02d}/{config.epochs} | "
            f"train loss {train_loss:.4f}, train acc {train_accuracy * 100:.2f}% | "
            f"test loss {test_loss:.4f}, test acc {test_accuracy * 100:.2f}%"
        )

    return model


def parse_args() -> argparse.Namespace:
    """命令行参数。

    默认参数偏“学习友好”：数据量较小、只跑 1 个 epoch。
    想认真训练时，可以把 --train-limit 和 --test-limit 设为 none。
    """

    parser = argparse.ArgumentParser(description="A small, readable ResNet MVP for CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=128, help="每个 batch 的图片数量。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam 学习率。")
    parser.add_argument("--train-limit", type=str, default="4096", help="训练样本数；写 none 使用完整训练集。")
    parser.add_argument("--test-limit", type=str, default="1024", help="测试样本数；写 none 使用完整测试集。")
    parser.add_argument("--verbose-shapes", action="store_true", help="打印第一批图片穿过网络时的 shape。")
    return parser.parse_args()


def parse_optional_limit(raw_value: str) -> int | None:
    """把命令行里的样本数量转换成 int 或 None。"""

    if raw_value.lower() == "none":
        return None
    limit: int = int(raw_value)
    if limit <= 0:
        raise ValueError("limit 必须是正整数，或者写 none。")
    return limit


def main() -> None:
    file_dir: Final[Path] = Path(__file__).resolve().parent
    root_dir: Final[Path] = file_dir.parent
    args = parse_args()
    config = TrainConfig(
        data_dir=root_dir / "data",
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_limit=parse_optional_limit(args.train_limit),
        test_limit=parse_optional_limit(args.test_limit),
        verbose_shapes=args.verbose_shapes,
    )
    train(config)


if __name__ == "__main__":
    main()
