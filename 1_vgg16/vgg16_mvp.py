from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.vgg import VGG

# ============================================================
# 0. 这个文件在做什么？
# ============================================================
# 这是一个 VGG16 的 MVP 教学脚本。
#
# 你可以把整个文件复制到 Colab 的一个 code cell 里直接运行。
# 它不会读取命令行参数，因为在 Colab notebook 里命令行参数兼容性很差。
#
# 运行逻辑很简单：
# - 如果检测到 GPU，也就是 torch.cuda.is_available() == True：
#   使用完整 CIFAR-10 训练集和测试集，加载 ImageNet 预训练 VGG16。
# - 如果没有 GPU：
#   自动只取很小的数据量做 smoke test，确认数据流、模型 forward、训练循环都能跑通。
#
# 为什么 CPU 不跑完整训练？
# VGG16 参数量很大，CPU 上完整训练会非常慢。
# 对学习来说，CPU 模式先确认流程能跑通就够了；真正训练交给 Colab GPU。
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

IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

# 类型别名不是必须的，但很适合教学：
# - CIFAR10 单个样本：一张图片 tensor + 一个类别编号。
# - CIFAR10 一个 batch：多张图片 tensor + 多个类别编号 tensor。
# - CifarDataLoader：每次迭代都会吐出一个 CifarBatch。
#
# 小细节：
# PyTorch 类型标注会把 DataLoader[T] 的 T 当成 dataset 单样本类型。
# 但默认 collate_fn 在运行时会把多个样本合并成 batch：
# - 多个 image tensor -> 一个 image batch tensor
# - 多个 int label -> 一个 label batch tensor
# 所以后面创建 DataLoader 时会用 cast 告诉 Pylance：
# “这里迭代出来的是 batch，不是单个样本”。
CifarSample: TypeAlias = tuple[torch.Tensor, int]
CifarBatch: TypeAlias = tuple[torch.Tensor, torch.Tensor]
CifarDataset: TypeAlias = Dataset[CifarSample]
CifarDataLoader: TypeAlias = DataLoader[CifarBatch]


# ============================================================
# 1. 训练配置
# ============================================================
# dataclass 用来把训练参数集中放在一起。
# 好处是：
# - main() 里不会散落一堆魔法数字。
# - 函数之间传参更清楚。
# - 后面你想调 epoch、batch size、learning rate，很容易找到位置。
# ============================================================


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    checkpoint_path: Path
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    train_limit: int | None
    test_limit: int | None
    num_workers: int
    pin_memory: bool
    use_pretrained_weights: bool
    save_best_checkpoint: bool
    verbose_shapes: bool


@dataclass
class TrainingHistory:
    train_losses: list[float]
    train_accuracies: list[float]
    test_losses: list[float]
    test_accuracies: list[float]


@dataclass
class TrainingResult:
    model: VGG
    history: TrainingHistory
    best_test_accuracy: float


def default_data_dir() -> Path:
    """选择数据目录。

    Colab 里使用 /content/data，符合 Colab 的临时文件习惯。
    本地运行时，如果这个文件在项目目录里，就使用项目根目录下的 data。
    """

    if Path("/content").exists():
        return Path("/content/data")

    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent / "data"

    return Path.cwd() / "data"


def default_checkpoint_path() -> Path:
    """选择模型保存位置。

    GPU 完整训练时会保存测试集准确率最高的权重。
    CPU smoke test 默认不会保存，因为它只是流程检查。
    """

    if Path("/content").exists():
        return Path("/content/vgg16_cifar10_best.pt")

    return Path.cwd() / "vgg16_cifar10_best.pt"


def make_auto_config() -> TrainConfig:
    """根据当前机器是否有 CUDA，自动选择训练配置。

    这里没有任何命令行参数，复制到 Colab notebook 里最稳。
    """

    if torch.cuda.is_available():
        return TrainConfig(
            data_dir=default_data_dir(),
            checkpoint_path=default_checkpoint_path(),
            image_size=96,
            batch_size=32,
            epochs=3,
            learning_rate=1e-3,
            train_limit=None,
            test_limit=None,
            num_workers=2,
            pin_memory=True,
            use_pretrained_weights=True,
            save_best_checkpoint=True,
            verbose_shapes=True,
        )

    return TrainConfig(
        data_dir=default_data_dir(),
        checkpoint_path=default_checkpoint_path(),
        image_size=64,
        batch_size=8,
        epochs=1,
        learning_rate=1e-3,
        train_limit=32,
        test_limit=32,
        num_workers=0,
        pin_memory=False,
        use_pretrained_weights=False,
        save_best_checkpoint=False,
        verbose_shapes=True,
    )


def set_random_seed(seed: int = 42) -> None:
    """固定随机种子，让每次运行结果更接近。

    注意：深度学习在 GPU 上仍可能有一点非确定性。
    但固定 seed 后，数据打乱、随机增强、初始化会更稳定。
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. 数据预处理和 DataLoader
# ============================================================
# CIFAR-10 原图是 32x32。
# VGG16 最初是为 ImageNet 224x224 图片设计的。
#
# 这里做两件事：
# 1. Resize 到更适合 VGG16 的尺寸。
#    MVP 里用 96x96，比 224x224 快很多，也足够展示迁移学习流程。
# 2. 使用 ImageNet 的 mean/std 做 Normalize。
#    因为预训练 VGG16 在 ImageNet 归一化方式上学到特征，
#    输入分布尽量保持一致，迁移效果会更好。
# ============================================================


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    test_transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, test_transform


def maybe_limit_dataset(dataset: CifarDataset, limit: int | None) -> CifarDataset:
    """如果 limit 不是 None，就只取前 limit 张图片。

    GPU 模式下 limit=None，使用完整 CIFAR-10。
    CPU 模式下 limit 很小，用来快速确认代码能跑通。
    """

    if limit is None:
        return dataset

    return Subset(dataset, range(limit))


def build_dataloaders(config: TrainConfig) -> tuple[CifarDataLoader, CifarDataLoader]:
    train_transform: transforms.Compose
    test_transform: transforms.Compose
    train_transform, test_transform = build_transforms(config.image_size)

    train_dataset: CifarDataset = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset: CifarDataset = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_dataset = maybe_limit_dataset(train_dataset, config.train_limit)
    test_dataset = maybe_limit_dataset(test_dataset, config.test_limit)

    train_loader: CifarDataLoader = cast(
        CifarDataLoader,
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
    )

    test_loader: CifarDataLoader = cast(
        CifarDataLoader,
        DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
    )

    return train_loader, test_loader


# ============================================================
# 3. 创建 VGG16 迁移学习模型
# ============================================================
# VGG16 的结构可以粗略分成两部分：
#
# 1. features:
#    一堆 Conv2d + ReLU + MaxPool2d，用来提取图像特征。
# 2. classifier:
#    一堆 Linear 层，用来把特征变成类别分数。
#
# 迁移学习的常见 MVP 做法：
# - 加载 ImageNet 预训练权重。
# - 冻结大部分旧参数，让它们先不要训练。
# - 替换最后一层，让输出类别从 ImageNet 的 1000 类变成 CIFAR-10 的 10 类。
#
# 这样训练很快，因为真正从头学习的参数少很多。
# ============================================================


def build_vgg16_for_cifar10(config: TrainConfig, device: torch.device) -> VGG:
    if config.use_pretrained_weights:
        print("Loading ImageNet pretrained VGG16 weights...")
        weights: VGG16_Weights | None = VGG16_Weights.DEFAULT
    else:
        print("CPU smoke test: using random VGG16 weights to avoid a large download.")
        weights = None

    model: VGG = vgg16(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    last_layer: nn.Module = model.classifier[6]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError("Expected VGG16 classifier[6] to be nn.Linear.")

    model.classifier[6] = nn.Linear(last_layer.in_features, len(CIFAR10_CLASSES))

    return model.to(device)


def count_trainable_parameters(model: nn.Module) -> int:
    """统计需要训练的参数量。

    迁移学习里这个数字通常远小于模型总参数量。
    """

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def log_vgg16_shapes_once(
    model: VGG, images: torch.Tensor, device: torch.device
) -> None:
    """打印一小批图片在 VGG16 里的 shape 变化。

    shape 是理解 CNN 的关键：
    - batch 维度通常不变。
    - channel 维度会随着卷积层变化。
    - height/width 会随着池化层逐渐变小。
    """

    print("\nShape walkthrough for the first mini-batch:")

    model.eval()
    x: torch.Tensor = images[:2].to(device)
    print(f"{'input':<28} {tuple(x.shape)}")

    with torch.no_grad():
        features: nn.Sequential = cast(nn.Sequential, model.features)
        avgpool: nn.Module = model.avgpool
        classifier: nn.Sequential = cast(nn.Sequential, model.classifier)

        for index, layer in enumerate(features):
            x = layer(x)

            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                layer_name: str = layer.__class__.__name__
                print(f"features[{index:02d}] {layer_name:<13} {tuple(x.shape)}")

        x = avgpool(x)
        print(f"{'avgpool':<28} {tuple(x.shape)}")

        x = torch.flatten(x, start_dim=1)
        print(f"{'flatten':<28} {tuple(x.shape)}")

        for index, layer in enumerate(classifier):
            x = layer(x)

            if isinstance(layer, nn.Linear):
                print(f"classifier[{index}] Linear{'':<8} {tuple(x.shape)}")

    print()


# ============================================================
# 4. 训练和测试
# ============================================================
# 一个标准 PyTorch 训练循环通常有这些步骤：
#
# 1. model.train()
#    告诉模型现在是训练模式。
# 2. images, labels -> device
#    把数据放到 GPU 或 CPU。
# 3. outputs = model(images)
#    前向传播，得到每个类别的分数。
# 4. loss = criterion(outputs, labels)
#    计算预测和真实标签之间的差距。
# 5. optimizer.zero_grad()
#    清空上一轮的梯度。
# 6. loss.backward()
#    反向传播，计算当前 batch 的梯度。
# 7. optimizer.step()
#    根据梯度更新可训练参数。
# ============================================================


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
    predictions: torch.Tensor = logits.argmax(dim=1)
    return int((predictions == labels).sum().item())


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()

    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for batch_index, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        outputs: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size: int = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += accuracy_from_logits(outputs, labels)
        total += batch_size

        if batch_index == 1 or batch_index % 100 == 0:
            avg_loss: float = running_loss / total
            avg_accuracy: float = correct / total
            print(
                f"Epoch {epoch:02d} | batch {batch_index:04d}/{len(train_loader):04d} "
                f"| train loss {avg_loss:.4f} | train acc {avg_accuracy:.2%}"
            )

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    data_loader: CifarDataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs, labels)

            batch_size: int = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += accuracy_from_logits(outputs, labels)
            total += batch_size

    return running_loss / total, correct / total


def train_model(
    model: VGG,
    train_loader: CifarDataLoader,
    test_loader: CifarDataLoader,
    config: TrainConfig,
    device: torch.device,
) -> TrainingResult:
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    optimizer: optim.Adam = optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )

    history: TrainingHistory = TrainingHistory(
        train_losses=[],
        train_accuracies=[],
        test_losses=[],
        test_accuracies=[],
    )

    best_test_accuracy: float = 0.0

    for epoch in range(1, config.epochs + 1):
        train_loss: float
        train_accuracy: float
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        test_loss: float
        test_accuracy: float
        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        history.train_losses.append(train_loss)
        history.train_accuracies.append(train_accuracy)
        history.test_losses.append(test_loss)
        history.test_accuracies.append(test_accuracy)

        print(
            f"Epoch {epoch:02d} summary "
            f"| train loss {train_loss:.4f} | train acc {train_accuracy:.2%} "
            f"| test loss {test_loss:.4f} | test acc {test_accuracy:.2%}"
        )

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

            if config.save_best_checkpoint:
                torch.save(model.state_dict(), config.checkpoint_path)
                print(f"Saved best checkpoint to: {config.checkpoint_path}")

    return TrainingResult(
        model=model,
        history=history,
        best_test_accuracy=best_test_accuracy,
    )


# ============================================================
# 5. 可视化和单张预测
# ============================================================
# 训练完成后，只看数字有点抽象。
# 这里加两个小工具：
# - plot_history(): 画 loss 和 accuracy 曲线。
# - show_predictions(): 随机拿几张测试图，显示预测是否正确。
# ============================================================


def plot_history(history: TrainingHistory) -> None:
    epochs: range = range(1, len(history.train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.train_losses, label="train loss")
    plt.plot(epochs, history.test_losses, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.train_accuracies, label="train acc")
    plt.plot(epochs, history.test_accuracies, label="test acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    show_or_close_plot()


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """把 Normalize 后的图片还原到大致可显示的 [0, 1] 范围。"""

    mean: torch.Tensor = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std: torch.Tensor = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image.cpu() * std + mean
    return image.clamp(0, 1)


def is_running_in_notebook() -> bool:
    """判断当前是不是在 notebook 环境。

    Colab 和 Jupyter 里通常有 get_ipython。
    普通 .py 脚本里通常没有。
    """

    return "get_ipython" in globals()


def show_or_close_plot() -> None:
    """在 Colab 显示图，在普通脚本里避免 plt.show() 阻塞。

    你复制到 Colab 后会正常看到图。
    本地命令行运行时，图像窗口不会卡住整个脚本。
    """

    if is_running_in_notebook():
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(0.001)

    plt.close()


def show_predictions(
    model: VGG,
    test_loader: CifarDataLoader,
    device: torch.device,
    max_images: int = 8,
) -> None:
    model.eval()

    images: torch.Tensor
    labels: torch.Tensor
    images, labels = next(iter(test_loader))
    images_for_model: torch.Tensor = images.to(device)

    with torch.no_grad():
        outputs: torch.Tensor = model(images_for_model)
        predictions: torch.Tensor = outputs.argmax(dim=1).cpu()

    image_count: int = min(max_images, images.size(0))
    plt.figure(figsize=(12, 4))

    for index in range(image_count):
        plt.subplot(2, 4, index + 1)
        image: torch.Tensor = denormalize_image(images[index])
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")

        pred_name: str = CIFAR10_CLASSES[int(predictions[index])]
        true_name: str = CIFAR10_CLASSES[int(labels[index])]
        color: str = "green" if pred_name == true_name else "red"
        plt.title(f"pred: {pred_name}\ntrue: {true_name}", color=color)

    plt.tight_layout()
    show_or_close_plot()


# ============================================================
# 6. main: 把整条流程串起来
# ============================================================
# Colab 使用方式：
# 1. Runtime -> Change runtime type -> T4 GPU。
# 2. 把这个文件整段复制到一个 code cell。
# 3. 直接运行。
#
# 本地使用方式：
# .venv\Scripts\python.exe 1_vgg16\vgg16_mvp.py
#
# 不需要传任何命令行参数。
# ============================================================


def main() -> TrainingResult:
    set_random_seed(42)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config: TrainConfig = make_auto_config()

    print("=" * 72)
    print("VGG16 CIFAR-10 MVP")
    print("=" * 72)
    print(f"device: {device}")
    print(f"data_dir: {config.data_dir}")
    print(f"image_size: {config.image_size}")
    print(f"batch_size: {config.batch_size}")
    print(f"epochs: {config.epochs}")
    print(f"train_limit: {config.train_limit}")
    print(f"test_limit: {config.test_limit}")
    print(f"use_pretrained_weights: {config.use_pretrained_weights}")
    print("=" * 72)

    train_loader: CifarDataLoader
    test_loader: CifarDataLoader
    train_loader, test_loader = build_dataloaders(config)
    model: VGG = build_vgg16_for_cifar10(config, device)

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    if config.verbose_shapes:
        first_images: torch.Tensor
        first_images, _ = next(iter(train_loader))
        log_vgg16_shapes_once(model, first_images, device)

    result: TrainingResult = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )

    print(f"\nBest test accuracy: {result.best_test_accuracy:.2%}")

    plot_history(result.history)
    show_predictions(result.model, test_loader, device)

    return result


if __name__ == "__main__":
    result = main()
