from __future__ import annotations

import sys
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

CifarSample = tuple[torch.Tensor, int]
CifarBatch = tuple[torch.Tensor, torch.Tensor]

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
#
# Colab 使用方式：
# 1. 在 Colab 菜单里选择 Runtime -> Change runtime type -> T4 GPU。
# 2. 把这个文件整段复制到 Colab 的一个 code cell。
# 3. 直接运行这个 cell。代码会检测到 CUDA，然后自动跑完整 CIFAR-10。
#
# 本地使用方式：
#        .venv\Scripts\python.exe 3_resnet\resnet_mvp.py
#
# 本地没有 GPU 时，代码会自动只跑小数据量，确认流程能跑通。
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

# ============================================================
# 1. 你最常改的参数都在这里
# ============================================================
# CPU 模式：给本地无显卡电脑快速验证用。
CPU_EPOCHS: Final[int] = 1
CPU_BATCH_SIZE: Final[int] = 128
CPU_LEARNING_RATE: Final[float] = 1e-3
CPU_TRAIN_LIMIT: Final[int] = 4096
CPU_TEST_LIMIT: Final[int] = 1024

# GPU 模式：给 Colab 正式训练用。
GPU_EPOCHS: Final[int] = 20
GPU_BATCH_SIZE: Final[int] = 128
GPU_LEARNING_RATE: Final[float] = 1e-3

# True 表示第一批图片会打印每一层 shape。学习阶段建议开着。
VERBOSE_SHAPES: Final[bool] = True


# 数据配置类
@dataclass(frozen=True)
class TrainConfig:
    """训练配置。

    dataclass 的好处是：训练参数集中放在一起，函数签名不会变得很长。
    frozen=True 表示创建后不再修改，避免训练中途悄悄改配置。
    """

    data_dir: Path
    batch_size: int 
    epochs: int 
    learning_rate: float 
    train_limit: int | None 
    test_limit: int | None 
    num_workers: int 
    pin_memory: bool = False
    checkpoint_path: Path | None = None
    save_best: bool = False
    verbose_shapes: bool = False


@dataclass
class TrainingHistory:
    """每个 epoch 的训练记录，方便在 notebook 里画曲线。"""

    train_losses: list[float]
    train_accuracies: list[float]
    test_losses: list[float]
    test_accuracies: list[float]


@dataclass
class TrainingResult:
    """训练完成后的返回值。

    脚本模式可以只看打印输出；notebook 模式通常还需要 model 和 history。
    """

    model: MiniResNet
    history: TrainingHistory
    best_test_accuracy: float
    best_checkpoint_path: Path | None


def is_running_in_notebook() -> bool:
    """判断当前代码是不是在 Jupyter / Colab notebook cell 里运行。

    为什么需要它？
    - 作为 .py 文件运行时，我们希望自动进入命令行 main()。
    - 直接粘贴到 Colab cell 时，也可以让 __main__ 自动进入训练。

    所以 notebook 里会只定义类和函数，然后由你显式调用 run_colab_training()。
    """

    return "ipykernel" in sys.modules


def default_data_dir() -> Path:
    """根据运行环境选择数据目录。

    - Colab: 使用 /content/data，下载速度快，也符合 Colab 的临时文件习惯。
    - 本地 .py: 使用项目根目录下的 data。
    - 直接粘贴到普通 notebook: 使用当前工作目录下的 data。
    """

    if Path("/content").exists():
        return Path("/content/data")

    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent / "data"

    return Path.cwd() / "data"


def default_checkpoint_path() -> Path:
    """根据运行环境选择 checkpoint 保存位置。"""

    if Path("/content").exists():
        return Path("/content/resnet_cifar10_best.pt")

    return Path.cwd() / "resnet_cifar10_best.pt"


def make_auto_config() -> TrainConfig:
    """自动根据 CPU/GPU 选择训练配置。

    这里按你的使用场景做一个简单约定：
    - 检测到 CUDA：认为是在 Colab 上跑，使用完整 CIFAR-10 和正式 epoch。
    - 没有 CUDA：认为是在本地 CPU 上跑，只取小数据做 smoke test。
    """

    if torch.cuda.is_available():
        return make_colab_config(
            data_dir=default_data_dir(),
            checkpoint_path=default_checkpoint_path(),
        )

    return make_local_config(data_dir=default_data_dir())


def make_local_config(
    data_dir: Path,
    epochs: int = CPU_EPOCHS,
    batch_size: int = CPU_BATCH_SIZE,
    learning_rate: float = CPU_LEARNING_RATE,
    verbose_shapes: bool = VERBOSE_SHAPES,
) -> TrainConfig:
    """本地 CPU 友好的配置。

    这个配置只取一小部分 CIFAR-10，用来确认代码能跑通。
    你的本地没有显卡时，默认就用它。
    """

    return TrainConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        train_limit=CPU_TRAIN_LIMIT,
        test_limit=CPU_TEST_LIMIT,
        num_workers=0,
        pin_memory=False,
        checkpoint_path=None,
        save_best=False,
        verbose_shapes=verbose_shapes,
    )


def make_colab_config(
    data_dir: Path = Path("/content/data"),
    epochs: int = GPU_EPOCHS,
    batch_size: int = GPU_BATCH_SIZE,
    learning_rate: float = GPU_LEARNING_RATE,
    checkpoint_path: Path = Path("/content/resnet_cifar10_best.pt"),
    verbose_shapes: bool = VERBOSE_SHAPES,
) -> TrainConfig:
    """Colab GPU 友好的配置。

    这个配置默认使用完整 CIFAR-10，并保存测试集准确率最高的 checkpoint。
    在 Colab 菜单里选 Runtime -> Change runtime type -> T4 GPU 后使用它。
    """

    return TrainConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        train_limit=None,
        test_limit=None,
        num_workers=2,
        pin_memory=True,
        checkpoint_path=checkpoint_path,
        save_best=True,
        verbose_shapes=verbose_shapes,
    )


# 3*3 卷积工厂
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


# 网络核心模块（ResNet 的基础残差块）
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
        self.stage1: nn.Sequential = self._make_stage(
            out_channels=16, blocks=2, stride=1
        )
        self.stage2: nn.Sequential = self._make_stage(
            out_channels=32, blocks=2, stride=2
        )
        self.stage3: nn.Sequential = self._make_stage(
            out_channels=64, blocks=2, stride=2
        )

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


def explain_forward_shapes(
    model: MiniResNet, images: torch.Tensor, verbose: bool
) -> None:
    """手动走一遍 forward，用来展示每个 stage 的 shape。

    这不是训练必须的函数，只是教学辅助。
    如果你想理解 ResNet，可以保持 VERBOSE_SHAPES=True 跑一次。
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

    # 构建【训练集】图像预处理 + 数据增强流水线
    train_transform: transforms.Compose = transforms.Compose(
        [
            # ----------------- VERBOSE 详细解释 -----------------
            # 1. 随机裁剪：把 32x32 图片先 padding 4 像素，再随机切回 32x32
            # 作用：让模型学不同位置的特征，防止过拟合
            transforms.RandomCrop(32, padding=4),

            # 2. 随机水平翻转：50%概率把图片左右翻转
            # 作用：数据增强，扩充样本多样性
            transforms.RandomHorizontalFlip(),

            # 3. 转张量：把 PIL 图片 → PyTorch 张量
            # 同时把像素值 0~255 → 归一化到 0~1
            transforms.ToTensor(),

            # 4. 标准化：用数据集的均值、方差做 Z-score 归一化
            # 作用：让模型训练更稳定、收敛更快
            transforms.Normalize(mean, std),
        ]
    )

    # 构建【测试集】图像预处理流水线（**没有数据增强**）
    test_transform: transforms.Compose = transforms.Compose(
        [
            # ----------------- VERBOSE 详细解释 -----------------
            # 1. 只转张量，不做任何随机增强
            # 测试集必须保持原图，不能随机变换
            transforms.ToTensor(),

            # 2. 和训练集用**完全一样**的归一化
            # 保证训练/测试数据分布一致
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform


def maybe_limit_dataset(
    dataset: Dataset[CifarSample], limit: int | None
) -> Dataset[CifarSample]:
    """为了 MVP 快速跑通，可以只取一小部分数据。

    limit=None 表示使用完整 CIFAR-10。
    默认只训练 4096 张、测试 1024 张，是为了让 CPU 上也能快速看到完整流程。
    """

    if limit is None:
        return dataset

    safe_limit: int = min(limit, len(cast(Sized, cast(object, dataset))))
    indices: range = range(safe_limit)
    return Subset(dataset, indices)


def build_dataloaders(
    config: TrainConfig,
) -> tuple[
    DataLoader[CifarBatch],
    DataLoader[CifarBatch],
]:
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

    # CIFAR10 returns one sample as (image_tensor, label_int).
    # DataLoader's default collate function stacks a batch into
    # (images_tensor, labels_tensor). Pyright cannot infer that conversion,
    # so we cast at this boundary and keep the training loop honestly typed.
    train_loader = cast(
        DataLoader[CifarBatch],
        DataLoader(
            limited_train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
    )

    test_loader = cast(
        DataLoader[CifarBatch],
        DataLoader(
            limited_test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        ),
    )
    return train_loader, test_loader


def choose_device() -> torch.device:
    """优先使用 CUDA；没有 GPU 时自动回到 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(
    logits: torch.Tensor,  # 模型输出的原始分数 (batch, 10)
    labels: torch.Tensor   # 真实标签 (batch,) 比如 [3,5,1,0...]
) -> float:               # 返回值：0~1 之间的准确率（小数）    
    """根据 logits 计算一个 batch 的准确率。"""
    predictions: torch.Tensor = torch.argmax(logits, dim=1) 
    correct: int = int((predictions == labels).sum().item())
    return correct / labels.numel()


def train_one_epoch(
    model: MiniResNet,
    train_loader: DataLoader[CifarBatch],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    verbose_shapes: bool,
    non_blocking: bool = False,
) -> tuple[float, float]:
    """训练一个 epoch，返回平均 loss 和平均 accuracy。"""

    model.train()
    total_loss: float = 0.0
    total_correct: int = 0
    total_examples: int = 0

    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

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
    test_loader: DataLoader[CifarBatch],
    criterion: nn.Module,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[float, float]:
    """在测试集上评估模型。

    @torch.no_grad() 表示这里不需要保存梯度，速度更快，也更省内存。
    """

    model.eval()
    total_loss: float = 0.0
    total_correct: int = 0
    total_examples: int = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        logits: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(logits, labels)

        batch_size: int = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_examples += batch_size

    average_loss: float = total_loss / total_examples
    average_accuracy: float = total_correct / total_examples
    return average_loss, average_accuracy


def save_checkpoint(
    model: MiniResNet,
    checkpoint_path: Path,
    epoch: int,
    test_accuracy: float,
    config: TrainConfig,
) -> None:
    """保存 checkpoint，方便 Colab 训练结束后下载或挂载到 Google Drive。"""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "test_accuracy": test_accuracy,
            "model_state_dict": model.state_dict(),
            "config": config,
            "classes": CIFAR10_CLASSES,
        },
        checkpoint_path,
    )


def plot_history(history: TrainingHistory) -> None:
    """画 loss / accuracy 曲线。

    这个函数主要给 Colab 用。matplotlib 放在函数内部 import，
    这样本地只跑快速训练时，不会因为画图环境问题影响主流程。
    """

    import matplotlib.pyplot as plt

    epochs = range(1, len(history.train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.train_losses, label="train loss")
    plt.plot(epochs, history.test_losses, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        [value * 100 for value in history.train_accuracies],
        label="train acc",
    )
    plt.plot(
        epochs,
        [value * 100 for value in history.test_accuracies],
        label="test acc",
    )
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train(config: TrainConfig) -> TrainingResult:
    """完整训练入口：准备数据、创建模型、训练、评估。"""

    device: torch.device = choose_device()
    print(f"device = {device}")
    print(f"data_dir = {config.data_dir}")

    train_loader, test_loader = build_dataloaders(config)
    model = MiniResNet(num_classes=len(CIFAR10_CLASSES)).to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=config.learning_rate)
    history = TrainingHistory(
        train_losses=[],
        train_accuracies=[],
        test_losses=[],
        test_accuracies=[],
    )
    best_test_accuracy: float = 0.0
    best_checkpoint_path: Path | None = None
    non_blocking: bool = config.pin_memory and device.type == "cuda"

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            verbose_shapes=config.verbose_shapes,
            non_blocking=non_blocking,
        )
        test_loss, test_accuracy = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            non_blocking=non_blocking,
        )

        history.train_losses.append(train_loss)
        history.train_accuracies.append(train_accuracy)
        history.test_losses.append(test_loss)
        history.test_accuracies.append(test_accuracy)

        if config.save_best and config.checkpoint_path is not None:
            if test_accuracy >= best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_checkpoint_path = config.checkpoint_path
                save_checkpoint(
                    model=model,
                    checkpoint_path=config.checkpoint_path,
                    epoch=epoch,
                    test_accuracy=test_accuracy,
                    config=config,
                )
                checkpoint_note: str = f" | saved {config.checkpoint_path}"
            else:
                checkpoint_note = ""
        else:
            best_test_accuracy = max(best_test_accuracy, test_accuracy)
            checkpoint_note = ""

        print(
            f"epoch {epoch:02d}/{config.epochs} | "
            f"train loss {train_loss:.4f}, train acc {train_accuracy * 100:.2f}% | "
            f"test loss {test_loss:.4f}, test acc {test_accuracy * 100:.2f}%"
            f"{checkpoint_note}"
        )

    return TrainingResult(
        model=model,
        history=history,
        best_test_accuracy=best_test_accuracy,
        best_checkpoint_path=best_checkpoint_path,
    )


def run_colab_training(
    epochs: int = GPU_EPOCHS,
    batch_size: int = GPU_BATCH_SIZE,
    learning_rate: float = GPU_LEARNING_RATE,
    data_dir: Path = Path("/content/data"),
    checkpoint_path: Path = Path("/content/resnet_cifar10_best.pt"),
    plot: bool = True,
) -> TrainingResult:
    """Colab 一行启动函数。

    你把这个文件复制到 Colab cell 并运行后，再执行：

        result = run_colab_training(epochs=20)

    它会：
    - 使用完整 CIFAR-10
    - 自动选择 CUDA
    - 保存测试集准确率最高的 checkpoint
    - 训练结束后画 loss / accuracy 曲线
    """

    config = make_colab_config(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path,
        verbose_shapes=True,
    )
    result = train(config)

    if plot:
        plot_history(result.history)

    print(f"best test accuracy = {result.best_test_accuracy * 100:.2f}%")
    print(f"best checkpoint = {result.best_checkpoint_path}")
    return result


def run_auto_training(plot: bool | None = None) -> TrainingResult:
    """自动训练入口。

    这是现在最推荐的入口：不需要命令行参数。

    - 本地 CPU：自动使用小数据量，快速确认代码能跑。
    - Colab GPU：自动使用完整 CIFAR-10，跑正式 epoch，并保存最佳 checkpoint。

    plot=None 表示：GPU/Colab 时画曲线，CPU 本地快速验证时不画。
    """

    config = make_auto_config()
    if torch.cuda.is_available():
        print("CUDA detected: using full CIFAR-10 training settings.")
    else:
        print("CUDA not detected: using small CPU smoke-test settings.")

    result = train(config)
    should_plot: bool = torch.cuda.is_available() if plot is None else plot
    if should_plot:
        plot_history(result.history)

    print(f"best test accuracy = {result.best_test_accuracy * 100:.2f}%")
    print(f"best checkpoint = {result.best_checkpoint_path}")
    return result


def run_shape_self_test() -> None:
    """内置最小测试。

    这个函数替代单独的 test 文件，让本章可以保持“一份文件即可复制到 Colab”。
    它不下载 CIFAR-10，只用随机 tensor 验证 ResNet 最重要的 shape 逻辑：
    - stride=1 且通道不变时，BasicBlock 保持 shape。
    - stride=2 或通道改变时，projection shortcut 对齐 shape。
    - MiniResNet 对每张图片输出 10 个 logits。
    """

    torch.manual_seed(0)

    same_shape_block = BasicBlock(in_channels=16, out_channels=16, stride=1)
    same_shape_input = torch.randn(2, 16, 32, 32)
    same_shape_output = same_shape_block(same_shape_input)
    assert tuple(same_shape_output.shape) == (2, 16, 32, 32)

    projected_block = BasicBlock(in_channels=16, out_channels=32, stride=2)
    projected_input = torch.randn(2, 16, 32, 32)
    projected_output = projected_block(projected_input)
    assert tuple(projected_output.shape) == (2, 32, 16, 16)

    model = MiniResNet(num_classes=10)
    images = torch.randn(4, 3, 32, 32)
    logits = model(images)
    assert tuple(logits.shape) == (4, 10)

    local_config = make_local_config(data_dir=Path("data"))
    assert local_config.train_limit == 4096
    assert local_config.test_limit == 1024

    colab_config = make_colab_config()
    assert colab_config.train_limit is None
    assert colab_config.test_limit is None
    assert colab_config.save_best

    print("shape self-test passed")


if __name__ == "__main__":
    result = run_auto_training()
