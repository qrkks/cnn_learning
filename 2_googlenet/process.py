from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path


# ============================================================
# 0. 一些基础设置
# ============================================================

# 如果有 GPU 就用 GPU，没有就用 CPU
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

# 训练轮数先设小一点，方便你快速跑通。
num_epochs: int
if device.type == "cuda":
    num_epochs = 30  # Colab 免费 T4 GPU 上，30 个 epoch 大概 10 分钟，性价比不错
num_epochs = 1  # 对于这个简单的图像分类任务，10 到 50 个 epoch 是性价比最高的选择。

# batch size：每次喂给模型多少张图片
batch_size: int = 128
"""常用参考（直接照抄）
小模型 + CIFAR10：64 ~ 128（最好 128）
Colab 免费 T4 GPU：128 最稳
更大模型：32 / 64"""

# 学习率
learning_rate: float = 0.001  # Adam 优化器万能默认值，基本不会出错。

# 是否打印每一层 shape
verbose: bool = True

# 设定数据目录
root_dir: Final[Path] = Path(__file__).parent.parent
print("根目录:", root_dir)
data_dir: Final[Path] = root_dir / "data"
print("数据目录:", data_dir)


# ============================================================
# 1. 数据预处理
# ============================================================
# CIFAR-10 的图片大小是 32x32，RGB 三通道
#
# ToTensor():
#   把图片从 PIL / numpy 转成 PyTorch tensor
#   同时把像素从 [0,255] 缩放到 [0,1]
#
# Normalize():
#   做标准化，让训练更稳定
#   CIFAR-10 常用均值/方差可以先简单写成 0.5 / 0.5
#   标准化后大致落到 [-1,1]
# ============================================================

# 打包 transform：图片 → 张量 → 居中归一化
transform_train: transforms.Compose = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # 所有像素从 [0, 1] → 变成 [-1, 1], 神经网络最喜欢输入在 0 附近对称的数据！
    ]
)

transform_test: transforms.Compose = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


# ============================================================
# 2. 下载 / 读取 CIFAR-10 数据集
# ============================================================
# train=True  -> 训练集
# train=False -> 测试集
# ============================================================

train_dataset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=transform_train
)

test_dataset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=transform_test
)

train_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]] = (
    torch.utils.data.DataLoader(  # DataLoader 是 PyTorch 提供的一个工具，负责把 dataset 打包成一个个 batch，方便训练循环使用
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 使用训练集，一次打包128张图片，打乱顺序
    )
)

test_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]] = (
    torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
)

# CIFAR-10 的 10 个类别，将数字标签映射成文字标签，方便看结果
classes: Final[tuple[str, ...]] = (
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
# 3. 一个打印 shape 的小工具
# ============================================================
# 方便你看张量在网络里怎么流动
# ============================================================


def log_shape(name: str, x: torch.Tensor, verbose: bool = True) -> None:
    if verbose:
        print(f"{name:<25} shape = {tuple(x.shape)}")


# ============================================================
# 4. 定义模型参数
# ============================================================
# 你要求“不用类”，所以这里不写 class GoogLeNet(nn.Module)
#
# 但训练时，PyTorch 还是需要知道有哪些可训练参数。
# 所以我们用 ModuleDict / ModuleList 来收集所有层。
#
# 你可以把它理解成：
# “虽然没写 class，但还是要把所有层装起来，方便 optimizer 管理参数”
# ============================================================

layers = nn.ModuleDict(  # 用 ModuleDict 把所有层装起来，方便管理参数。nn.ModuleDict = 给神经网络层起名字 + 装起来的 “带名字工具箱”。供 forward() 使用，也供 optimizer 管理参数。只是存放顺序，forward 函数里的调用顺序，才是真正的执行顺序。
    {
        # --------------------------------------------------------
        # Stem：网络最前面几层
        #
        # 注意：
        # CIFAR-10 图片只有 32x32，很小
        # 所以不能像 ImageNet 那样一上来就 7x7 stride=2
        # 不然尺寸会掉太快
        #
        # 更适合 CIFAR-10 的做法：
        # - 用 3x3 卷积
        # - stride=1
        # - 温和地下采样
        # --------------------------------------------------------
        "conv1": nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        "relu1": nn.ReLU(
            inplace=True
        ),  # ReLU 激活函数，inplace=True 表示直接在输入上修改，节省内存。Rectified Linear Unit，线性整流函数，常用的激活函数之一，能引入非线性，使模型能够学习复杂的函数关系。
        # 卷积在“找模式”，ReLU在“筛选有用模式”，多层叠加就变成“从简单到复杂识别”。
        # 卷积是超级神经元；ReLU 是神经递质化学传导，既有开关，又有浓度信息。
        "conv2": nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        "relu2": nn.ReLU(inplace=True),
        "maxpool1": nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
        # --------------------------------------------------------
        # Inception 3a
        # 输入通道: 128
        #
        # branch1: 1x1 -> 输出 32
        # branch2: 1x1(降维到32) -> 3x3 -> 输出 64
        # branch3: 1x1(降维到16) -> 5x5 -> 输出 16
        # branch4: pool -> 1x1 -> 输出 16
        #
        # concat 后总通道:
        # 32 + 64 + 16 + 16 = 128
        # --------------------------------------------------------
        "inc3a_b1_conv1": nn.Conv2d(128, 32, kernel_size=1),
        "inc3a_b1_relu1": nn.ReLU(inplace=True),
        "inc3a_b2_conv1": nn.Conv2d(128, 32, kernel_size=1),
        "inc3a_b2_relu1": nn.ReLU(inplace=True),
        "inc3a_b2_conv2": nn.Conv2d(32, 64, kernel_size=3, padding=1),
        "inc3a_b2_relu2": nn.ReLU(inplace=True),
        "inc3a_b3_conv1": nn.Conv2d(128, 16, kernel_size=1),
        "inc3a_b3_relu1": nn.ReLU(inplace=True),
        "inc3a_b3_conv2": nn.Conv2d(16, 16, kernel_size=5, padding=2),
        "inc3a_b3_relu2": nn.ReLU(inplace=True),
        "inc3a_b4_pool": nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        "inc3a_b4_conv1": nn.Conv2d(128, 16, kernel_size=1),
        "inc3a_b4_relu1": nn.ReLU(inplace=True),
        # --------------------------------------------------------
        # 再下一次池化
        # 16x16 -> 8x8
        # --------------------------------------------------------
        "maxpool2": nn.MaxPool2d(kernel_size=2, stride=2),
        # --------------------------------------------------------
        # Inception 3b
        # 输入通道: 128
        #
        # 这里稍微把通道做大一点
        # concat 后:
        # 64 + 96 + 32 + 32 = 224
        # --------------------------------------------------------
        "inc3b_b1_conv1": nn.Conv2d(128, 64, kernel_size=1),
        "inc3b_b1_relu1": nn.ReLU(inplace=True),
        "inc3b_b2_conv1": nn.Conv2d(128, 48, kernel_size=1),
        "inc3b_b2_relu1": nn.ReLU(inplace=True),
        "inc3b_b2_conv2": nn.Conv2d(48, 96, kernel_size=3, padding=1),
        "inc3b_b2_relu2": nn.ReLU(inplace=True),
        "inc3b_b3_conv1": nn.Conv2d(128, 16, kernel_size=1),
        "inc3b_b3_relu1": nn.ReLU(inplace=True),
        "inc3b_b3_conv2": nn.Conv2d(16, 32, kernel_size=5, padding=2),
        "inc3b_b3_relu2": nn.ReLU(inplace=True),
        "inc3b_b4_pool": nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        "inc3b_b4_conv1": nn.Conv2d(128, 32, kernel_size=1),
        "inc3b_b4_relu1": nn.ReLU(inplace=True),
        # --------------------------------------------------------
        # 分类头
        # --------------------------------------------------------
        "avgpool": nn.AdaptiveAvgPool2d((1, 1)),
        "dropout": nn.Dropout(0.3),
        "fc": nn.Linear(224, 10),
    }
)

# 把所有层搬到 device
layers = layers.to(device)  # 学习地点
layers: nn.ModuleDict


# ============================================================
# 5. 定义 forward 函数
# ============================================================
# 因为我们没写 class，所以手工写一个 forward(images)
#
# 注意：这就是“前向传播”全过程
# 输入一批图片，输出每张图片属于 10 类的 logits
#
# logits:
#   就是最后全连接层的原始输出
#   还没经过 softmax
#
# CrossEntropyLoss 内部会自动做 softmax 相关计算，
# 所以这里不要自己再 softmax
# ============================================================


def forward(x: torch.Tensor, verbose: bool = False) -> torch.Tensor:  # 干活步骤说明书
    # --------------------------------------------------------
    # 输入
    # CIFAR-10: (batch, 3, 32, 32)
    # --------------------------------------------------------
    log_shape("input", x, verbose)

    # --------------------------------------------------------
    # Stem
    # --------------------------------------------------------
    x = layers["conv1"](x)
    x = layers["relu1"](x)
    log_shape("conv1 + relu1", x, verbose)  # (B,64,32,32)

    x = layers["conv2"](x)
    x = layers["relu2"](x)
    log_shape("conv2 + relu2", x, verbose)  # (B,128,32,32)

    x = layers["maxpool1"](x)
    log_shape("maxpool1", x, verbose)  # (B,128,16,16)

    # ========================================================
    # Inception 3a
    # ========================================================
    inception_input = x
    log_shape("inception3a input", inception_input, verbose)

    # -------------------------
    # branch1: 1x1
    # -------------------------
    b1 = layers["inc3a_b1_conv1"](inception_input)
    b1 = layers["inc3a_b1_relu1"](b1)
    log_shape("inc3a branch1", b1, verbose)  # (B,32,16,16)

    # -------------------------
    # branch2: 1x1 -> 3x3
    # 先降维，再卷积
    # -------------------------
    b2 = layers["inc3a_b2_conv1"](inception_input)
    b2 = layers["inc3a_b2_relu1"](b2)
    b2 = layers["inc3a_b2_conv2"](b2)
    b2 = layers["inc3a_b2_relu2"](b2)
    log_shape("inc3a branch2", b2, verbose)  # (B,64,16,16)

    # -------------------------
    # branch3: 1x1 -> 5x5
    # 感受野更大，但参数更贵
    # 所以前面先用 1x1 压通道
    # -------------------------
    b3 = layers["inc3a_b3_conv1"](inception_input)
    b3 = layers["inc3a_b3_relu1"](b3)
    b3 = layers["inc3a_b3_conv2"](b3)
    b3 = layers["inc3a_b3_relu2"](b3)
    log_shape("inc3a branch3", b3, verbose)  # (B,16,16,16)

    # -------------------------
    # branch4: pool -> 1x1
    # 池化之后再做通道投影
    # -------------------------
    b4 = layers["inc3a_b4_pool"](inception_input)
    b4 = layers["inc3a_b4_conv1"](b4)
    b4 = layers["inc3a_b4_relu1"](b4)
    log_shape("inc3a branch4", b4, verbose)  # (B,16,16,16)

    # -------------------------
    # 把四个分支沿着 channel 维拼接
    #
    # dim=1 表示在“通道维”拼接
    # 为什么不是相加？
    # 因为 GoogLeNet 的想法不是把信息混成一份，
    # 而是把不同尺度提取出来的特征并排放在一起
    # -------------------------
    x = torch.cat([b1, b2, b3, b4], dim=1)
    log_shape("inception3a output", x, verbose)  # (B,128,16,16)

    # --------------------------------------------------------
    # 池化
    # --------------------------------------------------------
    x = layers["maxpool2"](x)
    log_shape("maxpool2", x, verbose)  # (B,128,8,8)

    # ========================================================
    # Inception 3b
    # ========================================================
    inception_input = x
    log_shape("inception3b input", inception_input, verbose)

    # branch1: 1x1
    b1 = layers["inc3b_b1_conv1"](inception_input)
    b1 = layers["inc3b_b1_relu1"](b1)
    log_shape("inc3b branch1", b1, verbose)  # (B,64,8,8)

    # branch2: 1x1 -> 3x3
    b2 = layers["inc3b_b2_conv1"](inception_input)
    b2 = layers["inc3b_b2_relu1"](b2)
    b2 = layers["inc3b_b2_conv2"](b2)
    b2 = layers["inc3b_b2_relu2"](b2)
    log_shape("inc3b branch2", b2, verbose)  # (B,96,8,8)

    # branch3: 1x1 -> 5x5
    b3 = layers["inc3b_b3_conv1"](inception_input)
    b3 = layers["inc3b_b3_relu1"](b3)
    b3 = layers["inc3b_b3_conv2"](b3)
    b3 = layers["inc3b_b3_relu2"](b3)
    log_shape("inc3b branch3", b3, verbose)  # (B,32,8,8)

    # branch4: pool -> 1x1
    b4 = layers["inc3b_b4_pool"](inception_input)
    b4 = layers["inc3b_b4_conv1"](b4)
    b4 = layers["inc3b_b4_relu1"](b4)
    log_shape("inc3b branch4", b4, verbose)  # (B,32,8,8)

    # concat
    x = torch.cat([b1, b2, b3, b4], dim=1)
    log_shape("inception3b output", x, verbose)  # (B,224,8,8)

    # --------------------------------------------------------
    # Global Average Pooling
    #
    # 作用：
    # 把每个通道的 8x8 特征图压成 1x1
    #
    # 输入:  (B,224,8,8)
    # 输出:  (B,224,1,1)
    #
    # 这是 GoogLeNet 很经典的设计之一：
    # 用全局平均池化代替巨大的全连接层
    # --------------------------------------------------------
    x = layers["avgpool"](x)
    log_shape("global avgpool", x, verbose)

    # --------------------------------------------------------
    # 展平
    # (B,224,1,1) -> (B,224)
    # --------------------------------------------------------
    x = torch.flatten(x, 1)
    log_shape("flatten", x, verbose)

    # --------------------------------------------------------
    # Dropout：防止过拟合
    # --------------------------------------------------------
    x = layers["dropout"](x)
    log_shape("dropout", x, verbose)

    # --------------------------------------------------------
    # 最终分类层
    # 输出 10 个数，对应 CIFAR-10 的 10 个类别
    # --------------------------------------------------------
    x = layers["fc"](x)
    log_shape("fc logits", x, verbose)

    return x


# ============================================================
# 6. 定义损失函数和优化器
# ============================================================
# CrossEntropyLoss:
#   多分类任务最常用
#
# Adam:
#   入门时通常很好用，收敛比纯 SGD 更省心
# ============================================================

criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.Adam = optim.Adam(layers.parameters(), lr=learning_rate)


# ============================================================
# 7. 先拿一个 batch 试跑，看看 shape 对不对
# ============================================================
# 这里只打印一次，防止训练时刷屏太厉害
# ============================================================

sample_images, sample_labels = next(iter(train_loader))
sample_images = sample_images.to(device)
sample_labels = sample_labels.to(device)

print("\n================ SHAPE CHECK ================\n")
with torch.no_grad():
    sample_outputs = forward(sample_images[:4], verbose=verbose)  # 只看前4张
print("\n=============================================\n")


# ============================================================
# 8. 训练循环
# ============================================================
# 一个 epoch = 把整个训练集完整看一遍
#
# 每一步训练做的事：
# 1. 把图片和标签拿出来
# 2. 前向传播，算预测
# 3. 算 loss
# 4. 梯度清零
# 5. 反向传播
# 6. optimizer 更新参数
# ============================================================

for epoch in range(num_epochs):
    layers.train()  # 开启训练模式（dropout 生效）
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = forward(images, verbose=False)

        # 计算损失
        loss = criterion(outputs, labels)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播：计算每个参数的梯度
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

        # 每 100 个 batch 打印一次
        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{batch_idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}")


# ============================================================
# 9. 测试集评估
# ============================================================
# 不需要梯度，所以用 torch.no_grad()
# ============================================================

layers.eval()  # 切换到评估模式（dropout 关闭）

correct: int = 0
total: int = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = forward(images, verbose=False)

        # 取每一行中最大的那个类别下标
        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy: float = 100.0 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")
