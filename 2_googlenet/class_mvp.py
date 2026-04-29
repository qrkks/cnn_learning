import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 定义Inception模块
class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()

        # 1x1卷积
        self.conv1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        # 1x1卷积后接3x3卷积
        self.conv1x1_3x3 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.conv3x3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # 1x1卷积后接5x5卷积
        self.conv1x1_5x5 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.conv5x5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        # 3x3最大池化后接1x1卷积
        self.pool3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_channels, 32, kernel_size=1)

    def forward(self, x):
        # 各种并行操作
        conv1x1_out = self.conv1x1(x)
        conv1x1_3x3_out = self.conv3x3(F.relu(self.conv1x1_3x3(x)))
        conv1x1_5x5_out = self.conv5x5(F.relu(self.conv1x1_5x5(x)))
        pool_out = self.conv_pool(self.pool3x3(x))

        # 合并各部分输出
        return torch.cat([conv1x1_out, conv1x1_3x3_out, conv1x1_5x5_out, pool_out], 1)


# 定义GoogleNet网络结构
class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 堆叠Inception模块
        self.inception1 = InceptionModule(192)
        self.inception2 = InceptionModule(256)
        self.inception3 = InceptionModule(320)
        self.inception4 = InceptionModule(384)

        # 最后几个层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# CIFAR-10 数据集
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 创建GoogleNet模型
model = GoogleNet(num_classes=10)

# 使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%"
    )

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
