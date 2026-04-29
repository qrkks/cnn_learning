# ResNet MVP

这个目录是一个最小但完整的 ResNet 学习版本。重点不是追求最高准确率，而是让你看懂 ResNet 到底怎么用、为什么能训练更深的网络。

## 你要先抓住的一句话

普通网络层想直接学习：

```text
H(x)
```

ResNet 让网络学习残差：

```text
F(x) = H(x) - x
```

所以残差块的输出是：

```text
H(x) = F(x) + x
```

代码里对应这一句：

```python
out = out + identity
```

其中：

- `out` 是卷积主分支学到的 `F(x)`
- `identity` 是 shortcut 分支传过来的 `x`
- 相加后再过一次 `ReLU`

## 文件说明

- `resnet_mvp.py`: 从零实现的 Mini ResNet，可以训练 CIFAR-10。
- `test_resnet_mvp.py`: 验证残差块和模型输出 shape 的最小测试。

## 快速运行

在项目根目录执行：

```powershell
.venv\Scripts\python.exe 3_resnet\resnet_mvp.py --verbose-shapes
```

默认只训练 4096 张训练图、测试 1024 张测试图，并且只跑 1 个 epoch。这样 CPU 上也能较快看完整流程。

如果你想多训练几轮：

```powershell
.venv\Scripts\python.exe 3_resnet\resnet_mvp.py --epochs 5 --verbose-shapes
```

如果你想使用完整 CIFAR-10：

```powershell
.venv\Scripts\python.exe 3_resnet\resnet_mvp.py --epochs 10 --train-limit none --test-limit none
```

## 建议阅读顺序

1. 看 `BasicBlock.__init__`
   重点理解什么时候 shortcut 是 `nn.Identity()`，什么时候需要 `1x1 Conv`。

2. 看 `BasicBlock.forward`
   重点看 `identity = self.shortcut(x)` 和 `out = out + identity`。

3. 看 `MiniResNet.__init__`
   理解 CIFAR-10 为什么使用 `3x3 stride=1` 的开头，而不是 ImageNet ResNet 的 `7x7 stride=2`。

4. 运行 `--verbose-shapes`
   看图片 tensor 如何从 `3 x 32 x 32` 逐步变成 `64 x 8 x 8`，最后变成 10 个分类 logits。

## 运行测试

```powershell
.venv\Scripts\python.exe -m unittest 3_resnet\test_resnet_mvp.py -v
```

测试覆盖三个最重要的学习点：

- stride=1 且通道不变时，BasicBlock 保持 shape。
- stride=2 或通道改变时，BasicBlock 用 projection shortcut 对齐 shape。
- MiniResNet 对每张图片输出 10 个 logits。

## 怎么继续学习

你可以按这个顺序改代码：

1. 把 `MiniResNet` 每个 stage 的 `blocks=2` 改成 `blocks=3`，观察训练速度和准确率。
2. 把 `train_limit` 改成 `none`，看看完整 CIFAR-10 上的表现。
3. 在 `BasicBlock.forward` 里临时删掉 `out = out + identity`，比较没有残差连接时训练是否变差。
4. 把 `MiniResNet` 的通道数从 `16, 32, 64` 改成 `32, 64, 128`，观察参数量和速度变化。
