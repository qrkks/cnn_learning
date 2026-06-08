# Dataset

## Kvasir-SEG

数据集下载地址：<https://datasets.simula.no/kvasir-seg/>

Kvasir-SEG 是结肠镜息肉图像分割数据集，包含 1000 张息肉图像及对应的像素级分割掩膜。每张图像都有一张人工标注 mask，适合用于医学图像病灶分割、辅助诊断和分割模型评价。

本项目不把数据集提交到 GitHub。Colab 训练时下载压缩包并解压到：

```text
4_medical_segmentation/data/Kvasir-SEG/
  images/
  masks/
```

如果官方压缩包解压后的目录名不同，只需要让 `configs/colab.yaml` 中的 `data_root` 指向包含 `images` 和 `masks` 的目录即可。

## Why This Dataset Fits The Assignment

- 任务类型是医学影像病灶分割，不是简单分类。
- 图像和 mask 是常规图片格式，适合课程项目快速实现。
- 数据量适中，Google Colab 免费 GPU 可以完成训练。
- 输出结果直观，便于在报告中展示原图、真实 mask、预测 mask 和叠加图。

