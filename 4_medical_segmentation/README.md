# Kvasir-SEG Polyp Segmentation

本目录实现课程作业：**基于 Attention U-Net 的结肠镜息肉图像自动分割研究**。

设计目标是本地只做 smoke test，正式训练放到 Google Colab GPU。代码、文档和配置都放在仓库中，Kvasir-SEG 数据集通过官方链接下载，不提交到 GitHub。

## 目录结构

```text
4_medical_segmentation/
  configs/
    smoke.yaml
    colab.yaml
  src/
    train.py
    evaluate.py
    segmentation/
      datasets.py
      losses.py
      metrics.py
      visualize.py
      models/attention_unet.py
  tests/test_core.py
  notebooks/kvasir_attention_unet_colab.ipynb
  dataset.md
  report/report.md
```

## 本地 smoke test

本地不需要真实数据集，也不需要显卡。下面命令会生成少量玩具样本，只检查训练链路能否跑通。

```powershell
cd C:\Coding\Curriculum\cnn_learning
.\.venv\Scripts\python.exe -m pytest 4_medical_segmentation/tests -q
.\.venv\Scripts\python.exe 4_medical_segmentation/src/train.py --config 4_medical_segmentation/configs/smoke.yaml --make-sample-data
.\.venv\Scripts\python.exe 4_medical_segmentation/src/evaluate.py --config 4_medical_segmentation/configs/smoke.yaml --checkpoint 4_medical_segmentation/outputs/smoke/best_model.pth --num-visuals 2
```

## Colab 正式训练

把当前仓库推到 GitHub 后，在 Colab 中运行：

```python
!git clone https://github.com/qrkks/cnn_learning.git
%cd cnn_learning/4_medical_segmentation
!pip install -r requirements.txt
```

下载并解压 Kvasir-SEG：

```python
!mkdir -p data
!wget -O data/kvasir-seg.zip https://datasets.simula.no/downloads/kvasir-seg.zip
!unzip -q data/kvasir-seg.zip -d data
!ls data/Kvasir-SEG
```

训练和评估：

```python
!python src/train.py --config configs/colab.yaml
!python src/evaluate.py --config configs/colab.yaml --checkpoint outputs/colab/best_model.pth --num-visuals 8
```

主要输出：

- `outputs/colab/best_model.pth`
- `outputs/colab/history.csv`
- `outputs/colab/metrics_val.json`
- `outputs/colab/predictions/*.png`

## 方法概述

模型采用 Attention U-Net。编码器提取多尺度特征，解码器逐级恢复空间分辨率，跳跃连接前加入注意力门控，用于突出息肉区域并抑制背景干扰。损失函数采用 `BCEWithLogitsLoss + Dice Loss`，评价指标包括 Dice、IoU、Precision 和 Recall。
