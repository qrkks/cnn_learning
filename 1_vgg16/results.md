# Results

```bash
========================================================================
VGG16 CIFAR-10 MVP
========================================================================
device: cuda
data_dir: /content/data
image_size: 96
batch_size: 32
epochs: 30
train_limit: None
test_limit: None
use_pretrained_weights: True
========================================================================
Loading ImageNet pretrained VGG16 weights...
Trainable parameters: 40,970

Shape walkthrough for the first mini-batch:
input                        (2, 3, 96, 96)
features[00] Conv2d        (2, 64, 96, 96)
features[02] Conv2d        (2, 64, 96, 96)
features[04] MaxPool2d     (2, 64, 48, 48)
features[05] Conv2d        (2, 128, 48, 48)
features[07] Conv2d        (2, 128, 48, 48)
features[09] MaxPool2d     (2, 128, 24, 24)
features[10] Conv2d        (2, 256, 24, 24)
features[12] Conv2d        (2, 256, 24, 24)
features[14] Conv2d        (2, 256, 24, 24)
features[16] MaxPool2d     (2, 256, 12, 12)
features[17] Conv2d        (2, 512, 12, 12)
features[19] Conv2d        (2, 512, 12, 12)
features[21] Conv2d        (2, 512, 12, 12)
features[23] MaxPool2d     (2, 512, 6, 6)
features[24] Conv2d        (2, 512, 6, 6)
features[26] Conv2d        (2, 512, 6, 6)
features[28] Conv2d        (2, 512, 6, 6)
features[30] MaxPool2d     (2, 512, 3, 3)
avgpool                      (2, 512, 7, 7)
flatten                      (2, 25088)
classifier[0] Linear         (2, 4096)
classifier[3] Linear         (2, 4096)
classifier[6] Linear         (2, 10)

Epoch 01 | batch 0001/1563 | train loss 2.4181 | train acc 9.38%
Epoch 01 | batch 0100/1563 | train loss 0.9675 | train acc 66.66%
Epoch 01 | batch 0200/1563 | train loss 0.8875 | train acc 70.00%
Epoch 01 | batch 0300/1563 | train loss 0.8646 | train acc 71.14%
Epoch 01 | batch 0400/1563 | train loss 0.8508 | train acc 71.94%
Epoch 01 | batch 0500/1563 | train loss 0.8397 | train acc 72.65%
Epoch 01 | batch 0600/1563 | train loss 0.8253 | train acc 73.21%
Epoch 01 | batch 0700/1563 | train loss 0.8254 | train acc 73.58%
Epoch 01 | batch 0800/1563 | train loss 0.8273 | train acc 73.81%
Epoch 01 | batch 0900/1563 | train loss 0.8290 | train acc 73.80%
Epoch 01 | batch 1000/1563 | train loss 0.8301 | train acc 73.96%
Epoch 01 | batch 1100/1563 | train loss 0.8322 | train acc 74.08%
Epoch 01 | batch 1200/1563 | train loss 0.8334 | train acc 74.24%
Epoch 01 | batch 1300/1563 | train loss 0.8368 | train acc 74.32%
Epoch 01 | batch 1400/1563 | train loss 0.8406 | train acc 74.30%
Epoch 01 | batch 1500/1563 | train loss 0.8422 | train acc 74.35%
Epoch 01 summary | train loss 0.8440 | train acc 74.34% | test loss 0.6217 | test acc 79.86%
Saved best checkpoint to: /content/vgg16_cifar10_best.pt
Epoch 02 | batch 0001/1563 | train loss 0.6116 | train acc 78.12%
Epoch 02 | batch 0100/1563 | train loss 0.8639 | train acc 75.25%
Epoch 02 | batch 0200/1563 | train loss 0.8543 | train acc 75.89%
Epoch 02 | batch 0300/1563 | train loss 0.8583 | train acc 75.57%
Epoch 02 | batch 0400/1563 | train loss 0.8559 | train acc 75.84%
Epoch 02 | batch 0500/1563 | train loss 0.8534 | train acc 75.91%
Epoch 02 | batch 0600/1563 | train loss 0.8611 | train acc 75.84%
Epoch 02 | batch 0700/1563 | train loss 0.8646 | train acc 75.89%
Epoch 02 | batch 0800/1563 | train loss 0.8711 | train acc 75.90%
Epoch 02 | batch 0900/1563 | train loss 0.8706 | train acc 75.90%
Epoch 02 | batch 1000/1563 | train loss 0.8784 | train acc 75.84%
Epoch 02 | batch 1100/1563 | train loss 0.8806 | train acc 75.78%
Epoch 02 | batch 1200/1563 | train loss 0.8849 | train acc 75.74%
Epoch 02 | batch 1300/1563 | train loss 0.8847 | train acc 75.75%
Epoch 02 | batch 1400/1563 | train loss 0.8863 | train acc 75.73%
Epoch 02 | batch 1500/1563 | train loss 0.8902 | train acc 75.66%
Epoch 02 summary | train loss 0.8903 | train acc 75.70% | test loss 0.6842 | test acc 79.40%
Epoch 03 | batch 0001/1563 | train loss 0.8854 | train acc 68.75%
Epoch 03 | batch 0100/1563 | train loss 0.8460 | train acc 77.69%
Epoch 03 | batch 0200/1563 | train loss 0.8429 | train acc 77.11%
Epoch 03 | batch 0300/1563 | train loss 0.8608 | train acc 77.00%
Epoch 03 | batch 0400/1563 | train loss 0.8837 | train acc 76.53%
Epoch 03 | batch 0500/1563 | train loss 0.8931 | train acc 76.35%
Epoch 03 | batch 0600/1563 | train loss 0.8939 | train acc 76.12%
Epoch 03 | batch 0700/1563 | train loss 0.9037 | train acc 76.01%
Epoch 03 | batch 0800/1563 | train loss 0.9122 | train acc 75.77%
Epoch 03 | batch 0900/1563 | train loss 0.9064 | train acc 75.89%
Epoch 03 | batch 1000/1563 | train loss 0.9076 | train acc 75.94%
Epoch 03 | batch 1100/1563 | train loss 0.9085 | train acc 75.97%
Epoch 03 | batch 1200/1563 | train loss 0.9084 | train acc 76.01%
Epoch 03 | batch 1300/1563 | train loss 0.9094 | train acc 75.97%
Epoch 03 | batch 1400/1563 | train loss 0.9137 | train acc 75.95%
Epoch 03 | batch 1500/1563 | train loss 0.9200 | train acc 75.90%
Epoch 03 summary | train loss 0.9208 | train acc 75.90% | test loss 0.6588 | test acc 80.80%
Saved best checkpoint to: /content/vgg16_cifar10_best.pt
Epoch 04 | batch 0001/1563 | train loss 0.6724 | train acc 81.25%
Epoch 04 | batch 0100/1563 | train loss 0.8430 | train acc 76.66%
Epoch 04 | batch 0200/1563 | train loss 0.9137 | train acc 75.72%
Epoch 04 | batch 0300/1563 | train loss 0.9022 | train acc 75.78%
Epoch 04 | batch 0400/1563 | train loss 0.9161 | train acc 75.79%
Epoch 04 | batch 0500/1563 | train loss 0.9248 | train acc 75.59%
Epoch 04 | batch 0600/1563 | train loss 0.9246 | train acc 75.66%
Epoch 04 | batch 0700/1563 | train loss 0.9321 | train acc 75.62%
Epoch 04 | batch 0800/1563 | train loss 0.9243 | train acc 75.92%
Epoch 04 | batch 0900/1563 | train loss 0.9223 | train acc 76.04%
Epoch 04 | batch 1000/1563 | train loss 0.9288 | train acc 76.01%
Epoch 04 | batch 1100/1563 | train loss 0.9301 | train acc 76.02%
Epoch 04 | batch 1200/1563 | train loss 0.9333 | train acc 75.99%
Epoch 04 | batch 1300/1563 | train loss 0.9320 | train acc 76.11%
Epoch 04 | batch 1400/1563 | train loss 0.9338 | train acc 76.07%
Epoch 04 | batch 1500/1563 | train loss 0.9367 | train acc 76.03%
Epoch 04 summary | train loss 0.9370 | train acc 76.05% | test loss 0.7267 | test acc 79.38%
Epoch 05 | batch 0001/1563 | train loss 1.2093 | train acc 81.25%
Epoch 05 | batch 0100/1563 | train loss 0.9181 | train acc 75.94%
Epoch 05 | batch 0200/1563 | train loss 0.9296 | train acc 76.22%
Epoch 05 | batch 0300/1563 | train loss 0.9129 | train acc 76.49%
Epoch 05 | batch 0400/1563 | train loss 0.9238 | train acc 76.20%
Epoch 05 | batch 0500/1563 | train loss 0.9305 | train acc 76.29%
Epoch 05 | batch 0600/1563 | train loss 0.9385 | train acc 76.15%
Epoch 05 | batch 0700/1563 | train loss 0.9432 | train acc 76.15%
Epoch 05 | batch 0800/1563 | train loss 0.9468 | train acc 76.14%
Epoch 05 | batch 0900/1563 | train loss 0.9490 | train acc 76.21%
Epoch 05 | batch 1000/1563 | train loss 0.9438 | train acc 76.26%
Epoch 05 | batch 1100/1563 | train loss 0.9464 | train acc 76.26%
Epoch 05 | batch 1200/1563 | train loss 0.9540 | train acc 76.12%
Epoch 05 | batch 1300/1563 | train loss 0.9561 | train acc 76.10%
Epoch 05 | batch 1400/1563 | train loss 0.9625 | train acc 76.01%
Epoch 05 | batch 1500/1563 | train loss 0.9600 | train acc 76.04%
Epoch 05 summary | train loss 0.9597 | train acc 76.02% | test loss 0.7083 | test acc 79.85%
Epoch 06 | batch 0001/1563 | train loss 1.0101 | train acc 78.12%
Epoch 06 | batch 0100/1563 | train loss 0.9906 | train acc 75.50%
Epoch 06 | batch 0200/1563 | train loss 0.9495 | train acc 76.27%
Epoch 06 | batch 0300/1563 | train loss 0.9501 | train acc 76.27%
Epoch 06 | batch 0400/1563 | train loss 0.9428 | train acc 76.21%
Epoch 06 | batch 0500/1563 | train loss 0.9409 | train acc 76.34%
Epoch 06 | batch 0600/1563 | train loss 0.9422 | train acc 76.29%
Epoch 06 | batch 0700/1563 | train loss 0.9374 | train acc 76.31%
Epoch 06 | batch 0800/1563 | train loss 0.9290 | train acc 76.58%
Epoch 06 | batch 0900/1563 | train loss 0.9278 | train acc 76.58%
Epoch 06 | batch 1000/1563 | train loss 0.9317 | train acc 76.52%
Epoch 06 | batch 1100/1563 | train loss 0.9343 | train acc 76.53%
Epoch 06 | batch 1200/1563 | train loss 0.9360 | train acc 76.51%
Epoch 06 | batch 1300/1563 | train loss 0.9344 | train acc 76.56%
Epoch 06 | batch 1400/1563 | train loss 0.9431 | train acc 76.51%
Epoch 06 | batch 1500/1563 | train loss 0.9470 | train acc 76.45%
Epoch 06 summary | train loss 0.9482 | train acc 76.41% | test loss 0.6964 | test acc 80.14%
Epoch 07 | batch 0001/1563 | train loss 0.9213 | train acc 71.88%
Epoch 07 | batch 0100/1563 | train loss 0.9117 | train acc 76.38%
Epoch 07 | batch 0200/1563 | train loss 0.9181 | train acc 76.95%
Epoch 07 | batch 0300/1563 | train loss 0.9270 | train acc 76.82%
Epoch 07 | batch 0400/1563 | train loss 0.9312 | train acc 76.54%
Epoch 07 | batch 0500/1563 | train loss 0.9382 | train acc 76.56%
Epoch 07 | batch 0600/1563 | train loss 0.9544 | train acc 76.28%
Epoch 07 | batch 0700/1563 | train loss 0.9524 | train acc 76.38%
Epoch 07 | batch 0800/1563 | train loss 0.9552 | train acc 76.24%
Epoch 07 | batch 0900/1563 | train loss 0.9552 | train acc 76.16%
Epoch 07 | batch 1000/1563 | train loss 0.9574 | train acc 76.14%
Epoch 07 | batch 1100/1563 | train loss 0.9603 | train acc 76.12%
Epoch 07 | batch 1200/1563 | train loss 0.9614 | train acc 76.21%
Epoch 07 | batch 1300/1563 | train loss 0.9609 | train acc 76.21%
Epoch 07 | batch 1400/1563 | train loss 0.9603 | train acc 76.25%
Epoch 07 | batch 1500/1563 | train loss 0.9574 | train acc 76.27%
Epoch 07 summary | train loss 0.9566 | train acc 76.30% | test loss 0.6620 | test acc 81.24%
Saved best checkpoint to: /content/vgg16_cifar10_best.pt
Epoch 08 | batch 0001/1563 | train loss 0.3786 | train acc 78.12%
Epoch 08 | batch 0100/1563 | train loss 0.9103 | train acc 77.00%
Epoch 08 | batch 0200/1563 | train loss 0.9372 | train acc 76.30%
Epoch 08 | batch 0300/1563 | train loss 0.9276 | train acc 76.68%
Epoch 08 | batch 0400/1563 | train loss 0.9339 | train acc 76.54%
Epoch 08 | batch 0500/1563 | train loss 0.9391 | train acc 76.53%
Epoch 08 | batch 0600/1563 | train loss 0.9375 | train acc 76.47%
Epoch 08 | batch 0700/1563 | train loss 0.9463 | train acc 76.42%
Epoch 08 | batch 0800/1563 | train loss 0.9435 | train acc 76.45%
Epoch 08 | batch 0900/1563 | train loss 0.9484 | train acc 76.47%
Epoch 08 | batch 1000/1563 | train loss 0.9560 | train acc 76.48%
Epoch 08 | batch 1100/1563 | train loss 0.9582 | train acc 76.42%
Epoch 08 | batch 1200/1563 | train loss 0.9607 | train acc 76.47%
Epoch 08 | batch 1300/1563 | train loss 0.9596 | train acc 76.50%
Epoch 08 | batch 1400/1563 | train loss 0.9556 | train acc 76.61%
Epoch 08 | batch 1500/1563 | train loss 0.9559 | train acc 76.64%
Epoch 08 summary | train loss 0.9579 | train acc 76.58% | test loss 0.7228 | test acc 80.18%
Epoch 09 | batch 0001/1563 | train loss 0.6963 | train acc 81.25%
Epoch 09 | batch 0100/1563 | train loss 0.9806 | train acc 77.38%
Epoch 09 | batch 0200/1563 | train loss 0.9708 | train acc 77.14%
Epoch 09 | batch 0300/1563 | train loss 0.9650 | train acc 77.05%
Epoch 09 | batch 0400/1563 | train loss 0.9468 | train acc 77.26%
Epoch 09 | batch 0500/1563 | train loss 0.9357 | train acc 77.31%
Epoch 09 | batch 0600/1563 | train loss 0.9417 | train acc 77.17%
Epoch 09 | batch 0700/1563 | train loss 0.9458 | train acc 77.12%
Epoch 09 | batch 0800/1563 | train loss 0.9418 | train acc 77.16%
Epoch 09 | batch 0900/1563 | train loss 0.9414 | train acc 77.19%
Epoch 09 | batch 1000/1563 | train loss 0.9465 | train acc 76.98%
Epoch 09 | batch 1100/1563 | train loss 0.9490 | train acc 76.89%
Epoch 09 | batch 1200/1563 | train loss 0.9497 | train acc 76.87%
Epoch 09 | batch 1300/1563 | train loss 0.9483 | train acc 76.88%
Epoch 09 | batch 1400/1563 | train loss 0.9510 | train acc 76.84%
Epoch 09 | batch 1500/1563 | train loss 0.9521 | train acc 76.80%
Epoch 09 summary | train loss 0.9538 | train acc 76.81% | test loss 0.7417 | test acc 79.77%
Epoch 10 | batch 0001/1563 | train loss 0.8212 | train acc 81.25%
Epoch 10 | batch 0100/1563 | train loss 0.9521 | train acc 76.00%
Epoch 10 | batch 0200/1563 | train loss 0.9545 | train acc 76.30%
Epoch 10 | batch 0300/1563 | train loss 0.9557 | train acc 76.21%
Epoch 10 | batch 0400/1563 | train loss 0.9416 | train acc 76.61%
Epoch 10 | batch 0500/1563 | train loss 0.9459 | train acc 76.58%
Epoch 10 | batch 0600/1563 | train loss 0.9449 | train acc 76.77%
Epoch 10 | batch 0700/1563 | train loss 0.9478 | train acc 76.67%
Epoch 10 | batch 0800/1563 | train loss 0.9445 | train acc 76.73%
Epoch 10 | batch 0900/1563 | train loss 0.9518 | train acc 76.65%
Epoch 10 | batch 1000/1563 | train loss 0.9603 | train acc 76.63%
Epoch 10 | batch 1100/1563 | train loss 0.9673 | train acc 76.43%
Epoch 10 | batch 1200/1563 | train loss 0.9607 | train acc 76.52%
Epoch 10 | batch 1300/1563 | train loss 0.9640 | train acc 76.41%
Epoch 10 | batch 1400/1563 | train loss 0.9662 | train acc 76.42%
Epoch 10 | batch 1500/1563 | train loss 0.9626 | train acc 76.52%
Epoch 10 summary | train loss 0.9635 | train acc 76.52% | test loss 0.6648 | test acc 81.81%
Saved best checkpoint to: /content/vgg16_cifar10_best.pt
Epoch 11 | batch 0001/1563 | train loss 0.3792 | train acc 87.50%
Epoch 11 | batch 0100/1563 | train loss 0.9100 | train acc 76.91%
Epoch 11 | batch 0200/1563 | train loss 0.9087 | train acc 77.00%
Epoch 11 | batch 0300/1563 | train loss 0.9164 | train acc 77.15%
Epoch 11 | batch 0400/1563 | train loss 0.9259 | train acc 77.17%
Epoch 11 | batch 0500/1563 | train loss 0.9250 | train acc 77.19%
Epoch 11 | batch 0600/1563 | train loss 0.9350 | train acc 77.01%
Epoch 11 | batch 0700/1563 | train loss 0.9422 | train acc 76.94%
Epoch 11 | batch 0800/1563 | train loss 0.9497 | train acc 76.84%
Epoch 11 | batch 0900/1563 | train loss 0.9531 | train acc 76.76%
Epoch 11 | batch 1000/1563 | train loss 0.9568 | train acc 76.69%
Epoch 11 | batch 1100/1563 | train loss 0.9609 | train acc 76.71%
Epoch 11 | batch 1200/1563 | train loss 0.9583 | train acc 76.72%
Epoch 11 | batch 1300/1563 | train loss 0.9607 | train acc 76.63%
Epoch 11 | batch 1400/1563 | train loss 0.9638 | train acc 76.65%
Epoch 11 | batch 1500/1563 | train loss 0.9681 | train acc 76.64%
Epoch 11 summary | train loss 0.9704 | train acc 76.61% | test loss 0.7142 | test acc 80.30%
Epoch 12 | batch 0001/1563 | train loss 0.6471 | train acc 87.50%
Epoch 12 | batch 0100/1563 | train loss 0.9469 | train acc 76.97%
Epoch 12 | batch 0200/1563 | train loss 0.9653 | train acc 76.94%
Epoch 12 | batch 0300/1563 | train loss 0.9418 | train acc 77.22%
Epoch 12 | batch 0400/1563 | train loss 0.9422 | train acc 77.27%
Epoch 12 | batch 0500/1563 | train loss 0.9506 | train acc 77.14%
Epoch 12 | batch 0600/1563 | train loss 0.9525 | train acc 77.17%
Epoch 12 | batch 0700/1563 | train loss 0.9512 | train acc 77.23%
Epoch 12 | batch 0800/1563 | train loss 0.9462 | train acc 77.26%
Epoch 12 | batch 0900/1563 | train loss 0.9455 | train acc 77.25%
Epoch 12 | batch 1000/1563 | train loss 0.9489 | train acc 77.15%
Epoch 12 | batch 1100/1563 | train loss 0.9565 | train acc 77.04%
Epoch 12 | batch 1200/1563 | train loss 0.9578 | train acc 76.97%
Epoch 12 | batch 1300/1563 | train loss 0.9589 | train acc 76.92%
Epoch 12 | batch 1400/1563 | train loss 0.9618 | train acc 76.85%
Epoch 12 | batch 1500/1563 | train loss 0.9664 | train acc 76.80%
Epoch 12 summary | train loss 0.9668 | train acc 76.78% | test loss 0.7000 | test acc 80.97%
Epoch 13 | batch 0001/1563 | train loss 0.4774 | train acc 84.38%
Epoch 13 | batch 0100/1563 | train loss 0.9141 | train acc 77.66%
Epoch 13 | batch 0200/1563 | train loss 0.9042 | train acc 77.50%
Epoch 13 | batch 0300/1563 | train loss 0.9069 | train acc 77.43%
Epoch 13 | batch 0400/1563 | train loss 0.9249 | train acc 77.07%
Epoch 13 | batch 0500/1563 | train loss 0.9395 | train acc 76.62%
Epoch 13 | batch 0600/1563 | train loss 0.9407 | train acc 76.63%
Epoch 13 | batch 0700/1563 | train loss 0.9493 | train acc 76.63%
Epoch 13 | batch 0800/1563 | train loss 0.9545 | train acc 76.62%
Epoch 13 | batch 0900/1563 | train loss 0.9575 | train acc 76.60%
Epoch 13 | batch 1000/1563 | train loss 0.9660 | train acc 76.50%
Epoch 13 | batch 1100/1563 | train loss 0.9736 | train acc 76.37%
Epoch 13 | batch 1200/1563 | train loss 0.9723 | train acc 76.40%
Epoch 13 | batch 1300/1563 | train loss 0.9737 | train acc 76.37%
Epoch 13 | batch 1400/1563 | train loss 0.9750 | train acc 76.42%
Epoch 13 | batch 1500/1563 | train loss 0.9733 | train acc 76.46%
Epoch 13 summary | train loss 0.9752 | train acc 76.45% | test loss 0.7182 | test acc 80.23%
Epoch 14 | batch 0001/1563 | train loss 1.3286 | train acc 75.00%
Epoch 14 | batch 0100/1563 | train loss 0.9533 | train acc 76.91%
Epoch 14 | batch 0200/1563 | train loss 0.9651 | train acc 76.55%
Epoch 14 | batch 0300/1563 | train loss 0.9530 | train acc 76.92%
Epoch 14 | batch 0400/1563 | train loss 0.9426 | train acc 77.06%
Epoch 14 | batch 0500/1563 | train loss 0.9548 | train acc 76.96%
Epoch 14 | batch 0600/1563 | train loss 0.9567 | train acc 76.85%
Epoch 14 | batch 0700/1563 | train loss 0.9542 | train acc 76.99%
Epoch 14 | batch 0800/1563 | train loss 0.9601 | train acc 76.91%
Epoch 14 | batch 0900/1563 | train loss 0.9589 | train acc 76.98%
Epoch 14 | batch 1000/1563 | train loss 0.9595 | train acc 76.90%
Epoch 14 | batch 1100/1563 | train loss 0.9638 | train acc 76.84%
Epoch 14 | batch 1200/1563 | train loss 0.9665 | train acc 76.79%
Epoch 14 | batch 1300/1563 | train loss 0.9673 | train acc 76.72%
Epoch 14 | batch 1400/1563 | train loss 0.9670 | train acc 76.71%
Epoch 14 | batch 1500/1563 | train loss 0.9688 | train acc 76.67%
Epoch 14 summary | train loss 0.9695 | train acc 76.69% | test loss 0.7329 | test acc 80.06%
Epoch 15 | batch 0001/1563 | train loss 1.2630 | train acc 68.75%
Epoch 15 | batch 0100/1563 | train loss 1.0094 | train acc 76.03%
Epoch 15 | batch 0200/1563 | train loss 0.9521 | train acc 76.89%
Epoch 15 | batch 0300/1563 | train loss 0.9433 | train acc 76.98%
Epoch 15 | batch 0400/1563 | train loss 0.9313 | train acc 77.12%
Epoch 15 | batch 0500/1563 | train loss 0.9335 | train acc 77.11%
Epoch 15 | batch 0600/1563 | train loss 0.9353 | train acc 77.12%
Epoch 15 | batch 0700/1563 | train loss 0.9407 | train acc 77.24%
Epoch 15 | batch 0800/1563 | train loss 0.9566 | train acc 76.99%
Epoch 15 | batch 0900/1563 | train loss 0.9620 | train acc 76.87%
Epoch 15 | batch 1000/1563 | train loss 0.9638 | train acc 76.89%
Epoch 15 | batch 1100/1563 | train loss 0.9649 | train acc 76.85%
Epoch 15 | batch 1200/1563 | train loss 0.9654 | train acc 76.86%
Epoch 15 | batch 1300/1563 | train loss 0.9681 | train acc 76.87%
Epoch 15 | batch 1400/1563 | train loss 0.9690 | train acc 76.85%
Epoch 15 | batch 1500/1563 | train loss 0.9746 | train acc 76.77%
Epoch 15 summary | train loss 0.9743 | train acc 76.75% | test loss 0.7014 | test acc 81.12%
Epoch 16 | batch 0001/1563 | train loss 0.5419 | train acc 81.25%
Epoch 16 | batch 0100/1563 | train loss 0.9057 | train acc 77.41%
Epoch 16 | batch 0200/1563 | train loss 0.9212 | train acc 76.98%
Epoch 16 | batch 0300/1563 | train loss 0.8952 | train acc 77.62%
Epoch 16 | batch 0400/1563 | train loss 0.8859 | train acc 77.78%
Epoch 16 | batch 0500/1563 | train loss 0.9030 | train acc 77.70%
Epoch 16 | batch 0600/1563 | train loss 0.9051 | train acc 77.71%
Epoch 16 | batch 0700/1563 | train loss 0.9126 | train acc 77.66%
Epoch 16 | batch 0800/1563 | train loss 0.9241 | train acc 77.52%
Epoch 16 | batch 0900/1563 | train loss 0.9342 | train acc 77.38%
Epoch 16 | batch 1000/1563 | train loss 0.9403 | train acc 77.36%
Epoch 16 | batch 1100/1563 | train loss 0.9456 | train acc 77.26%
Epoch 16 | batch 1200/1563 | train loss 0.9519 | train acc 77.11%
Epoch 16 | batch 1300/1563 | train loss 0.9503 | train acc 77.15%
Epoch 16 | batch 1400/1563 | train loss 0.9539 | train acc 77.08%
Epoch 16 | batch 1500/1563 | train loss 0.9626 | train acc 76.91%
Epoch 16 summary | train loss 0.9664 | train acc 76.90% | test loss 0.7098 | test acc 80.28%
Epoch 17 | batch 0001/1563 | train loss 0.5338 | train acc 84.38%
Epoch 17 | batch 0100/1563 | train loss 0.9231 | train acc 76.91%
Epoch 17 | batch 0200/1563 | train loss 0.9307 | train acc 77.30%
Epoch 17 | batch 0300/1563 | train loss 0.9511 | train acc 77.08%
Epoch 17 | batch 0400/1563 | train loss 0.9546 | train acc 77.14%
Epoch 17 | batch 0500/1563 | train loss 0.9580 | train acc 77.12%
Epoch 17 | batch 0600/1563 | train loss 0.9626 | train acc 77.16%
Epoch 17 | batch 0700/1563 | train loss 0.9595 | train acc 77.13%
Epoch 17 | batch 0800/1563 | train loss 0.9561 | train acc 77.14%
Epoch 17 | batch 0900/1563 | train loss 0.9596 | train acc 77.10%
Epoch 17 | batch 1000/1563 | train loss 0.9598 | train acc 77.10%
Epoch 17 | batch 1100/1563 | train loss 0.9639 | train acc 77.09%
Epoch 17 | batch 1200/1563 | train loss 0.9626 | train acc 77.05%
Epoch 17 | batch 1300/1563 | train loss 0.9669 | train acc 76.94%
Epoch 17 | batch 1400/1563 | train loss 0.9720 | train acc 76.82%
Epoch 17 | batch 1500/1563 | train loss 0.9733 | train acc 76.80%
Epoch 17 summary | train loss 0.9754 | train acc 76.78% | test loss 0.6806 | test acc 80.72%
Epoch 18 | batch 0001/1563 | train loss 0.7543 | train acc 78.12%
Epoch 18 | batch 0100/1563 | train loss 0.9426 | train acc 75.97%
Epoch 18 | batch 0200/1563 | train loss 0.9377 | train acc 76.73%
Epoch 18 | batch 0300/1563 | train loss 0.9293 | train acc 76.95%
Epoch 18 | batch 0400/1563 | train loss 0.9252 | train acc 77.08%
Epoch 18 | batch 0500/1563 | train loss 0.9435 | train acc 76.96%
Epoch 18 | batch 0600/1563 | train loss 0.9440 | train acc 76.91%
Epoch 18 | batch 0700/1563 | train loss 0.9427 | train acc 77.00%
Epoch 18 | batch 0800/1563 | train loss 0.9379 | train acc 77.19%
Epoch 18 | batch 0900/1563 | train loss 0.9449 | train acc 77.09%
Epoch 18 | batch 1000/1563 | train loss 0.9451 | train acc 77.19%
Epoch 18 | batch 1100/1563 | train loss 0.9557 | train acc 77.01%
Epoch 18 | batch 1200/1563 | train loss 0.9602 | train acc 76.94%
Epoch 18 | batch 1300/1563 | train loss 0.9625 | train acc 76.86%
Epoch 18 | batch 1400/1563 | train loss 0.9666 | train acc 76.79%
Epoch 18 | batch 1500/1563 | train loss 0.9692 | train acc 76.80%
Epoch 18 summary | train loss 0.9682 | train acc 76.79% | test loss 0.7153 | test acc 80.59%
Epoch 19 | batch 0001/1563 | train loss 0.5122 | train acc 81.25%
Epoch 19 | batch 0100/1563 | train loss 0.9379 | train acc 77.81%
Epoch 19 | batch 0200/1563 | train loss 0.9549 | train acc 76.88%
Epoch 19 | batch 0300/1563 | train loss 0.9310 | train acc 77.24%
Epoch 19 | batch 0400/1563 | train loss 0.9409 | train acc 77.10%
Epoch 19 | batch 0500/1563 | train loss 0.9471 | train acc 77.04%
Epoch 19 | batch 0600/1563 | train loss 0.9499 | train acc 76.90%
Epoch 19 | batch 0700/1563 | train loss 0.9475 | train acc 76.80%
Epoch 19 | batch 0800/1563 | train loss 0.9467 | train acc 76.80%
Epoch 19 | batch 0900/1563 | train loss 0.9540 | train acc 76.81%
Epoch 19 | batch 1000/1563 | train loss 0.9565 | train acc 76.78%
Epoch 19 | batch 1100/1563 | train loss 0.9643 | train acc 76.74%
Epoch 19 | batch 1200/1563 | train loss 0.9651 | train acc 76.74%
Epoch 19 | batch 1300/1563 | train loss 0.9697 | train acc 76.70%
Epoch 19 | batch 1400/1563 | train loss 0.9698 | train acc 76.72%
Epoch 19 | batch 1500/1563 | train loss 0.9701 | train acc 76.72%
Epoch 19 summary | train loss 0.9707 | train acc 76.73% | test loss 0.7361 | test acc 80.24%
Epoch 20 | batch 0001/1563 | train loss 0.8956 | train acc 75.00%
Epoch 20 | batch 0100/1563 | train loss 0.8927 | train acc 77.19%
Epoch 20 | batch 0200/1563 | train loss 0.9423 | train acc 76.58%
Epoch 20 | batch 0300/1563 | train loss 0.9326 | train acc 76.96%
Epoch 20 | batch 0400/1563 | train loss 0.9546 | train acc 76.85%
Epoch 20 | batch 0500/1563 | train loss 0.9546 | train acc 77.03%
Epoch 20 | batch 0600/1563 | train loss 0.9590 | train acc 77.06%
Epoch 20 | batch 0700/1563 | train loss 0.9594 | train acc 77.08%
Epoch 20 | batch 0800/1563 | train loss 0.9557 | train acc 77.05%
Epoch 20 | batch 0900/1563 | train loss 0.9640 | train acc 76.93%
Epoch 20 | batch 1000/1563 | train loss 0.9637 | train acc 76.88%
Epoch 20 | batch 1100/1563 | train loss 0.9667 | train acc 76.87%
Epoch 20 | batch 1200/1563 | train loss 0.9688 | train acc 76.77%
Epoch 20 | batch 1300/1563 | train loss 0.9674 | train acc 76.78%
Epoch 20 | batch 1400/1563 | train loss 0.9679 | train acc 76.74%
Epoch 20 | batch 1500/1563 | train loss 0.9729 | train acc 76.68%
Epoch 20 summary | train loss 0.9712 | train acc 76.71% | test loss 0.6885 | test acc 80.96%
Epoch 21 | batch 0001/1563 | train loss 1.1926 | train acc 65.62%
Epoch 21 | batch 0100/1563 | train loss 0.9267 | train acc 76.53%
Epoch 21 | batch 0200/1563 | train loss 0.9058 | train acc 77.56%
Epoch 21 | batch 0300/1563 | train loss 0.9214 | train acc 77.21%
Epoch 21 | batch 0400/1563 | train loss 0.9256 | train acc 77.45%
Epoch 21 | batch 0500/1563 | train loss 0.9396 | train acc 77.19%
Epoch 21 | batch 0600/1563 | train loss 0.9564 | train acc 76.94%
Epoch 21 | batch 0700/1563 | train loss 0.9648 | train acc 76.87%
Epoch 21 | batch 0800/1563 | train loss 0.9587 | train acc 76.95%
Epoch 21 | batch 0900/1563 | train loss 0.9626 | train acc 77.00%
Epoch 21 | batch 1000/1563 | train loss 0.9666 | train acc 76.95%
Epoch 21 | batch 1100/1563 | train loss 0.9701 | train acc 76.93%
Epoch 21 | batch 1200/1563 | train loss 0.9677 | train acc 76.93%
Epoch 21 | batch 1300/1563 | train loss 0.9685 | train acc 76.98%
Epoch 21 | batch 1400/1563 | train loss 0.9740 | train acc 76.86%
Epoch 21 | batch 1500/1563 | train loss 0.9773 | train acc 76.80%
Epoch 21 summary | train loss 0.9774 | train acc 76.75% | test loss 0.7264 | test acc 80.48%
Epoch 22 | batch 0001/1563 | train loss 0.9165 | train acc 84.38%
Epoch 22 | batch 0100/1563 | train loss 0.9555 | train acc 77.06%
Epoch 22 | batch 0200/1563 | train loss 0.9908 | train acc 76.52%
Epoch 22 | batch 0300/1563 | train loss 0.9660 | train acc 76.96%
Epoch 22 | batch 0400/1563 | train loss 0.9626 | train acc 76.73%
Epoch 22 | batch 0500/1563 | train loss 0.9714 | train acc 76.46%
Epoch 22 | batch 0600/1563 | train loss 0.9767 | train acc 76.34%
Epoch 22 | batch 0700/1563 | train loss 0.9746 | train acc 76.48%
Epoch 22 | batch 0800/1563 | train loss 0.9761 | train acc 76.63%
Epoch 22 | batch 0900/1563 | train loss 0.9792 | train acc 76.55%
Epoch 22 | batch 1000/1563 | train loss 0.9808 | train acc 76.54%
Epoch 22 | batch 1100/1563 | train loss 0.9836 | train acc 76.55%
Epoch 22 | batch 1200/1563 | train loss 0.9807 | train acc 76.58%
Epoch 22 | batch 1300/1563 | train loss 0.9764 | train acc 76.68%
Epoch 22 | batch 1400/1563 | train loss 0.9776 | train acc 76.65%
Epoch 22 | batch 1500/1563 | train loss 0.9803 | train acc 76.72%
Epoch 22 summary | train loss 0.9788 | train acc 76.75% | test loss 0.7341 | test acc 80.10%
Epoch 23 | batch 0001/1563 | train loss 0.1490 | train acc 96.88%
Epoch 23 | batch 0100/1563 | train loss 0.9704 | train acc 76.53%
Epoch 23 | batch 0200/1563 | train loss 0.9819 | train acc 76.08%
Epoch 23 | batch 0300/1563 | train loss 0.9896 | train acc 76.34%
Epoch 23 | batch 0400/1563 | train loss 0.9727 | train acc 76.84%
Epoch 23 | batch 0500/1563 | train loss 0.9562 | train acc 77.08%
Epoch 23 | batch 0600/1563 | train loss 0.9648 | train acc 76.84%
Epoch 23 | batch 0700/1563 | train loss 0.9654 | train acc 76.78%
Epoch 23 | batch 0800/1563 | train loss 0.9581 | train acc 76.98%
Epoch 23 | batch 0900/1563 | train loss 0.9568 | train acc 76.92%
Epoch 23 | batch 1000/1563 | train loss 0.9448 | train acc 77.12%
Epoch 23 | batch 1100/1563 | train loss 0.9526 | train acc 77.05%
Epoch 23 | batch 1200/1563 | train loss 0.9605 | train acc 76.86%
Epoch 23 | batch 1300/1563 | train loss 0.9622 | train acc 76.85%
Epoch 23 | batch 1400/1563 | train loss 0.9639 | train acc 76.83%
Epoch 23 | batch 1500/1563 | train loss 0.9676 | train acc 76.79%
Epoch 23 summary | train loss 0.9685 | train acc 76.79% | test loss 0.7249 | test acc 80.23%
Epoch 24 | batch 0001/1563 | train loss 0.5076 | train acc 84.38%
Epoch 24 | batch 0100/1563 | train loss 0.9529 | train acc 77.19%
Epoch 24 | batch 0200/1563 | train loss 0.9286 | train acc 77.48%
Epoch 24 | batch 0300/1563 | train loss 0.9566 | train acc 76.93%
Epoch 24 | batch 0400/1563 | train loss 0.9662 | train acc 76.88%
Epoch 24 | batch 0500/1563 | train loss 0.9658 | train acc 77.04%
Epoch 24 | batch 0600/1563 | train loss 0.9743 | train acc 76.74%
Epoch 24 | batch 0700/1563 | train loss 0.9694 | train acc 76.84%
Epoch 24 | batch 0800/1563 | train loss 0.9727 | train acc 76.86%
Epoch 24 | batch 0900/1563 | train loss 0.9788 | train acc 76.70%
Epoch 24 | batch 1000/1563 | train loss 0.9777 | train acc 76.75%
Epoch 24 | batch 1100/1563 | train loss 0.9797 | train acc 76.72%
Epoch 24 | batch 1200/1563 | train loss 0.9857 | train acc 76.62%
Epoch 24 | batch 1300/1563 | train loss 0.9824 | train acc 76.63%
Epoch 24 | batch 1400/1563 | train loss 0.9803 | train acc 76.72%
Epoch 24 | batch 1500/1563 | train loss 0.9791 | train acc 76.69%
Epoch 24 summary | train loss 0.9790 | train acc 76.69% | test loss 0.6915 | test acc 81.16%
Epoch 25 | batch 0001/1563 | train loss 0.2680 | train acc 90.62%
Epoch 25 | batch 0100/1563 | train loss 0.8777 | train acc 78.44%
Epoch 25 | batch 0200/1563 | train loss 0.9235 | train acc 77.77%
Epoch 25 | batch 0300/1563 | train loss 0.9458 | train acc 77.45%
Epoch 25 | batch 0400/1563 | train loss 0.9361 | train acc 77.55%
Epoch 25 | batch 0500/1563 | train loss 0.9331 | train acc 77.64%
Epoch 25 | batch 0600/1563 | train loss 0.9439 | train acc 77.40%
Epoch 25 | batch 0700/1563 | train loss 0.9485 | train acc 77.26%
Epoch 25 | batch 0800/1563 | train loss 0.9522 | train acc 77.16%
Epoch 25 | batch 0900/1563 | train loss 0.9663 | train acc 76.95%
Epoch 25 | batch 1000/1563 | train loss 0.9727 | train acc 76.77%
Epoch 25 | batch 1100/1563 | train loss 0.9751 | train acc 76.81%
Epoch 25 | batch 1200/1563 | train loss 0.9772 | train acc 76.82%
Epoch 25 | batch 1300/1563 | train loss 0.9811 | train acc 76.74%
Epoch 25 | batch 1400/1563 | train loss 0.9826 | train acc 76.68%
Epoch 25 | batch 1500/1563 | train loss 0.9898 | train acc 76.57%
Epoch 25 summary | train loss 0.9877 | train acc 76.60% | test loss 0.7175 | test acc 80.37%
Epoch 26 | batch 0001/1563 | train loss 0.7790 | train acc 78.12%
Epoch 26 | batch 0100/1563 | train loss 0.9160 | train acc 78.38%
Epoch 26 | batch 0200/1563 | train loss 0.9292 | train acc 77.64%
Epoch 26 | batch 0300/1563 | train loss 0.9492 | train acc 77.24%
Epoch 26 | batch 0400/1563 | train loss 0.9450 | train acc 77.41%
Epoch 26 | batch 0500/1563 | train loss 0.9450 | train acc 77.38%
Epoch 26 | batch 0600/1563 | train loss 0.9443 | train acc 77.33%
Epoch 26 | batch 0700/1563 | train loss 0.9436 | train acc 77.28%
Epoch 26 | batch 0800/1563 | train loss 0.9487 | train acc 77.21%
Epoch 26 | batch 0900/1563 | train loss 0.9617 | train acc 76.99%
Epoch 26 | batch 1000/1563 | train loss 0.9638 | train acc 76.97%
Epoch 26 | batch 1100/1563 | train loss 0.9650 | train acc 76.93%
Epoch 26 | batch 1200/1563 | train loss 0.9687 | train acc 76.91%
Epoch 26 | batch 1300/1563 | train loss 0.9711 | train acc 76.87%
Epoch 26 | batch 1400/1563 | train loss 0.9740 | train acc 76.85%
Epoch 26 | batch 1500/1563 | train loss 0.9750 | train acc 76.85%
Epoch 26 summary | train loss 0.9760 | train acc 76.83% | test loss 0.6740 | test acc 81.27%
Epoch 27 | batch 0001/1563 | train loss 1.2059 | train acc 78.12%
Epoch 27 | batch 0100/1563 | train loss 1.0598 | train acc 75.53%
Epoch 27 | batch 0200/1563 | train loss 1.0197 | train acc 75.95%
Epoch 27 | batch 0300/1563 | train loss 0.9970 | train acc 76.26%
Epoch 27 | batch 0400/1563 | train loss 0.9983 | train acc 76.20%
Epoch 27 | batch 0500/1563 | train loss 1.0003 | train acc 76.17%
Epoch 27 | batch 0600/1563 | train loss 0.9943 | train acc 76.38%
Epoch 27 | batch 0700/1563 | train loss 0.9917 | train acc 76.32%
Epoch 27 | batch 0800/1563 | train loss 0.9867 | train acc 76.47%
Epoch 27 | batch 0900/1563 | train loss 0.9874 | train acc 76.38%
Epoch 27 | batch 1000/1563 | train loss 0.9841 | train acc 76.41%
Epoch 27 | batch 1100/1563 | train loss 0.9846 | train acc 76.32%
Epoch 27 | batch 1200/1563 | train loss 0.9795 | train acc 76.42%
Epoch 27 | batch 1300/1563 | train loss 0.9763 | train acc 76.52%
Epoch 27 | batch 1400/1563 | train loss 0.9772 | train acc 76.52%
Epoch 27 | batch 1500/1563 | train loss 0.9789 | train acc 76.54%
Epoch 27 summary | train loss 0.9817 | train acc 76.58% | test loss 0.6935 | test acc 81.23%
Epoch 28 | batch 0001/1563 | train loss 1.2774 | train acc 71.88%
Epoch 28 | batch 0100/1563 | train loss 0.9426 | train acc 78.00%
Epoch 28 | batch 0200/1563 | train loss 0.9166 | train acc 77.92%
Epoch 28 | batch 0300/1563 | train loss 0.9501 | train acc 77.47%
Epoch 28 | batch 0400/1563 | train loss 0.9361 | train acc 77.63%
Epoch 28 | batch 0500/1563 | train loss 0.9332 | train acc 77.64%
Epoch 28 | batch 0600/1563 | train loss 0.9372 | train acc 77.68%
Epoch 28 | batch 0700/1563 | train loss 0.9414 | train acc 77.48%
Epoch 28 | batch 0800/1563 | train loss 0.9366 | train acc 77.51%
Epoch 28 | batch 0900/1563 | train loss 0.9395 | train acc 77.39%
Epoch 28 | batch 1000/1563 | train loss 0.9471 | train acc 77.28%
Epoch 28 | batch 1100/1563 | train loss 0.9474 | train acc 77.34%
Epoch 28 | batch 1200/1563 | train loss 0.9509 | train acc 77.25%
Epoch 28 | batch 1300/1563 | train loss 0.9575 | train acc 77.17%
Epoch 28 | batch 1400/1563 | train loss 0.9680 | train acc 77.01%
Epoch 28 | batch 1500/1563 | train loss 0.9703 | train acc 76.92%
Epoch 28 summary | train loss 0.9728 | train acc 76.90% | test loss 0.7088 | test acc 80.82%
Epoch 29 | batch 0001/1563 | train loss 2.1886 | train acc 75.00%
Epoch 29 | batch 0100/1563 | train loss 0.9913 | train acc 76.56%
Epoch 29 | batch 0200/1563 | train loss 0.9592 | train acc 77.05%
Epoch 29 | batch 0300/1563 | train loss 0.9491 | train acc 77.00%
Epoch 29 | batch 0400/1563 | train loss 0.9662 | train acc 76.75%
Epoch 29 | batch 0500/1563 | train loss 0.9751 | train acc 76.68%
Epoch 29 | batch 0600/1563 | train loss 0.9762 | train acc 76.65%
Epoch 29 | batch 0700/1563 | train loss 0.9765 | train acc 76.59%
Epoch 29 | batch 0800/1563 | train loss 0.9787 | train acc 76.68%
Epoch 29 | batch 0900/1563 | train loss 0.9723 | train acc 76.75%
Epoch 29 | batch 1000/1563 | train loss 0.9717 | train acc 76.82%
Epoch 29 | batch 1100/1563 | train loss 0.9722 | train acc 76.78%
Epoch 29 | batch 1200/1563 | train loss 0.9729 | train acc 76.88%
Epoch 29 | batch 1300/1563 | train loss 0.9664 | train acc 76.93%
Epoch 29 | batch 1400/1563 | train loss 0.9675 | train acc 76.86%
Epoch 29 | batch 1500/1563 | train loss 0.9709 | train acc 76.88%
Epoch 29 summary | train loss 0.9759 | train acc 76.75% | test loss 0.7057 | test acc 80.73%
Epoch 30 | batch 0001/1563 | train loss 0.9544 | train acc 68.75%
Epoch 30 | batch 0100/1563 | train loss 0.9484 | train acc 76.69%
Epoch 30 | batch 0200/1563 | train loss 0.9468 | train acc 77.14%
Epoch 30 | batch 0300/1563 | train loss 0.9289 | train acc 77.41%
Epoch 30 | batch 0400/1563 | train loss 0.9283 | train acc 77.52%
Epoch 30 | batch 0500/1563 | train loss 0.9432 | train acc 77.32%
Epoch 30 | batch 0600/1563 | train loss 0.9577 | train acc 77.19%
Epoch 30 | batch 0700/1563 | train loss 0.9681 | train acc 77.03%
Epoch 30 | batch 0800/1563 | train loss 0.9694 | train acc 76.97%
Epoch 30 | batch 0900/1563 | train loss 0.9719 | train acc 76.88%
Epoch 30 | batch 1000/1563 | train loss 0.9729 | train acc 76.87%
Epoch 30 | batch 1100/1563 | train loss 0.9752 | train acc 76.86%
Epoch 30 | batch 1200/1563 | train loss 0.9740 | train acc 76.88%
Epoch 30 | batch 1300/1563 | train loss 0.9758 | train acc 76.88%
Epoch 30 | batch 1400/1563 | train loss 0.9793 | train acc 76.84%
Epoch 30 | batch 1500/1563 | train loss 0.9814 | train acc 76.78%
Epoch 30 summary | train loss 0.9840 | train acc 76.75% | test loss 0.6767 | test acc 81.49%

Best test accuracy: 81.81%
```

![alt text](image.png)
