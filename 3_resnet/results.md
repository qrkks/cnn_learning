```bash
CUDA detected: using full CIFAR-10 training settings.
device = cuda
data_dir = /content/data

Shape trace for one batch:
input              shape = (128, 3, 32, 32) # (BatchSize, Channels, Height, Width)
stem               shape = (128, 16, 32, 32)
stage1             shape = (128, 16, 32, 32)
stage2             shape = (128, 32, 16, 16)
stage3             shape = (128, 64, 8, 8)
avg_pool           shape = (128, 64, 1, 1)
flatten            shape = (128, 64)
classifier         shape = (128, 10)

epoch 01/100 | train loss 1.4327, train acc 47.25% | test loss 1.2974, test acc 53.55% | saved /content/resnet_cifar10_best.pt
epoch 02/100 | train loss 1.0195, train acc 63.65% | test loss 1.4385, test acc 53.54%
epoch 03/100 | train loss 0.8559, train acc 69.85% | test loss 0.8577, test acc 70.37% | saved /content/resnet_cifar10_best.pt
epoch 04/100 | train loss 0.7536, train acc 73.60% | test loss 0.7960, test acc 73.02% | saved /content/resnet_cifar10_best.pt
epoch 05/100 | train loss 0.6872, train acc 76.19% | test loss 0.7214, test acc 74.94% | saved /content/resnet_cifar10_best.pt
epoch 06/100 | train loss 0.6350, train acc 77.84% | test loss 0.7281, test acc 75.93% | saved /content/resnet_cifar10_best.pt
epoch 07/100 | train loss 0.5917, train acc 79.59% | test loss 0.6591, test acc 78.44% | saved /content/resnet_cifar10_best.pt
epoch 08/100 | train loss 0.5615, train acc 80.44% | test loss 0.7358, test acc 75.08%
epoch 09/100 | train loss 0.5340, train acc 81.42% | test loss 0.7295, test acc 76.34%
epoch 10/100 | train loss 0.5091, train acc 82.33% | test loss 0.6434, test acc 78.82% | saved /content/resnet_cifar10_best.pt
epoch 11/100 | train loss 0.4909, train acc 82.88% | test loss 0.6512, test acc 79.22% | saved /content/resnet_cifar10_best.pt
epoch 12/100 | train loss 0.4674, train acc 83.83% | test loss 0.5300, test acc 81.93% | saved /content/resnet_cifar10_best.pt
epoch 13/100 | train loss 0.4524, train acc 84.33% | test loss 0.6230, test acc 79.88%
epoch 14/100 | train loss 0.4377, train acc 85.00% | test loss 0.5233, test acc 82.44% | saved /content/resnet_cifar10_best.pt
epoch 15/100 | train loss 0.4297, train acc 84.99% | test loss 0.5475, test acc 82.00%
epoch 16/100 | train loss 0.4064, train acc 85.98% | test loss 0.4816, test acc 83.95% | saved /content/resnet_cifar10_best.pt
epoch 17/100 | train loss 0.4017, train acc 86.09% | test loss 0.5442, test acc 82.79%
epoch 18/100 | train loss 0.3893, train acc 86.40% | test loss 0.4762, test acc 83.96% | saved /content/resnet_cifar10_best.pt
epoch 19/100 | train loss 0.3773, train acc 86.79% | test loss 0.5403, test acc 81.96%
epoch 20/100 | train loss 0.3677, train acc 87.08% | test loss 0.4835, test acc 83.75%
epoch 21/100 | train loss 0.3564, train acc 87.64% | test loss 0.4962, test acc 83.40%
epoch 22/100 | train loss 0.3494, train acc 87.93% | test loss 0.4834, test acc 83.62%
epoch 23/100 | train loss 0.3408, train acc 88.04% | test loss 0.4979, test acc 83.34%
epoch 24/100 | train loss 0.3352, train acc 88.19% | test loss 0.4726, test acc 84.70% | saved /content/resnet_cifar10_best.pt
epoch 25/100 | train loss 0.3249, train acc 88.54% | test loss 0.4404, test acc 85.62% | saved /content/resnet_cifar10_best.pt
epoch 26/100 | train loss 0.3176, train acc 89.16% | test loss 0.4671, test acc 85.07%
epoch 27/100 | train loss 0.3137, train acc 89.00% | test loss 0.5282, test acc 83.25%
epoch 28/100 | train loss 0.3063, train acc 89.33% | test loss 0.4871, test acc 84.44%
epoch 29/100 | train loss 0.2990, train acc 89.49% | test loss 0.5348, test acc 83.29%
epoch 30/100 | train loss 0.2946, train acc 89.80% | test loss 0.4256, test acc 85.85% | saved /content/resnet_cifar10_best.pt
epoch 31/100 | train loss 0.2887, train acc 89.81% | test loss 0.4390, test acc 85.77%
epoch 32/100 | train loss 0.2842, train acc 90.03% | test loss 0.4642, test acc 85.04%
epoch 33/100 | train loss 0.2762, train acc 90.35% | test loss 0.4609, test acc 85.56%
epoch 34/100 | train loss 0.2717, train acc 90.46% | test loss 0.4361, test acc 86.23% | saved /content/resnet_cifar10_best.pt
epoch 35/100 | train loss 0.2681, train acc 90.54% | test loss 0.4864, test acc 85.18%
epoch 36/100 | train loss 0.2632, train acc 90.77% | test loss 0.4176, test acc 86.89% | saved /content/resnet_cifar10_best.pt
epoch 37/100 | train loss 0.2635, train acc 90.71% | test loss 0.4467, test acc 85.82%
epoch 38/100 | train loss 0.2537, train acc 91.19% | test loss 0.4183, test acc 86.61%
epoch 39/100 | train loss 0.2490, train acc 91.16% | test loss 0.4612, test acc 86.03%
epoch 40/100 | train loss 0.2506, train acc 91.11% | test loss 0.4283, test acc 86.41%
epoch 41/100 | train loss 0.2427, train acc 91.44% | test loss 0.4149, test acc 87.19% | saved /content/resnet_cifar10_best.pt
epoch 42/100 | train loss 0.2356, train acc 91.62% | test loss 0.4221, test acc 87.12%
epoch 43/100 | train loss 0.2329, train acc 91.73% | test loss 0.4382, test acc 86.54%
epoch 44/100 | train loss 0.2390, train acc 91.58% | test loss 0.4110, test acc 87.14%
epoch 45/100 | train loss 0.2300, train acc 91.90% | test loss 0.4227, test acc 86.96%
epoch 46/100 | train loss 0.2241, train acc 92.09% | test loss 0.5306, test acc 84.50%
epoch 47/100 | train loss 0.2223, train acc 92.26% | test loss 0.4634, test acc 86.23%
epoch 48/100 | train loss 0.2148, train acc 92.45% | test loss 0.4587, test acc 86.85%
epoch 49/100 | train loss 0.2145, train acc 92.46% | test loss 0.4464, test acc 86.78%
epoch 50/100 | train loss 0.2108, train acc 92.48% | test loss 0.4144, test acc 87.34% | saved /content/resnet_cifar10_best.pt
epoch 51/100 | train loss 0.2135, train acc 92.50% | test loss 0.4703, test acc 86.19%
epoch 52/100 | train loss 0.2056, train acc 92.87% | test loss 0.4740, test acc 86.21%
epoch 53/100 | train loss 0.2020, train acc 92.89% | test loss 0.4463, test acc 86.84%
epoch 54/100 | train loss 0.2015, train acc 92.87% | test loss 0.4617, test acc 86.73%
epoch 55/100 | train loss 0.1976, train acc 93.06% | test loss 0.4425, test acc 86.87%
epoch 56/100 | train loss 0.1987, train acc 92.98% | test loss 0.4195, test acc 87.73% | saved /content/resnet_cifar10_best.pt
epoch 57/100 | train loss 0.1938, train acc 93.07% | test loss 0.4508, test acc 86.62%
epoch 58/100 | train loss 0.1911, train acc 93.19% | test loss 0.4295, test acc 87.18%
epoch 59/100 | train loss 0.1911, train acc 93.21% | test loss 0.4429, test acc 86.97%
epoch 60/100 | train loss 0.1909, train acc 93.21% | test loss 0.4556, test acc 87.20%
epoch 61/100 | train loss 0.1838, train acc 93.41% | test loss 0.4730, test acc 86.56%
epoch 62/100 | train loss 0.1831, train acc 93.45% | test loss 0.4447, test acc 87.22%
epoch 63/100 | train loss 0.1771, train acc 93.69% | test loss 0.4485, test acc 87.28%
epoch 64/100 | train loss 0.1796, train acc 93.60% | test loss 0.4439, test acc 87.21%
epoch 65/100 | train loss 0.1764, train acc 93.72% | test loss 0.4769, test acc 86.69%
epoch 66/100 | train loss 0.1762, train acc 93.68% | test loss 0.4848, test acc 86.53%
epoch 67/100 | train loss 0.1730, train acc 93.76% | test loss 0.4295, test acc 87.63%
epoch 68/100 | train loss 0.1688, train acc 93.95% | test loss 0.4445, test acc 87.19%
epoch 69/100 | train loss 0.1666, train acc 94.05% | test loss 0.4863, test acc 87.14%
epoch 70/100 | train loss 0.1676, train acc 94.05% | test loss 0.4716, test acc 87.03%
epoch 71/100 | train loss 0.1666, train acc 93.95% | test loss 0.4411, test acc 87.51%
epoch 72/100 | train loss 0.1625, train acc 94.25% | test loss 0.4503, test acc 87.31%
epoch 73/100 | train loss 0.1628, train acc 94.28% | test loss 0.4581, test acc 87.36%
epoch 74/100 | train loss 0.1583, train acc 94.30% | test loss 0.4621, test acc 87.08%
epoch 75/100 | train loss 0.1579, train acc 94.42% | test loss 0.4843, test acc 86.60%
epoch 76/100 | train loss 0.1544, train acc 94.40% | test loss 0.4359, test acc 87.79% | saved /content/resnet_cifar10_best.pt
epoch 77/100 | train loss 0.1512, train acc 94.55% | test loss 0.4718, test acc 87.27%
epoch 78/100 | train loss 0.1552, train acc 94.44% | test loss 0.4618, test acc 87.31%
epoch 79/100 | train loss 0.1512, train acc 94.48% | test loss 0.4392, test acc 87.78%
epoch 80/100 | train loss 0.1498, train acc 94.63% | test loss 0.4546, test acc 87.93% | saved /content/resnet_cifar10_best.pt
epoch 81/100 | train loss 0.1481, train acc 94.59% | test loss 0.4557, test acc 87.36%
epoch 82/100 | train loss 0.1473, train acc 94.79% | test loss 0.4633, test acc 87.37%
epoch 83/100 | train loss 0.1468, train acc 94.73% | test loss 0.4772, test acc 87.47%
epoch 84/100 | train loss 0.1412, train acc 94.91% | test loss 0.4921, test acc 87.23%
epoch 85/100 | train loss 0.1430, train acc 94.85% | test loss 0.4972, test acc 87.44%
epoch 86/100 | train loss 0.1405, train acc 94.89% | test loss 0.4462, test acc 87.82%
epoch 87/100 | train loss 0.1412, train acc 94.98% | test loss 0.4768, test acc 87.77%
epoch 88/100 | train loss 0.1421, train acc 94.91% | test loss 0.5038, test acc 87.47%
epoch 89/100 | train loss 0.1374, train acc 94.98% | test loss 0.4727, test acc 87.72%
epoch 90/100 | train loss 0.1342, train acc 95.21% | test loss 0.5121, test acc 87.09%
epoch 91/100 | train loss 0.1354, train acc 95.12% | test loss 0.4898, test acc 87.63%
epoch 92/100 | train loss 0.1372, train acc 95.12% | test loss 0.4887, test acc 87.05%
epoch 93/100 | train loss 0.1362, train acc 95.09% | test loss 0.4933, test acc 87.17%
epoch 94/100 | train loss 0.1314, train acc 95.19% | test loss 0.4773, test acc 87.70%
epoch 95/100 | train loss 0.1326, train acc 95.25% | test loss 0.4903, test acc 87.39%
epoch 96/100 | train loss 0.1270, train acc 95.42% | test loss 0.4988, test acc 87.55%
epoch 97/100 | train loss 0.1298, train acc 95.33% | test loss 0.5185, test acc 87.42%
epoch 98/100 | train loss 0.1296, train acc 95.42% | test loss 0.5260, test acc 86.81%
epoch 99/100 | train loss 0.1297, train acc 95.34% | test loss 0.4842, test acc 87.53%
epoch 100/100 | train loss 0.1238, train acc 95.58% | test loss 0.4865, test acc 87.37%
best test accuracy = 87.93%
best checkpoint = /content/resnet_cifar10_best.pt
```

![alt text](image.png)
