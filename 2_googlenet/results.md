# Results

```shell
device = cuda

================ SHAPE CHECK ================

input                     shape = (4, 3, 32, 32)
conv1 + relu1             shape = (4, 64, 32, 32)
conv2 + relu2             shape = (4, 128, 32, 32)
maxpool1                  shape = (4, 128, 16, 16)
inception3a input         shape = (4, 128, 16, 16)
inc3a branch1             shape = (4, 32, 16, 16)
inc3a branch2             shape = (4, 64, 16, 16)
inc3a branch3             shape = (4, 16, 16, 16)
inc3a branch4             shape = (4, 16, 16, 16)
inception3a output        shape = (4, 128, 16, 16)
maxpool2                  shape = (4, 128, 8, 8)
inception3b input         shape = (4, 128, 8, 8)
inc3b branch1             shape = (4, 64, 8, 8)
inc3b branch2             shape = (4, 96, 8, 8)
inc3b branch3             shape = (4, 32, 8, 8)
inc3b branch4             shape = (4, 32, 8, 8)
inception3b output        shape = (4, 224, 8, 8)
global avgpool            shape = (4, 224, 1, 1)
flatten                   shape = (4, 224)
dropout                   shape = (4, 224)
fc logits                 shape = (4, 10)

=============================================

Epoch [1/30], Step [100/391], Loss: 1.9965
Epoch [1/30], Step [200/391], Loss: 1.7043
Epoch [1/30], Step [300/391], Loss: 1.9129
Epoch [1/30] Average Loss: 1.8279
Epoch [2/30], Step [100/391], Loss: 1.5314
Epoch [2/30], Step [200/391], Loss: 1.4042
Epoch [2/30], Step [300/391], Loss: 1.4319
Epoch [2/30] Average Loss: 1.4776
Epoch [3/30], Step [100/391], Loss: 1.2979
Epoch [3/30], Step [200/391], Loss: 1.2391
Epoch [3/30], Step [300/391], Loss: 1.2208
Epoch [3/30] Average Loss: 1.2800
Epoch [4/30], Step [100/391], Loss: 1.1354
Epoch [4/30], Step [200/391], Loss: 1.0174
Epoch [4/30], Step [300/391], Loss: 1.0572
Epoch [4/30] Average Loss: 1.1436
Epoch [5/30], Step [100/391], Loss: 1.0882
Epoch [5/30], Step [200/391], Loss: 1.1479
Epoch [5/30], Step [300/391], Loss: 1.1106
Epoch [5/30] Average Loss: 1.0502
Epoch [6/30], Step [100/391], Loss: 0.9044
Epoch [6/30], Step [200/391], Loss: 1.0163
Epoch [6/30], Step [300/391], Loss: 1.0923
Epoch [6/30] Average Loss: 0.9892
Epoch [7/30], Step [100/391], Loss: 0.9406
Epoch [7/30], Step [200/391], Loss: 0.8447
Epoch [7/30], Step [300/391], Loss: 0.9185
Epoch [7/30] Average Loss: 0.9310
Epoch [8/30], Step [100/391], Loss: 0.8490
Epoch [8/30], Step [200/391], Loss: 0.8690
Epoch [8/30], Step [300/391], Loss: 0.8797
Epoch [8/30] Average Loss: 0.8922
Epoch [9/30], Step [100/391], Loss: 0.8024
Epoch [9/30], Step [200/391], Loss: 0.8495
Epoch [9/30], Step [300/391], Loss: 0.8542
Epoch [9/30] Average Loss: 0.8546
Epoch [10/30], Step [100/391], Loss: 0.7539
Epoch [10/30], Step [200/391], Loss: 0.8033
Epoch [10/30], Step [300/391], Loss: 0.8291
Epoch [10/30] Average Loss: 0.8259
Epoch [11/30], Step [100/391], Loss: 0.8349
Epoch [11/30], Step [200/391], Loss: 0.8758
Epoch [11/30], Step [300/391], Loss: 0.7799
Epoch [11/30] Average Loss: 0.7907
Epoch [12/30], Step [100/391], Loss: 0.7098
Epoch [12/30], Step [200/391], Loss: 0.8761
Epoch [12/30], Step [300/391], Loss: 0.9576
Epoch [12/30] Average Loss: 0.7628
Epoch [13/30], Step [100/391], Loss: 0.8135
Epoch [13/30], Step [200/391], Loss: 0.6500
Epoch [13/30], Step [300/391], Loss: 0.6676
Epoch [13/30] Average Loss: 0.7371
Epoch [14/30], Step [100/391], Loss: 0.6685
Epoch [14/30], Step [200/391], Loss: 0.5708
Epoch [14/30], Step [300/391], Loss: 0.7846
Epoch [14/30] Average Loss: 0.7134
Epoch [15/30], Step [100/391], Loss: 0.7941
Epoch [15/30], Step [200/391], Loss: 0.6581
Epoch [15/30], Step [300/391], Loss: 0.7781
Epoch [15/30] Average Loss: 0.6895
Epoch [16/30], Step [100/391], Loss: 0.8238
Epoch [16/30], Step [200/391], Loss: 0.8321
Epoch [16/30], Step [300/391], Loss: 0.6386
Epoch [16/30] Average Loss: 0.6672
Epoch [17/30], Step [100/391], Loss: 0.6832
Epoch [17/30], Step [200/391], Loss: 0.6832
Epoch [17/30], Step [300/391], Loss: 0.7566
Epoch [17/30] Average Loss: 0.6396
Epoch [18/30], Step [100/391], Loss: 0.5879
Epoch [18/30], Step [200/391], Loss: 0.6385
Epoch [18/30], Step [300/391], Loss: 0.6391
Epoch [18/30] Average Loss: 0.6295
Epoch [19/30], Step [100/391], Loss: 0.5165
Epoch [19/30], Step [200/391], Loss: 0.4709
Epoch [19/30], Step [300/391], Loss: 0.6268
Epoch [19/30] Average Loss: 0.6042
Epoch [20/30], Step [100/391], Loss: 0.5628
Epoch [20/30], Step [200/391], Loss: 0.4978
Epoch [20/30], Step [300/391], Loss: 0.6873
Epoch [20/30] Average Loss: 0.5815
Epoch [21/30], Step [100/391], Loss: 0.5161
Epoch [21/30], Step [200/391], Loss: 0.7027
Epoch [21/30], Step [300/391], Loss: 0.5148
Epoch [21/30] Average Loss: 0.5661
Epoch [22/30], Step [100/391], Loss: 0.5229
Epoch [22/30], Step [200/391], Loss: 0.5172
Epoch [22/30], Step [300/391], Loss: 0.5523
Epoch [22/30] Average Loss: 0.5442
Epoch [23/30], Step [100/391], Loss: 0.5298
Epoch [23/30], Step [200/391], Loss: 0.4987
Epoch [23/30], Step [300/391], Loss: 0.5597
Epoch [23/30] Average Loss: 0.5302
Epoch [24/30], Step [100/391], Loss: 0.3743
Epoch [24/30], Step [200/391], Loss: 0.6314
Epoch [24/30], Step [300/391], Loss: 0.4723
Epoch [24/30] Average Loss: 0.5158
Epoch [25/30], Step [100/391], Loss: 0.4899
Epoch [25/30], Step [200/391], Loss: 0.4929
Epoch [25/30], Step [300/391], Loss: 0.5008
Epoch [25/30] Average Loss: 0.5004
Epoch [26/30], Step [100/391], Loss: 0.5630
Epoch [26/30], Step [200/391], Loss: 0.6702
Epoch [26/30], Step [300/391], Loss: 0.3682
Epoch [26/30] Average Loss: 0.4831
Epoch [27/30], Step [100/391], Loss: 0.5472
Epoch [27/30], Step [200/391], Loss: 0.4953
Epoch [27/30], Step [300/391], Loss: 0.4934
Epoch [27/30] Average Loss: 0.4702
Epoch [28/30], Step [100/391], Loss: 0.4571
Epoch [28/30], Step [200/391], Loss: 0.4653
Epoch [28/30], Step [300/391], Loss: 0.4814
Epoch [28/30] Average Loss: 0.4520
Epoch [29/30], Step [100/391], Loss: 0.5721
Epoch [29/30], Step [200/391], Loss: 0.4763
Epoch [29/30], Step [300/391], Loss: 0.4263
Epoch [29/30] Average Loss: 0.4501
Epoch [30/30], Step [100/391], Loss: 0.5616
Epoch [30/30], Step [200/391], Loss: 0.2991
Epoch [30/30], Step [300/391], Loss: 0.4134
Epoch [30/30] Average Loss: 0.4297

Test Accuracy: 81.50%
```
