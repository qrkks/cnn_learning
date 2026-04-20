# Results

## 30 epochs

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

## 100 epoches

```shell
device = cuda
100%|██████████| 170M/170M [00:04<00:00, 42.1MB/s]

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

Epoch [1/100], Step [100/391], Loss: 1.9304
Epoch [1/100], Step [200/391], Loss: 1.7913
Epoch [1/100], Step [300/391], Loss: 1.7105
Epoch [1/100] Average Loss: 1.8293
Epoch [2/100], Step [100/391], Loss: 1.5219
Epoch [2/100], Step [200/391], Loss: 1.4869
Epoch [2/100], Step [300/391], Loss: 1.3635
Epoch [2/100] Average Loss: 1.4687
Epoch [3/100], Step [100/391], Loss: 1.4360
Epoch [3/100], Step [200/391], Loss: 1.2650
Epoch [3/100], Step [300/391], Loss: 1.1479
Epoch [3/100] Average Loss: 1.2829
Epoch [4/100], Step [100/391], Loss: 1.0620
Epoch [4/100], Step [200/391], Loss: 1.0696
Epoch [4/100], Step [300/391], Loss: 1.0759
Epoch [4/100] Average Loss: 1.1473
Epoch [5/100], Step [100/391], Loss: 0.9058
Epoch [5/100], Step [200/391], Loss: 1.0097
Epoch [5/100], Step [300/391], Loss: 1.0696
Epoch [5/100] Average Loss: 1.0577
Epoch [6/100], Step [100/391], Loss: 0.9645
Epoch [6/100], Step [200/391], Loss: 0.9924
Epoch [6/100], Step [300/391], Loss: 0.9920
Epoch [6/100] Average Loss: 0.9859
Epoch [7/100], Step [100/391], Loss: 0.7933
Epoch [7/100], Step [200/391], Loss: 0.9311
Epoch [7/100], Step [300/391], Loss: 0.8915
Epoch [7/100] Average Loss: 0.9300
Epoch [8/100], Step [100/391], Loss: 0.6799
Epoch [8/100], Step [200/391], Loss: 0.9222
Epoch [8/100], Step [300/391], Loss: 1.0013
Epoch [8/100] Average Loss: 0.8838
Epoch [9/100], Step [100/391], Loss: 0.9102
Epoch [9/100], Step [200/391], Loss: 0.9189
Epoch [9/100], Step [300/391], Loss: 0.8841
Epoch [9/100] Average Loss: 0.8471
Epoch [10/100], Step [100/391], Loss: 0.8233
Epoch [10/100], Step [200/391], Loss: 0.7494
Epoch [10/100], Step [300/391], Loss: 0.8809
Epoch [10/100] Average Loss: 0.8092
Epoch [11/100], Step [100/391], Loss: 0.7424
Epoch [11/100], Step [200/391], Loss: 0.7138
Epoch [11/100], Step [300/391], Loss: 0.7945
Epoch [11/100] Average Loss: 0.7850
Epoch [12/100], Step [100/391], Loss: 0.8239
Epoch [12/100], Step [200/391], Loss: 0.7314
Epoch [12/100], Step [300/391], Loss: 0.7245
Epoch [12/100] Average Loss: 0.7489
Epoch [13/100], Step [100/391], Loss: 0.7140
Epoch [13/100], Step [200/391], Loss: 0.6895
Epoch [13/100], Step [300/391], Loss: 0.6659
Epoch [13/100] Average Loss: 0.7268
Epoch [14/100], Step [100/391], Loss: 0.7349
Epoch [14/100], Step [200/391], Loss: 0.7004
Epoch [14/100], Step [300/391], Loss: 0.6953
Epoch [14/100] Average Loss: 0.6932
Epoch [15/100], Step [100/391], Loss: 0.7038
Epoch [15/100], Step [200/391], Loss: 0.5606
Epoch [15/100], Step [300/391], Loss: 0.6182
Epoch [15/100] Average Loss: 0.6695
Epoch [16/100], Step [100/391], Loss: 0.6067
Epoch [16/100], Step [200/391], Loss: 0.5645
Epoch [16/100], Step [300/391], Loss: 0.5448
Epoch [16/100] Average Loss: 0.6461
Epoch [17/100], Step [100/391], Loss: 0.6323
Epoch [17/100], Step [200/391], Loss: 0.5866
Epoch [17/100], Step [300/391], Loss: 0.6632
Epoch [17/100] Average Loss: 0.6292
Epoch [18/100], Step [100/391], Loss: 0.5884
Epoch [18/100], Step [200/391], Loss: 0.5878
Epoch [18/100], Step [300/391], Loss: 0.6325
Epoch [18/100] Average Loss: 0.6049
Epoch [19/100], Step [100/391], Loss: 0.5988
Epoch [19/100], Step [200/391], Loss: 0.6386
Epoch [19/100], Step [300/391], Loss: 0.4422
Epoch [19/100] Average Loss: 0.5807
Epoch [20/100], Step [100/391], Loss: 0.5255
Epoch [20/100], Step [200/391], Loss: 0.5211
Epoch [20/100], Step [300/391], Loss: 0.7333
Epoch [20/100] Average Loss: 0.5693
Epoch [21/100], Step [100/391], Loss: 0.5631
Epoch [21/100], Step [200/391], Loss: 0.6676
Epoch [21/100], Step [300/391], Loss: 0.7477
Epoch [21/100] Average Loss: 0.5502
Epoch [22/100], Step [100/391], Loss: 0.4784
Epoch [22/100], Step [200/391], Loss: 0.5300
Epoch [22/100], Step [300/391], Loss: 0.4741
Epoch [22/100] Average Loss: 0.5294
Epoch [23/100], Step [100/391], Loss: 0.4523
Epoch [23/100], Step [200/391], Loss: 0.6649
Epoch [23/100], Step [300/391], Loss: 0.5474
Epoch [23/100] Average Loss: 0.5155
Epoch [24/100], Step [100/391], Loss: 0.5319
Epoch [24/100], Step [200/391], Loss: 0.4848
Epoch [24/100], Step [300/391], Loss: 0.5406
Epoch [24/100] Average Loss: 0.5067
Epoch [25/100], Step [100/391], Loss: 0.4116
Epoch [25/100], Step [200/391], Loss: 0.4832
Epoch [25/100], Step [300/391], Loss: 0.5280
Epoch [25/100] Average Loss: 0.4886
Epoch [26/100], Step [100/391], Loss: 0.3575
Epoch [26/100], Step [200/391], Loss: 0.3959
Epoch [26/100], Step [300/391], Loss: 0.5257
Epoch [26/100] Average Loss: 0.4746
Epoch [27/100], Step [100/391], Loss: 0.5609
Epoch [27/100], Step [200/391], Loss: 0.4153
Epoch [27/100], Step [300/391], Loss: 0.4838
Epoch [27/100] Average Loss: 0.4587
Epoch [28/100], Step [100/391], Loss: 0.3896
Epoch [28/100], Step [200/391], Loss: 0.3781
Epoch [28/100], Step [300/391], Loss: 0.4361
Epoch [28/100] Average Loss: 0.4444
Epoch [29/100], Step [100/391], Loss: 0.3882
Epoch [29/100], Step [200/391], Loss: 0.4617
Epoch [29/100], Step [300/391], Loss: 0.4650
Epoch [29/100] Average Loss: 0.4369
Epoch [30/100], Step [100/391], Loss: 0.4302
Epoch [30/100], Step [200/391], Loss: 0.4414
Epoch [30/100], Step [300/391], Loss: 0.5384
Epoch [30/100] Average Loss: 0.4225
Epoch [31/100], Step [100/391], Loss: 0.4893
Epoch [31/100], Step [200/391], Loss: 0.4595
Epoch [31/100], Step [300/391], Loss: 0.5342
Epoch [31/100] Average Loss: 0.4129
Epoch [32/100], Step [100/391], Loss: 0.5559
Epoch [32/100], Step [200/391], Loss: 0.3445
Epoch [32/100], Step [300/391], Loss: 0.4972
Epoch [32/100] Average Loss: 0.3995
Epoch [33/100], Step [100/391], Loss: 0.2794
Epoch [33/100], Step [200/391], Loss: 0.3667
Epoch [33/100], Step [300/391], Loss: 0.3293
Epoch [33/100] Average Loss: 0.3878
Epoch [34/100], Step [100/391], Loss: 0.3179
Epoch [34/100], Step [200/391], Loss: 0.4104
Epoch [34/100], Step [300/391], Loss: 0.3400
Epoch [34/100] Average Loss: 0.3798
Epoch [35/100], Step [100/391], Loss: 0.3301
Epoch [35/100], Step [200/391], Loss: 0.3891
Epoch [35/100], Step [300/391], Loss: 0.3099
Epoch [35/100] Average Loss: 0.3707
Epoch [36/100], Step [100/391], Loss: 0.2784
Epoch [36/100], Step [200/391], Loss: 0.2753
Epoch [36/100], Step [300/391], Loss: 0.3136
Epoch [36/100] Average Loss: 0.3602
Epoch [37/100], Step [100/391], Loss: 0.3327
Epoch [37/100], Step [200/391], Loss: 0.4195
Epoch [37/100], Step [300/391], Loss: 0.3652
Epoch [37/100] Average Loss: 0.3529
Epoch [38/100], Step [100/391], Loss: 0.3853
Epoch [38/100], Step [200/391], Loss: 0.2718
Epoch [38/100], Step [300/391], Loss: 0.3046
Epoch [38/100] Average Loss: 0.3399
Epoch [39/100], Step [100/391], Loss: 0.3275
Epoch [39/100], Step [200/391], Loss: 0.3560
Epoch [39/100], Step [300/391], Loss: 0.3905
Epoch [39/100] Average Loss: 0.3306
Epoch [40/100], Step [100/391], Loss: 0.2218
Epoch [40/100], Step [200/391], Loss: 0.3370
Epoch [40/100], Step [300/391], Loss: 0.2921
Epoch [40/100] Average Loss: 0.3255
Epoch [41/100], Step [100/391], Loss: 0.2750
Epoch [41/100], Step [200/391], Loss: 0.2708
Epoch [41/100], Step [300/391], Loss: 0.3319
Epoch [41/100] Average Loss: 0.3133
Epoch [42/100], Step [100/391], Loss: 0.3106
Epoch [42/100], Step [200/391], Loss: 0.2670
Epoch [42/100], Step [300/391], Loss: 0.2914
Epoch [42/100] Average Loss: 0.3031
Epoch [43/100], Step [100/391], Loss: 0.2417
Epoch [43/100], Step [200/391], Loss: 0.2210
Epoch [43/100], Step [300/391], Loss: 0.2592
Epoch [43/100] Average Loss: 0.2991
Epoch [44/100], Step [100/391], Loss: 0.2509
Epoch [44/100], Step [200/391], Loss: 0.3464
Epoch [44/100], Step [300/391], Loss: 0.2403
Epoch [44/100] Average Loss: 0.2898
Epoch [45/100], Step [100/391], Loss: 0.2892
Epoch [45/100], Step [200/391], Loss: 0.2443
Epoch [45/100], Step [300/391], Loss: 0.4246
Epoch [45/100] Average Loss: 0.2791
Epoch [46/100], Step [100/391], Loss: 0.3695
Epoch [46/100], Step [200/391], Loss: 0.2663
Epoch [46/100], Step [300/391], Loss: 0.2640
Epoch [46/100] Average Loss: 0.2724
Epoch [47/100], Step [100/391], Loss: 0.2175
Epoch [47/100], Step [200/391], Loss: 0.3532
Epoch [47/100], Step [300/391], Loss: 0.3100
Epoch [47/100] Average Loss: 0.2626
Epoch [48/100], Step [100/391], Loss: 0.3234
Epoch [48/100], Step [200/391], Loss: 0.2533
Epoch [48/100], Step [300/391], Loss: 0.2392
Epoch [48/100] Average Loss: 0.2582
Epoch [49/100], Step [100/391], Loss: 0.2643
Epoch [49/100], Step [200/391], Loss: 0.2164
Epoch [49/100], Step [300/391], Loss: 0.4484
Epoch [49/100] Average Loss: 0.2499
Epoch [50/100], Step [100/391], Loss: 0.1990
Epoch [50/100], Step [200/391], Loss: 0.2472
Epoch [50/100], Step [300/391], Loss: 0.2434
Epoch [50/100] Average Loss: 0.2446
Epoch [51/100], Step [100/391], Loss: 0.2438
Epoch [51/100], Step [200/391], Loss: 0.2083
Epoch [51/100], Step [300/391], Loss: 0.2823
Epoch [51/100] Average Loss: 0.2385
Epoch [52/100], Step [100/391], Loss: 0.2894
Epoch [52/100], Step [200/391], Loss: 0.1888
Epoch [52/100], Step [300/391], Loss: 0.2530
Epoch [52/100] Average Loss: 0.2253
Epoch [53/100], Step [100/391], Loss: 0.2847
Epoch [53/100], Step [200/391], Loss: 0.2200
Epoch [53/100], Step [300/391], Loss: 0.2508
Epoch [53/100] Average Loss: 0.2231
Epoch [54/100], Step [100/391], Loss: 0.2421
Epoch [54/100], Step [200/391], Loss: 0.2109
Epoch [54/100], Step [300/391], Loss: 0.2922
Epoch [54/100] Average Loss: 0.2211
Epoch [55/100], Step [100/391], Loss: 0.2522
Epoch [55/100], Step [200/391], Loss: 0.1307
Epoch [55/100], Step [300/391], Loss: 0.2105
Epoch [55/100] Average Loss: 0.2154
Epoch [56/100], Step [100/391], Loss: 0.2994
Epoch [56/100], Step [200/391], Loss: 0.1710
Epoch [56/100], Step [300/391], Loss: 0.1487
Epoch [56/100] Average Loss: 0.2081
Epoch [57/100], Step [100/391], Loss: 0.2188
Epoch [57/100], Step [200/391], Loss: 0.1744
Epoch [57/100], Step [300/391], Loss: 0.2328
Epoch [57/100] Average Loss: 0.1999
Epoch [58/100], Step [100/391], Loss: 0.1792
Epoch [58/100], Step [200/391], Loss: 0.1746
Epoch [58/100], Step [300/391], Loss: 0.1704
Epoch [58/100] Average Loss: 0.1985
Epoch [59/100], Step [100/391], Loss: 0.1131
Epoch [59/100], Step [200/391], Loss: 0.1565
Epoch [59/100], Step [300/391], Loss: 0.1596
Epoch [59/100] Average Loss: 0.1940
Epoch [60/100], Step [100/391], Loss: 0.3102
Epoch [60/100], Step [200/391], Loss: 0.2132
Epoch [60/100], Step [300/391], Loss: 0.1322
Epoch [60/100] Average Loss: 0.1937
Epoch [61/100], Step [100/391], Loss: 0.1932
Epoch [61/100], Step [200/391], Loss: 0.1341
Epoch [61/100], Step [300/391], Loss: 0.1459
Epoch [61/100] Average Loss: 0.1765
Epoch [62/100], Step [100/391], Loss: 0.1619
Epoch [62/100], Step [200/391], Loss: 0.1979
Epoch [62/100], Step [300/391], Loss: 0.2574
Epoch [62/100] Average Loss: 0.1827
Epoch [63/100], Step [100/391], Loss: 0.1618
Epoch [63/100], Step [200/391], Loss: 0.2568
Epoch [63/100], Step [300/391], Loss: 0.1752
Epoch [63/100] Average Loss: 0.1780
Epoch [64/100], Step [100/391], Loss: 0.1936
Epoch [64/100], Step [200/391], Loss: 0.0909
Epoch [64/100], Step [300/391], Loss: 0.1065
Epoch [64/100] Average Loss: 0.1739
Epoch [65/100], Step [100/391], Loss: 0.1869
Epoch [65/100], Step [200/391], Loss: 0.2030
Epoch [65/100], Step [300/391], Loss: 0.2186
Epoch [65/100] Average Loss: 0.1723
Epoch [66/100], Step [100/391], Loss: 0.2437
Epoch [66/100], Step [200/391], Loss: 0.1046
Epoch [66/100], Step [300/391], Loss: 0.2440
Epoch [66/100] Average Loss: 0.1639
Epoch [67/100], Step [100/391], Loss: 0.1362
Epoch [67/100], Step [200/391], Loss: 0.2265
Epoch [67/100], Step [300/391], Loss: 0.1113
Epoch [67/100] Average Loss: 0.1621
Epoch [68/100], Step [100/391], Loss: 0.2191
Epoch [68/100], Step [200/391], Loss: 0.1748
Epoch [68/100], Step [300/391], Loss: 0.1429
Epoch [68/100] Average Loss: 0.1592
Epoch [69/100], Step [100/391], Loss: 0.1310
Epoch [69/100], Step [200/391], Loss: 0.0642
Epoch [69/100], Step [300/391], Loss: 0.1541
Epoch [69/100] Average Loss: 0.1586
Epoch [70/100], Step [100/391], Loss: 0.1545
Epoch [70/100], Step [200/391], Loss: 0.1648
Epoch [70/100], Step [300/391], Loss: 0.1886
Epoch [70/100] Average Loss: 0.1532
Epoch [71/100], Step [100/391], Loss: 0.1624
Epoch [71/100], Step [200/391], Loss: 0.1194
Epoch [71/100], Step [300/391], Loss: 0.1578
Epoch [71/100] Average Loss: 0.1518
Epoch [72/100], Step [100/391], Loss: 0.1442
Epoch [72/100], Step [200/391], Loss: 0.1651
Epoch [72/100], Step [300/391], Loss: 0.0888
Epoch [72/100] Average Loss: 0.1475
Epoch [73/100], Step [100/391], Loss: 0.1276
Epoch [73/100], Step [200/391], Loss: 0.1298
Epoch [73/100], Step [300/391], Loss: 0.1016
Epoch [73/100] Average Loss: 0.1413
Epoch [74/100], Step [100/391], Loss: 0.1955
Epoch [74/100], Step [200/391], Loss: 0.1159
Epoch [74/100], Step [300/391], Loss: 0.1528
Epoch [74/100] Average Loss: 0.1456
Epoch [75/100], Step [100/391], Loss: 0.0716
Epoch [75/100], Step [200/391], Loss: 0.1169
Epoch [75/100], Step [300/391], Loss: 0.1314
Epoch [75/100] Average Loss: 0.1437
Epoch [76/100], Step [100/391], Loss: 0.1082
Epoch [76/100], Step [200/391], Loss: 0.0902
Epoch [76/100], Step [300/391], Loss: 0.1839
Epoch [76/100] Average Loss: 0.1365
Epoch [77/100], Step [100/391], Loss: 0.0936
Epoch [77/100], Step [200/391], Loss: 0.1809
Epoch [77/100], Step [300/391], Loss: 0.1062
Epoch [77/100] Average Loss: 0.1424
Epoch [78/100], Step [100/391], Loss: 0.0930
Epoch [78/100], Step [200/391], Loss: 0.0778
Epoch [78/100], Step [300/391], Loss: 0.1181
Epoch [78/100] Average Loss: 0.1331
Epoch [79/100], Step [100/391], Loss: 0.2048
Epoch [79/100], Step [200/391], Loss: 0.1618
Epoch [79/100], Step [300/391], Loss: 0.1224
Epoch [79/100] Average Loss: 0.1320
Epoch [80/100], Step [100/391], Loss: 0.1632
Epoch [80/100], Step [200/391], Loss: 0.1142
Epoch [80/100], Step [300/391], Loss: 0.1199
Epoch [80/100] Average Loss: 0.1265
Epoch [81/100], Step [100/391], Loss: 0.1348
Epoch [81/100], Step [200/391], Loss: 0.0982
Epoch [81/100], Step [300/391], Loss: 0.1427
Epoch [81/100] Average Loss: 0.1316
Epoch [82/100], Step [100/391], Loss: 0.0876
Epoch [82/100], Step [200/391], Loss: 0.0568
Epoch [82/100], Step [300/391], Loss: 0.1421
Epoch [82/100] Average Loss: 0.1318
Epoch [83/100], Step [100/391], Loss: 0.1803
Epoch [83/100], Step [200/391], Loss: 0.1812
Epoch [83/100], Step [300/391], Loss: 0.1621
Epoch [83/100] Average Loss: 0.1289
Epoch [84/100], Step [100/391], Loss: 0.1234
Epoch [84/100], Step [200/391], Loss: 0.1554
Epoch [84/100], Step [300/391], Loss: 0.1197
Epoch [84/100] Average Loss: 0.1243
Epoch [85/100], Step [100/391], Loss: 0.1325
Epoch [85/100], Step [200/391], Loss: 0.1395
Epoch [85/100], Step [300/391], Loss: 0.1349
Epoch [85/100] Average Loss: 0.1217
Epoch [86/100], Step [100/391], Loss: 0.2149
Epoch [86/100], Step [200/391], Loss: 0.1017
Epoch [86/100], Step [300/391], Loss: 0.0791
Epoch [86/100] Average Loss: 0.1180
Epoch [87/100], Step [100/391], Loss: 0.1437
Epoch [87/100], Step [200/391], Loss: 0.0600
Epoch [87/100], Step [300/391], Loss: 0.0852
Epoch [87/100] Average Loss: 0.1152
Epoch [88/100], Step [100/391], Loss: 0.0999
Epoch [88/100], Step [200/391], Loss: 0.0633
Epoch [88/100], Step [300/391], Loss: 0.1592
Epoch [88/100] Average Loss: 0.1212
Epoch [89/100], Step [100/391], Loss: 0.0949
Epoch [89/100], Step [200/391], Loss: 0.0700
Epoch [89/100], Step [300/391], Loss: 0.1081
Epoch [89/100] Average Loss: 0.1188
Epoch [90/100], Step [100/391], Loss: 0.1545
Epoch [90/100], Step [200/391], Loss: 0.1147
Epoch [90/100], Step [300/391], Loss: 0.1221
Epoch [90/100] Average Loss: 0.1132
Epoch [91/100], Step [100/391], Loss: 0.0893
Epoch [91/100], Step [200/391], Loss: 0.1164
Epoch [91/100], Step [300/391], Loss: 0.1365
Epoch [91/100] Average Loss: 0.1160
Epoch [92/100], Step [100/391], Loss: 0.0584
Epoch [92/100], Step [200/391], Loss: 0.0896
Epoch [92/100], Step [300/391], Loss: 0.1559
Epoch [92/100] Average Loss: 0.1059
Epoch [93/100], Step [100/391], Loss: 0.0844
Epoch [93/100], Step [200/391], Loss: 0.1346
Epoch [93/100], Step [300/391], Loss: 0.1323
Epoch [93/100] Average Loss: 0.1113
Epoch [94/100], Step [100/391], Loss: 0.0869
Epoch [94/100], Step [200/391], Loss: 0.2100
Epoch [94/100], Step [300/391], Loss: 0.1052
Epoch [94/100] Average Loss: 0.1149
Epoch [95/100], Step [100/391], Loss: 0.1137
Epoch [95/100], Step [200/391], Loss: 0.1012
Epoch [95/100], Step [300/391], Loss: 0.0816
Epoch [95/100] Average Loss: 0.1063
Epoch [96/100], Step [100/391], Loss: 0.1184
Epoch [96/100], Step [200/391], Loss: 0.1002
Epoch [96/100], Step [300/391], Loss: 0.1503
Epoch [96/100] Average Loss: 0.1063
Epoch [97/100], Step [100/391], Loss: 0.0832
Epoch [97/100], Step [200/391], Loss: 0.2193
Epoch [97/100], Step [300/391], Loss: 0.1203
Epoch [97/100] Average Loss: 0.1074
Epoch [98/100], Step [100/391], Loss: 0.1044
Epoch [98/100], Step [200/391], Loss: 0.1400
Epoch [98/100], Step [300/391], Loss: 0.1427
Epoch [98/100] Average Loss: 0.1134
Epoch [99/100], Step [100/391], Loss: 0.1164
Epoch [99/100], Step [200/391], Loss: 0.0835
Epoch [99/100], Step [300/391], Loss: 0.1075
Epoch [99/100] Average Loss: 0.1036
Epoch [100/100], Step [100/391], Loss: 0.0605
Epoch [100/100], Step [200/391], Loss: 0.1369
Epoch [100/100], Step [300/391], Loss: 0.0872
Epoch [100/100] Average Loss: 0.1044

Test Accuracy: 81.24%
```

## End
