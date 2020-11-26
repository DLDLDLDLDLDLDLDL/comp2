# EfficientDet

Our implement is under ```ours``` directory.

## Reference

[Tensorflow Custom layers](https://www.tensorflow.org/tutorials/customization/custom_layers)

[如何使用Keras fit和fit_generator（动手教程)](https://blog.csdn.net/learning_tortosie/article/details/85243310)

[How to Train Your Keras Model (Dramatically Faster)](https://towardsdatascience.com/how-to-train-your-model-dramatically-faster-9ad063f0f718)

[Distributed training with TensorFlow](https://www.tensorflow.org/guide/distributed_training)

## Model Training Guide

### ***Command flags***
- --weighted-bifpn: Use Weighted BiFPN 
- --snapshot: specify checkpoint or pretrained model, you can use imagenet as pretrained backbone
- --phi: determine which model to use, you can choose 0(EfficientDet D0) ~ 6(EfficientDet D6) 
- --freeze-backbone: Fix backbone weight
- --steps: determine the number of steps in a epoch
- --epochs: training epochs
- pascal: use Pascal VOC dataset format and give dataset path to VOC2007 directory

### ***Command***
For example, train EfficicentDet D0 with ImageNet pretrained backbone and Weighted BiFPN

```python3 train.py --snapshot imagenet --phi 1 --weighted-bifpn --gpu 3 --random-transform --compute-val-loss --freeze-backbone --batch-size 32 --steps 150 --epochs 100 pascal /opt/shared-disk2/sychou/comp2/VOCdevkit/VOC2007/```

## Model Evaluation Guide
### ***Processes***

因為檔案路徑(import utils)的關係，必須把Testing的檔案eval/common.py移到上一層資料夾，方可執行使用。並且注意檔案內以下幾個參數
1. phi: 選擇EfficientDet的backbone，必須和training時python3 train.py --phi參數選擇的相同
2. weighted_bifpn: 使否使用weighted BiFPN
3. PascalVocGenerator: 填上testing dataset的path，例如: '/opt/shared-disk2/sychou/comp2/VOCdevkit/test/VOC2007'
4. model_path: 填上訓練好的模型weights，例如: '/home/ccchen/sychou/comp2/comp2/efficent_series/EfficientDet/old_checkpoint/pascal_45_0.3772_0.3917.h5'
   
### ***Evaluation Results***

BUJO+ Environment:
- GPU: 2080Ti 10986MB
- Python 3.6.8 :: Anaconda, Inc.
- Tensorflow: 2.2.0, with GPU support
- Cuda compilation tools, release 10.0, V10.0.130

[***Here***](https://drive.google.com/drive/folders/1xFXLxnsf_HqB7hy66OikNNzLpItV3ctG) is the  trained checkpoint.

### 11/24 Training Checkpoints

- command:

```python3 train.py --snapshot imagenet --phi 1 --weighted-bifpn --gpu 3 --random-transform --compute-val-loss --freeze-backbone --batch-size 32 --steps 150 --epochs 100 pascal /opt/shared-disk2/sychou/comp2/VOCdevkit/VOC2007/```

- Backbone: EfficeintNet B1, phi 1

#### ***pascal_99_0.1957_0.1596.h5***
The final result(Epoch 100) of the first time training. Training: mAp 0.97, Testing: mAp 0.74

#### ***pascal_98_0.1906_0.1555.h5***
The last two result(Epoch 99) of the first time training. mAp 0.97
However, I forget to split train and test into different folders. It probably train on test dataset.
It perhaps overfits.

---
### 11/25 Training Checkpoints

- Phi 6: Ran out of memory with batch size 8
- Phi 5: Ran out of memory with batch size 32
- Phi 3: Ran out of memory with batch size 16, start training successfully with batch size 8

- command:

```python3 train.py --snapshot imagenet --phi 3 --weighted-bifpn --gpu 3 --random-transform --compute-val-loss --freeze-backbone --batch-size 8 --steps 150 --epochs 100 pascal /opt/shared-disk2/sychou/comp2/VOCdevkit/trainval/VOC2007/```

- Backbone: EfficeintNet B3, phi 3

#### ***pascal_45_0.3772_0.3917.h5***
Epoch 45, mAp 0.73. Train 7.5 hours

#### ***pascal_100_0.2464_0.2470.h5***
Epoch 100, 0.9399, Train 15 hours

Testing mAP: 0.7704
