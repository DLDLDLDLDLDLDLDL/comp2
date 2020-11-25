# Model Evaluation Guide

# Processes

因為檔案路徑(import utils)的關係，必須把Testing的檔案eval/common.py移到上一層資料夾，方可執行使用。並且注意檔案內以下幾個參數
1. phi: 選擇EfficientDet的backbone，必須和training時python3 train.py --phi參數選擇的相同
2. weighted_bifpn: 使否使用weighted BiFPN
3. PascalVocGenerator: 填上testing dataset的path，例如: '/opt/shared-disk2/sychou/comp2/VOCdevkit/test/VOC2007'
4. model_path: 填上訓練好的模型weights，例如: '/home/ccchen/sychou/comp2/comp2/efficent_series/EfficientDet/old_checkpoint/pascal_45_0.3772_0.3917.h5'


# Model Parameters

BUJO+ Environment:
GPU: 2080Ti 10986MB

## 11/24 Training Checkpoints

command:
python3 train.py --snapshot imagenet --phi 1 --weighted-bifpn --gpu 3 --random-transform --compute-val-loss --freeze-backbone --batch-size 32 --steps 150 --epochs 100 pascal /opt/shared-disk2/sychou/comp2/VOCdevkit/VOC2007/

Model:
Backbone: EfficeintNet B1, phi 1

### pascal_99_0.1957_0.1596.h5
The final result(Epoch 100) of the first time training. Training: mAp 0.97, Testing: mAp 0.74

### pascal_98_0.1906_0.1555.h5
The last two result(Epoch 99) of the first time training. mAp 0.97
However, I forget to split train and test into different folders. It probably train on test dataset.
It perhaps overfits.

## 11/25 Training Checkpoints

Phi 6: Ran out of memory with batch size 8
Phi 5: Ran out of memory with batch size 32
Phi 3: Ran out of memory with batch size 16, start training successfully with batch size 8

command:


Model:
Backbone: EfficeintNet B3, phi 3

### pascal_45_0.3772_0.3917.h5
Epoch 45, mAp 0.73. Train 7.5 hours