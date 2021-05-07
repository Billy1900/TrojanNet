# TrojanNet
This is the Pytorch implementation of [TrojanNet: Embedding Hidden Trojan Horse Models in Neural Networks](https://arxiv.org/abs/2002.10078).

## 1. Dependencies:
<pre>
Pytorch 1.8.0
imgaug 0.4.0
</pre>

## 2. Example: training on public task and secret task. 

### 2.1 Run TrojanResnet50 on CIFAR10 and SVHN. 
```
python train.py --epochs 300 --datasets_name cifar10 svhn --model trojan_resnet50 --seed 0 --data_root ./data --save_dir checkpoint
```
### 2.2 Run TrojanResnet50 on CIFAR10, CIFAR100, SVHN and GTSRB.
```
python train.py --epochs 300 --datasets_name cifar10 cifar100 svhn gtsrb --model trojan_resnet50 --seed 0 --data_root ./data --save_dir checkpoint
```
