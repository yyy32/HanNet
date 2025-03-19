### This repository is the official implementation of [A Lightweight and Gradient-Stable Neural Layer].

## Requirements
```
--python >= 3.6
--pytorch >=1.8
--numpy
--pandas
--matplotlib
```
## Experiments on Checkerboard

```
cd checkerboard_experiments/
```

To train HanNet and FCNet on Checkerboard dataset, run commands:

```
python main.py --model hannet --activation ABS --initial orth 

python main.py --model fcnet --activation ReLU --initial kaiming
```

## Experiments on Regression Datasets
```
cd regession_experiments/
```
To train HanNet and FCNet in the paper on Elevators dataset, run commands:
```
python main.py --model hannet --prob elevators  --depth 20 --width 100

python main.py --model fcnet --prob elevators --depth 6 --width 100 
```

## Experiments on Image Datasets

Please save datasets in the `../data/`, then
```
pip install randaugment
cd image_experiments/
```
To train MLP/Han-Mixer and ResNet in the paper on Cifar10 dataset, run commands:
```
python train.py --dataset cifar10  --batch-size 256 --epoch 600 --optim adam --lr 0.001 --model hanmixer --channel 256 --mixerblock 4 --hanblock 0 

python train.py --dataset cifar10  --batch-size 256 --epoch 600 --optim adam --lr 0.001 --model hanmixer --channel 256 --mixerblock 4 --hanblock 12

python train.py --dataset cifar10  --batch-size 256 --epoch 600 --optim adam --lr 0.001 --model resnet32 
```

To train Mobile-ViT XXS w/o Han block on Cifar10 dataset, please 
```
pip install einops
```
Then run commands:
```
python train-mobilevit.py --dataset cifar10  --batch-size 128 --epoch 300 --optim adam --model xxs --lr 0.001 --augment 1 

python train-mobilevit.py --dataset cifar10  --batch-size 128 --epoch 300 --optim adam --model hxxs --lr 0.001 --augment 1 
```

## Citations
```
@article{yu2024lightweight,
  title={A lightweight and gradient-stable neural layer},
  author={Yu, Yueyao and Zhang, Yin},
  journal={Neural Networks},
  volume={175},
  pages={106269},
  year={2024},
  publisher={Elsevier}
}
```
