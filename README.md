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

To train HanNet and FCNet on Checkerboard dataset, run this command:

```
python main.py --model hannet --activation ABS --initial orth 
python main.py --model fcnet --activation ReLU --initial kaiming
```

## Experiments on Regression Datasets
```
cd regession_experiments/
```
To train HanNet and FCNet in the paper on Elevators dataset, run this command:
```
python main.py --model hannet --prob elevators  --depth 20 --width 100
python main.py --model fcnet --prob elevators --depth 6 --width 100 
```

## Experiments on Image Datasets

Please save data in the `../data/` directory
```
pip install randaugment
cd regession_experiments/
```
To train MLP/Han-Mixer and ResNet in the paper on Cifar10 dataset, run this command:
```
python train.py --dataset cifar10  --batch-size 256 --epoch 600 --optim adam --lr 0.001 --model hanmixer --channel 256 --mixerblock 4 --hanblock 0 

python train.py --dataset cifar10  --batch-size 256 --epoch 600 --optim adam --lr 0.001 --model hanmixer --channel 256 --mixerblock 4 --hanblock 12

python train.py --dataset cifar10  --batch-size 256 --epoch 600 --optim adam --lr 0.001 --model resnet32 
```
