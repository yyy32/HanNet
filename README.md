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

