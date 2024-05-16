@echo off
python train.py -l ssim -p 0.1 -c 9 -d cifar10 -t True -e 100
python train.py -l ssim -p 0.2 -c 0  -d cifar10 -t True -e 100
python train.py -l ssim -p 0.5 -c 0 -d cifar10 -t True -e 100
python train.py -l ssim -p 1.0 -c 0 -d cifar10 -t True -e 100