#!/bin/bash

. /home/transcend/code/BANGKOK/pytorch/bin/activate
cd src || exit
python train.py --batch-size 1000 --epochs 10 --lr 0.0001 --clip 4.0
