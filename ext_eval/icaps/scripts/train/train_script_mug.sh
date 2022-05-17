#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_net.py --cfg ./config/training_cfg/train_mug.yml --epochs 300 --save 100