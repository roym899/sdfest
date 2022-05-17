#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_net_pn.py --cfg ./config/training_cfg/train_laptop.yml --epochs 300 --save 100