#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_net_pn.py --cfg ./config/training_cfg/train_bowl.yml --epochs 61 --save 100