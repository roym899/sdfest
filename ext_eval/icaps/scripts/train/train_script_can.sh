#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_net.py --cfg ./config/training_cfg/train_can.yml --epochs 100 --save 100 \
                              --pretrained ./checkpoints/can20200605T201536_default/ckpt_can_0200.pth