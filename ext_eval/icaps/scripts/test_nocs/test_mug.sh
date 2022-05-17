#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python test_net_nocs.py \
                              --obj_ctg mug \
                              --ckpt_folder mug20200529T111737_default\
                              --n_seq $2;