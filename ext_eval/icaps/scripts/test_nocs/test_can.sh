#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python test_net_nocs.py \
                              --obj_ctg can \
                              --ckpt_folder can20200605T201536_default\
                              --n_seq $2;