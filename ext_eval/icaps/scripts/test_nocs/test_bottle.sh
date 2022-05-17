#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python test_net_nocs.py \
                              --obj_ctg bottle \
                              --ckpt_folder bottle20200608T172228_default\
                              --n_seq $2;