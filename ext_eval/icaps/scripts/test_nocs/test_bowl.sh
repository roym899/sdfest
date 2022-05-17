#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python test_net_nocs.py \
                              --obj_ctg bowl \
                              --ckpt_folder bowl20200603T175721_default\
                              --n_seq $2;