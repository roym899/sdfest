#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python test_net_nocs.py \
                              --obj_ctg camera \
                              --ckpt_folder camera20200603T175729_default\
                              --n_seq $2;