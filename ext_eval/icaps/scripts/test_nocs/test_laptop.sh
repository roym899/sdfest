#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python test_net_nocs.py \
                              --obj_ctg laptop \
                              --ckpt_folder laptop20200605T205824_default\
                              --n_seq $2;