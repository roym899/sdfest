import argparse
import matplotlib
import sys
from pose_rbpf.pose_rbpf import *
from datasets.nocs_real_dataset import *
from config.config import cfg, cfg_from_file
import pprint
import glob
import copy

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test Pose Estimation Network with Multiple Object Models')
    parser.add_argument('--obj_ctg', dest='obj_ctg',
                        help='test object category',
                        required=True, type=str)
    parser.add_argument('--ckpt_folder', dest='ckpt_folder',
                        help='checkpoint folder',
                        required=True, type=str)
    parser.add_argument('--pf_config_dir', dest='pf_cfg_dir',
                        help='directory for poserbpf configuration files',
                        default='./config/pf_cfgs/', type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--dataset_dir', dest='dataset_dir',
                        help='relative dir of the NOCS dataset',
                        default='../NOCS_dataset/real_test/',
                        type=str)
    parser.add_argument('--dataset_gt_dir', dest='dataset_gt_dir',
                        help='relative dir of the ground truth pose of NOCS dataset',
                        default='../NOCS_dataset/gts/real_test/',
                        type=str)
    parser.add_argument('--model_real_dir', dest='model_real_dir',
                        help='directory of objects',
                        default='../obj_models/real_test/',
                        type=str)
    parser.add_argument('--n_seq', dest='n_seq',
                        help='index of sequence',
                        default=1,
                        type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    print(args)

    # load the configurations
    ckpt_folder_full = './checkpoints/aae_ckpts/{}/'.format(args.ckpt_folder)
    train_cfg_file = ckpt_folder_full + 'config.yml'
    cfg_from_file(train_cfg_file)
    test_cfg_file = args.pf_cfg_dir+args.obj_ctg+'.yml'
    cfg_from_file(test_cfg_file)

    # load the test objects
    print('Testing with objects: ')
    print(cfg.TEST.OBJECTS)
    obj_list = cfg.TEST.OBJECTS

    # pf config files
    cfg_list = []
    cfg_list.append(copy.deepcopy(cfg))
    pprint.pprint(cfg_list)

    # dataset
    data_list_file = './datasets/nocs_real_eval/{}/seq{}.txt'.format(args.obj_ctg, args.n_seq)
    dataset = nocs_real_dataset(args.obj_ctg, data_list_file, args.dataset_dir, args.dataset_gt_dir, args.model_real_dir)
    
    # setup the poserbpf
    pose_rbpf = PoseRBPF(obj_list, cfg_list, args.ckpt_folder, dataset.obj_instance)
    target_obj = cfg.TEST.OBJECTS[0]
    pose_rbpf.set_target_obj(target_obj)
    
    # run nocs dataset
    pose_rbpf.run_nocs_dataset(dataset, args.n_seq)
