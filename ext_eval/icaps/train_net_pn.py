from __future__ import division
import matplotlib
import argparse
from datasets.render_shapenet_dataset import *
from models.pn_trainer import *
from config.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
import pprint


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train Category-level Augmented Auto-encoder')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train',
                        default=100, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--save', dest='save_frequency',
                        help='checkpoint saving frequency',
                        default=5, type=int)
    parser.add_argument('--model_dir', dest='model_dir',
                        help='CAD model of the objects',
                        default='../latentnet_dataset/',
                        type=str)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lr_decay', type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    # device
    print('GPU device {:d}'.format(args.gpu_id))
   
    cfg.MODE = 'TRAIN'
    dataset_val = pointnet_render_dataset(args.model_dir, cfg.TRAIN.OBJECTS_CTG, gpu_id=cfg.GPU_ID, mode='val')
    dataset_train = pointnet_render_dataset(args.model_dir, cfg.TRAIN.OBJECTS_CTG, gpu_id=cfg.GPU_ID)

    trainer = PointNet_Trainer(cfg_path=args.cfg_file,
                          model_category=cfg.TRAIN.OBJECTS_CTG,
                          ckpt_path=args.pretrained)

    trainer.train_model(dataset_train, dataset_val,
                      epochs=args.epochs,
                      save_frequency=args.save_frequency,
                      )


