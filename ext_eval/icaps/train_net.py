from __future__ import division
import matplotlib
matplotlib.use('Agg')
import argparse
from datasets.render_shapenet_dataset import *
from datasets.distractor import *
from models.aae_trainer import *
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
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='ycb', type=str)
    parser.add_argument('--train_dir', dest='train_dir',
                        help='relative dir of training set',
                        default='../Dataset/train/',
                        type=str)
    parser.add_argument('--val_dir', dest='val_dir',
                        help='relative dir of validation set',
                        default='../Dataset/val/',
                        type=str)
    parser.add_argument('--dis_dir', dest='dis_dir',
                        help='relative dir of the distration set',
                        default='../coco/val2017',
                        type=str)
    parser.add_argument('--model_dir', dest='model_dir',
                        help='CAD model of the objects',
                        default='../category-level_models/',
                        type=str)

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
    print(args.dataset_name)
   
    cfg.MODE = 'TRAIN'
    dataset_val = shapenet_render_dataset(args.model_dir, cfg.TRAIN.OBJECTS_CTG, gpu_id=cfg.GPU_ID, mode='val')
    dataset_train = shapenet_render_dataset(args.model_dir, cfg.TRAIN.OBJECTS_CTG, gpu_id=cfg.GPU_ID)
    
    trainer = AAE_Trainer(cfg_path=args.cfg_file,
                          model_category=cfg.TRAIN.OBJECTS_CTG,
                          ckpt_path=args.pretrained)

    dataset_dis = DistractorDataset(args.dis_dir, cfg.TRAIN.CHM_RAND_LEVEL,
                                    size_crop=(cfg.TRAIN.INPUT_IM_SIZE[1],
                                               cfg.TRAIN.INPUT_IM_SIZE[0]))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        trainer.AAE.encoder = nn.DataParallel(trainer.AAE.encoder)
        trainer.AAE.decoder = nn.DataParallel(trainer.AAE.decoder)

    trainer.train_model(dataset_train, dataset_val,
                      epochs=args.epochs,
                      dstr_dataset=dataset_dis,
                      save_frequency=args.save_frequency,
                      )


