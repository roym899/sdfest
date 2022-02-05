# for loading NOCS dataset
import os, sys
import numpy as np
sys.path.append("../NOCS_loader")
from config import Config

class ScenesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ShapeNetTOI"
    OBJ_MODEL_DIR = os.path.join('..', 'dataset', 'obj_models')
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 6 object categories
    MEAN_PIXEL = np.array([[ 120.66209412, 114.70348358, 105.81269836]])

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    WEIGHT_DECAY = 0.0001
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    COORD_LOSS_SCALE = 1
    
    COORD_USE_BINS = True
    if COORD_USE_BINS:
         COORD_NUM_BINS = 32
    else:
        COORD_REGRESS_LOSS   = 'Soft_L1'
   
    COORD_SHARE_WEIGHTS = False
    COORD_USE_DELTA = False

    COORD_POOL_SIZE = 14
    COORD_SHAPE = [28, 28]

    USE_BN = True
#     if COORD_SHARE_WEIGHTS:
#         USE_BN = False

    USE_SYMMETRY_LOSS = True


    RESNET = "resnet50"
    TRAINING_AUGMENTATION = True
    SOURCE_WEIGHT = [3, 1, 1] #'ShapeNetTOI', 'Real', 'coco'

use_regression = False
use_delta = False

class InferenceConfig(ScenesConfig):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    COORD_USE_REGRESSION = use_regression
    if COORD_USE_REGRESSION:
        COORD_REGRESS_LOSS   = 'Soft_L1' 
    else:
        COORD_NUM_BINS = 32
    COORD_USE_DELTA = use_delta

    USE_SYMMETRY_LOSS = True
    TRAINING_AUGMENTATION = False


synset_names = ['BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ]

class_map = {
    'bottle': 'bottle',
    'bowl':'bowl',
    'cup':'mug',
    'laptop': 'laptop',
}
