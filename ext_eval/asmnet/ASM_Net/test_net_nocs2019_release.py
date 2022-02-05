import sys
import datetime
import os
import os.path as osp

sys.path.append("..")
import numpy as np
import glob
import copy
import random
import cv2
import open3d as o3
import argparse

import torch
import common3Dfunc as c3D
from loss import quaternion2rotationPT

import _pickle as cPickle
from nocs_configs import InferenceConfig, synset_names, class_map
from NOCS_loader.dataset import NOCSDataset
from NOCS_loader import utils
import cr6d_utils

seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def get_arguments():
    """
    Parse arguments from command line
    """

    parser = argparse.ArgumentParser(description="Test net on NOCS CVPR2019 dataset")
    parser.add_argument(
        "--proot",
        type=str,
        default="../params",
        help="path to the parameter root (asm params and weights)",
    )
    parser.add_argument(
        "--droot",
        type=str,
        default="../dataset",
        help="path to the dataset root (nocs2019 dataset and object masks)",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=800,
        help="number of point to be input the network",
    )
    parser.add_argument(
        "--ddim", type=int, default=3, help="dimension of deformation parameter"
    )
    parser.add_argument(
        "--use_icp",
        action="store_true",
        help="flag to use ICP algorithm for post processing",
    )
    parser.add_argument(
        "--use_gt_mask",
        action="store_true",
        help="flag to use identical mask of object",
    )
    parser.add_argument(
        "--use_mean_shape", action="store_true", help="flag to use mean shape"
    )
    parser.add_argument("--out_dir", type=str, default="None", help="output dir name")
    parser.add_argument("--vis", action="store_true", help="make result image")

    return parser.parse_args()


args = get_arguments()
print(args)

dataset_root = args.droot
param_root = args.proot
asm_dim = args.ddim
n_points = args.npoints
use_icp = args.use_icp
mode = "pred"  # pred/gt
if args.use_gt_mask:
    mode = "gt"
vis = args.vis
use_mean_shape = False
use_mean_shape = args.use_mean_shape


config = InferenceConfig()


def random_sample(data, n_sample):

    if n_sample < data.shape[0]:
        choice = random.sample(list(np.arange(0, data.shape[0], 1)), k=n_sample)
    else:
        choice = random.choices(list(np.arange(0, data.shape[0], 1)), k=n_sample)

    sampled = np.array(data[choice])
    return copy.deepcopy(sampled)


# set path
dataset_real_test = NOCSDataset(synset_names, "test", config)
real_dir = os.path.join(dataset_root, "real")
dataset_real_test.load_real_scenes(real_dir)
dataset_real_test.prepare(class_map)
dataset = dataset_real_test
scene_mask_root = osp.join(dataset_root, "masks_real_test")
scene_img_root = osp.join(dataset_root, "real_test")
model_root = osp.join(dataset_root, "obj_models", "real_test")
intrinsic_path = osp.join(dataset_root, "intrinsic.json")
camera_intrinsic = o3.io.read_pinhole_camera_intrinsic(intrinsic_path)
gt_dir = os.path.join(dataset_root, "gts", "real_test")
image_ids = dataset.image_ids
print(len(image_ids), "images")

now = datetime.datetime.now().strftime("%Y%m%d%H%M")
if args.out_dir != "None":
    now = args.out_dir
if use_mean_shape:
    save_dir = os.path.join("release_mean_out", "{}_woICP_{}".format("real_test", now))
    if use_icp:
        save_dir = os.path.join(
            "release_mean_out", "{}_wICP_{}".format("real_test", now)
        )
else:
    save_dir = os.path.join(
        "release_proposed_out", "{}_woICP_{}".format("real_test", now)
    )
    if use_icp:
        save_dir = os.path.join(
            "release_proposed_out", "{}_wICP_{}".format("real_test", now)
        )

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load params
asmds = cr6d_utils.load_asmds(osp.join(param_root, "asm_params"), synset_names)
masks_list = sorted(glob.glob(osp.join(scene_mask_root, "*.npz")))
len(masks_list)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models = cr6d_utils.load_models_release(
    osp.join(param_root, "weights"), synset_names, asm_dim, n_points, device
)

# pose estimation
count = 0
mapping = c3D.Mapping(intrinsic_path)
for i, image_id in enumerate(image_ids):
    count += 1
    image_path = dataset.image_info[image_id]["path"]
    print(image_path)
    # Mask-RCNN path
    ipp = image_path.split("/")
    mask_path = osp.join(
        scene_mask_root, "results_{}_{}_{}.npz".format(ipp[-3], ipp[-2], ipp[-1])
    )

    # record results
    result = {}
    # loading ground truth
    im_c = dataset.load_image(image_id)  # H, W, 3
    im_d = dataset.load_depth(image_id)
    mask_info = np.load(mask_path)
    scale_coeffs, model_cids, gt_pcds = cr6d_utils.get_model_scale(
        image_path, model_root
    )

    gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(
        image_id
    )
    gt_bbox = utils.extract_bboxes(gt_mask)

    result["image_id"] = image_id
    result["image_path"] = image_path

    result["gt_class_ids"] = gt_class_ids
    result["gt_bboxes"] = gt_bbox
    result["gt_RTs"] = None
    result["gt_scales"] = gt_scales

    # Get GT pose
    image_path_parsing = image_path.split("/")
    gt_pkl_path = os.path.join(
        gt_dir,
        "results_{}_{}_{}.pkl".format(
            "real_test", image_path_parsing[-2], image_path_parsing[-1]
        ),
    )
    print(gt_pkl_path)
    if os.path.exists(gt_pkl_path):
        with open(gt_pkl_path, "rb") as f:
            gt = cPickle.load(f)
        result["gt_RTs"] = gt["gt_RTs"]
        if "handle_visibility" in gt:
            result["gt_handle_visibility"] = gt["handle_visibility"]
            assert len(gt["handle_visibility"]) == len(gt_class_ids)
            print("got handle visibiity.")
        else:
            result["gt_handle_visibility"] = np.ones_like(gt_class_ids)

    # Detection
    im_masks, detected_ids = cr6d_utils.get_mask(mask_info, mode)
    if len(im_masks) == 0:
        # print("CAUTION!! object mask is not detected.")
        continue
    # bbox = utils.extract_bboxes(im_m[:,:,np.newaxis])
    bbox = utils.extract_bboxes(im_masks)

    pred_scales = []  # Predicted scales
    pred_rts = []  # Predicted poses
    pred_scores = []  # 1
    bboxes = []  # bounding box
    class_ids = []  # class ID
    pcd_pred_vises = []  # Reconstructed pcd
    pred_pcds = []  # Reconstructed pcd (initial pose)
    im_masks = im_masks.transpose((2, 0, 1))
    for m, im_m in enumerate(im_masks):

        # get instance id
        overlap_pix = []
        for n in range(gt_mask.shape[2]):
            overlap_pix.append(np.sum(im_m * gt_mask[:, :, n]))
        instance_id = np.argmax(overlap_pix)

        # Generate and noise reduction for detected masks.
        im_md = im_d * im_m  # apply mask
        im_md = c3D.image_statistical_outlier_removal(im_md, factor=2.0)
        pcd_obj = cr6d_utils.get_pcd_from_rgbd(
            im_c.copy(), im_md.copy(), camera_intrinsic
        )
        [pcd_obj, _] = pcd_obj.remove_statistical_outlier(100, 2.0)
        pcd_in = copy.deepcopy(pcd_obj)
        pcd_c, offset = c3D.centering(pcd_in)
        pcd_n, scale = c3D.size_normalization(pcd_c)

        # open3d pcd -> tensor
        np_pcd = np.array(pcd_n.points)
        np_input = random_sample(np_pcd, n_points)
        np_input = np_input.astype(np.float32)
        t_input = torch.from_numpy(np_input)

        with torch.no_grad():
            # perform inference on GPU
            class_name = synset_names[detected_ids[m]]
            points = t_input
            points = points.unsqueeze(0)
            points = points.transpose(2, 1)
            if torch.cuda.is_available():
                points = points.cuda()
            dparam_pred, q_pred = models[class_name](points)
            dparam_pred = dparam_pred.cpu().numpy().squeeze()
            pred_rot = quaternion2rotationPT(q_pred)
            pred_rot = pred_rot.cpu().numpy().squeeze()
            pred_dp_param = dparam_pred[:-1]
            pred_scaling_param = dparam_pred[-1]

            pcd_pred = None
            if use_mean_shape is True:
                pcd_pred = asmds[class_name].deformation([0])
            else:
                pcd_pred = asmds[class_name].deformation(pred_dp_param)
                pcd_pred = pcd_pred.remove_statistical_outlier(20, 1.0)[0]
                print(pred_scaling_param)
                pcd_pred.scale(pred_scaling_param, (0.0, 0.0, 0.0))

            # convert real scale
            pcd_pred_vis = copy.deepcopy(pcd_pred)
            pcd_pred_vis.scale(scale, (0.0, 0.0, 0.0))

            pred_pcd = copy.deepcopy(pcd_pred_vis)
            pred_pcds.append(pred_pcd)

            bb = pcd_pred_vis.get_axis_aligned_bounding_box()
            bbox3d = bb.get_max_bound() - bb.get_min_bound()

            pcd_pred_vis.rotate(pred_rot)
            pcd_pred_vis.translate(offset)

            pcd_pred.rotate(pred_rot)

            # ICP
            rt_icp = np.identity(4)
            if use_icp:
                pcd_pred_vis_ds = pcd_pred_vis.voxel_down_sample(0.005)
                pcd_pred_visible = c3D.applyHPR(pcd_pred_vis_ds)
                pcd_in = pcd_in.voxel_down_sample(0.005)
                reg_result = o3.pipelines.registration.registration_icp(
                    pcd_pred_visible, pcd_in, max_correspondence_distance=0.02
                )
                pcd_pred_vis = copy.deepcopy(pcd_pred_vis_ds).transform(
                    reg_result.transformation
                )
                rt_icp = reg_result.transformation
            pcd_pred_vises.append(pcd_pred_vis)

            # 4x4 matrix
            pred_rt = np.identity(4)
            pred_rt[:3, :3] = pred_rot.copy()
            if use_icp:  # If apply ICP, then update pose
                pred_rt = np.dot(reg_result.transformation, pred_rt)
            # Compute translation as bounding box center
            maxb = pcd_pred_vis.get_max_bound()  # bbox max
            minb = pcd_pred_vis.get_min_bound()  # bbox min
            center = (maxb - minb) / 2 + minb  # bbox center
            pred_rt[:3, 3] = center.copy()

            # convert to NOCS format matrix
            s = np.identity(4)
            s[0, 0] = s[1, 1] = s[2, 2] = 1 / scale_coeffs[instance_id]
            nocs_rt = np.dot(pred_rt, np.linalg.inv(s))

            o3.visualization.draw_geometries([pcd_pred_vis, pcd_in])

            # Set result
            pred_rts.append(nocs_rt.copy())
            pred_scales.append(bbox3d / scale_coeffs[instance_id])
            pred_scores.append(1.0)
            class_ids.append(detected_ids[m])
            bboxes.append(bbox[m])

    result["pred_class_ids"] = np.asarray(class_ids)
    result["pred_bboxes"] = np.asarray(bboxes)
    result["pred_RTs"] = np.asarray(pred_rts)
    result["pred_scales"] = np.asarray(pred_scales)
    result["pred_scores"] = np.asarray(pred_scores)

    # Save result
    path_parse = image_path.split("/")
    image_short_path = "_".join(path_parse[-3:])

    save_path = os.path.join(save_dir, "results_{}.pkl".format(image_short_path))
    with open(save_path, "wb") as f:
        cPickle.dump(result, f)
    print(
        "Results of image {} has been saved to {}.".format(image_short_path, save_path)
    )

    # Visualization
    im_imposed = im_c.copy()
    pcd_scene = None
    if vis is True:
        pcd_scene = cr6d_utils.get_pcd_from_rgbd(
            im_c.copy(), im_d.copy(), camera_intrinsic
        )
        im_objs = np.zeros(im_c.shape).astype(np.float)
        for pcd in pcd_pred_vises:
            img = mapping.Cloud2Image(pcd, True).astype(np.float)
            im_objs += img
        im_imposed = ((im_objs.astype(np.float) + im_c.astype(np.float)) / 2).astype(
            np.uint8
        )
        ofname = "results_{}.png".format(image_short_path)
        opath = osp.join(save_dir, ofname)
        cv2.imwrite(opath, im_imposed)
        # open3d visualization
        # o3.visualization.draw_geometries([pcd_scene, pcd_pred_vis]))
