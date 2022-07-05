"""Simple script to run inference on real data.

Usage (evaluation on random RGB-D images from folder):
    python -m sdfest.estimation.scripts.real_data \
        --config estimation/configs/rgbd_objects_uw.yaml estimation/configs/mug.yaml \
        --folder data/rgbd_objects_uw/coffee_mug/

Usage (evaluation on single RGB image from Redwood or RGB-D objects dataset):
    python -m sdfest.estimation.scripts.real_data \
        --config configs/rgbd_objects_uw.yaml configs/mug.yaml \
        --input rgbd_objects_uw/coffee_mug/coffee_mug_1/coffee_mug_1_1_103.png

Specific parameters:
    measure_runtime:
        if True, a breakdown of the runtime will be generated
        only supported for single input
    out_folder:
        if provided and measure_runtime is true, the runtime results are written to file
    visualize_optimization: whether to visualize optimization while at it
    visualize_input: whether to visualize the input
    create_animation:
        If true, three animations will be created. One for depth optimization, depth
        error, and mesh.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import copy
import random
import os
import glob
import time
from typing import Dict, List, Callable, Tuple
import functools

import detectron2
import detectron2.engine
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import open3d as o3d
import pickle
import torch
import matplotlib.pyplot as plt
import yoco

import sdfest
from sdfest.estimation.simple_setup import SDFPipeline, NoDepthError


def load_real275_rgbd(rgb_path: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load RGB-D image from RGB path.

    Args:
        rgb_path: path to RGB image

    Returns:
        Tuple containing:
            - The color image, float32, RGB, 0-1, shape (H,W,C).
            - The depth image, float32, in meters, shape (H,W).
            - The color path.
            - The depth path.
    """
    depth_path = rgb_path[:-10] + "_depth.png"
    color_img = np.asarray(o3d.io.read_image(rgb_path), dtype=np.float32) / 255
    depth_img = (
        np.asarray(
            o3d.io.read_image(depth_path),
            dtype=np.float32,
        )
        * 0.001
    )
    return color_img, depth_img, rgb_path, depth_path


def load_real275_sample(folder: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load a sample from RGBD Object dataset.

    https://rgbd-dataset.cs.washington.edu/dataset/

    Args:
        folder: The root folder of the dataset.

    Returns:
        See load_real275_rgbd.
    """
    files = glob.glob(folder + "/**/*color.png", recursive=True)
    rgb_path = random.choice(files)
    return load_real275_rgbd(rgb_path)


def load_rgbd_object_uw_rgbd(rgb_path: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load RGB-D image from RGB path.

    Args:
        rgb_path: path to RGB image

    Returns:
        Tuple containing:
            - The color image, float32, RGB, 0-1, shape (H,W,C).
            - The depth image, float32, in meters, shape (H,W).
            - The color path.
            - The depth path.
    """
    depth_path = rgb_path[:-4] + "_depth" + rgb_path[-4:]
    color_img = np.asarray(o3d.io.read_image(rgb_path), dtype=np.float32) / 255
    depth_img = (
        np.asarray(
            o3d.io.read_image(depth_path),
            dtype=np.float32,
        )
        * 0.001
    )
    return color_img, depth_img, rgb_path, depth_path


def load_rgbd_object_uw_sample(folder: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load a sample from RGBD Object dataset.

    https://rgbd-dataset.cs.washington.edu/dataset/

    Args:
        folder: The root folder of the dataset.

    Returns:
        See load_rgbd_object_uw_rgbd.
    """
    files = glob.glob(folder + "/**/*[0-9].png", recursive=True)
    rgb_path = random.choice(files)
    return load_rgbd_object_uw_rgbd(rgb_path)


def load_redwood_rgbd(rgb_path: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load RGB-D image from RGB path of Redwood dataset.

    Args:
        rgb_path: path to RGB image

    Returns:
        Tuple containing:
            - The color image, float32, RGB, 0-1, shape (H,W,C).
            - The depth image, float32, in meters, shape (H,W).
            - The color path.
            - The depth path.
    """
    depth_dir = os.path.join(os.path.dirname(rgb_path), "..", "depth")

    rgb_timestamp = int(rgb_path[-16:-4])

    # find closest depth image in time
    depth_paths = glob.glob(depth_dir + "/*.png")
    depth_timestamps = np.array([int(p[-16:-4]) for p in depth_paths])
    ind = np.argmin(np.abs(depth_timestamps - rgb_timestamp))
    depth_path = depth_paths[ind]

    color_img = np.asarray(o3d.io.read_image(rgb_path), dtype=np.float32) / 255

    depth_img = (
        np.asarray(
            o3d.io.read_image(depth_path),
            dtype=np.float32,
        )
        * 0.001
    )
    return color_img, depth_img, rgb_path, depth_path


def load_redwood_sample(folder: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load a sample from Redwood dataset.

    Args:
        folder: The root folder of the dataset.

    Returns:
        Tuple containing:
            - The color image, float32, RGB, 0-1, shape (H,W,C).
            - The depth image, float32, in meters, shape (H,W).
            - The color path.
            - The depth path.
    """
    sequence_paths = glob.glob(folder + "/*")
    sequence_path = random.choice(sequence_paths)

    rgb_dir = os.path.join(sequence_path, "rgb")
    rgb_paths = glob.glob(rgb_dir + "/*.jpg")
    rgb_path = random.choice(rgb_paths)

    return load_redwood_rgbd(rgb_path)


def load_sample_from_folder(config: dict) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load a sample from dataset specified in config.

    See the dataset specific load functions for more details of the expected folder
    structure.

    Params:
        config: Configuration dictionary that must contain the following keys:
            "dataset": one of "redwood" | "rgbd_object_uw"
            "folder": the root folder of the dataset
    Returns:
        Tuple containing:
            - The color image, float32, RGB, 0-1, shape (H,W,C).
            - The depth image, float32, in meters, shape (H,W).
    """
    if config["dataset"] == "redwood":
        return load_redwood_sample(config["folder"])
    elif config["dataset"] == "rgbd_object_uw":
        return load_rgbd_object_uw_sample(config["folder"])
    elif config["dataset"] == "real275":
        return load_real275_sample(config["folder"])
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} is not supported")


def _timing_decorator(timing_dict: defaultdict, name: str, f: Callable) -> Callable:
    @functools.wraps(f)
    def timing_wrapper(*args, **kwargs):
        before = time.time()
        result = f(*args, **kwargs)
        torch.cuda.synchronize()
        after = time.time()
        timing_dict[name].append([before, after])
        return result

    return timing_wrapper


def add_timing_decorators(pipeline: SDFPipeline) -> defaultdict:
    timing_dict = defaultdict(list)
    pipeline._nn_init = _timing_decorator(timing_dict, "init", pipeline._nn_init)
    pipeline.vae.decode = _timing_decorator(timing_dict, "decode", pipeline.vae.decode)
    pipeline.render = _timing_decorator(timing_dict, "render", pipeline.render)
    pipeline._compute_gradients = _timing_decorator(
        timing_dict, "backward", pipeline._compute_gradients
    )
    pipeline._compute_view_losses = _timing_decorator(
        timing_dict, "losses", pipeline._compute_view_losses
    )
    return timing_dict


def load_rgbd(config: dict) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load a single RGB-D image from path and dataset specified in config.

    See the dataset specific load functions for more details of the expected folder
    structure.

    Params:
        config: Configuration dictionary that must contain the following keys:
            "dataset": one of "redwood" | "rgbd_object_uw"
            "input": the path to the RGB image
    Returns:
        Tuple containing:
            - The color image, float32, RGB, 0-1, shape (H,W,C).
            - The depth image, float32, in meters, shape (H,W).
            - The color path.
            - The depth path.
    """
    if config["dataset"] == "redwood":
        return load_redwood_rgbd(config["input"])
    elif config["dataset"] == "rgbd_object_uw":
        return load_rgbd_object_uw_rgbd(config["input"])
    elif config["dataset"] == "real275":
        return load_real275_rgbd(config["input"])
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} is not supported")


def str_to_bool(v: str) -> bool:
    """Try to convert string to boolean.

    From: https://stackoverflow.com/a/43357954
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def generate_runtime_overview(config, timing_dicts: List[Dict]) -> None:
    total_runs = len(timing_dicts) / 2  # runs per setting, 2 settings
    stats_dicts = {
        True: defaultdict(lambda: {"total": 0.0, "total_calls": 0}),
        False: defaultdict(lambda: {"total": 0.0, "total_calls": 0}),
    }
    for timing_dict in timing_dicts:
        for name, timings in timing_dict.items():
            if name == "shape_optimization":
                continue
            for timing in timings:
                so = timing_dict["shape_optimization"]
                stats_dicts[so][name]["total"] += timing[1] - timing[0]
                stats_dicts[so][name]["total_calls"] += 1

    for stats_dict in stats_dicts.values():
        for name, stats in stats_dict.items():
            stats["mean"] = stats["total"] / stats["total_calls"]
            stats["calls_per_run"] = stats["total_calls"] / total_runs
            stats["total_per_run"] = stats["total"] / total_runs

    # save to file
    if "out_folder" in config and config["out_folder"] is not None:
        os.makedirs(config["out_folder"], exist_ok=True)
        filename = (
            f"runtime_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.yaml"
        )
        out_path = os.path.join(config["out_folder"], filename)
        combined_dict = {
            **config,
            "results_with_decode": dict(stats_dicts[True]),
            "results_without_decode": dict(stats_dicts[False]),
        }
        yoco.save_config_to_file(out_path, combined_dict)


def main() -> None:
    """Entry point of the program."""
    # define the arguments
    parser = argparse.ArgumentParser(description="SDF pose estimation in real data")

    # parse arguments
    parser.add_argument("--device")
    parser.add_argument("--input")
    parser.add_argument("--folder")
    parser.add_argument("--measure_runtime", type=str_to_bool, default=False)
    parser.add_argument("--visualize_optimization", type=str_to_bool, default=False)
    parser.add_argument("--visualize_input", type=str_to_bool, default=False)
    parser.add_argument("--cached_segmentation", action="store_true")
    parser.add_argument("--segmentation_dir", default="./cached_segmentations/")
    parser.add_argument("--config", default="configs/default.yaml", nargs="+")

    config = yoco.load_config_from_args(
        parser, search_paths=[".", "~/.sdfest/", sdfest.__path__[0]]
    )

    if "input" in config and "folder" in config:
        print("Only one of input and folder can be specified.")
        exit()

    if config["measure_runtime"] and config["visualize_optimization"]:
        print("Visualization not supported while measuring runtime.")
        exit()

    pipeline = SDFPipeline(config)
    create_animation = (
        config["create_animation"] if "create_animation" in config else False
    )

    timing_dict = None
    timing_dicts = []
    if config["measure_runtime"]:
        timing_dict = add_timing_decorators(pipeline)

    # Segmentation using detectron2
    print("Loading segmentation model...")
    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )

    predictor = detectron2.engine.DefaultPredictor(cfg)
    print("Segmentation model loaded.")

    completed_runs = 0
    shape_optimization = True

    while True:
        if timing_dict is not None:
            timing_dict.clear()
        if "folder" in config:
            color_img, depth_img, color_path, _ = load_sample_from_folder(config)
        elif "input" in config:
            color_img, depth_img, color_path, _ = load_rgbd(config)
        else:
            print("No folder or input file specified.")
            exit()

        if timing_dict is not None:
            timing_dict["pipeline"].append([time.time(), None])
            timing_dict["segmentation"].append([time.time(), None])

        if config["cached_segmentation"]:
            # check if segmentation exists
            color_name, _ = os.path.splitext(color_path)
            color_dir = os.path.dirname(color_name)
            segmentation_dir = os.path.join(config["segmentation_dir"], color_dir)
            segmentation_path = (
                os.path.join(config["segmentation_dir"], color_name) + ".pickle"
            )
            os.makedirs(segmentation_dir, exist_ok=True)

            if os.path.isfile(segmentation_path):
                with open(segmentation_path, "rb") as f:
                    outputs = pickle.load(f)
            else:
                # compute segmentation and save
                # detectron expects (H,C,W), BGR, 0-255 as input
                detectron_color_img = color_img[:, :, ::-1] * 255
                outputs = predictor(detectron_color_img)
                with open(segmentation_path, "wb") as f:
                    pickle.dump(outputs, f)
        else:
            # detectron expects (H,C,W), BGR, 0-255 as input
            detectron_color_img = color_img[:, :, ::-1] * 255
            outputs = predictor(detectron_color_img)

        if timing_dict is not None:
            torch.cuda.synchronize()
            timing_dict["segmentation"][-1][1] = time.time()

        category_id = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes.index(
            config["category"]
        )
        matching_instances = []
        for i in range(len(outputs["instances"])):
            instance = outputs["instances"][i]
            if instance.pred_classes != category_id:
                continue
            matching_instances.append(instance)

        matching_instances.sort(key=lambda k: k.pred_masks.sum())

        if not matching_instances:
            print("Warning: category not detected in input")
        else:
            print("Category detected")

        for instance in matching_instances:
            if create_animation:
                animation_path = os.path.join(
                    os.getcwd(),
                    f"animation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                )
            else:
                animation_path = None

            if config["visualize_input"]:
                v = Visualizer(
                    color_img * 255,
                    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                    scale=1.2,
                )
                out = v.draw_instance_predictions(instance.to("cpu"))
                plt.imshow(out.get_image())
                plt.show()
                plt.imshow(depth_img)
                plt.show()
            depth_tensor = torch.from_numpy(depth_img).to(config["device"])
            instance_mask = instance.pred_masks.cuda()[0]
            color_img_tensor = torch.from_numpy(color_img).to(config["device"])

            try:
                position, orientation, scale, shape = pipeline(
                    depth_tensor,
                    instance_mask,
                    color_img_tensor,
                    visualize=config["visualize_optimization"],
                    animation_path=animation_path,
                    shape_optimization=shape_optimization,
                )
            except NoDepthError:
                print("No depth data, skipping")

            break  # comment to evaluate all instances, instead of largest only

        if timing_dict is not None:
            torch.cuda.synchronize()
            timing_dict["pipeline"][-1][1] = time.time()

            if completed_runs != 0 or not config["skip_first_run"]:
                timing_dicts.append(copy.deepcopy(timing_dict))
                timing_dicts[-1]["shape_optimization"] = shape_optimization

            print(f"\r{completed_runs+1}/{config['runs']}", end="")

        completed_runs += 1

        # only run single evaluation for single file
        if "input" in config and not config["measure_runtime"]:
            break
        elif config["measure_runtime"] and completed_runs == config["runs"]:
            if shape_optimization:
                shape_optimization = False
                completed_runs = 0
                print("")
            else:
                print("")
                break

    if config["measure_runtime"]:
        generate_runtime_overview(config, timing_dicts)


if __name__ == "__main__":
    main()
