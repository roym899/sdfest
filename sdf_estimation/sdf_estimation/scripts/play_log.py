"""Script to play back log file and generate animation.

Usage:
  python play_log.py --log_file filename.pkl

Log files in the required format is generated by render_evaluation.py.
"""

import argparse
from datetime import datetime
import ffmpeg
import os
import pickle
from shutil import copyfile
import time

import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch

from sdf_estimation.simple_setup import SDFPipeline
from sdf_estimation import synthetic
from sdf_single_shot import pointset_utils, quaternion_utils

reset = False
realtime = True
reconstruction_type = "mesh" or "pointcloud"
animation_queued = False
color = True
camera_frames = True


def quit_program(_: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    """Quit program."""
    exit()


def reset_bounding_box(_: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    """Schedule resetting of bounding box."""
    global reset
    print("Reset bounding box.")
    reset = True
    return False


def toggle_realtime(_: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    """Toggle realtime playback."""
    global realtime
    realtime = not realtime
    print(f"Realtime: {realtime}")
    return False


def toggle_color(_: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    """Toggle realtime playback."""
    global color
    color = not color
    print(f"Color: {color}")
    return False


def toggle_camera_frames(_: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    """Toggle realtime playback."""
    global camera_frames
    camera_frames = not camera_frames
    print(f"Camera frames: {camera_frames}")
    return False


def switch_reconstruction_type(_: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    """Switch between mesh and pointcloud reconstruction."""
    global reconstruction_type
    reconstruction_type = (
        "mesh" if reconstruction_type == "pointcloud" else "pointcloud"
    )
    print(f"Reconstruction: {reconstruction_type}")
    return False


def queue_animation(_: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    """Switch between mesh and pointcloud reconstruction."""
    global animation_queued
    animation_queued = True
    print("Saving animation for next loop.")
    return False


def main() -> None:
    """Entry point of the evaluation program."""
    global reset
    global animation_queued
    parser = argparse.ArgumentParser(
        description="Play log file and generate animation."
    )
    parser.add_argument("--log_file", required=True)
    args = parser.parse_args()

    with open(args.log_file, "rb") as f:
        data = pickle.load(f)
        config = data["config"]
        log_entries = data["log"]

    pipeline = SDFPipeline(config)

    # precompute meshes for each timestep
    for log_entry in tqdm(log_entries):
        # print(log_entry)
        if "latent_shape" in log_entry:
            out_mesh = pipeline.generate_mesh(
                log_entry["latent_shape"], 1 / log_entry["scale_inv"], True
            )
            out_mesh.position = log_entry["position"][0].detach().cpu().numpy()
            out_mesh.orientation = log_entry["orientation"][0].detach().cpu().numpy()
            log_entry["mesh"] = out_mesh.get_transformed_o3d_geometry()
            log_entry["mesh"].paint_uniform_color([0.2, 0.4, 0.7])
            log_entry["pointcloud"] = log_entry["mesh"].sample_points_uniformly(20000)

    # visualize log entries
    cam_meshes = []
    pointclouds = []
    KEY_ESCAPE = 256
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(key=ord("A"), callback_func=reset_bounding_box)
    vis.register_key_callback(key=ord("R"), callback_func=toggle_realtime)
    vis.register_key_callback(key=ord("S"), callback_func=switch_reconstruction_type)
    vis.register_key_callback(key=ord("N"), callback_func=queue_animation)
    vis.register_key_callback(key=ord("C"), callback_func=toggle_color)
    vis.register_key_callback(key=ord("F"), callback_func=toggle_camera_frames)
    vis.register_key_callback(key=KEY_ESCAPE, callback_func=quit_program)
    vis.create_window(width=640, height=480)
    print(
        "Controls\n\ta: reset view point & bounding box\n"
        "\tr: toggle realtime\n"
        "\ts: switch reconstruction_type\n",
        "\tn: queue animation\n",
        "\tc: toggle color\n",
        "\tf: toggle camera frames\n",
    )
    first = True
    while True:
        vis.clear_geometries()
        animation_folder = None
        animation_files = []
        animation_timestamps = []
        if animation_queued:
            animation_queued = False
            animation_folder = (
                f"animation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            os.makedirs(animation_folder, exist_ok=True)
        start_time = time.time()
        for log_entry in log_entries:
            vis.poll_events()
            vis.update_renderer()
            if realtime and animation_folder is None:
                while log_entry["timestamp"] > time.time() - start_time:
                    vis.poll_events()
                    vis.update_renderer()

            if "camera_positions" in log_entry:
                cam_meshes = []
                for t_c2w, quat_c2w in zip(
                    log_entry["camera_positions"], log_entry["camera_orientations"]
                ):
                    frame_mesh = synthetic.Mesh(
                        mesh=o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.1, origin=[0, 0, 0]
                        ),
                        rel_scale=True,
                    )
                    frame_mesh.position = t_c2w.cpu().numpy()
                    frame_mesh.orientation = quaternion_utils.quaternion_multiply(
                        quat_c2w.cpu(), torch.tensor([1.0, 0, 0, 0])
                    ).numpy()
                    cam_meshes.append(frame_mesh.get_transformed_o3d_geometry())

            if "depth_images" in log_entry:
                pointclouds = []
                for depth_image, color_image, t_c2w, quat_c2w in zip(
                    log_entry["depth_images"],
                    log_entry["color_images"],
                    log_entry["camera_positions"],
                    log_entry["camera_orientations"],
                ):
                    points_c = pointset_utils.depth_to_pointcloud(
                        depth_image, pipeline.cam, normalize=False
                    )
                    pointcloud_torch = (
                        quaternion_utils.quaternion_apply(quat_c2w, points_c) + t_c2w
                    )
                    pointcloud_numpy = pointcloud_torch.cpu().numpy()
                    pointcloud_o3d = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(pointcloud_numpy)
                    )
                    if color:
                        pointcloud_colors_torch = color_image[depth_image != 0]
                        pointcloud_colors_numpy = pointcloud_colors_torch.cpu().numpy()
                        pointcloud_o3d.colors = o3d.utility.Vector3dVector(
                            pointcloud_colors_numpy
                        )
                    else:
                        pointcloud_o3d.colors = o3d.utility.Vector3dVector(
                            np.ones_like(pointcloud_numpy) * np.array([1.0, 0.2, 0.2])
                        )

                    pointclouds.append(pointcloud_o3d)

            geometries = [] + pointclouds
            if camera_frames:
                geometries += cam_meshes
            if "mesh" in log_entry:
                geometries.append(log_entry[reconstruction_type])
            vis.clear_geometries()
            for i, geometry in enumerate(geometries):
                vis.add_geometry(geometry, reset_bounding_box=first or reset)
                if i == len(geometries) - 1:
                    reset = first = False

            if animation_folder is not None:
                vis.poll_events()
                vis.update_renderer()
                timestamp = log_entry["timestamp"]
                filename = f"{timestamp}.png"
                vis.capture_screen_image(os.path.join(animation_folder, filename))
                animation_files.append(filename)
                animation_timestamps.append(timestamp)

        # create constant framerate video
        if animation_folder is not None:
            fps = 30
            frame_folder = os.path.join(animation_folder, "constant_framerate")
            os.makedirs(frame_folder, exist_ok=True)
            video_name = f"{animation_folder}.mp4"
            current_frame = animation_files.pop(0)
            current_time = animation_timestamps.pop(0)
            frame_number = 0
            while animation_files:
                if animation_timestamps[0] <= current_time:
                    current_frame = animation_files.pop(0)
                    animation_timestamps.pop(0)
                copyfile(
                    os.path.join(animation_folder, current_frame),
                    os.path.join(frame_folder, f"{frame_number:06d}.png"),
                )

                current_time += 1 / fps
                frame_number += 1

            ffmpeg.input(
                os.path.join(frame_folder, "*.png"), pattern_type="glob", framerate=fps
            ).output(video_name).run()


if __name__ == "__main__":
    main()
