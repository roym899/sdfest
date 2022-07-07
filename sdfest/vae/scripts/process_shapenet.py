"""Script to preprocess shapenet like dataset.

This script allows to interactively go through a ShapeNet dataset and decide
whether to keep or remove a mesh. All textures are removed, only the obj file
and a converted SDF volume are stored in the output folder."""
import argparse
import json
from multiprocessing import Pool, Process, Manager
import os
import shutil

from sdfest.vae import sdf_utils  # needs to be imported before pyrender

from typing import List
import numpy as np
import time
from tqdm import tqdm
import trimesh
from trimesh import Trimesh
from trimesh.scene.scene import Scene
from trimesh.visual.material import SimpleMaterial
import pyrender


class Object3D:
    """Representation of a 3D object."""

    def __init__(self, mesh_path: str):
        """Initialize the 3D object.

        Args:
            mesh_path: the full file path of the mesh
        """
        self.mesh_path = mesh_path
        self.simplified_mesh = None
        self.sdf_volume = None
        self.reconstructed_mesh = None

    def convert_to_sdf(self, cells_per_dim, padding):
        self.sdf_volume = sdf_utils.mesh_to_sdf(
            self.simplified_mesh, cells_per_dim, padding
        )
        return self

    def load_mesh(self):
        loaded_obj = trimesh.load(self.mesh_path, process=False)
        if isinstance(loaded_obj, Trimesh):
            self.simplified_mesh = Trimesh(
                loaded_obj.vertices,
                loaded_obj.faces,
                vertex_normals=loaded_obj.vertex_normals,
                visual=trimesh.visual.TextureVisuals(material=SimpleMaterial()),
            )
        elif isinstance(loaded_obj, Scene):
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(
                        vertices=g.vertices,
                        faces=g.faces,
                        vertex_normals=g.vertex_normals,
                    )
                    for g in loaded_obj.geometry.values()
                )
            )
            self.simplified_mesh = Trimesh(
                mesh.vertices,
                mesh.faces,
                vertex_normals=mesh.vertex_normals,
                visual=trimesh.visual.TextureVisuals(material=SimpleMaterial()),
            )
        else:
            print(f"Not supported: {type(loaded_obj)}")
            return False
        return True

    def reconstruct_from_sdf(self):
        level = 1.0 / self.sdf_volume.shape[-1]
        self.reconstructed_mesh = sdf_utils.mesh_from_sdf(self.sdf_volume, level)
        return self


def trimesh_decision_viewer(mesh, return_dict):
    """View a trimesh with pyrender adding support to press left and right."""

    def on_press(key):
        nonlocal inp
        try:
            inp = key.char  # single-char keys
        except AttributeError:
            inp = key.name

    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])
    scene.add(pyrender_mesh)
    v = pyrender.Viewer(scene, run_in_thread=True, use_raymond_lighting=True)
    inp = None
    from pynput import keyboard

    decision = None

    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread
    while True:
        time.sleep(0.1)
        if inp == "left":
            decision = "remove"
            break
        elif inp == "right":
            decision = "keep"
            break
        elif inp == "down":
            decision = "stop"
            break
    # TODO: listener.stop() hangs randomly (especially under load)
    v.close_external()
    return_dict["decision"] = decision
    return decision


def check_meshes(path):
    objects = []
    for root, _, files in os.walk(path):
        for file in files:
            _, file_extension = os.path.splitext(file)
            if file_extension == ".obj":
                full_file_path = os.path.join(root, file)
                objects.append(Object3D(full_file_path))

    print("Press left to discard the mesh, press right to keep it.")
    # first filtering
    print(
        "Decide which shapes generally fit your criteria? (irrespective of SDF "
        "conversion)"
    )
    total_objects = len(objects)
    filtered_objects = []
    for i, object in enumerate(objects):
        if object.load_mesh():
            manager = Manager()
            return_dict = manager.dict()
            p = Process(
                target=trimesh_decision_viewer,
                args=(object.simplified_mesh, return_dict),
            )
            p.start()
            p.join()
            decision = return_dict["decision"]
            if decision == "keep":
                filtered_objects.append(object)
            if decision == "stop":
                break
            progress = i / total_objects * 100
            print(f"Progress: {progress:.2f}% Kept: {len(filtered_objects)}")
    return filtered_objects


def conv_to_sdf(obj: Object3D):
    print(obj.mesh_path)
    return obj.convert_to_sdf(resolution, padding)


def convert_to_sdf(objects: List[Object3D]) -> List[Object3D]:
    # convert to SDF
    # TODO do not keep everything in memory, store to disk instead

    num_objects = len(objects)
    with Pool() as p:
        objects = list(
            tqdm(
                p.imap(conv_to_sdf, objects),
                total=len(objects),
            )
        )
    # only keep objects with valid SDF volume
    objects = [obj for obj in objects if obj.sdf_volume is not None]

    # convert to Mesh
    for obj in objects:
        obj.reconstruct_from_sdf()
    # only keep objects with valid mesh reconstruction
    objects = [obj for obj in objects if obj.reconstructed_mesh is not None]

    print(f"{len(objects)} / {num_objects} have been successfully converted.")

    return objects


if __name__ == "__main__":
    # define the arguments
    parser = argparse.ArgumentParser(description="Training script for init network.")

    # parse arguments int(float(x)) to support exponential notation
    parser.add_argument("--resolution", required=True, type=lambda x: int(x))
    parser.add_argument("--inpath", required=True)
    parser.add_argument("--outpath", required=True)
    parser.add_argument("--all", default=False, action="store_true")
    parser.add_argument("--padding", required=True, type=lambda x: int(x))
    parser.add_argument("--good_meshes", required=False, default="./good_meshes.json")
    parser.add_argument("--final_meshes", required=False, default="./final_meshes.json")

    args = parser.parse_args()

    path = args.inpath
    output_path = args.outpath
    resolution = args.resolution
    padding = args.padding
    no_filtering = args.all
    good_mesh_paths_file_path = args.good_meshes
    final_mesh_paths_file_path = args.final_meshes

    if not os.path.isfile(good_mesh_paths_file_path):
        good_mesh_paths_dict = {}
    else:
        with open(good_mesh_paths_file_path) as f:
            good_mesh_paths_dict = json.load(f)
    if not os.path.isfile(final_mesh_paths_file_path):
        final_mesh_paths_dict = {}
    else:
        with open(final_mesh_paths_file_path) as f:
            final_mesh_paths_dict = json.load(f)

    if no_filtering:
        objects = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if file_extension == ".obj":
                    full_file_path = os.path.join(root, file)
                    objects.append(full_file_path)
                    objects.append(Object3D(full_file_path))
                    objects[-1].load_mesh()
    else:
        if path in final_mesh_paths_dict:
            final_mesh_paths = final_mesh_paths_dict[path]
            objects = [Object3D(p) for p in final_mesh_paths]
            for object in objects:
                object.load_mesh()
        elif path in good_mesh_paths_dict:
            good_mesh_paths = good_mesh_paths_dict[path]
            objects = [Object3D(good_mesh_path) for good_mesh_path in good_mesh_paths]
            for object in objects:
                object.load_mesh()
        else:
            objects = check_meshes(path)
            while True:
                print(f"Store decisions? ({good_mesh_paths_file_path}) (y/n)")
                decision = input()
                if decision == "y":
                    good_mesh_paths = [obj.mesh_path for obj in objects]
                    good_mesh_paths_dict[path] = good_mesh_paths
                    with open(good_mesh_paths_file_path, "w") as f:
                        json.dump(good_mesh_paths_dict, f, indent=4)
                    break
                elif decision == "n":
                    break

    objects = convert_to_sdf(objects)

    # manually check if sdf volume is okay
    if no_filtering or path in final_mesh_paths_dict:
        final_objects = objects
    else:
        print("Decide which shapes have an okay sdf reconstruction?")
        final_objects = []
        for i, object in enumerate(objects):
            manager = Manager()
            return_dict = manager.dict()
            p = Process(
                target=trimesh_decision_viewer,
                args=(object.reconstructed_mesh, return_dict),
            )
            p.start()
            p.join()
            decision = return_dict["decision"]
            if decision == "keep":
                final_objects.append(object)
            if decision == "stop":
                break
            print(f"Progress: {i/len(objects) * 100:.2f}% Kept: {len(final_objects)}")
        while True:
            print(f"Store final decisions? ({final_mesh_paths_file_path}) (y/n)")
            decision = input()
            if decision == "y":
                final_mesh_paths = [obj.mesh_path for obj in final_objects]
                final_mesh_paths_dict[path] = final_mesh_paths
                with open(final_mesh_paths_file_path, "w") as f:
                    json.dump(final_mesh_paths_dict, f, indent=4)
                break
            elif decision == "n":
                break

    # save the objects
    os.makedirs(output_path, exist_ok=True)
    counter = 0
    for object in final_objects:
        output_file_path_obj = os.path.join(output_path, f"{counter:05}.obj")
        shutil.copyfile(object.mesh_path, output_file_path_obj)
        output_file_path_npy = os.path.join(output_path, f"{counter:05}.npy")
        np.save(output_file_path_npy, object.sdf_volume)
        counter += 1
