import json
import glob
import copy
import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

fix_paths = glob.glob("./*_fix.obj")
fix_paths.sort()

test_count = 0

for fix_path in fix_paths:
    original_path = fix_path.replace("_fix", "")
    print(fix_path, original_path)
    org_mesh = o3d.io.read_triangle_mesh(original_path)
    fix_mesh = o3d.io.read_triangle_mesh(fix_path)
    org_pts = org_mesh.sample_points_uniformly(10000)
    fix_pts = fix_mesh.sample_points_uniformly(10000)
    org_pts.paint_uniform_color([0, 0.651, 0.929])
    fix_pts.paint_uniform_color([0.641, 0.651, 0])

    o3d.visualization.draw_geometries([org_pts, fix_pts])
    org_vertices = np.asarray(org_mesh.vertices)
    org_extents = org_vertices.max(axis=0) - org_vertices.min(axis=0)
    fix_vertices = np.asarray(fix_mesh.vertices)
    fix_extents = fix_vertices.max(axis=0) - fix_vertices.min(axis=0)
    print(org_extents, fix_extents)

    fix_center = fix_extents / 2 + fix_vertices.min(axis=0)
    fix_mesh.translate(-fix_center, relative=True)

    o3d.io.write_triangle_mesh(original_path, fix_mesh)

    # adjust position and rotation
    T_fix = np.eye(4)
    T_fix[0:3, 3] = fix_center

    with open("./annotations.json", "r") as f:
        anns_dict = json.load(f)

    # fix scale
    num = original_path[-9:-4]

    anns_dict[num]["scale"] = (fix_extents / 2).tolist()

    # if num in category_dict:
    #     data_dict[num]["category"] = category_dict[num]

    for pose_ann in anns_dict[num]["pose_anns"]:
        T_original = np.eye(4)
        org_orientation_m = Rotation.from_quat(pose_ann["orientation"]).as_matrix()
        org_position = pose_ann["position"]
        T_original[0:3, 0:3] = org_orientation_m
        T_original[0:3, 3] = org_position

        T_final = T_original @ T_fix

        position = T_final[0:3, 3].tolist()
        orientation = Rotation.from_matrix(T_final[0:3, 0:3]).as_quat().tolist()

        # save posed org and fix
        # posed_org_mesh = copy.deepcopy(org_mesh)
        # posed_fix_mesh = copy.deepcopy(fix_mesh)

        # posed_org_mesh.rotate(org_orientation_m, center=np.asarray([0.0, 0, 0]))
        # posed_fix_mesh.rotate(
        #     Rotation.from_quat(orientation).as_matrix(), center=np.asarray([0.0, 0, 0])
        # )
        # posed_org_mesh.translate(org_position)
        # posed_fix_mesh.translate(position)

        # o3d.io.write_triangle_mesh(f"./test/{test_count}_org.obj", posed_org_mesh)
        # o3d.io.write_triangle_mesh(f"./test/{test_count}_fix.obj", posed_fix_mesh)

        # test_count += 1

        pose_ann["position"] = position
        pose_ann["orientation"] = orientation

    with open("./annotations.json", "w") as f :
        json.dump(anns_dict, f, indent=2)
