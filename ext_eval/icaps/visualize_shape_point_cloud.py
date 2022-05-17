import numpy as np
import open3d as o3d 
import os
import argparse

def custom_draw_geometry_load_option_single(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600, height=600)
    vis.add_geometry(pcd)
    vis.get_view_control()
    vis.run()
    vis.destroy_window()

def custom_draw_geometry_load_double(pcd, pcd2):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600, height=600)
    vis.add_geometry(pcd)
    vis.add_geometry(pcd2)
    vis.get_view_control()
    vis.run()
    vis.destroy_window()

def main(args):
    print(args)

    pcd_est_file = os.path.join('./results', args.exp, args.category, "seq_{}".format(args.seq), args.shape_est_file)
    pcd_meas_file = os.path.join('./results', args.exp, args.category, "seq_{}".format(args.seq), args.partial_point_cloud_file)

    pcd_est = o3d.io.read_point_cloud(pcd_est_file)
    color = np.zeros_like(pcd_est.points)
    color[:, 1] = 0.2
    color[:, 2] = 0.8
    pcd_est.colors = o3d.utility.Vector3dVector(color)

    pcd_meas = o3d.io.read_point_cloud(pcd_meas_file)
    color = np.zeros_like(pcd_meas.points)
    color[:, 0] = 0.8
    color[:, 1] = 0.2
    pcd_meas.colors = o3d.utility.Vector3dVector(color)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    
    custom_draw_geometry_load_double(pcd_est, pcd_meas)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Visualize Point Cloud Estimation')
    parser.add_argument('--exp', dest='exp',
                        help='experiment_folder',
                        required=True, type=str)
    parser.add_argument('--category', dest='category',
                        help='object_category',
                        required=True, type=str)
    parser.add_argument('--seq', dest='seq',
                        help='sequence_id',
                        required=True, type=str)
    parser.add_argument('--shape_est_file', dest='shape_est_file',
                        help='estimated shape file name',
                        required=True, type=str)
    parser.add_argument('--partial_point_cloud_file', dest='partial_point_cloud_file',
                        help='partial point cloud file name',
                        required=True, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)