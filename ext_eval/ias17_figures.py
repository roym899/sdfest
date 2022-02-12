"""Script to produce figures for IAS17 figures."""
import numpy as np
from sdf_estimation import metrics
import open3d as o3d
import matplotlib.pyplot as plt
import tikzplotlib

def compute_chamfer_distances(samples):
    gt = o3d.io.read_triangle_mesh("./ias17_obj/1.obj")
    o2 = o3d.io.read_triangle_mesh("./ias17_obj/2.obj")
    o3 = o3d.io.read_triangle_mesh("./ias17_obj/3.obj")

    gt_pts = np.asarray(gt.sample_points_uniformly(samples).points)
    o2_pts = np.asarray(o2.sample_points_uniformly(samples).points)
    o3_pts = np.asarray(o3.sample_points_uniformly(samples).points)
    cd_2 = metrics.symmetric_chamfer(gt_pts, o2_pts)
    cd_3 = metrics.symmetric_chamfer(gt_pts, o3_pts)
    print(cd_2, cd_3)

def compute_cd_plot_1():
    samples = np.logspace(2,6,num=50)
    gt = o3d.io.read_triangle_mesh("./ias17_obj/1.obj")
    o2 = o3d.io.read_triangle_mesh("./ias17_obj/4.obj")
    o3 = o3d.io.read_triangle_mesh("./ias17_obj/3.obj")

    cd_2s = []
    cd_3s = []
    cd_gt = []

    for sample in samples:
        gt_pts = np.asarray(gt.sample_points_uniformly(int(sample)).points)
        gt_pts_2 = np.asarray(gt.sample_points_uniformly(int(sample)).points)
        o2_pts = np.asarray(o2.sample_points_uniformly(int(sample)).points)
        o3_pts = np.asarray(o3.sample_points_uniformly(int(sample)).points)
        cd_2s.append(metrics.symmetric_chamfer(gt_pts, o2_pts))
        cd_3s.append(metrics.symmetric_chamfer(gt_pts, o3_pts))
        cd_gt.append(metrics.symmetric_chamfer(gt_pts, gt_pts_2))

    print(cd_2s)
    print(samples)

    plt.figure(figsize=(3,3))
    plt.plot(np.asarray(samples), np.asarray(cd_gt), label="GT")
    plt.plot(np.asarray(samples), np.asarray(cd_2s), label="S1")
    plt.plot(samples, cd_3s, label="S2")
    plt.xscale("log")
    plt.grid()
    plt.xlabel("# Samples")
    plt.ylabel("CD")

    plt.show()
    # tikzplotlib.save("cd_samples.tex")
    # plt.close()

if __name__ == "__main__":
    # compute_chamfer_distances(50000)
    compute_cd_plot_1()
