"""Script to produce figures for IAS17 figures."""
from matplotlib import cm
import numpy as np
from sdf_estimation import metrics
import open3d as o3d
import matplotlib.pyplot as plt
import tikzplotlib
import yoco
from sdf_single_shot.utils import str_to_object
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import random
import healpy as hp


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


def compute_cd_plot() -> None:
    samples = np.logspace(2, 6, num=50)
    gt = o3d.io.read_triangle_mesh("./ias17_obj/1.obj")
    o2 = o3d.io.read_triangle_mesh("./ias17_obj/2.obj")
    o3 = o3d.io.read_triangle_mesh("./ias17_obj/3.obj")

    # metric scaling
    extents = np.asarray(gt.vertices).max(0) - np.asarray(gt.vertices).min(0)
    height = extents[1]
    scale = 0.1 / height
    gt = gt.scale(scale, np.array([0.0, 0, 0]))
    o2 = o2.scale(scale, np.array([0.0, 0, 0]))
    o3 = o3.scale(scale, np.array([0.0, 0, 0]))

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

    plt.figure(figsize=(3, 3))
    plt.plot(np.asarray(samples), np.asarray(cd_gt), label="GT")
    plt.plot(np.asarray(samples), np.asarray(cd_2s), label="S1")
    plt.plot(samples, cd_3s, label="S2")
    plt.xscale("log")
    plt.grid()
    plt.xlabel("# Samples")
    plt.ylabel("CD")

    # plt.show()
    tikzplotlib.save("cd_samples.tex")
    plt.close()


def compute_f_plot() -> None:
    samples = np.logspace(2, 6, num=50)
    gt = o3d.io.read_triangle_mesh("./ias17_obj/1.obj")
    o2 = o3d.io.read_triangle_mesh("./ias17_obj/2.obj")
    o3 = o3d.io.read_triangle_mesh("./ias17_obj/3.obj")

    # metric scaling
    extents = np.asarray(gt.vertices).max(0) - np.asarray(gt.vertices).min(0)
    height = extents[1]
    scale = 0.1 / height
    gt = gt.scale(scale, np.array([0.0, 0, 0]))
    o2 = o2.scale(scale, np.array([0.0, 0, 0]))
    o3 = o3.scale(scale, np.array([0.0, 0, 0]))

    cd_2s = []
    cd_3s = []
    cd_gt = []

    for sample in samples:
        gt_pts = np.asarray(gt.sample_points_uniformly(int(sample)).points)
        gt_pts_2 = np.asarray(gt.sample_points_uniformly(int(sample)).points)
        o2_pts = np.asarray(o2.sample_points_uniformly(int(sample)).points)
        o3_pts = np.asarray(o3.sample_points_uniformly(int(sample)).points)
        cd_2s.append(metrics.reconstruction_fscore(gt_pts, o2_pts, 0.01))
        cd_3s.append(metrics.reconstruction_fscore(gt_pts, o3_pts, 0.01))
        cd_gt.append(metrics.reconstruction_fscore(gt_pts, gt_pts_2, 0.01))

    plt.figure(figsize=(3, 3))
    plt.plot(np.asarray(samples), np.asarray(cd_gt), label="GT")
    plt.plot(np.asarray(samples), np.asarray(cd_2s), label="S1")
    plt.plot(samples, cd_3s, label="S2")
    plt.xscale("log")
    plt.grid()
    plt.xlabel("# Samples")
    plt.ylabel("F_1cm")

    tikzplotlib.save("fscore_samples.tex")
    plt.close()


def compute_orientation_hist() -> None:
    config = yoco.load_config_from_file("./config/real275.yaml")
    dataset_config = config["dataset_config"]
    dataset_name = dataset_config["name"]
    dataset_type = str_to_object(dataset_config["type"])
    dataset = dataset_type(config=dataset_config["config_dict"])
    # Faster but probably only worth it if whole evaluation supports batches
    # self._dataloader = DataLoader(self._dataset, 1, num_workers=8)
    print(f"{len(dataset)} samples found for dataset {dataset_name}.")

    
    pointList = []

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    # indices = indices[:500]

    # NPIX = hp.nside2npix(32)
    # print(NPIX)
    # m = np.arange(NPIX)
    # hp.mollview(m, title="Mollview image RING")
    # hp.graticule()
    # plt.show()

    for i in tqdm(indices):
        sample = dataset[i]
        pt = Rotation.from_quat(sample["quaternion"]).apply([0, 1, 0])
        print(pt, np.linalg.norm(pt))
        pointList.append(pt)

    pointList = np.array(pointList)

    # def random_point(r=1):
    #     ct = 2 * np.random.rand() - 1
    #     st = np.sqrt(1 - ct ** 2)
    #     phi = 2 * np.pi * np.random.rand()
    #     x = r * st * np.cos(phi)
    #     y = r * st * np.sin(phi)
    #     z = r * ct
    #     return np.array([x, y, z])

    def near(p, pntList, d0):
        cnt = 0

        d = np.linalg.norm(p - np.array(pntList), axis=1)
        cnt = (1 - d[d<d0] / d0).sum()
        # for pj in pntList:
        #     dist = np.linalg.norm(p - pj)
        #     if dist < d0:
        #         cnt += 1 - dist / d0
        #     if cnt > 1:
        #         break
        return min(cnt, 1)

    # pointList = np.array([random_point(1) for i in range(65)])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d", proj_type = 'ortho')

    u = np.linspace(0, 2 * np.pi, 480)
    v = np.linspace(0, np.pi, 240)

    # create the sphere surface
    XX = np.outer(np.cos(u), np.sin(v))
    YY = np.outer(np.sin(u), np.sin(v))
    ZZ = np.outer(np.ones(np.size(u)), np.cos(v))

    WW = XX.copy()
    for i in range(len(XX)):
        for j in range(len(XX[0])):
            x = XX[i, j]
            y = YY[i, j]
            z = ZZ[i, j]
            WW[i, j] = near(np.array([x, y, z]), pointList, 0.1)
    WW = WW / np.amax(WW)
    myheatmap = WW

    ax.plot_surface(XX, YY, ZZ, cstride=1, rstride=1, facecolors=cm.jet(myheatmap))
    ax.axis("off")
    ax.set_box_aspect([1,1,1])

    # for jj in range(0,360,180):
    for ii in [80,260]:
        ax.view_init(elev=180, azim=ii)
        plt.savefig("movie%d_%d.png" % (225, ii))

    # plt.imshow(WW)
    # plt.show()


if __name__ == "__main__":
    # gt = o3d.io.read_triangle_mesh("./ias17_obj/1.obj")
    # compute_cd_plot()
    # compute_f_plot()
    compute_orientation_hist()
