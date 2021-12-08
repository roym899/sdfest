import torch
import torch.nn as nn
from sdf_single_shot.pointnet_util import *
from sdf_single_shot.transformer import TransformerBlock
from typing import TypedDict

class Cfg(TypedDict):
    num_point: int
    nblocks: int
    nneighbor: int
    num_class: int
    input_dim: int
    transformer_dim: int


default_cfg: Cfg = {
    "num_point": 1024,
    "nblocks": 4,
    "nneighbor": 16,
    "num_class": 10,
    "input_dim": 3,
    "transformer_dim": 512
}

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)

class Backbone(nn.Module):
    def __init__(self, cfg: Cfg = default_cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg['num_point'], cfg['nblocks'], cfg['nneighbor'], cfg[
            'num_class'], cfg['input_dim']
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg['transformer_dim'], nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg['transformer_dim'], nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        #apply mlp and then transformer
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformer(nn.Module):
    def __init__(self, cfg: Cfg = default_cfg):
        super().__init__()

        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg['num_point'], cfg['nblocks'], cfg['nneighbor'], cfg['num_class'], cfg['input_dim']

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks

    def forward(self, x):
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        return res







# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # cfg: Cfg = dict()
    # cfg['num_point'] = 1024
    # cfg['nblocks'] = 4
    # cfg['nneighbor'] = 16
    # cfg['num_class'] = 10
    # cfg['input_dim'] = 3
    # cfg['transformer_dim'] = 512


    inp = torch.randn(2, 4096, 3)

    point_transformer = PointTransformer()
    out = point_transformer(inp)

    print(f"out shape is {out.shape}")




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
