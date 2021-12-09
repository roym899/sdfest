"""PointTransformer based on
Zhao, Hengshuang, et al. "Point transformer."
Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

Generally following:
https://github.com/qq456cvb/Point-Transformers/tree/master/models/Hengshuang
"""
import torch
import torch.nn as nn
from sdf_single_shot.pointnet_util import *
from sdf_single_shot.transformer import TransformerBlock
from typing import TypedDict


class Cfg(TypedDict):
    """Configuration dictionary of various point_transformer classes

    Attributes:
        n_points: Number of points in each batch
        n_blocks: Number of (TransitionDown, Transformer) blocks in step 3) of the Backbone class
        n_neighbors: Number of neighbours for transformer point sampling (k in the paper)
        n_class: Number of output classes
        input_dim: Dimension of the input points.
        transformer_dim: Size of embedding which the transformer operates in.
    """

    n_points: int
    n_blocks: int
    n_neighbors: int
    n_class: int
    input_dim: int
    transformer_dim: int


default_cfg: Cfg = {
    "n_points": 1024,
    "n_blocks": 4,
    "n_neighbors": 16,
    "n_class": 10,
    "input_dim": 3,
    "transformer_dim": 512,
}


class TransitionDown(nn.Module):
    """TransitionDown block.

    This block reduces cardinality of the point set.

    Farthest point sampling, k nearest neighbours, linear
    transformation along with Relu and maxpooling is handled by class
    PointNetSetAbstraction.
    """

    def __init__(self, n_points, n_neighbors, channels) -> None:
        """Initialize the TransitionDown module.

        Args:
            n_points: output number of points after reduction of cardinality
            n_neighbors: k of k nearest neighbors in the algorithm
            channels: list like [in_shape, out1_shape, out2_shape ..] to parameterize MLP layer
        """

        super().__init__()
        self.sa = PointNetSetAbstraction(
            n_points,
            0,
            n_neighbors,
            channels[0],
            channels[1:],
            group_all=False,
            knn=True,
        )

    def forward(self, xyz, points) -> (torch.Tensor, torch.Tensor):
        """Forward pass of the module."""
        return self.sa(xyz, points)


class Backbone(nn.Module):
    """Backbone structure of the PointTransformer.

    It is composed of a series of steps:
        1) an MLP 2) Transformer block 3) a series of TransitionDown and Transformer blocks. i.e.,
            features = MLP(xyz)
            features = Transformer(xyz, features)
            for i in range(n_blocks):
                xyz, features = TransitionDown(xyz, features)
                features = Transformer_i(xyz, features)
    """

    def __init__(self, cfg: Cfg = default_cfg) -> None:
        """Initialize the Backbone module.

        Args:
            cfg: a dictionary comprising of class Cfg
        """
        super().__init__()
        n_points, n_blocks, n_neighbors, n_c, d_points = (
            cfg["n_points"],
            cfg["n_blocks"],
            cfg["n_neighbors"],
            cfg["n_class"],
            cfg["input_dim"],
        )
        self.fc1 = nn.Sequential(nn.Linear(d_points, 32), nn.ReLU(), nn.Linear(32, 32))
        self.transformer1 = TransformerBlock(32, cfg["transformer_dim"], n_neighbors)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(n_blocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(
                    n_points // 4 ** (i + 1),
                    n_neighbors,
                    [channel // 2 + 3, channel, channel],
                )
            )
            self.transformers.append(
                TransformerBlock(channel, cfg["transformer_dim"], n_neighbors)
            )
        self.n_blocks = n_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Input has dimension NxMxC, where N is the batch size, M the number of points per
        set, and C the number of channels per point.

        Args:
            x: batch of point sets
        """
        # take first 3 rows as point coordinates
        xyz = x[..., :3]

        # apply MLP and then Transformer
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]

        # apply series of TransitionDown and Transformers
        for i in range(self.n_blocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformer(nn.Module):
    """PointTransformer Classifier"""

    def __init__(self, cfg: Cfg = default_cfg) -> None:
        """Initialize PointTransformer module.

        Args:
            cfg: a dictionary comprising of class Cfg
        """
        super().__init__()

        # initialize backbone
        self.backbone = Backbone(cfg)
        n_points, n_blocks, n_neighbors, n_c, d_points = (
            cfg["n_points"],
            cfg["n_blocks"],
            cfg["n_neighbors"],
            cfg["n_class"],
            cfg["input_dim"],
        )

        # define MLP for final classification
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** n_blocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c),
        )
        self.n_blocks = n_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Input has dimension NxMxC, where N is the batch size, M the number of points per
        set, and C the number of channels per point.

        Args:
            x: batch of point sets
        """
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        return res


if __name__ == "__main__":

    # cfg: Cfg = dict()
    # cfg['n_points'] = 1024
    # cfg['n_blocks'] = 4
    # cfg['n_neighbors'] = 16
    # cfg['n_class'] = 10
    # cfg['input_dim'] = 3
    # cfg['transformer_dim'] = 512

    inp = torch.randn(2, 4096, 3)

    point_transformer = PointTransformer()
    out = point_transformer(inp)

    print(f"out shape is {out.shape}")
