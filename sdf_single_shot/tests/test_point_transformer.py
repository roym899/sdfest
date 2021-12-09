import torch

from sdf_single_shot.point_transformer import *
from sdf_single_shot.pointnet import VanillaPointNet


def test_default_transformer() -> None:
    """Test if default parameters loads"""

    inp1 = torch.randn(2, 1024, 3)
    point_transformer = PointTransformer()
    out_t = point_transformer(inp1)
    assert out_t.shape == (2, 10)


def test_shape_equality_transformer() -> None:
    """Test various equalities w.r.t. shape of PointTransformer output."""

    # Test whether VanillaPointNet and PointTransformer
    # outputs same shape.

    inp1 = torch.randn(2, 512, 3)

    pointnet = VanillaPointNet(3, [64, 64, 10], True)
    out_p = pointnet(inp1)

    cfg: Cfg = dict()
    cfg["n_points"] = 512
    cfg["n_blocks"] = 4
    cfg["n_neighbors"] = 16
    cfg["n_class"] = 10
    cfg["input_dim"] = 3
    cfg["transformer_dim"] = 512

    point_transformer = PointTransformer(cfg)
    out_t = point_transformer(inp1)

    assert out_p.shape == out_t.shape

    # Test whether PointTransformer outputs required shape.
    inp2 = torch.randn(2, 1000, 3)

    cfg2: Cfg = dict()
    cfg2["n_points"] = 1000
    cfg2["n_blocks"] = 4
    cfg2["n_neighbors"] = 16
    cfg2["n_class"] = 12
    cfg2["input_dim"] = 3
    cfg2["transformer_dim"] = 512

    point_transformer = PointTransformer(cfg2)
    out_t2 = point_transformer(inp2)

    assert out_t2.shape == (2, 12)


def test_backward_transformer() -> None:
    """Test whether backward function is possible to call on
    PointTransformer."""

    inp1 = torch.randn(2, 512, 3)

    cfg: Cfg = dict()
    cfg["n_points"] = 512
    cfg["n_blocks"] = 4
    cfg["n_neighbors"] = 16
    cfg["n_class"] = 10
    cfg["input_dim"] = 3
    cfg["transformer_dim"] = 512

    point_transformer = PointTransformer(cfg)
    out_t = point_transformer(inp1)

    out_sum = torch.sum(out_t)

    out_sum.backward()
