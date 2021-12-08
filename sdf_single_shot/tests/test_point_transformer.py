import torch

from sdf_single_shot.point_transformer import *
from sdf_single_shot.pointnet import VanillaPointNet
# from sdf_single_shot.pointnet_util import *


def test_shape_equality_transformer() -> None:
    """Test various equalities w.r.t. shape of PointTransformer output."""

    # Test whether VanillaPointNet and IterativePointNet
    # outputs same shape.

    inp1 = torch.randn(2, 512, 3)

    pointnet = VanillaPointNet(3, [64, 64, 10], True)
    out_p = pointnet(inp1)


    cfg: Cfg = dict()
    cfg['num_point'] = 512
    cfg['nblocks'] = 4
    cfg['nneighbor'] = 16
    cfg['num_class'] = 10
    cfg['input_dim'] = 3
    cfg['transformer_dim'] = 512

    point_transformer = PointTransformer(cfg)
    out_t = point_transformer(inp1)


    assert out_p.shape == out_t.shape

    # Test whether IterativePointnet outputs required shape.
    inp2 = torch.randn(2, 1000, 3)

    cfg2: Cfg = dict()
    cfg2['num_point'] = 1000
    cfg2['nblocks'] = 4
    cfg2['nneighbor'] = 16
    cfg2['num_class'] = 12
    cfg2['input_dim'] = 3
    cfg2['transformer_dim'] = 512

    point_transformer = PointTransformer(cfg2)
    out_t2 = point_transformer(inp2)


    assert out_t2.shape == (2, 12)

def test_backward_transformer() -> None:
    """Test whether backward function is possible to call on
    PointTransformer."""

    inp1 = torch.randn(2, 512, 3)

    cfg: Cfg = dict()
    cfg['num_point'] = 512
    cfg['nblocks'] = 4
    cfg['nneighbor'] = 16
    cfg['num_class'] = 10
    cfg['input_dim'] = 3
    cfg['transformer_dim'] = 512

    point_transformer = PointTransformer(cfg)
    out_t = point_transformer(inp1)

    out_sum = torch.sum(out_t)

    out_sum.backward()


