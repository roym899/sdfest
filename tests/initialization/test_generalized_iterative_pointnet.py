import torch

from sdfest.initialization.pointnet import VanillaPointNet, GeneralizedIterativePointNet


def test_shape_equality_generalized_iterative() -> None:
    """Test various equalities w.r.t. shape of GeneralizedIterativePointNet output."""

    # Test whether VanillaPointNet and GeneralizedIterativePointNet
    # outputs same shape.

    inp = torch.randn(100, 60, 3)
    pointnet = VanillaPointNet(3, [32, 64, 128], True)
    out_p = pointnet(inp)
    generalized_iterative_pointnet = GeneralizedIterativePointNet(
        [0], 3, [[32, 64, 128]], True
    )
    out_gip = generalized_iterative_pointnet(inp)
    assert out_gip.shape == out_p.shape

    # Test whether GeneralizedIterativePointNet outputs required shape.

    inp2 = torch.randn(100, 30, 2)
    generalized_iterative_pointnet = GeneralizedIterativePointNet(
        [1, 2, 3], 2, [[32, 64], [64, 128, 64], [128, 100]], True
    )
    out_gip2 = generalized_iterative_pointnet(inp2)
    assert out_gip2.shape == (100, 100)


def test_backward_generalized_iterative() -> None:
    """Test whether backward function is possible to call on
    GeneralizedIterativePointNet."""

    generalized_iterative_pointnet = GeneralizedIterativePointNet(
        [2, 4], 3, [[64, 64, 64, 128], [156, 128]], True
    )

    inp2 = torch.randn(100, 500, 3)
    out2 = generalized_iterative_pointnet(inp2)
    out_sum2 = torch.sum(out2)

    out_sum2.backward()
