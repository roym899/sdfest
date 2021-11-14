import torch


from sdf_single_shot.pointnet import (
    VanillaPointNet,
    IterativePointNet,
    GeneralizedIterativePointNet,
)


def test_shape_equality_pointnet_iteratative() -> None:
    """Test whether VanillaPointnet and IterativePointnet outputs same shape."""

    inp1 = torch.randn(2, 500, 3)

    pointnet = VanillaPointNet(3, [64, 64, 1024], True)
    out_p = pointnet(inp1)

    iterative_pointnet = IterativePointNet(0, 3, [64, 64, 1024], True)
    out_ip = iterative_pointnet(inp1)

    assert out_p.shape == out_ip.shape

    """Test whether IterativePointnet outputs required shape."""
    inp2 = torch.randn(100, 50, 2)
    iterative_pointnet2 = IterativePointNet(3, 2, [32, 64, 64, 1024], True)
    out_ip2 = iterative_pointnet2(inp2)

    assert out_ip2.shape == (100, 1024)


def test_shape_equality_pointnet_generalized() -> None:
    """Test whether VanillaPointnet and GeneralizedIterativePointnet
    outputs same shape."""

    inp = torch.randn(100, 60, 3)
    pointnet = VanillaPointNet(3, [32, 64, 128], True)
    out_p = pointnet(inp)
    generalized_iterative_pointnet = GeneralizedIterativePointNet(
        [0], 3, [[32, 64, 128]], True
    )
    out_gip = generalized_iterative_pointnet(inp)
    assert out_gip.shape == out_p.shape

    """Test whether IterativePointnet outputs required shape."""

    inp2 = torch.randn(100, 30, 2)
    generalized_iterative_pointnet = GeneralizedIterativePointNet(
        [1, 2, 3], 2, [[32, 64], [64, 128, 64], [128, 100]], True
    )
    out_gip = generalized_iterative_pointnet(inp2)
    assert out_gip.shape == (100, 100)


def test_backward() -> None:
    """Test whether backward function is possible to call on IterativePointnet."""

    iterative_pointnet = IterativePointNet(4, 3, [64, 64, 64, 128], True)
    inp = torch.randn(100, 500, 3)
    out1 = iterative_pointnet(inp)
    out_sum = torch.sum(out1)

    out_sum.backward()

    """Test whether backward function is possible to call on IterativePointnet."""

    generalized_iterative_pointnet = GeneralizedIterativePointNet(
        [2, 4], 3, [[64, 64, 64, 128], [156, 128]], True
    )

    inp2 = torch.randn(100, 500, 3)
    out2 = generalized_iterative_pointnet(inp2)
    out_sum2 = torch.sum(out2)

    out_sum2.backward()
