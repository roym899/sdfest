import torch


from sdfest.initialization.pointnet import VanillaPointNet, IterativePointNet


def test_shape_equality_iteratative() -> None:
    """Test various equalities w.r.t. shape of IterativePointNet output."""

    # Test whether VanillaPointNet and IterativePointNet
    # outputs same shape.

    inp1 = torch.randn(2, 500, 3)

    pointnet = VanillaPointNet(3, [64, 64, 1024], True)
    out_p = pointnet(inp1)

    iterative_pointnet = IterativePointNet(0, 3, [64, 64, 1024], True)
    out_ip = iterative_pointnet(inp1)

    assert out_p.shape == out_ip.shape

    # Test whether IterativePointnet outputs required shape.
    inp2 = torch.randn(100, 50, 2)
    iterative_pointnet2 = IterativePointNet(3, 2, [32, 64, 64, 1024], True)
    out_ip2 = iterative_pointnet2(inp2)

    assert out_ip2.shape == (100, 1024)


def test_backward_iterative() -> None:
    """Test whether backward function is possible to call on
    IterativePointNet."""

    iterative_pointnet = IterativePointNet(4, 3, [64, 64, 64, 128], True)
    inp = torch.randn(100, 500, 3)
    out1 = iterative_pointnet(inp)
    out_sum = torch.sum(out1)

    out_sum.backward()
