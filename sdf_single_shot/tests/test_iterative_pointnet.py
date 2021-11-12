import torch


from sdf_single_shot.pointnet import VanillaPointNet, IteratativePointNet


def test_shape_equality_pointnets():
    """Test whether VanillaPointnet and IterativePointnet outputs same shape"""
    inp1 = torch.randn(2, 500, 3)

    pointnet = VanillaPointNet(3, [64, 64, 1024], True)
    out_p = pointnet(inp1)

    iteratative_pointnet = IteratativePointNet(0, 3, [64, 64, 1024], True)
    out_ip = iteratative_pointnet(inp1)

    assert out_p.shape == out_ip.shape

    """Test whether IterativePointnet outputs required shape"""
    inp2 = torch.randn(100, 50, 2)
    iteratative_pointnet2 = IteratativePointNet(3, 2, [32, 64, 64, 1024], True)
    out_ip2 = iteratative_pointnet2(inp2)

    assert out_ip2.shape == (100, 1024)


def test_backward() -> None:
    """Test whether backward function is possible to call."""

    iteratative_pointnet = IteratativePointNet(4, 3, [64, 64, 64, 128], True)
    inp = torch.randn(100, 500, 3)
    out1 = iteratative_pointnet(inp)
    out_sum = torch.sum(out1)

    out_sum.backward()
