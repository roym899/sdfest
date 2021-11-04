


from sdf_single_shot.pointnet import VanillaPointNet, IteratativePointNet
def test_equality_between_pointnets():
    """Test whether VanillaPointnet and IterativePointnet with num_concat = 1 outputs same result"""
    pointnet = VanillaPointNet(3, [64, 64, 1024], True)
    iteratative_pointnet = IteratativePointNet(1, 3, [64, 64, 64, 128, 1024], True)

    inp = torch.randn(100, 500, 3)
    out1 = pointnet(inp)
    out2 = iteratative_pointnet(inp)
    assert out1 == out2
