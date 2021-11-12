"""Parametrized PointNet."""
import torch
import torch.nn as nn
from typing import List


class VanillaPointNet(nn.Module):
    """Parametrized PointNet without transformation layers (no T-nets).

    Generally following:
        PointNet Deep Learning on Point Sets for 3D Classification and Segmentation
        Qi, 2017
    """

    def __init__(self, in_size: int, mlp_out_sizes: List, batchnorm: bool):
        """Initialize the VanillaPointNet module.

        This module will only implements the MLP + MaxPooling part of the pointnet.

        It still requires a task specific head.

        Args:
            in_size:        dimension of the input points
            mlp_out_sizes:  output sizes of each linear layer
            batchnorm:      whether to use batchnorm or not
        """
        super().__init__()

        self._in_size = in_size
        self._mlp_out_sizes = mlp_out_sizes
        self._batchnorm = batchnorm

        # define layers
        self._linear_layers = torch.nn.ModuleList([])
        for i, out_size in enumerate(mlp_out_sizes):
            if i == 0:
                self._linear_layers.append(nn.Linear(self._in_size, out_size))
            else:
                self._linear_layers.append(nn.Linear(mlp_out_sizes[i - 1], out_size))

        self._bn_layers = torch.nn.ModuleList([])
        if self._batchnorm:
            for out_size in mlp_out_sizes:
                self._bn_layers.append(nn.BatchNorm1d(out_size))

    def forward(self, x):
        """Forward pass of the module.

        Input has dimension NxMxC, where N is the batch size, M the number of points
        per set, and C the number of channels per point.

        Args:
            x: batch of point sets
        """
        out = x
        for i, linear_layer in enumerate(self._linear_layers):
            out = linear_layer(out)
            if self._batchnorm:
                # BN over channels across all points and sets
                pts_per_set = out.shape[1]
                out_view = out.view(-1, self._mlp_out_sizes[i])
                out = self._bn_layers[i](out_view)
                out = out.view(-1, pts_per_set, self._mlp_out_sizes[i])
            out = nn.functional.relu(out)
        # Maximum over points in same set
        out, _ = torch.max(out, 1)
        return out


class IteratativePointNet(nn.Module):
    """Iterated PointNet which concatenates input with output of previous stage."""

    def __init__(
        self, num_concat: int, in_size: int, mlp_out_sizes: List, batchnorm: bool
    ):
        """Initialize the IteratativePointNet module.

        Args:
            num_concat:
                Number of concatenations of input and previous iteration.
                If 0 this module is the same as VanillaPointNet.
            in_size: Dimension of the input points.
            mlp_out_sizes: Output sizes of each linear layer.
            batchnorm: Whether to use batchnorm or not.
        """
        super().__init__()
        self.num_concat = num_concat
        # create 1st pointnet for taking points of channel = in_size
        self.pointnet_1 = VanillaPointNet(in_size, mlp_out_sizes, batchnorm)
        # create 2nd pointnet for taking points of channel = size of concatenated vector
        self.pointnet_2 = VanillaPointNet(
            in_size + mlp_out_sizes[-1], mlp_out_sizes, batchnorm
        )

    def forward(self, x):
        """Forward pass.

        Input has dimension NxMxC, where N is the batch size, M the number of points
        per set, and C the number of channels per point.

        Args:
            x: batch of point sets
        """
        out = self.pointnet_1(x)
        batchsize, set_size, channels = x.shape
        for concat_step in range(self.num_concat):
            # apply 1st pointnet to input
            # out has dim (batchsize, num_outputs)
            # repeat output vector across 2nd dimension (dim = batchsize, set_size, num_outputs)
            repeated_out = out.unsqueeze(1).repeat(1, set_size, 1)
            # concatenate input vector and repeated_out
            modified_x = torch.cat((repeated_out, x), 2)
            out = self.pointnet_2(modified_x)
        return out


if __name__ == "__main__":
    # check if dimensions are consistent across pointnets

    inp1 = torch.randn(2, 500, 3)

    pointnet = VanillaPointNet(3, [64, 64, 1024], True)
    out_p = pointnet(inp1)

    iteratative_pointnet = IteratativePointNet(0, 3, [64, 64, 1024], True)
    out_ip = iteratative_pointnet(inp1)

    assert out_p.shape == out_ip.shape

    # check if dimension is as expected

    inp2 = torch.randn(100, 50, 2)
    iteratative_pointnet2 = IteratativePointNet(3, 2, [32, 64, 64, 1024], True)
    out_ip2 = iteratative_pointnet2(inp2)

    assert out_ip2.shape == (100, 1024)
