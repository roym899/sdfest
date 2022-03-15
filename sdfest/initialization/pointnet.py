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

    def __init__(
        self,
        in_size: int,
        mlp_out_sizes: List,
        batchnorm: bool,
        residual: bool = False,
        dense: bool = False,
    ) -> None:
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
        self._residual = residual
        self._dense = dense

        # define layers
        self._linear_layers = torch.nn.ModuleList([])
        for i, out_size in enumerate(mlp_out_sizes):
            if i == 0:
                self._linear_layers.append(nn.Linear(self._in_size, out_size))
            else:
                if dense:
                    self._linear_layers.append(
                        nn.Linear(2 * mlp_out_sizes[i - 1], out_size)
                    )
                else:
                    self._linear_layers.append(
                        nn.Linear(mlp_out_sizes[i - 1], out_size)
                    )

        self._bn_layers = torch.nn.ModuleList([])
        if self._batchnorm:
            for out_size in mlp_out_sizes:
                self._bn_layers.append(nn.BatchNorm1d(out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Input has dimension NxMxC, where N is the batch size, M the number of points per
        set, and C the number of channels per point.

        Args:
            x: batch of point sets
        """
        set_size = x.shape[1]
        out = prev_out = x
        for i, linear_layer in enumerate(self._linear_layers):
            out = linear_layer(out)
            if self._batchnorm:
                # BN over channels across all points and sets
                pts_per_set = out.shape[1]
                out_view = out.view(-1, self._mlp_out_sizes[i])
                out = self._bn_layers[i](out_view)
                out = out.view(-1, pts_per_set, self._mlp_out_sizes[i])
            out = nn.functional.relu(out)

            if self._dense:
                out_max, _ = torch.max(out, 1, keepdim=True)
                if i != len(self._linear_layers) - 1:
                    out = torch.cat((out, out_max.expand(-1, set_size, -1)), dim=2)

            if self._residual:
                if prev_out.shape == out.shape:
                    out = prev_out + out
            prev_out = out

        # Maximum over points in same set
        out, _ = torch.max(out, 1)

        return out


class IterativePointNet(nn.Module):
    """Iterative PointNet which concatenates input.

    This is composed of 2 PointNets, where the first PointNet is applied once, the
    second PointNet a number of times, i.e.,
        out = PointNet1(in)
        for i in range(num_concat):
            out = PointNet2( concat( out, in ) )
    """

    def __init__(
        self, num_concat: int, in_size: int, mlp_out_sizes: List, batchnorm: bool
    ) -> None:
        """Initialize the IterativePointNet module.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input has dimension NxMxC, where N is the batch size, M the number of points
        per set, and C the number of channels per point.

        Args:
            x: batch of point sets
        """
        # apply 1st pointnet to input
        out = self.pointnet_1(x)  # shape (batch_size, num_outputs)
        set_size = x.shape[1]
        for _ in range(self.num_concat):
            # repeat output vector across 2nd dimension
            repeated_out = out.unsqueeze(1).repeat(1, set_size, 1)
            # concatenate input vector and repeated_out
            modified_x = torch.cat((repeated_out, x), 2)
            out = self.pointnet_2(modified_x)
        return out


class GeneralizedIterativePointNet(nn.Module):
    """Generalized Iterative PointNet composed of multiple IterativePointNet instances.

    This is a sequence of iterative pointnets, where the initial input will be
    concatenated to each input, e.g.,
        out = IterativePointNet1(in)
        out = IterativePointNet2(concat(out, in))
        out = IterativePointNet3(concat(out, in))
        ...
    """

    def __init__(
        self, list_concat: list, in_size: int, list_mlp_out_sizes: list, batchnorm: bool
    ) -> None:
        """Initialize GeneralizedIterativePointnet module.

        Args:
            list_concat:
                List of concatenations for each MLP.
            in_size: Dimension of the input points.
            list_mlp_out_sizes:
                List of Output sizes of each linear layer.
                It is a List of Lists.
            batchnorm: Whether to use batchnorm or not.
        """
        super().__init__()

        init_in_size = in_size
        self.iterative_pointnet_list = torch.nn.ModuleList([])
        temp_iterative_pointnet = IterativePointNet(
            list_concat[0], in_size, list_mlp_out_sizes[0], batchnorm
        )
        self.iterative_pointnet_list.append(temp_iterative_pointnet)
        for iterative_pointnet_num in range(1, len(list_mlp_out_sizes)):
            # the input size to new MLP should be the output size of the previous MLP
            # plus previous input size
            in_size = list_mlp_out_sizes[iterative_pointnet_num - 1][-1] + init_in_size
            temp_iterative_pointnet = IterativePointNet(
                list_concat[iterative_pointnet_num],
                in_size,
                list_mlp_out_sizes[iterative_pointnet_num],
                batchnorm,
            )
            self.iterative_pointnet_list.append(temp_iterative_pointnet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input has dimension NxMxC, where N is the batch size, M the number of points
        per set, and C the number of channels per point.

        Args:
            x: batch of point sets
        """
        set_size = x.shape[1]
        init_x = x
        for iterative_pointnet in self.iterative_pointnet_list:
            out = iterative_pointnet(x)  # shape (batch_size, num_outputs)
            # repeat output vector across 2nd dimension
            x = out.unsqueeze(1).repeat(1, set_size, 1)
            x = torch.cat((x, init_x), 2)
        return out


if __name__ == "__main__":
    # check if dimensions are consistent across pointnets

    inp1 = torch.randn(2, 500, 3)

    pointnet = VanillaPointNet(3, [64, 64, 1024], True)
    out_p = pointnet(inp1)

    iterative_pointnet = IterativePointNet(0, 3, [64, 64, 1024], True)
    out_ip = iterative_pointnet(inp1)

    assert out_p.shape == out_ip.shape

    # check if dimension is as expected

    inp2 = torch.randn(100, 50, 2)
    iterative_pointnet2 = IterativePointNet(3, 2, [32, 64, 64, 1024], True)
    out_ip2 = iterative_pointnet2(inp2)

    assert out_ip2.shape == (100, 1024)
