"""Parametrized PointNet."""
import torch
import torch.nn as nn
from typing import List, Optional


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
        tnet: Optional[dict] = None,
    ):
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
        self._end_1 = nn.Linear(mlp_out_sizes[-1], 1)

        # define layers
        self._linear_layers_2 = torch.nn.ModuleList([])
        for i, out_size in enumerate(mlp_out_sizes):
            if i == 0:
                self._linear_layers_2.append(nn.Linear(self._in_size, out_size))
            else:
                self._linear_layers_2.append(nn.Linear(mlp_out_sizes[i - 1], out_size))

        self._bn_layers_2 = torch.nn.ModuleList([])
        if self._batchnorm:
            for out_size in mlp_out_sizes:
                self._bn_layers_2.append(nn.BatchNorm1d(out_size))

        # define layers
        self._linear_layers_3 = torch.nn.ModuleList([])
        for i, out_size in enumerate(mlp_out_sizes):
            if i == 0:
                self._linear_layers_3.append(nn.Linear(self._in_size + mlp_out_sizes[-1], out_size))
            else:
                self._linear_layers_3.append(nn.Linear(mlp_out_sizes[i - 1], out_size))

        self._bn_layers_3 = torch.nn.ModuleList([])
        if self._batchnorm:
            for out_size in mlp_out_sizes:
                self._bn_layers_3.append(nn.BatchNorm1d(out_size))

        # define t-net layers

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
        global_feat, _ = torch.max(out, 1, keepdim=True)
        set_size = x.shape[1]
        print(x.shape)
        print(global_feat.shape)
        out = torch.cat((x, global_feat.expand(-1, set_size, -1)), dim=2)

        for i, linear_layer in enumerate(self._linear_layers_3):
            out = linear_layer(out)
            if self._batchnorm:
                # BN over channels across all points and sets
                pts_per_set = out.shape[1]
                out_view = out.view(-1, self._mlp_out_sizes[i])
                out = self._bn_layers_3[i](out_view)
                out = out.view(-1, pts_per_set, self._mlp_out_sizes[i])
            out = nn.functional.relu(out)
        factors = torch.sigmoid(self._end_1(out))

        out = x
        for i, linear_layer in enumerate(self._linear_layers_2):
            out = linear_layer(out)
            if self._batchnorm:
                # BN over channels across all points and sets
                pts_per_set = out.shape[1]
                out_view = out.view(-1, self._mlp_out_sizes[i])
                out = self._bn_layers_2[i](out_view)
                out = out.view(-1, pts_per_set, self._mlp_out_sizes[i])
            out = nn.functional.relu(out)

        # multiply features by factors
        out = out * factors

        # Maximum over points in same set
        out, _ = torch.max(out, 1)

        return out


if __name__ == "__main__":
    # simple sanity check
    pointnet = VanillaPointNet(3, [64, 64, 64, 128, 1024], True)
    inp = torch.randn(1, 500, 3)
    out = pointnet(inp)
    print(out.shape)
