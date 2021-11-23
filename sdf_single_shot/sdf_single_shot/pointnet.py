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
        dense: bool = False,
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

    def forward(self, x):
        """Forward pass of the module.

        Input has dimension NxMxC, where N is the batch size, M the number of points
        per set, and C the number of channels per point.

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
