"""Parametrized networks for pose and shape estimation."""
import torch
import torch.nn as nn
from typing import List, Callable, Optional

from sdfest.initialization.so3grid import SO3Grid


class SDFPoseHead(nn.Module):
    """Parametrized head to estimate pose and shape from feature vector."""

    def __init__(
        self,
        in_size: int,
        mlp_out_sizes: List,
        shape_dimension: int,
        batchnorm: bool,
        orientation_repr: Optional[str] = "quaternion",
        orientation_grid_resolution: Optional[int] = None,
    ):
        """Initialize the SDFPoseHead.

        Args:
            in_size:            number of input features
            mlp_out_sizes:      output sizes of each linear layer
            shape_dimension:    dimension of shape description
            batchnorm:          whether to use batchnorm or not
            orientation_repr:
                The orientation represention. One of "quaternion"|"discretized".
            orientation_grid_resolution:
                The resolution of the SO3 grid.
                Only used when orientation_repr == "discretized".
        """
        super().__init__()

        self._in_size = in_size
        self._mlp_out_sizes = mlp_out_sizes
        self._batchnorm = batchnorm
        self._shape_dimension = shape_dimension
        self._orientation_repr = orientation_repr

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

        if orientation_repr == "quaternion":
            self._grid = None
            self._final_layer = nn.Linear(mlp_out_sizes[-1], self._shape_dimension + 8)
        elif orientation_repr == "discretized":
            self._grid = SO3Grid(orientation_grid_resolution)
            self._final_layer = nn.Linear(
                mlp_out_sizes[-1], self._shape_dimension + 4 + self._grid.num_cells()
            )
        else:
            raise NotImplementedError(
                f"orientation_repr {orientation_repr} is not supported."
            )

    def forward(self, x):
        """Forward pass of the module.

        Input represents set of input features used to compute pose.

        Args:
            x: batch of input vectors
        Returns:
            Tuple with the following entries:
                The predicted shape vector.
                The predicted pose.
                The predicted scale.
                The predicted orientation in the specified orientation representation.
                    For "quaternion" this will be of shape (N,4) with each quaternion
                    having the order (x, y, z, w), i.e., scalar-last, and normalized.
                    For "discretized" this will be of shape (N,M) based on the grid
                    resolution. No activation function is applied. I.e., softmax has
                    to be used to get probabilities, and cross_entropy_loss should be
                    used during training.
        """
        out = x
        for i, linear_layer in enumerate(self._linear_layers):
            out = linear_layer(out)
            if self._batchnorm:
                out = self._bn_layers[i](out)
            out = nn.functional.relu(out)

        # Normalize quaternion
        if self._orientation_repr == "quaternion":
            out = self._final_layer(out)
            orientation = out[:, self._shape_dimension + 4 :]
            orientation = orientation / torch.sqrt(
                torch.sum(orientation ** 2, 1, keepdim=True)
            )
        elif self._orientation_repr == "discretized":
            out = self._final_layer(out)
            orientation = out[:, self._shape_dimension + 4 :]
        else:
            raise NotImplementedError(
                f"orientation_repr {self.orientation_repr} is not supported."
            )

        return (
            out[:, 0 : self._shape_dimension],
            out[:, self._shape_dimension : self._shape_dimension + 3],
            out[:, self._shape_dimension + 3],
            orientation,
        )


class SDFPoseNet(nn.Module):
    """Pose and shape estimation from sensor data.

    Composed of feature extraction backbone and shape/pose head.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        """Construct SDF pose and shape network.

        Args:
            backbone:       function or class representing the backbone
            backbone_dict:  parameters passed to backbone on construction
            head:           function or class representing the head
            head_dict:      parameters passed to head on construction
        """
        super().__init__()
        self._backbone = backbone
        self._head = head

    def forward(self, x):
        """Forward pass.

        Args:
            x: input compatible with backbone.
        Returns:
            output from head
        """
        out = self._backbone(x)
        out = self._head(out)
        return out


if __name__ == "__main__":
    # simple sanity check
    sdf_pose_head = SDFPoseHead(
        1024,
        [512, 256, 128],
        10,
        True,
        orientation_repr="discretized",
        orientation_grid_resolution=0,
    )
    inp = torch.rand(16, 1024)  # batch size 16 with 1024 features
    out = sdf_pose_head(inp)
    print([x.shape for x in out])
