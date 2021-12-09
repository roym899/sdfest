"""Point Transformer Block """
from sdf_single_shot.pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerBlock(nn.Module):
    """Point Transformer Block.

    Implements the steps linear, point transformer and linear layer along with the residual connection.

    """

    def __init__(self, d_points, d_model, k) -> None:
        """Initialize the TransformerBlock module.

        Args:
            d_points: length of feature Vector
            d_model: length of feature transformation vector
            k: k of the k nearest neighbor algorithm
        """
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        """Forward pass of the module.
        xyz is of dimension b x n x 3 and features is of dimension b x n x f where b is batch_size,
         n is points per batch and f is feature vector dimension.

        Args:
            xyz: batch of point sets
            features: batch of feature vectors corresponding to each point
        """

        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, : self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        # linear
        x = self.fc1(features)

        # point transformer
        q, k, v = (
            self.w_qs(x),
            index_points(self.w_ks(x), knn_idx),
            index_points(self.w_vs(x), knn_idx),
        )

        #
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x d_model

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x d_model

        res = torch.einsum("bnkd,bnkd->bnd", attn, v + pos_enc)

        # linear and residual
        res = self.fc2(res) + pre
        return res, attn
