"""Test quaternion_utils module."""

import torch

from sdf_single_shot import quaternion_utils


def test_quaternion_invert() -> None:
    """Test inverting of quaternions."""
    q = torch.tensor([0, 0, 0, 1.0])
    q_inv = quaternion_utils.quaternion_invert(q)
    assert torch.all(q == q_inv)

    q = torch.tensor([1.0, 0, 0, 0.0])
    q_inv = quaternion_utils.quaternion_invert(q)
    q_inv_exp = torch.tensor([-1.0, 0, 0, 0])
    assert torch.all(q_inv == q_inv_exp)

    # test other dimension
    q = torch.rand(5, 3, 4)
    q_inv = quaternion_utils.quaternion_invert(q)
    assert q.shape == q_inv.shape


def test_quaternion_apply() -> None:
    """Test inverting of quaternions."""
    q = torch.tensor([0, 0, 0, 1.0])
    p = torch.tensor([1.0, 1.0, 0])
    p_rot = quaternion_utils.quaternion_apply(q, p)
    assert torch.all(p_rot == p)

    q = torch.tensor([1.0, 0, 0, 0])  # rotate around x by pi
    p = torch.tensor([1.0, 1.0, 0])
    p_rot = quaternion_utils.quaternion_apply(q, p)
    p_rot_exp = torch.tensor([1.0, -1.0, 0])
    assert torch.all(p_rot == p_rot_exp)

    # test batched points
    q = torch.tensor([1.0, 0, 0, 0])  # rotate around x by pi
    p = torch.tensor([[1.0, 1.0, 0], [1.0, 2.0, 0]])
    p_rot = quaternion_utils.quaternion_apply(q, p)
    p_rot_exp = torch.tensor([[1.0, -1.0, 0], [1.0, -2.0, 0]])
    assert torch.all(p_rot == p_rot_exp)

    # test batched quaternions
    q = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    p = torch.tensor([1.0, 1.0, 0])
    p_rot = quaternion_utils.quaternion_apply(q, p)
    p_rot_exp = torch.tensor([[1.0, -1.0, 0], [-1.0, 1.0, 0]])
    assert torch.all(p_rot == p_rot_exp)
