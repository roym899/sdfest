"""Test for utils module."""
import inspect

from sdfest.initialization import utils


def test_str_to_object() -> None:
    """Test str_to_object function."""
    a = 1

    assert utils.str_to_object("a") is a

    assert utils.str_to_object("utils") is utils

    assert inspect.isclass(utils.str_to_object("torch.Tensor"))

    assert utils.str_to_object("____") is None
