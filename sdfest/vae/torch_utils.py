"""Generally useful modules for pytorch."""
import torch


class View(torch.nn.Module):
    """Wrapper of torch's view method to use with nn.Sequential."""

    def __init__(self, *shape):
        """Construct the module."""
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        """Reshape the tensor."""
        return x.view(*self.shape)
