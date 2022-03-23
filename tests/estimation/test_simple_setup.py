"""Test simple_setup module."""
from sdfest.estimation import simple_setup
import torch


def test_adjust_categorical_posterior() -> None:
    """Test adjustment of posterior."""
    # no change
    posterior = torch.tensor([0.8, 0.2, 0.0, 0.0])
    prior = torch.tensor([0.25, 0.25, 0.25, 0.25])
    train_prior = torch.tensor([0.4, 0.4, 0.1, 0.1])
    adjusted_posterior = simple_setup.SDFPipeline._adjust_categorical_posterior(
        posterior, prior, train_prior
    )
    assert torch.allclose(adjusted_posterior, posterior)

    # change
    posterior = torch.tensor([0.8, 0.2, 0.0, 0.0])
    prior = torch.tensor([0.1, 0.4, 0.25, 0.25])
    train_prior = torch.tensor([0.4, 0.4, 0.1, 0.1])
    adjusted_posterior = simple_setup.SDFPipeline._adjust_categorical_posterior(
        posterior, prior, train_prior
    )
    exp_adjusted_posterior = torch.tensor([0.8 * 0.1 / 0.4, 0.2 * 0.4 / 0.4, 0.0, 0.0])
    exp_adjusted_posterior /= torch.sum(exp_adjusted_posterior)  # renormalize
    assert torch.allclose(adjusted_posterior, exp_adjusted_posterior)
