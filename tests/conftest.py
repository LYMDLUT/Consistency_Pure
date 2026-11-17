import pytest
import torch
from torch import nn

import purification
from configs.cifar10_ncsnpp import get_config


def tiny_ncsnpp_config():
    config = get_config()
    config.model.nf = 32
    config.model.num_res_blocks = 1
    config.model.ch_mult = (1, 2)
    return config


@pytest.fixture
def tiny_ncsnpp(monkeypatch):
    """Makes ``purification`` build a small NCSN++ so that tests run quickly."""
    monkeypatch.setattr(purification, "get_config", tiny_ncsnpp_config)
    torch.manual_seed(0)
    return purification.build_consistency_model().eval().requires_grad_(False)


class ToyDenoiser(nn.Module):
    """Small deterministic stand-in for NCSN++ that depends on the noise input."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x, c_noise):
        return torch.tanh(self.conv(x)) * (1 + c_noise.view(-1, 1, 1, 1))


class ToyClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.linear = nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))
