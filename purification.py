"""Adversarial purification with a CIFAR-10 consistency model."""

import math

import torch
from torch import Tensor, nn

from common import PathLike, extract_state_dict
from configs.cifar10_ncsnpp import get_config
from models.ncsnpp import NCSNpp

SIGMA_MIN = 0.002
SIGMA_DATA = 0.5


def build_consistency_model() -> NCSNpp:
    return NCSNpp(get_config())


def load_consistency_model(path: PathLike, key: str = "ema") -> NCSNpp:
    """Loads NCSN++ weights from a plain state dict or a training checkpoint.

    For training checkpoints of improved_consistency_models_cifar10_pytorch, ``key``
    selects the EMA (``"ema"``) or the online (``"model"``) weights.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_consistency_model()
    model.load_state_dict(extract_state_dict(checkpoint, key))
    return model.eval().requires_grad_(False)


def consistency_function(
    model: nn.Module,
    x: Tensor,
    sigma: float,
    sigma_min: float = SIGMA_MIN,
    sigma_data: float = SIGMA_DATA,
) -> Tensor:
    """Maps ``x`` at noise level ``sigma`` to a clean image in ``[-1, 1]``.

    Uses the skip/output scalings of Song et al. (2023), which make the function the
    identity at ``sigma_min``.
    """
    c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
    c_out = sigma_data * (sigma - sigma_min) / math.sqrt(sigma**2 + sigma_data**2)
    c_in = 1 / math.sqrt(sigma**2 + sigma_data**2)
    c_noise = torch.full((x.shape[0],), 0.25 * math.log(sigma), device=x.device)
    return c_skip * x + c_out * model(c_in * x, c_noise)


def purification_sigmas(
    sigma_max: float, num_steps: int, sigma_min: float = SIGMA_MIN
) -> list[float]:
    """Noise levels of multi-step purification; each step moves 1/3 towards
    ``sigma_min``."""
    sigmas = [sigma_max]
    for _ in range(num_steps - 1):
        sigmas.append((2 * sigmas[-1] + sigma_min) / 3)
    return sigmas


class ConsistencyPurifier(nn.Module):
    """Diffuses images to ``sigma_max`` and maps them back with a consistency model.

    With ``num_steps > 1`` the result is re-noised to the next (smaller) noise level
    and denoised again, as in multi-step consistency sampling. The output is random and
    differentiable with respect to the input, which the EOT attacks rely on.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_max: float,
        num_steps: int = 1,
        sigma_min: float = SIGMA_MIN,
        sigma_data: float = SIGMA_DATA,
    ) -> None:
        super().__init__()
        if sigma_max <= sigma_min:
            raise ValueError(f"sigma_max must exceed sigma_min={sigma_min}")
        self.model = model
        self.sigmas = purification_sigmas(sigma_max, num_steps, sigma_min)
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

    def forward(self, images: Tensor) -> Tensor:
        """Purifies images in ``[0, 1]``."""
        x = images * 2 - 1
        for sigma in self.sigmas:
            noise_std = math.sqrt(sigma**2 - self.sigma_min**2)
            x = x + noise_std * torch.randn_like(x)
            x = consistency_function(
                self.model, x, sigma, self.sigma_min, self.sigma_data
            ).clamp(-1, 1)
        return (x + 1) / 2
