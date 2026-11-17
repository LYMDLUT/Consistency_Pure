"""PGD with Expectation over Transformation (EOT) against stochastic defenses."""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

NORMS = ("linf", "l2")


def _flat_norm(x: Tensor) -> Tensor:
    return x.flatten(start_dim=1).norm(p=2, dim=1).view(-1, *([1] * (x.ndim - 1)))


def pgd_eot(
    model_fn: Callable[[Tensor], Tensor],
    images: Tensor,
    labels: Tensor,
    norm: str = "linf",
    eps: float = 8 / 255,
    alpha: float = 1 / 255,
    steps: int = 20,
    eot_steps: int = 10,
    random_start: bool = False,
) -> Tensor:
    """Untargeted PGD on images in ``[0, 1]``.

    Each step averages the input gradient over ``eot_steps`` evaluations of the
    (stochastic) ``model_fn``, which maps images in ``[0, 1]`` to logits. ``eps`` and
    ``alpha`` are in the same ``[0, 1]`` pixel scale.
    """
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")

    delta = torch.zeros_like(images)
    if random_start:
        if norm == "linf":
            delta.uniform_(-eps, eps)
        else:
            delta.normal_()
            radius = torch.rand_like(_flat_norm(delta)) * eps
            delta = delta / _flat_norm(delta) * radius
        delta = delta.clamp(min=-images, max=1 - images)

    for _ in range(steps):
        delta.requires_grad_(True)
        grad = torch.zeros_like(images)
        for _ in range(eot_steps):
            loss = F.cross_entropy(model_fn(images + delta), labels)
            grad += torch.autograd.grad(loss, delta)[0]
        delta = delta.detach()

        if norm == "linf":
            delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
        else:
            delta = delta + alpha * grad / (_flat_norm(grad) + 1e-10)
            delta = delta * (eps / _flat_norm(delta)).clamp(max=1)
        delta = delta.clamp(min=-images, max=1 - images)

    return (images + delta).detach()
