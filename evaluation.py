"""Evaluation loop shared by the CIFAR and ImageNet attack scripts."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from tqdm import tqdm

from attacks import pgd_eot
from common import DistContext, all_reduce_sum

# clean:          classifier on clean images, no defense
# purified_clean: classifier on purified clean images
# adversarial:    classifier on the adversarial images, no defense
# robust:         classifier on purified adversarial images (the defended accuracy)
METRICS = ("clean", "purified_clean", "adversarial", "robust")


def evaluate_attack(
    loader: Iterable[tuple[Tensor, Tensor]],
    classifier: nn.Module,
    purifier: nn.Module,
    ctx: DistContext,
    **attack_kwargs,
) -> tuple[dict[str, float], int]:
    """Attacks ``classifier(purifier(x))`` with PGD-EOT on this process's shard.

    Returns the accuracies over all processes and the total number of images.
    """

    def defended(images: Tensor) -> Tensor:
        return classifier(purifier(images))

    correct = torch.zeros(len(METRICS), dtype=torch.long, device=ctx.device)
    total = torch.zeros((), dtype=torch.long, device=ctx.device)
    progress = tqdm(loader, disable=not ctx.is_main, dynamic_ncols=True)
    for images, labels in progress:
        images, labels = images.to(ctx.device), labels.to(ctx.device)
        with torch.no_grad():
            clean = classifier(images).argmax(dim=1) == labels
            purified_clean = defended(images).argmax(dim=1) == labels
        adversarial_images = pgd_eot(defended, images, labels, **attack_kwargs)
        with torch.no_grad():
            adversarial = classifier(adversarial_images).argmax(dim=1) == labels
            robust = defended(adversarial_images).argmax(dim=1) == labels

        batch_correct = (clean, purified_clean, adversarial, robust)
        correct += torch.stack([m.sum() for m in batch_correct])
        total += labels.numel()
        seen = int(total)
        progress.set_postfix(
            {m: f"{c / seen:.3f}" for m, c in zip(METRICS, correct.tolist())}
        )

    all_reduce_sum(correct)
    all_reduce_sum(total)
    num_images = int(total)
    accuracy = {m: c / max(num_images, 1) for m, c in zip(METRICS, correct.tolist())}
    return accuracy, num_images
