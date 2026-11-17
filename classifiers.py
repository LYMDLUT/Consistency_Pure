"""CIFAR classifiers protected by the purifier."""

from dataclasses import dataclass
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode

from common import PathLike, extract_state_dict, logger
from networks import ResNet18, WideResNet

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
NO_NORMALIZATION = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


@dataclass(frozen=True)
class ClassifierSpec:
    dataset: str
    default_ckpt: str
    description: str


CLASSIFIERS = {
    "t7": ClassifierSpec(
        "cifar10", "origin.t7", "WRN-28-10, pickled module saved as {'net': model}"
    ),
    "cifar100": ClassifierSpec(
        "cifar100",
        "wide-resnet-28x10-cifar100.t7",
        "WRN-28-10 on CIFAR-100, pickled module saved as {'net': model}",
    ),
    "kzcls": ClassifierSpec(
        "cifar10", "natural.pt", "WRN-28-10 (TRADES code), {'state_dict': ...}"
    ),
    "r18": ClassifierSpec("cifar10", "lastpoint.pth.tar", "ResNet-18 state dict"),
    "vit": ClassifierSpec(
        "cifar10",
        "vit-base-patch16-224-in21k-finetuned-cifar10",
        "Hugging Face ViT-B/16 checkpoint directory or hub id",
    ),
}


class NormalizedClassifier(nn.Module):
    """Takes images in ``[0, 1]``, applies the classifier's preprocessing and returns
    logits."""

    def __init__(
        self,
        net: nn.Module,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        input_size: Optional[int] = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        self.input_size = input_size
        self.interpolation = interpolation

    def forward(self, images: Tensor) -> Tensor:
        if self.input_size is not None:
            images = TF.resize(
                images,
                [self.input_size, self.input_size],
                interpolation=self.interpolation,
                antialias=True,
            )
        output = self.net((images - self.mean) / self.std)
        return getattr(output, "logits", output)


def _load_pickled_module(path: PathLike) -> nn.Module:
    # Whole pickled modules cannot be loaded with ``weights_only=True``; only load
    # trusted files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = checkpoint["net"] if isinstance(checkpoint, dict) else checkpoint
    return net.module if isinstance(net, nn.DataParallel) else net


def _load_vit(path: PathLike) -> NormalizedClassifier:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    net = AutoModelForImageClassification.from_pretrained(path)
    try:
        processor = AutoImageProcessor.from_pretrained(path)
        mean, std = tuple(processor.image_mean), tuple(processor.image_std)
        size = processor.size
        input_size = size if isinstance(size, int) else size["height"]
    except OSError:
        logger.warning("No image processor config in %s, using mean=std=0.5", path)
        mean, std, input_size = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 224
    return NormalizedClassifier(
        net, mean, std, input_size, interpolation=InterpolationMode.BICUBIC
    )


def load_classifier(name: str, path: Optional[PathLike] = None) -> NormalizedClassifier:
    """Builds one of ``CLASSIFIERS`` in eval mode with gradients disabled."""
    path = path or CLASSIFIERS[name].default_ckpt
    if name == "t7":
        classifier = NormalizedClassifier(
            _load_pickled_module(path), CIFAR10_MEAN, CIFAR10_STD
        )
    elif name == "cifar100":
        classifier = NormalizedClassifier(
            _load_pickled_module(path), CIFAR100_MEAN, CIFAR100_STD
        )
    elif name == "kzcls":
        net = WideResNet()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(extract_state_dict(checkpoint, key="state_dict"))
        classifier = NormalizedClassifier(net, *NO_NORMALIZATION)
    elif name == "r18":
        net = ResNet18()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(extract_state_dict(checkpoint, key="state_dict"))
        classifier = NormalizedClassifier(net, *NO_NORMALIZATION)
    elif name == "vit":
        classifier = _load_vit(path)
    else:
        raise ValueError(
            f"Unknown classifier {name!r}, choose from {list(CLASSIFIERS)}"
        )
    return classifier.eval().requires_grad_(False)
