import pytest
import torch
from conftest import ToyClassifier
from torch import nn

import classifiers
from classifiers import CIFAR10_MEAN, CIFAR10_STD, NormalizedClassifier, load_classifier
from common import DistContext, parse_number, shard_indices
from evaluation import METRICS, evaluate_attack
from networks import ResNet18, WideResNet
from networks.wide_resnet import Wide_ResNet


def test_normalized_classifier_normalizes_and_resizes():
    class Probe(nn.Module):
        def forward(self, x):
            self.seen = x
            return x.mean(dim=(2, 3))

    probe = Probe()
    classifier = NormalizedClassifier(probe, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), 16)
    classifier(torch.full((1, 3, 8, 8), 0.75))
    assert probe.seen.shape == (1, 3, 16, 16)
    torch.testing.assert_close(probe.seen, torch.ones(1, 3, 16, 16))


def test_load_pickled_t7_classifier(tmp_path):
    net = Wide_ResNet(depth=10, widen_factor=1, dropout_rate=0.0).eval()
    torch.save({"net": net, "acc": 95.0}, tmp_path / "origin.t7")

    classifier = load_classifier("t7", tmp_path / "origin.t7")

    x = torch.rand(2, 3, 32, 32)
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    torch.testing.assert_close(classifier(x), net((x - mean) / std))
    assert not any(p.requires_grad for p in classifier.parameters())


def small_wrn():
    return WideResNet(depth=10, widen_factor=1)


@pytest.mark.parametrize(
    "name,attr,build,wrap",
    [
        ("kzcls", "WideResNet", small_wrn, lambda sd: {"state_dict": sd, "epoch": 1}),
        (
            "r18",
            "ResNet18",
            ResNet18,
            lambda sd: {f"module.{k}": v for k, v in sd.items()},
        ),
    ],
)
def test_load_state_dict_classifiers_without_normalization(
    tmp_path, monkeypatch, name, attr, build, wrap
):
    monkeypatch.setattr(classifiers, attr, build)
    torch.manual_seed(0)
    net = build().eval()
    torch.save(wrap(net.state_dict()), tmp_path / "clf.pt")

    classifier = load_classifier(name, tmp_path / "clf.pt")

    x = torch.rand(2, 3, 32, 32)
    torch.testing.assert_close(classifier(x), net(x))


def test_load_vit_uses_processor_normalization(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=37,
        num_labels=10,
    )
    transformers.ViTForImageClassification(config).save_pretrained(tmp_path)
    transformers.ViTImageProcessor(
        size={"height": 32, "width": 32}, image_mean=[0.5] * 3, image_std=[0.5] * 3
    ).save_pretrained(tmp_path)

    classifier = load_classifier("vit", tmp_path)

    assert classifier.input_size == 32
    torch.testing.assert_close(classifier.mean.flatten(), torch.full((3,), 0.5))
    assert classifier(torch.rand(2, 3, 32, 32)).shape == (2, 10)


def test_shard_indices_cover_every_item_once():
    shards = [
        list(shard_indices(10, DistContext(r, 3, torch.device("cpu"))))
        for r in range(3)
    ]
    assert sorted(sum(shards, [])) == list(range(10))


def test_parse_number():
    assert parse_number("8/255") == pytest.approx(8 / 255)
    assert parse_number("0.5") == 0.5


def test_evaluate_attack_counts_every_image_once():
    classifier = ToyClassifier().requires_grad_(False)
    torch.manual_seed(0)
    images = torch.rand(7, 3, 8, 8)
    labels = classifier(images).argmax(dim=1)
    loader = [(images[:4], labels[:4]), (images[4:], labels[4:])]

    accuracy, num_images = evaluate_attack(
        loader,
        classifier,
        nn.Identity(),
        DistContext(0, 1, torch.device("cpu")),
        eps=0.0,
        alpha=0.0,
        steps=1,
        eot_steps=1,
    )

    assert num_images == 7
    assert accuracy == {name: 1.0 for name in METRICS}
