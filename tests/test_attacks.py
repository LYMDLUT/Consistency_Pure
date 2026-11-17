import pytest
import torch
import torch.nn.functional as F
from conftest import ToyClassifier

from attacks import pgd_eot


def original_linf(model_fn, sample_image, y, epsilon, alpha, pgd_step, eot_step):
    """PGD-Linf loop of ``cm_attack_eval_linf.py``; images in [-1, 1]."""
    lower_limit, upper_limit = -1.0, 1.0
    delta = torch.zeros_like(sample_image)
    delta.requires_grad = True
    for _ in range(pgd_step):
        eot = 0
        for _ in range(eot_step):
            loss = F.cross_entropy(model_fn(sample_image + delta), y)
            loss.backward()
            eot += delta.grad.detach()
            delta.grad.zero_()
        d = torch.clamp(delta + alpha * eot.sign(), -epsilon, epsilon)
        d = torch.max(
            torch.min(d, upper_limit - sample_image), lower_limit - sample_image
        )
        delta.data = d
    return sample_image + delta.detach()


def original_l2(model_fn, sample_image, y, epsilon, alpha, pgd_step, eot_step):
    """PGD-L2 loop of ``cm_attack_eval_l2.py``; images in [-1, 1]."""
    lower_limit, upper_limit = -1.0, 1.0
    batch_size = sample_image.shape[0]
    delta = torch.zeros_like(sample_image)
    delta.requires_grad = True
    for _ in range(pgd_step):
        eot = 0
        for _ in range(eot_step):
            loss = F.cross_entropy(model_fn(sample_image + delta), y)
            loss.backward()
            eot += delta.grad.detach()
            delta.grad.zero_()
        grad_norms = torch.norm(eot.view(batch_size, -1), p=2, dim=1) + 1e-10
        eot = eot / grad_norms.view(batch_size, 1, 1, 1)
        delta = delta + alpha * eot
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = torch.min(epsilon / delta_norms.view(-1, 1, 1, 1), torch.ones(1))
        delta = delta * factor
        delta = torch.max(
            torch.min(delta, upper_limit - sample_image), lower_limit - sample_image
        ).detach()
        delta.requires_grad = True
    return sample_image + delta.detach()


@pytest.fixture
def problem():
    torch.manual_seed(0)
    images = torch.rand(6, 3, 8, 8)
    labels = torch.randint(0, 10, (6,))
    return ToyClassifier().requires_grad_(False), images, labels


def test_linf_matches_original_scaled_to_unit_range(problem):
    classifier, images, labels = problem
    expected = original_linf(
        lambda u: classifier(u / 2 + 0.5),
        images * 2 - 1,
        labels,
        epsilon=(8 / 255) / 0.5,
        alpha=(1 / 255) / 0.5,
        pgd_step=5,
        eot_step=2,
    )
    actual = pgd_eot(classifier, images, labels, "linf", 8 / 255, 1 / 255, 5, 2)
    torch.testing.assert_close(actual, expected / 2 + 0.5, atol=1e-6, rtol=0)


def test_l2_matches_original_scaled_to_unit_range(problem):
    classifier, images, labels = problem
    expected = original_l2(
        lambda u: classifier(u / 2 + 0.5),
        images * 2 - 1,
        labels,
        epsilon=0.5 / 0.5,
        alpha=0.1 / 0.5,
        pgd_step=8,
        eot_step=2,
    )
    actual = pgd_eot(classifier, images, labels, "l2", 0.5, 0.1, 8, 2)
    torch.testing.assert_close(actual, expected / 2 + 0.5, atol=1e-6, rtol=0)


@pytest.mark.parametrize("norm,eps", [("linf", 8 / 255), ("l2", 0.5)])
@pytest.mark.parametrize("random_start", [False, True])
def test_attack_respects_budget_and_box(problem, norm, eps, random_start):
    classifier, images, labels = problem
    adversarial = pgd_eot(
        classifier,
        images,
        labels,
        norm,
        eps,
        eps / 4,
        steps=10,
        eot_steps=1,
        random_start=random_start,
    )
    delta = (adversarial - images).flatten(start_dim=1)
    size = delta.abs().amax(dim=1) if norm == "linf" else delta.norm(dim=1)
    assert torch.all(size <= eps + 1e-6)
    assert adversarial.min() >= 0 and adversarial.max() <= 1


def test_attack_increases_loss(problem):
    classifier, images, labels = problem
    adversarial = pgd_eot(classifier, images, labels, "linf", 8 / 255, 1 / 255, 10, 1)
    assert F.cross_entropy(classifier(adversarial), labels) > F.cross_entropy(
        classifier(images), labels
    )


def test_unknown_norm_is_rejected(problem):
    classifier, images, labels = problem
    with pytest.raises(ValueError):
        pgd_eot(classifier, images, labels, norm="l1")
