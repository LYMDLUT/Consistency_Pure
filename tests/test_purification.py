import pytest
import torch
from conftest import ToyDenoiser

from common import extract_state_dict
from purification import (
    SIGMA_MIN,
    ConsistencyPurifier,
    consistency_function,
    load_consistency_model,
    purification_sigmas,
)


def original_purify(model, images, sigma_max, steps, time_min=0.002, data_std=0.5):
    """The noising and ``ConsistencyPipeline`` of the original scripts, in [-1, 1]."""
    time_min = torch.tensor(time_min)
    time = torch.tensor(sigma_max)
    sample = images + torch.sqrt(time**2 - time_min**2) * torch.randn(images.shape)
    for step in range(steps):
        if step > 0:
            time = (2 * time + time_min) / 3
            sigma = torch.sqrt(time**2 - time_min**2 + 1e-6)
            sample = sample + sigma * torch.randn(sample.shape)
        out = model(
            sample * (1 / torch.sqrt(time**2 + data_std**2)),
            0.25 * torch.log(torch.ones(sample.shape[0]) * time),
        )
        skip_coef = data_std**2 / ((time - time_min) ** 2 + data_std**2)
        out_coef = data_std * (time - time_min) / torch.sqrt(time**2 + data_std**2)
        sample = (sample * skip_coef + out * out_coef).clamp(-1.0, 1.0)
    return (sample / 2 + 0.5).clamp(0, 1)


@pytest.mark.parametrize("steps", [1, 3])
def test_purifier_matches_original_pipeline(steps):
    model = ToyDenoiser()
    images = torch.rand(4, 3, 8, 8)

    torch.manual_seed(0)
    expected = original_purify(model, images * 2 - 1, 0.5, steps)
    torch.manual_seed(0)
    actual = ConsistencyPurifier(model, 0.5, steps)(images)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


def test_consistency_function_is_identity_at_sigma_min():
    x = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(consistency_function(ToyDenoiser(), x, SIGMA_MIN), x)


def test_purification_sigmas():
    assert purification_sigmas(0.5, 1) == [0.5]
    sigmas = purification_sigmas(0.5, 3)
    assert sigmas[1] == pytest.approx((2 * 0.5 + SIGMA_MIN) / 3)
    assert all(a > b > SIGMA_MIN for a, b in zip(sigmas, sigmas[1:]))


def test_purifier_rejects_sigma_below_sigma_min():
    with pytest.raises(ValueError):
        ConsistencyPurifier(ToyDenoiser(), SIGMA_MIN)


def test_ncsnpp_purifier_is_random_and_differentiable(tiny_ncsnpp):
    purifier = ConsistencyPurifier(tiny_ncsnpp, 0.5, 2)
    images = torch.rand(2, 3, 32, 32, requires_grad=True)

    out = purifier(images)
    out.sum().backward()

    assert out.shape == images.shape
    assert out.min() >= 0 and out.max() <= 1
    assert images.grad is not None and images.grad.abs().sum() > 0
    assert not torch.equal(out.detach(), purifier(images).detach())


def test_load_consistency_model_formats(tiny_ncsnpp, tmp_path):
    state_dict = tiny_ncsnpp.state_dict()
    plain, wrapped, training = (tmp_path / n for n in ("a.pth", "b.pth", "c.pt"))
    torch.save(state_dict, plain)
    torch.save({f"module.{k}": v for k, v in state_dict.items()}, wrapped)
    torch.save({"ema": state_dict, "model": state_dict, "step": 1}, training)

    for path in (plain, wrapped, training):
        loaded = load_consistency_model(path)
        assert not loaded.training
        assert not any(p.requires_grad for p in loaded.parameters())
        for name, value in loaded.state_dict().items():
            torch.testing.assert_close(value, state_dict[name])


def test_extract_state_dict_handles_wrappers_and_checkpoints():
    assert list(extract_state_dict({"module.a": torch.zeros(1)})) == ["a"]
    checkpoint = {"ema": {"a": torch.zeros(1)}, "model": {"b": torch.zeros(1)}}
    assert list(extract_state_dict(checkpoint)) == ["a"]
    assert list(extract_state_dict(checkpoint, key="model")) == ["b"]
