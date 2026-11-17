import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from lcm_purification import LCMPurifier  # noqa: E402


@pytest.fixture(scope="module")
def components():
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
    )
    vae = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(32, 64),
        latent_channels=4,
    )
    for module in (unet, vae):
        module.eval().requires_grad_(False)
    return vae, unet, torch.randn(1, 77, 32)


def make_purifier(components, **kwargs):
    vae, unet, prompt_embeds = components
    return LCMPurifier(
        vae, unet, diffusers.LCMScheduler(), prompt_embeds, resolution=16, **kwargs
    )


def test_runs_the_last_strength_fraction_of_steps(components):
    purifier = make_purifier(components, strength=0.1, num_inference_steps=50)
    calls = []
    handle = purifier.unet.register_forward_hook(lambda m, i, o: calls.append(i[1]))
    try:
        with torch.no_grad():
            out = purifier(torch.rand(3, 3, 8, 8))
    finally:
        handle.remove()

    assert out.shape == (3, 3, 16, 16)
    assert out.min() >= 0 and out.max() <= 1
    assert [int(t) for t in calls] == [99, 79, 59, 39, 19]


def test_gradient_checkpointing_does_not_change_outputs_or_gradients(components):
    images = torch.rand(2, 3, 16, 16)
    results = []
    for gradient_checkpointing in (True, False):
        x = images.clone().requires_grad_(True)
        torch.manual_seed(0)
        purifier = make_purifier(
            components, gradient_checkpointing=gradient_checkpointing
        )
        out = purifier(x)
        out.square().sum().backward()
        results.append((out.detach(), x.grad))

    (out_a, grad_a), (out_b, grad_b) = results
    torch.testing.assert_close(out_a, out_b)
    torch.testing.assert_close(grad_a, grad_b)
    assert grad_a.abs().sum() > 0


def test_rejects_strength_without_denoising_steps(components):
    with pytest.raises(ValueError):
        make_purifier(components, strength=0.01, num_inference_steps=50)
