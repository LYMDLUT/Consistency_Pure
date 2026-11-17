"""Adversarial purification with Stable Diffusion and LCM-LoRA (ImageNet)."""

import torch
import torchvision.transforms.functional as TF
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class LCMPurifier(nn.Module):
    """Differentiable img2img purification with a latent consistency model.

    Matches ``StableDiffusionImg2ImgPipeline`` with an ``LCMScheduler``, an empty
    prompt and ``guidance_scale=0``, but keeps the autograd graph so that attacks can
    backpropagate through it. Of the ``num_inference_steps`` scheduler steps only the
    last ``int(num_inference_steps * strength)`` are run, as in the pipeline.
    """

    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        scheduler,
        prompt_embeds: Tensor,
        strength: float = 0.1,
        num_inference_steps: int = 50,
        resolution: int = 512,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        num_denoising_steps = min(
            int(num_inference_steps * strength), num_inference_steps
        )
        if num_denoising_steps < 1:
            raise ValueError("strength * num_inference_steps must be at least 1")
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.register_buffer("prompt_embeds", prompt_embeds)
        self.num_inference_steps = num_inference_steps
        self.start_index = (num_inference_steps - num_denoising_steps) * scheduler.order
        self.resolution = resolution
        self.gradient_checkpointing = gradient_checkpointing

    def _timesteps(self, device: torch.device) -> Tensor:
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        self.scheduler.set_begin_index(self.start_index)
        return self.scheduler.timesteps[self.start_index :]

    def _predict_noise(self, latents: Tensor, t: Tensor, embeds: Tensor) -> Tensor:
        return self.unet(latents, t, encoder_hidden_states=embeds, return_dict=False)[0]

    def forward(self, images: Tensor) -> Tensor:
        """Purifies images in ``[0, 1]``; returns ``resolution``-sized images in
        ``[0, 1]``."""
        size = [self.resolution, self.resolution]
        x = TF.resize(images, size, antialias=True) * 2 - 1
        scaling = self.vae.config.scaling_factor

        timesteps = self._timesteps(x.device)
        latents = self.vae.encode(x).latent_dist.sample() * scaling
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(
            latents, noise, timesteps[:1].repeat(latents.shape[0])
        )
        embeds = self.prompt_embeds.expand(latents.shape[0], -1, -1)
        for t in timesteps:
            model_input = self.scheduler.scale_model_input(latents, t)
            if self.gradient_checkpointing and torch.is_grad_enabled():
                noise_pred = checkpoint(
                    self._predict_noise, model_input, t, embeds, use_reentrant=False
                )
            else:
                noise_pred = self._predict_noise(model_input, t, embeds)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        decoded = self.vae.decode(latents / scaling, return_dict=False)[0]
        return (decoded / 2 + 0.5).clamp(0, 1)


def load_lcm_purifier(
    sd_model: str,
    lcm_lora: str,
    device: torch.device,
    **purifier_kwargs,
) -> LCMPurifier:
    """Loads Stable Diffusion, swaps in the LCM scheduler and fuses the LCM-LoRA.

    An empty ``lcm_lora`` skips the LoRA, e.g. for checkpoints that are already
    distilled.
    """
    from diffusers import LCMScheduler, StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        sd_model,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if lcm_lora:
        pipe.load_lora_weights(lcm_lora)
        pipe.fuse_lora()
    pipe.to(device)
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(
            "", device, num_images_per_prompt=1, do_classifier_free_guidance=False
        )
    purifier = LCMPurifier(
        pipe.vae, pipe.unet, pipe.scheduler, prompt_embeds, **purifier_kwargs
    )
    return purifier.eval().requires_grad_(False)
