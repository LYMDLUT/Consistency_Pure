from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
import torch
import os
import timm
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from tqdm import tqdm
from loguru import logger
from torch.utils.checkpoint import checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from torch.distributed import all_gather
import argparse

import torch
from typing import Any, Callable, Dict, List, Optional, Union
from torchvision import transforms
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from torch.utils.checkpoint import checkpoint
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps
from torchvision.utils import save_image
from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler
from diffusers.utils import load_image, make_image_grid
resize_transform1 = transforms.Resize((512, 512))
resize_transform2 = transforms.Resize((224, 224))

class StableDiffusionImg2ImgPipeline_Warp(StableDiffusionImg2ImgPipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        #with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = checkpoint(
                self.unet,
                latent_model_input,
                t,
                prompt_embeds,
                None,
                timestep_cond,
                None,
                self.cross_attention_kwargs,
                added_cond_kwargs,
                None,
                None,
                None,
                None,
                False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        has_nsfw_concept = [False]
        do_denormalize = [True]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_test", type=int, default=64)
parser.add_argument("--pgd_step", type=int, default=20)
parser.add_argument("--eot_step", type=int, default=1)
parser.add_argument("--denoise_strength", type=float, default=0.1)
parser.add_argument("--inference_step", type=int, default=1000)
parser.add_argument("--model_id", type=str, default='../ckpt/ddpm_ema_cifar10')
parser.add_argument("--model_type", type=str, default='t7')
parser.add_argument("--fix_type", type=str, default='large')
parser.add_argument("--save_folder", type=str, default=None)
parser.add_argument("--train_data_dir", type=str, default=None)
FLAGS = parser.parse_args()

local_rank = FLAGS.local_rank
init_process_group(backend="nccl")
torch.cuda.set_device(FLAGS.local_rank)
world_size = torch.distributed.get_world_size()
device = torch.device("cuda", local_rank)


Denoise_strength = FLAGS.denoise_strength
Num_inference_steps = FLAGS.inference_step

model_id = FLAGS.model_id

batch_size = FLAGS.batch_size
num_test = FLAGS.num_test
pgd_step = FLAGS.pgd_step
eot_step = FLAGS.eot_step

logger.add(f"../../result_attack/{FLAGS.save_folder}/ddpm_linf_attack/{FLAGS.model_type}/log_test_fix_{FLAGS.fix_type}_ddpm_t_{Denoise_strength}_step_{Num_inference_steps}_|{model_id.split('/')[-1]}|_bs_{batch_size}_num_test_{num_test}_pgd_{pgd_step}_eot_{eot_step}.log")

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
ppp = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
upper_limit = ((1 - ppp)/ ppp)
lower_limit = ((0 - ppp)/ ppp)

transform_cifar10 = transforms.Compose(
    [transforms.Normalize(mean, std)]
)

augmentations = transforms.Compose(
    [
        transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
dataset = torchvision.datasets.ImageFolder(root=FLAGS.train_data_dir,transform=augmentations)
subset_indices = list(range(0, num_test))
# subset_indices = torch.load('./subset_indices.pth')
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

sampler = DistributedSampler(subset_dataset, num_replicas=world_size, rank=local_rank)
cifar10_test_loader = DataLoader(
    subset_dataset, shuffle=False, num_workers=5, batch_size=batch_size, sampler=sampler, drop_last=True)


network_clf = timm.create_model('resnet101', pretrained=True).to(device)
network_clf.eval()

pipe = StableDiffusionImg2ImgPipeline_Warp.from_pretrained(
    "/data3/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    variant="fp32",
).to(device)
# set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# load LCM-LoRA
pipe.load_lora_weights("/data3/lcm-lora-sdv1-5")
pipe.fuse_lora()

epsilon = (4 / 255.) / ppp
alpha = (0.5 / 255.) / ppp
def clamp1(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

cls_acc_list = []
pure_cls_acc_list = []
cls_init_acc_list = []

for sample_image, y_val in tqdm(cifar10_test_loader, colour='yellow'):
    sample_image = sample_image.to(device)
    y_val = y_val.to(device)
    noise_global = torch.randn(sample_image.shape).to(device)
    with torch.no_grad():
        yy_init= network_clf(transform_cifar10(sample_image/2+0.5))
        cls_acc_init = sum((yy_init.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("cls_init_acc:", str(cls_acc_init.cpu().item()))
        logger.info("cls_init_acc:"+str(cls_acc_init.cpu().item())+str({local_rank}))
        cls_init_acc_list.append(cls_acc_init.cpu().item())
        
    delta = torch.zeros_like(sample_image)
    delta.requires_grad = True
    for _ in range(pgd_step):
        eot = 0
        for _ in range(eot_step):
            images_1 = pipe(
                prompt="",
                image=resize_transform1((sample_image + delta)/2+0.5),
                strength=0.1,
                #generator=torch.manual_seed(42),
                num_inference_steps=50,
                guidance_scale=0,
                output_type="pt",
                return_dict=False
            )[0]
            tmp_in = transform_cifar10(resize_transform2(images_1))
            tmp_out = network_clf(tmp_in)
            loss = F.cross_entropy(tmp_out, y_val)
            loss.backward()
            grad = delta.grad.detach()
            eot += grad
            delta.grad.zero_()
            network_clf.zero_grad()
        d = clamp1(delta + alpha * eot.sign(), -epsilon, epsilon)
        d = clamp1(d, lower_limit - sample_image, upper_limit - sample_image)
        delta.data = d

    adv_out = (sample_image + delta)
    
    with torch.no_grad():
        yy = network_clf(transform_cifar10(adv_out/2+0.5))
        cls_acc = sum((yy.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("cls_acc:", str(cls_acc.cpu().item()))
        logger.info("cls_acc:"+str(cls_acc.cpu().item())+str({local_rank}))
        cls_acc_list.append(cls_acc.cpu().item())
    
    with torch.no_grad():
        images_1 = pipe(
            prompt="",
            image=resize_transform1((adv_out)/2+0.5),
            strength=0.1,
            #generator=torch.manual_seed(42),
            num_inference_steps=50,
            guidance_scale=0,
            output_type="pt",
            return_dict=False
        )[0]
             
    with torch.no_grad():
        yy = network_clf(transform_cifar10(resize_transform2(images_1).to(device)))
        pure_cls_acc = sum((yy.argmax(dim=-1) == y_val)) / y_val.shape[0]
        print("pure_cls_acc:", str(pure_cls_acc))
        logger.info("pure_cls_acc:" + str(pure_cls_acc) + str({local_rank}))
        pure_cls_acc_list.append(pure_cls_acc)
  
    del images_1, yy, adv_out, delta, d, eot, sample_image, y_val
    torch.distributed.barrier()
    torch.cuda.empty_cache()

print("=====================================")
cls_init_acc_list = torch.tensor(cls_init_acc_list).to(device)
gathered_cls_init_acc_list = [cls_init_acc_list.clone() for _ in range(world_size)]
all_gather(gathered_cls_init_acc_list, cls_init_acc_list)
cls_init_acc_list = torch.cat(gathered_cls_init_acc_list, dim=0)

cls_acc_list = torch.tensor(cls_acc_list).to(device)
gathered_cls_acc_list = [cls_acc_list.clone() for _ in range(world_size)]
all_gather(gathered_cls_acc_list, cls_acc_list)
cls_acc_list = torch.cat(gathered_cls_acc_list, dim=0)

pure_cls_acc_list = torch.tensor(pure_cls_acc_list).to(device)
gathered_pure_cls_acc_list = [pure_cls_acc_list.clone() for _ in range(world_size)]
all_gather(gathered_pure_cls_acc_list, pure_cls_acc_list)
pure_cls_acc_list = torch.cat(gathered_pure_cls_acc_list, dim=0)
if local_rank == 0:
    print("all_cls_init_acc:", "{:0.4f}".format(sum(cls_init_acc_list)/len(cls_init_acc_list)))
    logger.info("all_cls_init_acc"+"{:0.4f}".format(sum(cls_init_acc_list)/len(cls_init_acc_list)))
    print("all_cls_acc:", "{:0.4f}".format(sum(cls_acc_list)/len(cls_acc_list)))
    logger.info("all_cls_acc"+"{:0.4f}".format(sum(cls_acc_list)/len(cls_acc_list)))
    print("all_pure_cls_acc:", "{:0.4f}".format(sum(pure_cls_acc_list)/len(pure_cls_acc_list)))
    logger.info("all_pure_cls_acc"+"{:0.4f}".format(sum(pure_cls_acc_list)/len(pure_cls_acc_list)))
