"""Robustness of Stable Diffusion + LCM-LoRA purification on ImageNet under PGD-EOT.

Single process:  python attack_eval_imagenet_lcm.py --data-dir IMAGENET_VAL [options]
Multiple GPUs:   torchrun --nproc_per_node 4 attack_eval_imagenet_lcm.py ...
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from attacks import NORMS
from classifiers import NormalizedClassifier
from common import (
    cleanup_distributed,
    logger,
    parse_number,
    save_json,
    setup_distributed,
    setup_logging,
)
from evaluation import evaluate_attack
from lcm_purification import load_lcm_purifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="ImageNet validation set in ImageFolder layout.",
    )
    parser.add_argument(
        "--sd-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--lcm-lora",
        default="latent-consistency/lcm-lora-sdv1-5",
        help="LCM-LoRA weights; pass an empty string to skip.",
    )
    parser.add_argument("--classifier", default="resnet101", help="timm model name.")
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--norm", choices=NORMS, default="linf")
    parser.add_argument("--eps", type=parse_number, default=4 / 255)
    parser.add_argument("--alpha", type=parse_number, default=0.5 / 255)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--eot-steps", type=int, default=1)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--num-test", type=int, default=64)
    parser.add_argument(
        "--subset",
        choices=("random", "first"),
        default="random",
        help="Evaluate a seeded random subset, or the first --num-test images "
        "(which only cover the first few classes).",
    )
    parser.add_argument(
        "--subset-indices", help="JSON file with a list of image indices to evaluate."
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Per process.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/imagenet_lcm")
    parser.add_argument("--device", help="Default: cuda if available, else cpu.")
    return parser.parse_args()


def select_indices(args: argparse.Namespace, dataset_size: int) -> list[int]:
    if args.subset_indices:
        indices = json.loads(Path(args.subset_indices).read_text())
    elif args.subset == "first":
        indices = list(range(dataset_size))
    else:
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
    if args.num_test > len(indices):
        raise ValueError(f"--num-test exceeds the {len(indices)} available images")
    return indices[: args.num_test]


def load_timm_classifier(name: str) -> NormalizedClassifier:
    import timm

    net = timm.create_model(name, pretrained=True)
    config = timm.data.resolve_data_config(model=net)
    classifier = NormalizedClassifier(
        net, config["mean"], config["std"], input_size=224
    )
    return classifier.eval().requires_grad_(False)


def main() -> None:
    args = parse_args()
    ctx = setup_distributed(args.device)
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, ctx)
    logger.info("Arguments: %s", vars(args))
    torch.manual_seed(args.seed + ctx.rank)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    indices = select_indices(args, len(dataset))
    loader = DataLoader(
        Subset(dataset, indices[ctx.rank :: ctx.world_size]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=ctx.device.type == "cuda",
    )

    classifier = load_timm_classifier(args.classifier).to(ctx.device)
    purifier = load_lcm_purifier(
        args.sd_model,
        args.lcm_lora,
        ctx.device,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        resolution=args.resolution,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    accuracy, num_images = evaluate_attack(
        loader,
        classifier,
        purifier,
        ctx,
        norm=args.norm,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.pgd_steps,
        eot_steps=args.eot_steps,
        random_start=args.random_start,
    )

    if ctx.is_main:
        for name, value in accuracy.items():
            logger.info("%-15s %.4f", name, value)
        save_json(
            output_dir / "results.json",
            {"args": vars(args), "num_images": num_images, "accuracy": accuracy},
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
