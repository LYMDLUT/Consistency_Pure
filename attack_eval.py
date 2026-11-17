"""Robustness of consistency-model purification on CIFAR-10/100 under PGD-EOT.

Single process:  python attack_eval.py --cm-ckpt CKPT [options]
Multiple GPUs:   torchrun --nproc_per_node 4 attack_eval.py --cm-ckpt CKPT [options]
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from attacks import NORMS
from classifiers import CLASSIFIERS, load_classifier
from common import (
    cleanup_distributed,
    logger,
    main_process_first,
    parse_number,
    save_json,
    setup_distributed,
    setup_logging,
    shard_indices,
)
from evaluation import evaluate_attack
from purification import ConsistencyPurifier, load_consistency_model

DEFAULT_BUDGETS = {"linf": (8 / 255, 1 / 255), "l2": (0.5, 0.1)}
DATASETS = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cm-ckpt", required=True, help="Consistency model weights.")
    parser.add_argument(
        "--cm-ckpt-key",
        default="ema",
        help="Weights to use when --cm-ckpt is a training checkpoint (ema or model).",
    )
    parser.add_argument("--classifier", choices=CLASSIFIERS, default="t7")
    parser.add_argument(
        "--classifier-ckpt",
        help="Classifier weights (default depends on --classifier, see README).",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--norm", choices=NORMS, default="linf")
    parser.add_argument(
        "--eps",
        type=parse_number,
        help="Perturbation budget in [0, 1] pixel scale, e.g. 8/255 "
        "(default: 8/255 for linf, 0.5 for l2).",
    )
    parser.add_argument(
        "--alpha",
        type=parse_number,
        help="PGD step size (default: 1/255 for linf, 0.1 for l2).",
    )
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--eot-steps", type=int, default=10)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=0.5,
        help="Noise level the inputs are diffused to before purification.",
    )
    parser.add_argument("--purify-steps", type=int, default=1)
    parser.add_argument("--num-test", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16, help="Per process.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", help="Default: cuda if available, else cpu.")
    args = parser.parse_args()

    default_eps, default_alpha = DEFAULT_BUDGETS[args.norm]
    args.eps = default_eps if args.eps is None else args.eps
    args.alpha = default_alpha if args.alpha is None else args.alpha
    if args.output_dir is None:
        args.output_dir = (
            f"outputs/{args.classifier}_{args.norm}"
            f"_sigma{args.sigma_max}_steps{args.purify_steps}"
        )
    return args


def main() -> None:
    args = parse_args()
    ctx = setup_distributed(args.device)
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, ctx)
    logger.info("Arguments: %s", vars(args))
    torch.manual_seed(args.seed + ctx.rank)

    dataset_name = CLASSIFIERS[args.classifier].dataset
    with main_process_first(ctx):
        dataset = DATASETS[dataset_name](
            args.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
    if args.num_test > len(dataset):
        raise ValueError(f"--num-test exceeds the {len(dataset)} test images")
    loader = DataLoader(
        Subset(dataset, shard_indices(args.num_test, ctx)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=ctx.device.type == "cuda",
    )

    classifier = load_classifier(args.classifier, args.classifier_ckpt).to(ctx.device)
    model = load_consistency_model(args.cm_ckpt, args.cm_ckpt_key)
    purifier = ConsistencyPurifier(model, args.sigma_max, args.purify_steps)
    purifier.to(ctx.device)

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
