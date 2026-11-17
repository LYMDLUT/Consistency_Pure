"""Helpers shared by the evaluation scripts."""

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor

PathLike = Union[str, os.PathLike]

logger = logging.getLogger("consistency_pure")


@dataclass
class DistContext:
    rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(device: Optional[str] = None) -> DistContext:
    """Initialises the process group when launched with ``torchrun``.

    A single process is used when ``WORLD_SIZE`` is not set, so the scripts can also be
    started with plain ``python``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = f"cuda:{local_rank}"
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return DistContext(rank=rank, world_size=world_size, device=device)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def main_process_first(ctx: DistContext) -> Iterator[None]:
    """Lets the main process run the block (e.g. a download) before the others."""
    if not ctx.is_main and dist.is_initialized():
        dist.barrier()
    yield
    if ctx.is_main and dist.is_initialized():
        dist.barrier()


def shard_indices(num_items: int, ctx: DistContext) -> range:
    """Disjoint per-process indices that together cover ``range(num_items)`` once."""
    return range(ctx.rank, num_items, ctx.world_size)


def all_reduce_sum(tensor: Tensor) -> Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def setup_logging(output_dir: Path, ctx: DistContext) -> None:
    """Logs to stdout and ``output_dir/log.txt`` on the main process only."""
    logger.setLevel(logging.INFO if ctx.is_main else logging.WARNING)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / "log.txt"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def parse_number(text: str) -> float:
    """Parses ``"0.5"`` as well as fractions such as ``"8/255"``."""
    if "/" in text:
        numerator, denominator = text.split("/")
        return float(numerator) / float(denominator)
    return float(text)


def extract_state_dict(checkpoint: dict, key: str = "ema") -> dict[str, Tensor]:
    """Returns model weights from either a plain state dict or a training checkpoint.

    Prefixes added by wrappers such as DDP (``module.``) are removed.
    """
    if all(isinstance(v, Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint[key]
    for prefix in ("module.", "model_ema."):
        if all(k.startswith(prefix) for k in state_dict):
            state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
    return state_dict
