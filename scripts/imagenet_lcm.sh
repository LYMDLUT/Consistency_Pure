#!/usr/bin/env bash
# Stable Diffusion + LCM-LoRA purification on ImageNet (results/imagenet_lcm.md).
# Usage: IMAGENET_VAL=/path/to/val scripts/imagenet_lcm.sh [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${IMAGENET_VAL:?set IMAGENET_VAL to the ImageNet validation set (ImageFolder layout)}"
NUM_GPUS=${NUM_GPUS:-4}

run() {
  local name=$1
  shift
  torchrun --nproc_per_node "$NUM_GPUS" attack_eval_imagenet_lcm.py \
    --data-dir "$IMAGENET_VAL" --num-test 1024 --output-dir "outputs/imagenet_lcm/$name" \
    "$@"
}

run pgd1_eot1 --pgd-steps 1 --eot-steps 1 "$@"
run pgd20_eot1 --pgd-steps 20 --eot-steps 1 "$@"
run pgd20_eot10 --pgd-steps 20 --eot-steps 10 "$@"
