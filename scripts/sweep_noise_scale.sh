#!/usr/bin/env bash
# Robust accuracy against the noise level sigma_max (results/cifar10_noise_scale.md).
# Usage: CM_CKPT=path/to/cm.pth scripts/sweep_noise_scale.sh [extra attack_eval.py args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CM_CKPT:?set CM_CKPT to the consistency model checkpoint}"
NUM_GPUS=${NUM_GPUS:-4}
CLASSIFIER=${CLASSIFIER:-t7}

for norm in linf l2; do
  for sigma in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    torchrun --nproc_per_node "$NUM_GPUS" attack_eval.py \
      --cm-ckpt "$CM_CKPT" --classifier "$CLASSIFIER" --norm "$norm" \
      --sigma-max "$sigma" --purify-steps 1 --num-test 1024 --batch-size 32 \
      --output-dir "outputs/noise_scale/${CLASSIFIER}_${norm}_sigma${sigma}" "$@"
  done
done
