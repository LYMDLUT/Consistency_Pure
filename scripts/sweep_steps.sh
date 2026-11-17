#!/usr/bin/env bash
# Robust accuracy against the number of purification steps (results/cifar10_steps.md).
# Usage: CM_CKPT=path/to/cm.pth scripts/sweep_steps.sh [extra attack_eval.py args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CM_CKPT:?set CM_CKPT to the consistency model checkpoint}"
NUM_GPUS=${NUM_GPUS:-4}
CLASSIFIER=${CLASSIFIER:-t7}
SIGMA=${SIGMA:-0.4}

for norm in linf l2; do
  for steps in 1 2 3 4 5 6 7 8 9 10; do
    torchrun --nproc_per_node "$NUM_GPUS" attack_eval.py \
      --cm-ckpt "$CM_CKPT" --classifier "$CLASSIFIER" --norm "$norm" \
      --sigma-max "$SIGMA" --purify-steps "$steps" --num-test 1024 --batch-size 32 \
      --output-dir "outputs/steps/${CLASSIFIER}_${norm}_sigma${SIGMA}_steps${steps}" "$@"
  done
done
