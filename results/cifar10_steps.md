# CIFAR-10: robust accuracy (%) against the number of purification steps

Same model and attacks as in [cifar10_noise_scale.md](cifar10_noise_scale.md), with
`--sigma-max 0.4`.

| `--purify-steps` | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    |
|------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Linf             | 45.31 | 45.90 | 49.02 | 47.17 | 47.66 | 47.27 | 48.24 | 49.32 | 48.73 | 47.46 |
| L2               | 67.68 | 68.26 | 68.36 | 68.26 | 68.36 | 65.82 | 66.41 | 67.48 | 66.41 | 67.29 |

These numbers were obtained with the original scripts. Reproduce with
`scripts/sweep_steps.sh`.
