# CIFAR-10: accuracy (%) against the noise level

Consistency model: CT-LPIPS, checkpoint 74 (FID 8.8), one-step purification
(`--purify-steps 1`). Attack: PGD-20 + EOT-10; Linf with eps 8/255 and step 1/255,
L2 with eps 0.5 and step 0.1. 1024 test images.

| `--sigma-max` | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 0.8   | 0.9   | 1.0   |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| clean         | 93.26 | 90.82 | 88.57 | 87.40 | 84.38 | 81.74 | 79.69 | 74.80 | 75.10 | 71.00 |
| Linf          |  3.22 | 20.41 | 34.96 | 45.31 | 47.66 | 48.24 | 51.46 | 49.90 | 48.54 | 47.75 |
| L2            | 28.42 | 53.32 | 61.52 | 67.68 | 66.80 | 68.07 | 63.67 | 65.14 | 61.43 | 61.82 |

"clean" is the accuracy on purified clean images (`purified_clean` in
`results.json`), the attack rows are the `robust` accuracy.

These numbers were obtained with the original scripts; see the README for the
differences to the current code. Reproduce with `scripts/sweep_noise_scale.sh`.
