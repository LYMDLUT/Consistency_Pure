# ImageNet: Stable Diffusion v1.5 + LCM-LoRA purification

Classifier: timm ResNet-101. Purification: img2img with `--strength 0.1` and
`--num-inference-steps 50` (5 LCM steps) at 512x512. Attack: Linf PGD with eps 4/255
and step 0.5/255.

| Setting       | Accuracy |
|---------------|----------|
| Clean         | 85.55%   |
| PGD-1, EOT-1  | 82.13%   |
| PGD-20        | 39.84%   |
| PGD-20, EOT-10| 20.80%   |

These numbers were obtained with the original script, which evaluated the first
images of the validation set in ImageFolder order (so only the first few classes).
The current script evaluates a seeded random subset by default; pass `--subset first`
for the original selection. Reproduce with `scripts/imagenet_lcm.sh`.
