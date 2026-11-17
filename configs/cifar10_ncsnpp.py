"""NCSN++ architecture of the CIFAR-10 consistency models (CD/CT, Song et al. 2023).

The same architecture is used by the official JAX checkpoints and by
https://github.com/LYMDLUT/improved_consistency_models_cifar10_pytorch, so weights
from either can be loaded.
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.data = data = ml_collections.ConfigDict()
    data.image_size = 32
    data.num_channels = 3

    config.model = model = ml_collections.ConfigDict()
    model.name = "ncsnpp"
    # Only used to size the ``sigmas`` buffer stored in the checkpoints.
    model.sigma_min = 0.02
    model.sigma_max = 100.0
    model.num_scales = 18
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "none"
    model.progressive_input = "residual"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.embedding_type = "fourier"
    model.fourier_scale = 16.0
    model.init_scale = 0.0
    model.conv_size = 3
    model.dropout = 0.0

    return config
