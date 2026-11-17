import torch

from models.up_or_down_sampling import downsample_2d, upsample_2d


def test_fir_resampling_preserves_constant_images():
    x = torch.ones(2, 4, 8, 8)
    up = upsample_2d(x, k=[1, 3, 3, 1])
    down = downsample_2d(x, k=[1, 3, 3, 1])
    assert up.shape == (2, 4, 16, 16)
    assert down.shape == (2, 4, 4, 4)
    # Borders are affected by zero padding.
    torch.testing.assert_close(up[..., 2:-2, 2:-2], torch.ones(2, 4, 12, 12))
    torch.testing.assert_close(down[..., 1:-1, 1:-1], torch.ones(2, 4, 2, 2))
