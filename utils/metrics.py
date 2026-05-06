import math

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover
    skimage_ssim = None


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def mean_l1(a, b):
    return float(np.mean(np.abs(a - b)))


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim(a, b):
    if skimage_ssim is not None:
        try:
            return float(skimage_ssim(a, b, channel_axis=-1, data_range=1.0))
        except TypeError:
            return float(skimage_ssim(a, b, multichannel=True, data_range=1.0))

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))
