"""MNIST loading + fast batched augmentation.

The whole dataset fits in memory (60k x 28 x 28 uint8 = 47 MB), so instead of a
DataLoader with per-image PIL transforms we keep tensors in RAM and augment a
whole batch at once with `affine_grid` / `grid_sample`. On a CPU this is
~10x faster than torchvision's per-sample transforms.
"""
import math

import torch
import torch.nn.functional as F
from torchvision import datasets

MEAN, STD = 0.1307, 0.3081  # MNIST train-set statistics


def load_mnist(root="data"):
    """Official split: 60,000 train / 10,000 test (uint8 images, int64 labels)."""
    train = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)
    return (train.data, train.targets), (test.data, test.targets)


def train_val_split(x, y, val_size=5000, seed=0):
    """Hold out part of the *train* set for model selection.

    The test set is never used during training or checkpoint selection.
    """
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    val, tr = perm[:val_size], perm[val_size:]
    return (x[tr], y[tr]), (x[val], y[val])


def to_input(x_uint8):
    """uint8 (N, 28, 28) -> float (N, 1, 28, 28) in [0, 1]."""
    return x_uint8.float().div_(255.0).unsqueeze(1)


def normalize(x):
    return (x - MEAN) / STD


def iterate(x, y, batch_size, shuffle=False, generator=None):
    idx = torch.randperm(len(x), generator=generator) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), batch_size):
        j = idx[i:i + batch_size]
        yield x[j], y[j]


def _uniform(n, lo, hi, g):
    return torch.rand(n, generator=g) * (hi - lo) + lo


class Augment:
    """Batched MNIST augmentation applied to images in [0, 1].

    * random affine: rotation, isotropic scale, shear, translation
    * elastic distortion (Simard et al., 2003): smoothed random displacement field
    * random erasing: zero a random rectangle in a fraction of the images

    No horizontal flips: a mirrored digit is a different (or invalid) symbol.
    """

    def __init__(self, rotate=10.0, scale=(0.9, 1.1), shear=10.0, translate=0.1,
                 elastic_alpha=0.0, elastic_sigma=4.0, erase_p=0.0, erase_area=(0.05, 0.2)):
        self.rotate, self.scale, self.shear, self.translate = rotate, scale, shear, translate
        self.elastic_alpha, self.elastic_sigma = elastic_alpha, elastic_sigma
        self.erase_p, self.erase_area = erase_p, erase_area
        if elastic_alpha > 0:
            r = int(3 * elastic_sigma)
            k = torch.exp(-torch.arange(-r, r + 1).float() ** 2 / (2 * elastic_sigma ** 2))
            self.kernel = (k / k.sum()).view(1, 1, 1, -1)

    def __call__(self, x, g=None):
        b, _, h, w = x.shape
        # affine_grid's theta maps output coords -> input coords in [-1, 1].
        # All parameters are drawn from symmetric ranges, so sampling the
        # inverse transform directly is equivalent to sampling the forward one.
        a = torch.deg2rad(_uniform(b, -self.rotate, self.rotate, g))
        sh = torch.deg2rad(_uniform(b, -self.shear, self.shear, g))
        s = _uniform(b, *self.scale, g)
        t = _uniform(2 * b, -2 * self.translate, 2 * self.translate, g).view(b, 2)
        cos, sin, tan = torch.cos(a), torch.sin(a), torch.tan(sh)
        theta = torch.stack([
            torch.stack([cos, -sin + cos * tan, t[:, 0]], 1),
            torch.stack([sin, cos + sin * tan, t[:, 1]], 1),
        ], 1)
        theta[:, :, :2] /= s.view(b, 1, 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)

        if self.elastic_alpha > 0:
            d = torch.rand(b * 2, 1, h, w, generator=g) * 2 - 1
            r = self.kernel.shape[-1] // 2
            d = F.conv2d(F.pad(d, (r, r, 0, 0), mode="reflect"), self.kernel)
            d = F.conv2d(F.pad(d, (0, 0, r, r), mode="reflect"), self.kernel.transpose(2, 3))
            d = d.view(b, 2, h, w).permute(0, 2, 3, 1)
            grid = grid + d * (self.elastic_alpha * 2.0 / h)  # pixels -> normalized coords

        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        if self.erase_p > 0:
            area = _uniform(b, *self.erase_area, g)
            aspect = torch.exp(_uniform(b, math.log(0.5), math.log(2.0), g))
            eh = (torch.sqrt(area * aspect) * h).clamp(max=h)
            ew = (torch.sqrt(area / aspect) * w).clamp(max=w)
            y0 = torch.rand(b, generator=g) * (h - eh)
            x0 = torch.rand(b, generator=g) * (w - ew)
            ys = torch.arange(h).view(1, h, 1)
            xs = torch.arange(w).view(1, 1, w)
            mask = ((ys >= y0.view(b, 1, 1)) & (ys < (y0 + eh).view(b, 1, 1))
                    & (xs >= x0.view(b, 1, 1)) & (xs < (x0 + ew).view(b, 1, 1))
                    & (torch.rand(b, generator=g) < self.erase_p).view(b, 1, 1))
            x = x.masked_fill(mask.unsqueeze(1), 0.0)
        return x


AUGMENTATIONS = {
    "none": None,
    "affine": dict(rotate=10.0, scale=(0.9, 1.1), shear=10.0, translate=0.1),
    "full": dict(rotate=10.0, scale=(0.9, 1.1), shear=10.0, translate=0.1,
                 elastic_alpha=2.0, elastic_sigma=4.0, erase_p=0.3, erase_area=(0.02, 0.1)),
}


def build_augment(name):
    cfg = AUGMENTATIONS[name]
    return None if cfg is None else Augment(**cfg)
