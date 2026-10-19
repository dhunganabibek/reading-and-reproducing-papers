"""CIFAR-10 loading, preprocessing and the randomization tests of Zhang et al. (2017), Sec. 2.1.

Preprocessing follows Appendix A of the paper: pixels scaled to [0,1], centre crop to 28x28,
then per-image whitening (subtract the image mean, divide by the adjusted std).
"""
import os
import numpy as np
import torch


def load_cifar10(npz_path=None, root="./data"):
    """Return uint8 arrays x_train (50000,32,32,3), y_train, x_test, y_test."""
    if npz_path and os.path.exists(npz_path):
        d = np.load(npz_path)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    import torchvision  # downloaded on demand
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    return (tr.data, np.array(tr.targets), te.data, np.array(te.targets))


def preprocess(x_uint8):
    """[0,255] uint8 NHWC -> float32 NCHW, centre-cropped to 28x28 and whitened per image.

    tf.image.per_image_whitening: (x - mean) / max(std, 1/sqrt(num_pixels)).
    """
    x = torch.from_numpy(x_uint8).float() / 255.0
    x = x[:, 2:30, 2:30, :]                      # centre crop 32 -> 28
    x = x.permute(0, 3, 1, 2).contiguous()       # NCHW
    n = x[0].numel()
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    adj = torch.clamp(std, min=1.0 / np.sqrt(n))
    return (x - mean) / adj


def randomize(x_train, y_train, x_test, y_test, mode, p=0.0, seed=0, num_classes=10):
    """Apply one of the randomization tests. Labels are sampled ONCE here and are then fixed
    for the whole training run (the paper: 'random labels are fixed and consistent across epochs').

    mode:
      true            : nothing changed (p ignored)
      corrupt         : each label independently replaced w.p. p by a uniform class in {0..9}
                        (so only 0.9 p of the labels actually change; p=1 is 'random labels')
      shuffled_pixels : one fixed pixel permutation applied to every train and test image
      random_pixels   : an independent pixel permutation per image
      gaussian        : every image replaced by i.i.d. N(mu, sigma^2) pixels with the mean and
                        variance of the (whitened) training set
    """
    g = torch.Generator().manual_seed(seed)
    x_train, x_test = x_train.clone(), x_test.clone()
    y_train, y_test = y_train.clone(), y_test.clone()
    n = x_train.shape[0]
    if mode == "true":
        pass
    elif mode == "corrupt":
        mask = torch.rand(n, generator=g) < p
        rnd = torch.randint(0, num_classes, (n,), generator=g)
        y_train = torch.where(mask, rnd, y_train)
    elif mode == "shuffled_pixels":
        npix = 28 * 28
        perm = torch.randperm(npix, generator=g)
        x_train = x_train.flatten(2)[:, :, perm].view_as(x_train)
        x_test = x_test.flatten(2)[:, :, perm].view_as(x_test)
    elif mode == "random_pixels":
        npix = 28 * 28
        for x in (x_train, x_test):
            perms = torch.argsort(torch.rand(x.shape[0], npix, generator=g), dim=1)
            flat = x.flatten(2)                           # N,3,784
            idx = perms[:, None, :].expand(-1, 3, -1)
            x.copy_(torch.gather(flat, 2, idx).view_as(x))
    elif mode == "gaussian":
        mu, sigma = x_train.mean().item(), x_train.std().item()
        x_train = mu + sigma * torch.randn(x_train.shape, generator=g)
        x_test = mu + sigma * torch.randn(x_test.shape, generator=g)
    else:
        raise ValueError(mode)
    return x_train, y_train, x_test, y_test
