"""CIFAR-10 with the preprocessing of Appendix A, and the randomization tests of Section 2."""
import os

import numpy as np
import torch

LOCAL_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets", "cifar10.npz")


def load_cifar10(npz_path=None):
    """uint8 images (N, 32, 32, 3) and integer labels, from a local npz if there is one."""
    for path in (npz_path, LOCAL_NPZ):
        if path and os.path.exists(path):
            d = np.load(path)
            return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    import torchvision
    train = torchvision.datasets.CIFAR10("./data", train=True, download=True)
    test = torchvision.datasets.CIFAR10("./data", train=False, download=True)
    return train.data, np.array(train.targets), test.data, np.array(test.targets)


def preprocess(x):
    """Scale to [0, 1], center crop to 28x28, then whiten each image like tf.image.per_image_whitening."""
    x = torch.from_numpy(x).float() / 255.0
    x = x[:, 2:30, 2:30, :].permute(0, 3, 1, 2).contiguous()
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    std = torch.clamp(std, min=1.0 / np.sqrt(x[0].numel()))
    return (x - mean) / std


def randomize(x_train, y_train, x_test, y_test, mode, p=0.0, seed=0, num_classes=10):
    """Apply one randomization test. Everything is drawn once here, so random labels stay fixed across epochs."""
    g = torch.Generator().manual_seed(seed)
    x_train, x_test = x_train.clone(), x_test.clone()
    y_train, y_test = y_train.clone(), y_test.clone()
    n = len(y_train)

    if mode == "corrupt":
        # with probability p the label is replaced by a uniform class, which can be the true one
        replace = torch.rand(n, generator=g) < p
        y_train = torch.where(replace, torch.randint(0, num_classes, (n,), generator=g), y_train)
    elif mode == "shuffled_pixels":
        # one permutation of the 28x28 positions, shared by all train and test images
        perm = torch.randperm(28 * 28, generator=g)
        x_train = x_train.flatten(2)[:, :, perm].view_as(x_train)
        x_test = x_test.flatten(2)[:, :, perm].view_as(x_test)
    elif mode == "random_pixels":
        # a new permutation per image; the three color values of a pixel stay together
        for x in (x_train, x_test):
            perms = torch.argsort(torch.rand(len(x), 28 * 28, generator=g), dim=1)
            idx = perms[:, None, :].expand(-1, 3, -1)
            x.copy_(torch.gather(x.flatten(2), 2, idx).view_as(x))
    elif mode == "random_values":
        # a new permutation of all 3*28*28 values per image, so the colors are destroyed too
        for x in (x_train, x_test):
            perms = torch.argsort(torch.rand(len(x), 3 * 28 * 28, generator=g), dim=1)
            x.copy_(torch.gather(x.flatten(1), 1, perms).view_as(x))
    elif mode == "gaussian":
        mu, sigma = x_train.mean().item(), x_train.std().item()
        x_train = mu + sigma * torch.randn(x_train.shape, generator=g)
        x_test = mu + sigma * torch.randn(x_test.shape, generator=g)
    elif mode != "true":
        raise ValueError(f"unknown mode: {mode}")

    return x_train, y_train, x_test, y_test
