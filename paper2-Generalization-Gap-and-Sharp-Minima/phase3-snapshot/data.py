"""MNIST for F1 and CIFAR-10 for C1, C2. The paper does not give the preprocessing; I use pixels in [0, 1],
CIFAR-10 standardized per channel, and no augmentation."""
import os

import numpy as np
import torch

DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets")


def find_npz(env, fname):
    """A local copy of the dataset: $env, then data/<fname>, then the shared datasets/ folder."""
    for path in (os.environ.get(env), os.path.join("data", fname), os.path.join(DATASETS, fname)):
        if path and os.path.exists(path):
            return path
    return None


def load_mnist():
    path = find_npz("MNIST_NPZ", "mnist.npz")
    if path:
        d = np.load(path)
        x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    else:
        import torchvision
        train = torchvision.datasets.MNIST("data", train=True, download=True)
        test = torchvision.datasets.MNIST("data", train=False, download=True)
        x_train, y_train = train.data.numpy(), train.targets.numpy()
        x_test, y_test = test.data.numpy(), test.targets.numpy()

    def to_tensor(x):
        return torch.from_numpy(x).float().div(255.0).reshape(len(x), -1)

    return to_tensor(x_train), torch.from_numpy(y_train).long(), to_tensor(x_test), torch.from_numpy(y_test).long()


def load_cifar10():
    path = find_npz("CIFAR10_NPZ", "cifar10.npz")
    if path:
        d = np.load(path)
        x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    else:
        import torchvision
        train = torchvision.datasets.CIFAR10("data", train=True, download=True)
        test = torchvision.datasets.CIFAR10("data", train=False, download=True)
        x_train, y_train = train.data, np.array(train.targets)
        x_test, y_test = test.data, np.array(test.targets)

    def to_tensor(x):
        return torch.from_numpy(x).float().div(255.0).permute(0, 3, 1, 2).contiguous()

    x_train, x_test = to_tensor(x_train), to_tensor(x_test)
    mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
    std = x_train.std(dim=(0, 2, 3), keepdim=True)
    return ((x_train - mean) / std, torch.from_numpy(y_train).long(),
            (x_test - mean) / std, torch.from_numpy(y_test).long())


def load(net):
    return load_mnist() if net == "F1" else load_cifar10()
