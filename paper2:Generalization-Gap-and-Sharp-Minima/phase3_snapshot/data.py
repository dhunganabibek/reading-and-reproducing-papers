"""MNIST (network F1) and CIFAR-10 (networks C1, C2) for Keskar et al. (2017).

Preprocessing is not specified in the paper.  Choice: pixels scaled to [0,1]; CIFAR-10 is then
standardised per channel with the training-set mean/std; no data augmentation (the paper lists
augmentation only as a remedy in Appendix E).
"""
import os
import numpy as np
import torch


def load_mnist(npz_path=None, root="./data"):
    if npz_path and os.path.exists(npz_path):
        d = np.load(npz_path)
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    else:
        import torchvision
        tr = torchvision.datasets.MNIST(root, train=True, download=True)
        te = torchvision.datasets.MNIST(root, train=False, download=True)
        xtr, ytr, xte, yte = tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()
    f = lambda x: torch.from_numpy(x).float().div(255.0).reshape(x.shape[0], -1)   # 784-d
    return f(xtr), torch.from_numpy(ytr).long(), f(xte), torch.from_numpy(yte).long()


def load_cifar10(npz_path=None, root="./data"):
    if npz_path and os.path.exists(npz_path):
        d = np.load(npz_path)
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    else:
        import torchvision
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        xtr, ytr, xte, yte = tr.data, np.array(tr.targets), te.data, np.array(te.targets)
    f = lambda x: torch.from_numpy(x).float().div(255.0).permute(0, 3, 1, 2).contiguous()
    xtr, xte = f(xtr), f(xte)
    mean = xtr.mean(dim=(0, 2, 3), keepdim=True)
    std = xtr.std(dim=(0, 2, 3), keepdim=True)
    return (xtr - mean) / std, torch.from_numpy(ytr).long(), (xte - mean) / std, torch.from_numpy(yte).long()


def load(name, data_dir="data"):
    if name == "F1":
        return load_mnist(os.environ.get("MNIST_NPZ", os.path.join(data_dir, "mnist.npz")), data_dir)
    return load_cifar10(os.environ.get("CIFAR10_NPZ", os.path.join(data_dir, "cifar10.npz")), data_dir)
