"""The three CIFAR-10 architectures of Zhang et al. (2017), Appendix A / Figure 3.

Parameter counts reproduce Table 1 of the paper exactly:
  small Inception 1,649,402   small AlexNet 1,387,786   MLP 1x512 1,209,866   MLP 3x512 1,735,178
For Inception this requires convolutions without bias and batch-norm without the scale (gamma)
parameter (the TensorFlow-slim default the authors used); gamma is frozen at 1 below.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Conv -> BatchNorm (no scale) -> ReLU."""
    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.bn.weight.requires_grad_(False)   # scale=False as in tf.contrib.slim batch_norm

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    def __init__(self, cin, ch1, ch3):
        super().__init__()
        self.b1 = ConvModule(cin, ch1, 1)
        self.b3 = ConvModule(cin, ch3, 3)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x)], dim=1)


class DownsampleModule(nn.Module):
    def __init__(self, cin, ch3):
        super().__init__()
        self.conv = ConvModule(cin, ch3, 3, stride=2)

    def forward(self, x):
        return torch.cat([self.conv(x), F.max_pool2d(x, 3, stride=2, padding=1)], dim=1)


class SmallInception(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            ConvModule(3, 96, 3),
            InceptionModule(96, 32, 32),
            InceptionModule(64, 32, 48),
            DownsampleModule(80, 80),          # 28x28 -> 14x14, 80 + 80 = 160 channels
            InceptionModule(160, 112, 48),
            InceptionModule(160, 96, 64),
            InceptionModule(160, 80, 80),
            InceptionModule(160, 48, 96),
            DownsampleModule(144, 96),         # 14x14 -> 7x7, 96 + 144 = 240 channels
            InceptionModule(240, 176, 160),
            InceptionModule(336, 176, 160),
        )
        self.fc = nn.Linear(336, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.mean(dim=(2, 3))                 # global 7x7 mean pooling
        return self.fc(x)


class SmallAlexNet(nn.Module):
    """Two (conv5x5 -> maxpool3x3/2 -> LRN) blocks, FC 384, FC 192, FC 10 (TF cifar10 tutorial net)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.lrn = nn.LocalResponseNorm(size=9, alpha=0.001 / 9.0, beta=0.75, k=1.0)
        self.fc1 = nn.Linear(7 * 7 * 64, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.lrn(F.max_pool2d(x, 3, stride=2, padding=1))   # 28 -> 14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(self.lrn(x), 3, stride=2, padding=1)   # 14 -> 7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLP(nn.Module):
    def __init__(self, hidden_layers=1, width=512, num_classes=10, in_dim=3 * 28 * 28):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


def build(arch):
    if arch == "inception":
        return SmallInception()
    if arch == "alexnet":
        return SmallAlexNet()
    if arch == "mlp1":
        return MLP(1)
    if arch == "mlp3":
        return MLP(3)
    raise ValueError(arch)


def count_params(model):
    """Trainable parameters (the frozen BN scale is not a parameter in the authors' TF model)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for a, ref in [("inception", 1_649_402), ("alexnet", 1_387_786), ("mlp1", 1_209_866), ("mlp3", 1_735_178)]:
        n = count_params(build(a))
        print(f"{a:10s} params={n:,}  paper={ref:,}  match={n == ref}")
