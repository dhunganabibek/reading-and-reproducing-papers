"""The CIFAR-10 networks of Appendix A. Run `python3 models.py` to compare the parameter counts with Table 1."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        # no conv bias and no batch-norm scale (TF-slim defaults); without this the count is off by 1,712
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.bn.weight.requires_grad_(False)

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
            DownsampleModule(80, 80),    # 28x28 -> 14x14
            InceptionModule(160, 112, 48),
            InceptionModule(160, 96, 64),
            InceptionModule(160, 80, 80),
            InceptionModule(160, 48, 96),
            DownsampleModule(144, 96),   # 14x14 -> 7x7
            InceptionModule(240, 176, 160),
            InceptionModule(336, 176, 160),
        )
        self.fc = nn.Linear(336, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).mean(dim=(2, 3)))


class SmallAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.lrn = nn.LocalResponseNorm(size=9, alpha=0.001 / 9.0, beta=0.75, k=1.0)
        self.fc1 = nn.Linear(7 * 7 * 64, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.lrn(F.max_pool2d(F.relu(self.conv1(x)), 3, stride=2, padding=1))   # 28 -> 14
        x = F.max_pool2d(self.lrn(F.relu(self.conv2(x))), 3, stride=2, padding=1)   # 14 -> 7
        x = F.relu(self.fc1(x.flatten(1)))
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


ARCHS = {
    "inception": SmallInception,
    "alexnet": SmallAlexNet,
    "mlp1": lambda: MLP(hidden_layers=1),
}


def build(arch):
    return ARCHS[arch]()


def glorot_init(model):
    """TensorFlow's default init (Glorot uniform, zero bias), used for one diagnostic run."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    table1 = {"inception": 1_649_402, "alexnet": 1_387_786, "mlp1": 1_209_866}
    for arch, ref in table1.items():
        n = count_params(build(arch))
        print(f"{arch:10s} {n:>10,}   paper {ref:>10,}   {'ok' if n == ref else 'MISMATCH'}")
