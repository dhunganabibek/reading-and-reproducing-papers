"""Networks F1, C1, C2 of Keskar et al. (2017), Appendix B.

F1: 784 -> 5 x [FC 512, BN, ReLU] -> FC 10.
C1: 'AlexNet-like': 2 x ( conv 64 5x5 stride 2 -> maxpool 3 ) -> FC 384 -> FC 192 -> FC 10,
    batch-norm and ReLU everywhere, dropout (keep 0.5) on the two dense layers.
C2: 'VGG-like': conv stacks 2x64, 2x128, 3x256, 3x512, 3x512 (3x3, stride 1) each followed by
    maxpool 2 -> FC 512 -> FC 10, BN + ReLU everywhere, dropout on the dense layer.
Choices not fixed by the paper: 'same' padding; the stride of MaxPool(3) in C1 is taken as 2
with 'same' padding (so the map goes 32 -> 16 -> 8 -> 4 -> 2); BN is applied before ReLU.
"""
import torch
import torch.nn as nn


def fc_bn_relu(cin, cout, dropout=0.0):
    layers = [nn.Linear(cin, cout), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


def conv_bn_relu(cin, cout, k, stride=1):
    return [nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class F1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        layers, d = [], 784
        for _ in range(5):
            layers += fc_bn_relu(d, 512); d = 512
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class C1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn_relu(3, 64, 5, stride=2), nn.MaxPool2d(3, stride=2, padding=1),    # 32 -> 16 -> 8
            *conv_bn_relu(64, 64, 5, stride=2), nn.MaxPool2d(3, stride=2, padding=1),   # 8 -> 4 -> 2
        )
        self.classifier = nn.Sequential(
            *fc_bn_relu(64 * 2 * 2, 384, dropout=0.5),
            *fc_bn_relu(384, 192, dropout=0.5),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class C2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        cfg = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
        layers, cin = [], 3
        for n, c in cfg:
            for _ in range(n):
                layers += conv_bn_relu(cin, c, 3); cin = c
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)                       # 32 -> 1
        self.classifier = nn.Sequential(*fc_bn_relu(512, 512, dropout=0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def build(name):
    return {"F1": F1, "C1": C1, "C2": C2}[name]()


if __name__ == "__main__":
    for n in ["F1", "C1", "C2"]:
        m = build(n)
        print(n, f"{sum(p.numel() for p in m.parameters()):,} params")
