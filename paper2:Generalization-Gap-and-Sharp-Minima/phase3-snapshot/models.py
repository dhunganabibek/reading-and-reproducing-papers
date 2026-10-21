"""Networks F1, C1 and C2 from Appendix B of the paper. 'Same' padding and BN before ReLU are my choices."""
import torch.nn as nn


def fc_bn_relu(cin, cout, dropout=0.0):
    layers = [nn.Linear(cin, cout), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


def conv_bn_relu(cin, cout, k, stride=1):
    return [nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class F1(nn.Module):
    """MNIST: 784 -> 5 x [FC 512, BN, ReLU] -> 10."""
    def __init__(self, num_classes=10):
        super().__init__()
        layers, d = [], 784
        for _ in range(5):
            layers += fc_bn_relu(d, 512)
            d = 512
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class C1(nn.Module):
    """AlexNet-like: 2 x [conv 64 5x5, maxpool 3 stride 2] -> FC 384 -> FC 192 -> 10.

    The authors' code puts the stride 2 in the pooling. My Phase 3 version read "[64,5,5,2]" as a
    stride-2 convolution; it is kept as C1s2 (conv_stride=2).
    """
    def __init__(self, num_classes=10, conv_stride=1):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn_relu(3, 64, 5, stride=conv_stride), nn.MaxPool2d(3, stride=2, padding=1),
            *conv_bn_relu(64, 64, 5, stride=conv_stride), nn.MaxPool2d(3, stride=2, padding=1),
        )
        side = 32 // (4 * conv_stride * conv_stride)   # 8 with stride 1, 2 with stride 2
        self.classifier = nn.Sequential(
            *fc_bn_relu(64 * side * side, 384, dropout=0.5),
            *fc_bn_relu(384, 192, dropout=0.5),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class C2(nn.Module):
    """VGG-like: 3x3 conv stacks 2x64, 2x128, 3x256, 3x512, 3x512, each followed by maxpool -> FC 512 -> 10."""
    def __init__(self, num_classes=10):
        super().__init__()
        layers, cin = [], 3
        for n, c in [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]:
            for _ in range(n):
                layers += conv_bn_relu(cin, c, 3)
                cin = c
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)   # 32x32 -> 1x1
        self.classifier = nn.Sequential(*fc_bn_relu(512, 512, dropout=0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


NETS = {"F1": F1, "C1": C1, "C1s2": lambda: C1(conv_stride=2), "C2": C2}


def build(name):
    return NETS[name]()


if __name__ == "__main__":
    for name in NETS:
        print(f"{name:5s} {sum(p.numel() for p in build(name).parameters()):>11,} params")
