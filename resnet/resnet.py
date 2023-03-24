import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, channels, expand_dim=1, downsample=False):
        super().__init__()

        self.downsample = downsample
        initial_stride = 2 if self.downsample else 1

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * expand_dim,
                kernel_size=3,
                stride=initial_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels * expand_dim),
            nn.ReLU(),
            nn.Conv2d(
                channels * expand_dim,
                channels * expand_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels * expand_dim),
        )

        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.AvgPool2d(2, stride=2, padding=0, ceil_mode=True),
                nn.Conv2d(
                    channels,
                    channels * expand_dim,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * expand_dim),
            )
        else:
            self.downsample_layer = nn.Identity()

        self.final_relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.block(x)

        if self.downsample:
            identity = self.downsample_layer(identity)
            x += identity

        x = self.final_relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()

        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
        )

        self.layers = nn.Sequential(
            ResNetBlock(64),
            ResNetBlock(64, expand_dim=4, downsample=True),
            ResNetBlock(256, expand_dim=4, downsample=True),
            ResNetBlock(1024, downsample=True),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.AdaptiveAvgPool1d(1024),
            nn.Linear(1024,num_classes)
        )

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.layers(x)
        # x = self.head(x)

        return x


if __name__ == "__main__":
    model = ResNet(num_classes=1)
    print(model(torch.rand(1, 3, 50, 50)).shape)
    print(
        "params:",
        sum(p.numel() for i, p in model.named_parameters() if p.requires_grad),
    )
