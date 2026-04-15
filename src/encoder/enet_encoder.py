import torch.nn as nn
from ..blocks.initial import InitialBlock
from ..blocks.bottleneck import Bottleneck

class ENetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial = InitialBlock()

        self.stage1 = nn.Sequential(
            Bottleneck(16, 64, downsample=True),
            *[Bottleneck(64, 64) for _ in range(4)]
        )

        self.stage2 = nn.Sequential(
            Bottleneck(64, 128, downsample=True),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=2),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=4),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=8),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=16)
        )

        self.stage3 = nn.Sequential(
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=2),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=4),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=8),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=16)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x
