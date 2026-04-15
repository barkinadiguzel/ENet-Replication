import torch.nn as nn
from ..blocks.bottleneck import Bottleneck

class ENetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage4 = nn.Sequential(
            Bottleneck(128, 64, downsample=False),
            Bottleneck(64, 64),
            Bottleneck(64, 64)
        )

        self.stage5 = nn.Sequential(
            Bottleneck(64, 16, downsample=False),
            Bottleneck(16, 16)
        )

    def forward(self, x):
        x = self.stage4(x)
        x = self.stage5(x)
        return x
