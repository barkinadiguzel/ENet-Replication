import torch.nn as nn

from ..blocks.initial import InitialBlock
from ..blocks.bottleneck import Bottleneck


class ENet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ---------------- INITIAL ----------------
        self.initial = InitialBlock()

        # ---------------- STAGE 1 ----------------
        self.stage1_0 = Bottleneck(16, 64, downsample=True)
        self.stage1 = nn.Sequential(
            *[Bottleneck(64, 64) for _ in range(4)]
        )

        # ---------------- STAGE 2 ----------------
        self.stage2_0 = Bottleneck(64, 128, downsample=True)

        self.stage2 = nn.Sequential(
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=2),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=4),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=8),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=16),
        )

        # ---------------- STAGE 3 ----------------
        self.stage3 = nn.Sequential(
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=2),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=4),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilated=8),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilated=16),
        )

        # ---------------- STAGE 4 ----------------
        self.stage4 = nn.Sequential(
            Bottleneck(128, 64),
            Bottleneck(64, 64),
            Bottleneck(64, 64),
        )

        # ---------------- STAGE 5 ----------------
        self.stage5 = nn.Sequential(
            Bottleneck(64, 16),
            Bottleneck(16, 16),
        )

        # ---------------- CLASSIFIER ----------------
        self.classifier = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.initial(x)

        x = self.stage1_0(x)
        x = self.stage1(x)

        x = self.stage2_0(x)
        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)
        x = self.stage5(x)

        x = self.classifier(x)

        return x
