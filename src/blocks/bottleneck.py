import torch
import torch.nn as nn
from .layers import conv_1x1, conv_3x3, conv_asymmetric, SpatialDropout


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, downsample=False, dilated=0, asymmetric=False, p=0.1):
        super().__init__()

        self.downsample = downsample

        inter_c = out_c // 4

        self.reduce = conv_1x1(in_c, inter_c)
        self.bn1 = nn.BatchNorm2d(inter_c)
        self.prelu1 = nn.PReLU()

        if asymmetric:
            self.conv = conv_asymmetric(inter_c, inter_c)
        elif dilated > 0:
            self.conv = conv_3x3(inter_c, inter_c, dilation=dilated)
        else:
            stride = 2 if downsample else 1
            self.conv = conv_3x3(inter_c, inter_c, stride=stride)

        self.bn2 = nn.BatchNorm2d(inter_c)
        self.prelu2 = nn.PReLU()

        self.expand = conv_1x1(inter_c, out_c)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.dropout = SpatialDropout(p)

        self.skip = self._make_skip(in_c, out_c)

        self.prelu_out = nn.PReLU()

    def _make_skip(self, in_c, out_c):
        if self.downsample:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_c)
            )
        if in_c != out_c:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )
        return nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.reduce(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv(out)

        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.expand(out)
        out = self.bn3(out)

        out = self.dropout(out)

        out = out + identity
        return self.prelu_out(out)
