import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch - 3, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.maxpool(x)

        x = torch.cat([conv_out, pool_out], dim=1)
        x = self.bn(x)
        return self.prelu(x)
