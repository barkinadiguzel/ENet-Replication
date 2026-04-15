import torch.nn as nn

class SpatialDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        return self.dropout(x)


def conv_1x1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, 1, bias=False)


def conv_3x3(in_c, out_c, stride=1, dilation=1):
    padding = dilation
    return nn.Conv2d(in_c, out_c, 3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)


def conv_asymmetric(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, (5,1), padding=(2,0), bias=False),
        nn.Conv2d(out_c, out_c, (1,5), padding=(0,2), bias=False)
    )
