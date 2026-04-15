import torch.nn as nn
from ..encoder.enet_encoder import ENetEncoder
from ..decoder.enet_decoder import ENetDecoder

class ENet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = ENetEncoder()
        self.decoder = ENetDecoder()

        self.classifier = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x
