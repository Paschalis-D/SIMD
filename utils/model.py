"""This file contains the two stream model that will be used to predict the mask.
The same encoder modules will be applied to both RGB and Noise data. We will make sure
to apply only 3 SRM kernels to the RGB image so only 3 channels will be created.
The model will then share a bottleneck and decoder.
"""

from torch import nn


class Encoder(nn.Module):
    def __init__():
        super.__init__()
        pass

    def forward(self):
        pass


class Bottleneck(nn.Module):
    def __init__():
        super.__init__()
        pass

    def forward(self):
        pass


class Decoder(nn.Module):
    def __init__():
        super.__init__()
        pass

    def forward(self):
        pass


class SIMD(nn.Module):
    def __init__():
        super.__init__()
        pass

    def forward(self):
        pass
