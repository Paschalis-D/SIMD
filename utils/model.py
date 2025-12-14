"""This file contains the two stream model that will be used to predict the mask.
The same encoder modules will be applied to both RGB and Noise data. We will make sure
to apply only 3 SRM kernels to the RGB image so only 3 channels will be created.
The model will then share a bottleneck and decoder.
"""

import torch
import numpy as np
from torch import nn
import torchvision.transforms.functional as TF


class SRMConv2d(nn.Module):
    def __init__(self, truncation_threshold=3.0):
        super(SRMConv2d, self).__init__()
        self.truncation_threshold = truncation_threshold

        # 1. Define the 2D Kernels (Spatial Weights)
        # Kernel 1
        k1 = (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
            / 4.0
        )

        # Kernel 2
        k2 = (
            np.array(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=np.float32,
            )
            / 12.0
        )

        # Kernel 3
        k3 = (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
            / 2.0
        )

        filters = np.stack([k1, k2, k3], axis=0)  # Shape: (3, 5, 5)

        # 2. Define the RGB Weights (Channel Mixing)
        weights = np.zeros((3, 3, 5, 5), dtype=np.float32)

        # RGB Coefficients for Luminance
        rgb_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

        for i in range(3):  # For each output filter
            for j in range(3):  # For each input RGB channel
                weights[i, j, :, :] = filters[i] * rgb_weights[j]

        # 3. Initialize PyTorch Layer
        self.srm_layer = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.srm_layer.weight.data = torch.from_numpy(weights)

        # Freeze parameters
        for param in self.srm_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        assert x.shape[0] == 3, f"Expected three image channels but got {x.shape[0]}"
        # x: (Batch, 3, H, W)
        noise = self.srm_layer(x)

        # 4. TRUNCATION
        # This prevents strong edges from hiding weak tampering noise
        noise = torch.clamp(
            noise, -self.truncation_threshold, self.truncation_threshold
        )

        return noise


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, depths=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Construct the blocks
        for i, dim in enumerate(depths):
            # Input dim is 'in_channels' for first block, else 'previous_depth'
            input_dim = in_channels if i == 0 else depths[i - 1]
            output_dim = dim

            self.enc_blocks.append(ConvolutionBlock(input_dim, output_dim))

    def forward(self, x):
        skip_connections = []

        for block in self.enc_blocks:
            x = block(x)
            skip_connections.append(x)  # SAVE feature BEFORE pooling
            x = self.pool(x)

        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, depths=[1024, 512, 256, 128, 64], num_classes=1):
        super().__init__()
        self.dec_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        # We iterate backwards through depths: 1024 -> 512 -> 256...
        for i in range(len(depths) - 1):
            high_ch = depths[i]
            low_ch = depths[i + 1]

            # Upsamples 2x and reduces channels
            self.up_samples.append(
                nn.ConvTranspose2d(high_ch, low_ch, kernel_size=2, stride=2)
            )

            # Input channels = low_ch (from upsample) + low_ch (from skip connection)
            self.dec_blocks.append(ConvolutionBlock(low_ch * 2, low_ch))

        # Final classification layer (64 -> 1 for binary mask)
        self.final_conv = nn.Conv2d(depths[-1], num_classes, kernel_size=1)

    def forward(self, x, skip_connections):
        # x: The bottleneck output from Encoder (e.g., 1024 channels, tiny H,W)
        # skip_connections: List of features [64, 128, 256, 512, 1024]

        # We reverse the skips to match decoder direction
        # We assume the last element of skips is the same 'level' as x input if x was pooled
        skip_connections = skip_connections[::-1]

        for i in range(len(self.dec_blocks)):
            # 1. Upsample
            x = self.up_samples[i](x)

            # 2. Get corresponding skip connection
            skip = skip_connections[i]

            # 3. Handle Size Mismatch (Padding/Cropping)
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])

            # 4. Concatenate
            concat_features = torch.cat((skip, x), dim=1)

            # 5. Convolve
            x = self.dec_blocks[i](concat_features)

        # Final Binary Prediction
        return self.final_conv(x)


class SIMD(nn.Module):
    def __init__(self, depths: np.ndarray):
        super(SIMD, self).__init__()
        self.depths = depths
        self.encoder = Encoder(in_channels=3, depths=self.depths)
        self.decoder = Decoder(depths=self.depths[::-1], num_classes=1)
        self.srm_conv = SRMConv2d(truncation_threshold=3.0)

        # Define bottleneck channels
        self.bn_ch = self.depths[-1]
        self.bottleneck = ConvolutionBlock(
            in_channels=self.bn_ch, out_channels=self.bn_ch
        )

    def forward(self, rgb_tensor, noise_tensor):
        rgb_enc, skip = self.encoder.forward(rgb_tensor)

        # TODO: Apply here the SRM filter.
        noise_enc, skip = self.encoder.forward(noise_tensor)

        # TODO: Concatenate the encoded rgb and noise tensors.
        bottleneck = self.bottleneck.forward(encoded)

        out = self.decoder.forward(bottleneck, skip)

        return out
