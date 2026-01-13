import torch
import torch.nn as nn
from torch.nn import PixelShuffle

from .CCWF import ContentGuidedWaveletFusion
from .skip_connection import SkipConnectionModule
from .SMCNDB import SeparableMultiContentNetwork


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample

        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or downsample else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.relu(x + identity)


class SuperResolutionNetwork(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.encoder1 = ResidualBlock(base_channels, base_channels)
        self.encoder2 = ResidualBlock(base_channels, 2 * base_channels, downsample=True)
        self.encoder3 = ResidualBlock(2 * base_channels, 4 * base_channels, downsample=True)
        self.encoder4 = ResidualBlock(4 * base_channels, 8 * base_channels, downsample=True)

        self.content_blocks = nn.Sequential(
            SeparableMultiContentNetwork(8 * base_channels), 
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(8 * base_channels, 4 * base_channels, 3, padding=1),
            ResidualBlock(4 * base_channels, 4 * base_channels)
        )

        self.skip3 = SkipConnectionModule(4 * base_channels, 4 * base_channels, dilation_rates=[2, 4, 8])
        self.fusion3 = ContentGuidedWaveletFusion(4 * base_channels, wavelet_type='haar')
        self.channel_reduce3 = nn.Conv2d(4 * base_channels, 2 * base_channels, 3, padding=1)

        self.skip2 = SkipConnectionModule(2 * base_channels, 2 * base_channels, dilation_rates=[2, 4, 8])
        self.fusion2 = ContentGuidedWaveletFusion(2 * base_channels, wavelet_type='haar')
        self.channel_reduce2 = nn.Conv2d(2 * base_channels, base_channels, 3, padding=1)

        self.skip1 = SkipConnectionModule(base_channels, base_channels, dilation_rates=[2, 4, 8])
        self.fusion1 = ContentGuidedWaveletFusion(base_channels, wavelet_type='haar')

        self.reconstruction = nn.Sequential(
            nn.Conv2d(base_channels, 1 * (4 ** 2), 3, padding=1),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        feat1 = self.encoder1(x)

        feat2 = self.encoder2(feat1)
        feat3 = self.encoder3(feat2)
        feat4 = self.encoder4(feat3)

        feat4_identity = feat4
        feat4 = self.content_blocks(feat4)
        feat4 = feat4 + feat4_identity
        low_feat3 = self.bottleneck(feat4)

        enhanced_feat3 = self.skip3(feat3)
        fused_feat3 = self.fusion3(enhanced_feat3, low_feat3)
        low_feat2 = self.channel_reduce3(fused_feat3)

        enhanced_feat2 = self.skip2(feat2)
        fused_feat2 = self.fusion2(enhanced_feat2, low_feat2)
        low_feat1 = self.channel_reduce2(fused_feat2)

        enhanced_feat1 = self.skip1(feat1)
        reconstructed_feat = self.fusion1(enhanced_feat1, low_feat1)

        return self.reconstruction(reconstructed_feat)
 