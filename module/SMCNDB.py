import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.gelu(self.conv1(x) + x)
        x2 = self.gelu(self.conv2(x1) + x1 + x)
        x3 = self.gelu(self.conv3(x2) + x2 + x1 + x)
        return x3


class ContentExtractionNetwork(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.multi_scale_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4),
            nn.GELU(),
        )
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

        self.feature_fusion = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        multi_scale_feat = self.multi_scale_conv(x)

        channel_weight = self.channel_attn(multi_scale_feat)
        spatial_weight = self.spatial_attn(multi_scale_feat)

        channel_refined = multi_scale_feat * channel_weight
        spatial_refined = multi_scale_feat * spatial_weight

        fused_features = torch.cat([channel_refined, spatial_refined], dim=1)
        output = self.feature_fusion(fused_features)

        return self.activation(output + x)


class SeparableMultiContentNetwork(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.noise_extractor = DenseResidualBlock(in_channels)
        self.content_extractor = ContentExtractionNetwork(in_channels)
        
        self.adaptive_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 2, 1),
            nn.Softmax(dim=1)
        )

        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1)
        )

    def forward(self, x):
        noise_features = self.noise_extractor(x)
        content_path1 = x - noise_features

        content_path2 = self.content_extractor(x)

        weights = self.adaptive_weight(x)

        weighted_content = weights[:, 0:1] * content_path1 + weights[:, 1:2] * content_path2

        return self.refinement(weighted_content) + x 