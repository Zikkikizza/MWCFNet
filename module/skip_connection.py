import torch.nn as nn
import torch


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.spatial_conv = nn.Conv2d(2, 1, 1, padding=0, padding_mode='reflect', bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([x_avg, x_max], dim=1)
        spatial_attn = self.spatial_conv(x_concat)
        spatial_attn = self.sigmoid(spatial_attn)
        return spatial_attn


class EnhancedParallelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.spatial_attn = SpatialAttention()

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.pixel_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.fusion_mlp = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = torch.cat([self.spatial_attn(x) * x, self.channel_attn(x) * x, self.pixel_attn(x) * x], dim=1)
        x = self.fusion_mlp(x)
        x = identity + x
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class AtrousConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(AtrousConvBlock, self).__init__(*modules)


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, dilation_rates):
        super(MultiScaleFeatureExtractor, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = tuple(dilation_rates)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.branch2 = AtrousConvBlock(in_channels, out_channels, rate1)
        self.branch3 = AtrousConvBlock(in_channels, out_channels, rate2)
        self.branch4 = AtrousConvBlock(in_channels, out_channels, rate3)

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.squeeze_excite = ConvBlock(in_channels * 4, in_channels, 1, 0, 1)

    def forward(self, x):
        x_origin = x
        x = self.batch_norm(x)

        feat1 = self.branch1(x)
        feat2 = self.branch2(feat1 + x)
        feat3 = self.branch3(feat2 + x)
        feat4 = self.branch4(feat3 + x)

        concat_features = torch.cat([feat1, feat2, feat3, feat4], 1)
        compressed_features = self.squeeze_excite(concat_features)

        output = x_origin + compressed_features

        return output


class SkipConnectionModule(nn.Module):
    def __init__(self, in_channels, feature_dim, dilation_rates):
        super().__init__()
        self.feature_extractor = MultiScaleFeatureExtractor(in_channels, dilation_rates)
        self.parallel_attention = EnhancedParallelAttention(feature_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.parallel_attention(x)
        return x
 