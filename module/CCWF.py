import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class DWTFunction(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        channels = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(channels, -1, -1, -1), stride=2, groups=channels)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(channels, -1, -1, -1), stride=2, groups=channels)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(channels, -1, -1, -1), stride=2, groups=channels)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(channels, -1, -1, -1), stride=2, groups=channels)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None


class DiscreteWaveletTransform2D(nn.Module):
    def __init__(self, wavelet_type):
        super(DiscreteWaveletTransform2D, self).__init__()
        wavelet = pywt.Wavelet(wavelet_type)
        dec_hi = torch.Tensor(wavelet.dec_hi[::-1])
        dec_lo = torch.Tensor(wavelet.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWTFunction.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWTFunction(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class InverseDiscreteWaveletTransform2D(nn.Module):
    def __init__(self, wavelet_type):
        super(InverseDiscreteWaveletTransform2D, self).__init__()
        wavelet = pywt.Wavelet(wavelet_type)
        rec_hi = torch.Tensor(wavelet.rec_hi)
        rec_lo = torch.Tensor(wavelet.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWTFunction.apply(x, self.filters)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.gelu(self.conv1(x))
        out = F.gelu(self.conv2(out))
        out = out + residual
        return out


class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttentionModule, self).__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_features, key_features):
        B, N, C = query_features.shape

        Q = self.query_proj(query_features)
        K = self.key_proj(key_features)
        V = self.value_proj(key_features)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)
        attended_output = torch.matmul(attention_weights, V)

        return attended_output


class ContentGuidedWaveletFusion(nn.Module):
    def __init__(self, channels, wavelet_type):
        super(ContentGuidedWaveletFusion, self).__init__()
        self.dwt = DiscreteWaveletTransform2D(wavelet_type)
        self.high_freq_conv1 = nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.high_freq_block = ResidualBlock(channels)
        self.high_freq_conv2 = nn.Conv2d(channels, channels * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.low_freq_conv = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.low_freq_block = ResidualBlock(channels)

        self.cross_attn1 = CrossAttentionModule(channels)
        self.cross_attn2 = CrossAttentionModule(channels)
        self.activation = nn.ReLU()
        self.fusion_conv = nn.Conv2d(channels * 2, channels * 4, kernel_size=1, stride=1, padding=0, bias=True)

        self.idwt = InverseDiscreteWaveletTransform2D(wavelet_type)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        x_wavelet = self.dwt(x1)
        ll, lh, hl, hh = x_wavelet.split(c, 1)

        high_freq = torch.cat([lh, hl, hh], 1)
        high_feat1 = self.high_freq_conv1(high_freq)
        high_feat = self.high_freq_block(high_feat1)

        b1, c1, h1, w1 = ll.shape
        b2, c2, h2, w2 = x2.shape

        if h1 != h2:
            x2 = F.pad(x2, (0, 0, 1, 0), "constant", 0)

        low_freq = torch.cat([ll, x2], 1)
        low_freq = self.low_freq_conv(low_freq)
        low_feat = self.low_freq_block(low_freq)

        low_feat_orig = low_feat
        high_feat_orig = high_feat

        B, C, H, W = low_feat.shape
        low_feat_flat = low_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        high_feat_flat = high_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        low_enhanced = self.cross_attn1(high_feat_flat, low_feat_flat)
        high_enhanced = self.cross_attn2(low_feat_flat, high_feat_flat)

        low_enhanced = low_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)
        high_enhanced = high_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)

        low_combined = self.activation(low_feat_orig + low_enhanced)
        high_combined = self.activation(high_feat_orig + high_enhanced)

        fused_features = torch.cat((low_combined, high_combined), 1)
        fused_features = self.fusion_conv(fused_features)
        reconstructed = self.idwt(fused_features)

        return reconstructed 