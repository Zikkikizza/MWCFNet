import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class PerceptualLoss(nn.Module):
    def __init__(self, device, net='alex'):
        super().__init__()
        self.lpips_model = lpips.LPIPS(net=net).to(device)
        for param in self.lpips_model.parameters():
            param.requires_grad = False

    def forward(self, fake_batch, hr_batch):
        if fake_batch.size(1) == 1:
            fake_rgb = fake_batch.repeat(1, 3, 1, 1)
            hr_rgb = hr_batch.repeat(1, 3, 1, 1)
        else:
            fake_rgb = fake_batch
            hr_rgb = hr_batch

        fake_lpips_input = fake_rgb * 2 - 1
        hr_lpips_input = hr_rgb * 2 - 1

        return torch.mean(self.lpips_model(fake_lpips_input, hr_lpips_input))


class RegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        dh = torch.square(x[:, :, :-1, :] - x[:, :, 1:, :])
        dw = torch.square(x[:, :, :, :-1] - x[:, :, :, 1:])
        return torch.sum(dh) + torch.sum(dw)


class EdgeLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, generated, target):
        g_x = F.conv2d(generated, self.sobel_x, padding=1)
        g_y = F.conv2d(generated, self.sobel_y, padding=1)
        t_x = F.conv2d(target, self.sobel_x, padding=1)
        t_y = F.conv2d(target, self.sobel_y, padding=1)

        g_mag = torch.sqrt(g_x ** 2 + g_y ** 2 + 1e-8)
        t_mag = torch.sqrt(t_x ** 2 + t_y ** 2 + 1e-8)

        return F.l1_loss(g_mag, t_mag)