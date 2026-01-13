import torch
import math
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from piqa import MS_SSIM


class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        self.ssim = SSIM(
            data_range=1.0,
            kernel_size=11,
            sigma=1.5,
            k1=0.01,
            k2=0.03,
            reduction='elementwise_mean'
        ).to(device)
        self.ms_ssim = MS_SSIM(n_channels=1).to(device)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
    
    @staticmethod
    def calculate_psnr(fake_image, hr_image):
        mse = torch.mean((fake_image - hr_image) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))
        return psnr
    
    def calculate_lpips(self, fake, real):
        if fake.size(1) == 1:
            fake_rgb = fake.repeat(1, 3, 1, 1)
            real_rgb = real.repeat(1, 3, 1, 1)
        else:
            fake_rgb = fake
            real_rgb = real
        return self.lpips_model(fake_rgb, real_rgb).squeeze()
    
    @staticmethod
    def dynamic_normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val < 1e-8:
            return torch.full_like(tensor, 0.5)
        return (tensor - min_val) / (max_val - min_val + 1e-8)
    
    def compute_metrics(self, fake_images, hr_images):
        fake_norm = self.dynamic_normalize(fake_images)
        hr_norm = self.dynamic_normalize(hr_images)
        
        psnr_total = sum(self.calculate_psnr(fake, hr) 
                        for fake, hr in zip(fake_images, hr_images))
        
        ssim_value = self.ssim(fake_norm, hr_norm).item() * fake_images.size(0)
        ms_ssim_value = self.ms_ssim(fake_norm, hr_norm).item() * fake_images.size(0)
        
        lpips_total = 0
        for fake_img, hr_img in zip(fake_norm, hr_norm):
            fake_scaled = (fake_img - 0.5) * 2
            hr_scaled = (hr_img - 0.5) * 2
            lpips_total += self.calculate_lpips(fake_scaled, hr_scaled).item()
        
        return {
            'psnr': psnr_total,
            'ssim': ssim_value,
            'ms_ssim': ms_ssim_value,
            'lpips': lpips_total,
            'count': fake_images.size(0)
        }