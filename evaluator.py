import torch
import torch.nn as nn
from metrics import MetricsCalculator


class Evaluator:
    def __init__(self, metrics_calculator, pixel_loss):
        self.metrics_calculator = metrics_calculator
        self.pixel_loss = pixel_loss
    
    def validate(self, generator, dataloader, device):
        generator.eval()
        total_loss = 0
        
        aggregated_metrics = {
            'psnr': 0,
            'ssim': 0,
            'ms_ssim': 0,
            'lpips': 0,
            'count': 0
        }
        
        with torch.no_grad():
            for lr_images, hr_images, _, _ in dataloader:
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
                fake_images = generator(lr_images)
                
                total_loss += self.pixel_loss(fake_images, hr_images).item()
                
                batch_metrics = self.metrics_calculator.compute_metrics(fake_images, hr_images)
                for key in aggregated_metrics:
                    aggregated_metrics[key] += batch_metrics[key]
        
        num_images = aggregated_metrics['count']
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'psnr': aggregated_metrics['psnr'] / num_images,
            'ssim': aggregated_metrics['ssim'] / num_images,
            'ms_ssim': aggregated_metrics['ms_ssim'] / num_images,
            'lpips': aggregated_metrics['lpips'] / num_images
        }