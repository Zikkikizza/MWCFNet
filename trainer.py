import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from datetime import datetime

from metrics import MetricsCalculator
from evaluator import Evaluator 


class Trainer:
    def __init__(self, config, path_config, generator, dataloaders):
        self.config = config
        self.path_config = path_config
        self.generator = generator
        self.train_loader, self.val_loader  = dataloaders
        
        self.device = config.device
        self.pixel_loss = nn.L1Loss()
        self.metrics_calculator = MetricsCalculator(self.device)
        self.evaluator = Evaluator(self.metrics_calculator, self.pixel_loss)
        
        self.optimizer = optim.Adam(
            generator.parameters(), 
            lr=config.lr_generator, 
            betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=[100, 160], 
            gamma=0.5
        )
        
        self.best_metrics = {
            'psnr': {'value': -float('inf'), 'epoch': -1},
            'ssim': {'value': -float('inf'), 'epoch': -1},
            'ms_ssim': {'value': -float('inf'), 'epoch': -1},
            'lpips': {'value': float('inf'), 'epoch': -1},
            'val_loss': {'value': float('inf'), 'epoch': -1}
        }
        
        self.metrics_history = [] 
        
        self._setup_directories()
        self._setup_wandb()
    
    def _setup_directories(self):
        os.makedirs(self.path_config.saved_models, exist_ok=True)
        os.makedirs(self.path_config.validation_results, exist_ok=True) 
    
    def _setup_wandb(self):
        run_id = f"{self.config.model_name}_{self.path_config.dataset_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        wandb_name = f"{self.path_config.dataset_name}_{self.config.model_name}"
        
        wandb.login(key="........")
        wandb.init(
            project=self.config.wandb_project,
            name=wandb_name,
            id=run_id
        )
        
        wandb.define_metric("Epoch")
        wandb.define_metric("*", step_metric="Epoch")
        
        wandb.config.update({
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "lr_G": self.config.lr_generator,
        })
    
    def train_epoch(self, epoch):
        self.generator.train()
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch [{epoch + 1}/{self.config.epochs}] Training", 
            unit="batch"
        )
        
        for lr_images, hr_images, _, _ in progress_bar:
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)
            
            fake_images = self.generator(lr_images)
            g_loss = self.pixel_loss(fake_images, hr_images)
            
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()
            
            progress_bar.set_postfix({"G Loss": g_loss.item()})
        
        self.scheduler.step()
        
        return g_loss.item()
    
    def validate_epoch(self, epoch):
        val_metrics = self.evaluator.validate(self.generator, self.val_loader, self.device) 
        
        self.metrics_history.append({
            'epoch': epoch + 1,
            'val_loss': val_metrics['loss'],
            **{k: v for k, v in val_metrics.items() if k != 'loss'}
        })
         
        
        return val_metrics 
    
    def log_metrics(self, epoch, train_loss, val_metrics ):
        wandb.log({
            "Generator Loss": train_loss,
            "Validation Loss": val_metrics['loss'],
            "PSNR": val_metrics['psnr'],
            "SSIM": val_metrics['ssim'],
            "MS-SSIM": val_metrics['ms_ssim'],
            "LPIPS": val_metrics['lpips'], 
            "epoch": epoch + 1
        })
        
        print(
            f"Epoch [{epoch + 1}/{self.config.epochs}] | "
            f"G Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"PSNR: {val_metrics['psnr']:.2f} | "
            f"SSIM: {val_metrics['ssim']:.4f} | "
            f"MS-SSIM: {val_metrics['ms_ssim']:.4f} | "
            f"LPIPS: {val_metrics['lpips']:.4f} | " 
        )
    
    def save_best_models(self, epoch, val_metrics):
        current_metrics = {
            'psnr': val_metrics['psnr'],
            'ssim': val_metrics['ssim'],
            'ms_ssim': val_metrics['ms_ssim'],
            'lpips': val_metrics['lpips'],
            'val_loss': val_metrics['loss']
        }
        
        for metric in self.best_metrics:
            is_better = False
            
            if metric in ['psnr', 'ssim', 'ms_ssim']:
                is_better = current_metrics[metric] > self.best_metrics[metric]['value']
            elif metric in ['lpips', 'val_loss']:
                is_better = current_metrics[metric] < self.best_metrics[metric]['value']
            
            if is_better:
                self.best_metrics[metric]['value'] = current_metrics[metric]
                self.best_metrics[metric]['epoch'] = epoch + 1
                
                model_path = os.path.join(self.path_config.saved_models, f"best_{metric}.pth")
                if self.config.save_full_model:
                    torch.save(self.generator, model_path)
                else:
                    torch.save(self.generator.state_dict(), model_path)
                
                print(f"🏆 New best {metric.upper()} at epoch {epoch + 1}")
    
    def save_checkpoint(self, epoch):
        if (epoch + 1) % self.config.save_interval == 0:
            save_path = os.path.join(
                self.path_config.saved_models, 
                f"generator_epoch_{epoch + 1}.pth"
            )
            
            if self.config.save_full_model:
                torch.save(self.generator, save_path)
            else:
                torch.save(self.generator.state_dict(), save_path)
            
            print(f"Models saved at epoch {epoch + 1}")
    
    def generate_final_results(self):
        print("\nGenerating final results with best models:")
        
        for metric in self.best_metrics:
            model_path = os.path.join(self.path_config.saved_models, f"best_{metric}.pth")
            if not os.path.exists(model_path):
                continue
            
            try:
                if self.config.save_full_model:
                    best_generator = torch.load(model_path, map_location=self.device)
                else:
                    best_generator = self.generator.to(self.device)
                    best_generator.load_state_dict(torch.load(model_path, map_location=self.device))
                
                best_generator.eval()
 
                
                print(f"✅ Generated results for {metric.upper()} model")
            except Exception as e:
                print(f"❌ Error loading {metric.upper()} model: {str(e)}")
    
    def print_summary(self):
        def print_best(metric_name, history, higher_better=True):
            valid_metrics = [m for m in history if m.get(metric_name) is not None]
            if not valid_metrics:
                print(f"No valid entries for {metric_name}")
                return
            
            best = max(valid_metrics, key=lambda x: x[metric_name]) if higher_better else min(valid_metrics, key=lambda x: x[metric_name])
            print(f"\nBest {metric_name.upper()} at Epoch {best['epoch']}:")
            for key, value in best.items():
                if key != 'epoch':
                    print(f"  {key}: {value:.4f}")
        
        print("\n" + "="*50)
        print("Validation Set Best Metrics:")
        print_best('psnr', self.metrics_history)
        print_best('ssim', self.metrics_history)
        print_best('ms_ssim', self.metrics_history)
        print_best('lpips', self.metrics_history, higher_better=False)
        print_best('val_loss', self.metrics_history, higher_better=False) 
    
    def train(self):
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics  = self.validate_epoch(epoch)
            
            self.log_metrics(epoch, train_loss, val_metrics )
            self.save_best_models(epoch, val_metrics)
            self.save_checkpoint(epoch)
        
        self.generate_final_results()
        self.print_summary()
        wandb.finish()