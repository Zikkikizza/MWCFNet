import torch
from dataclasses import dataclass
from typing import List


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    save_interval: int = 50
    save_full_model: bool = True
    
    lr_generator: float = 0.0001
    lr_discriminator: float = 0.00001
    
    loss_adv_weight: float = 1e-2
    loss_reg_weight: float = 1e-8
    loss_pixel_weight: float = 0.01
    
    base_channels: int = 64
    in_channels: int = 1
    
    seed: int = 0
    num_workers: int = 2
    
    wandb_project: str = "work"
    model_name: str = "MWCF"
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DatasetConfig:
    datasets: List[str] = None
    data_root: str = "/datasets"
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["abdominal"]
    
    def get_paths(self, dataset_name: str):
        base_path = f"{self.data_root}/{dataset_name}"
        return {
            'train_lr': f"{base_path}/train/lr",
            'train_hr': f"{base_path}/train/hr",
            'valid_lr': f"{base_path}/valid/lr",
            'valid_hr': f"{base_path}/valid/hr"
        }


@dataclass
class PathConfig:
    model_name: str
    dataset_name: str
    
    @property
    def saved_models(self):
        return f"{self.model_name}_saved_models/{self.dataset_name}"
    
    @property
    def validation_results(self):
        return f"{self.model_name}_validation_results/{self.dataset_name}" 