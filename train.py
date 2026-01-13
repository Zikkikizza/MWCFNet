import torch
import numpy as np
import random
from thop import profile

from config import TrainingConfig, DatasetConfig, PathConfig
from dataset import DataLoaderFactory
from trainer import Trainer
from module.model import SuperResolutionNetwork


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def print_model_info(model, device):
    input_tensor = torch.randn(1, 1, 64, 64).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"参数量: {params / 1e6:.2f}M")
    print(f"FLOPs: {flops / 1e9:.2f}G")


def train_on_dataset(dataset_name, training_config, dataset_config):
    setup_seed(training_config.seed)
    
    path_config = PathConfig(training_config.model_name, dataset_name)
    dataset_paths = dataset_config.get_paths(dataset_name)
    
    generator = SuperResolutionNetwork(
        in_channels=training_config.in_channels,
        base_channels=training_config.base_channels
    ).to(training_config.device)
    
    print_model_info(generator, training_config.device)
    
    dataloader_factory = DataLoaderFactory(
        dataset_paths,
        training_config.batch_size,
        training_config.num_workers,
        use_augmentation=False
    )
    dataloaders = dataloader_factory.create_dataloaders()
    
    trainer = Trainer(training_config, path_config, generator, dataloaders)
    trainer.train()


def main():
    training_config = TrainingConfig()
    dataset_config = DatasetConfig()
    
    for dataset_name in dataset_config.datasets:
        print(f"\n{'=' * 40} 开始训练数据集: {dataset_name} {'=' * 40}")
        try:
            train_on_dataset(dataset_name, training_config, dataset_config)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"训练数据集 {dataset_name} 时发生错误: {str(e)}")
            continue


if __name__ == "__main__":
    main()