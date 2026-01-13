import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import SRDataset, RandomHorizontalFlipPair, RandomVerticalFlipPair, RandomRotatePair


class DataLoaderFactory:
    def __init__(self, dataset_paths, batch_size, num_workers=2, use_augmentation=False):
        self.dataset_paths = dataset_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        
        self.img_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        
        self.train_pair_transforms = [
            RandomHorizontalFlipPair(p=0.5),
            RandomVerticalFlipPair(p=0.5),
            RandomRotatePair(degrees=90),
        ] if use_augmentation else None
    
    def create_dataloaders(self):
        train_dataset = SRDataset(
            self.dataset_paths['train_lr'],
            self.dataset_paths['train_hr'],
            img_transform=self.img_transform,
            pair_transforms=self.train_pair_transforms
        )
        
        valid_dataset = SRDataset(
            self.dataset_paths['valid_lr'],
            self.dataset_paths['valid_hr'],
            img_transform=self.img_transform
        )
         
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        val_loader = DataLoader(
            valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
         
        
        return train_loader, val_loader