from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import random
from torchvision.transforms.functional import hflip, vflip, rotate


class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr):
        if random.random() < self.p:
            lr = hflip(lr)
            hr = hflip(hr)
        return lr, hr


class RandomVerticalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr):
        if random.random() < self.p:
            lr = vflip(lr)
            hr = vflip(hr)
        return lr, hr


class RandomRotatePair:
    def __init__(self, degrees=90):
        self.degrees = degrees

    def __call__(self, lr, hr):
        angle = random.uniform(-self.degrees, self.degrees)
        lr = rotate(lr, angle)
        hr = rotate(hr, angle)
        return lr, hr


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, img_transform=None, pair_transforms=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

        self.lr_filenames = [
            f for f in os.listdir(lr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
        self.hr_filenames = [
            f for f in os.listdir(hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]

        self.img_transform = img_transform
        self.pair_transforms = pair_transforms

        assert len(self.lr_filenames) == len(self.hr_filenames), (
            f"低分辨率图片数量({len(self.lr_filenames)})不等于高分辨率图片数量"
            f"({len(self.hr_filenames)})"
        )

    def __len__(self):
        return len(self.lr_filenames)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_filenames[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_filenames[idx])

        lr_image = Image.open(lr_path)
        hr_image = Image.open(hr_path)

        if self.img_transform:
            lr_image = self.img_transform(lr_image)
            hr_image = self.img_transform(hr_image)

        if self.pair_transforms:
            for transform in self.pair_transforms:
                lr_image, hr_image = transform(lr_image, hr_image)

        return lr_image, hr_image, self.lr_filenames[idx], self.hr_filenames[idx]