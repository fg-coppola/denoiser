from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_image

from ..transforms.prnu import ApplyPRNU
from .base import BaseDataModule


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_paths, clean_transform, noisy_transform):
        self.image_paths = image_paths
        self.clean_transform = clean_transform
        self.noisy_transform = noisy_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = decode_image(str(self.image_paths[idx]), mode=ImageReadMode.RGB)
        clean_image = self.clean_transform(image)
        noisy_image = self.noisy_transform(image)
        return noisy_image, clean_image


class PRNUDataModule(BaseDataModule):
    def __init__(self, data_dir, prnu_maps, map_augment=False, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir

        if prnu_maps is None:
            raise ValueError("Prnu_maps must be provided")

        self.clean_transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )
        self.noisy_transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ConvertImageDtype(torch.float32),
                ApplyPRNU(prnu_maps, map_augment=map_augment),
            ]
        )

    def setup(self, stage=None):
        image_paths = sorted(
            [
                path
                for path in Path(self.data_dir).iterdir()
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        )
        if not image_paths:
            raise ValueError(f"No images found in {self.data_dir}")

        full_ds = UnlabeledImageDataset(
            image_paths=image_paths,
            clean_transform=self.clean_transform,
            noisy_transform=self.noisy_transform,
        )

        self.train_ds, self.val_ds = random_split(full_ds, [0.8, 0.2])
