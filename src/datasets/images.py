from torch.utils.data import random_split
from torchvision import datasets, transforms

from ..transforms.prnu import ApplyPRNU
from .base import BaseDataModule


class PRNUDataModule(BaseDataModule):
    def __init__(self, data_dir, prnu_maps, map_augment=False, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir

        if prnu_maps is None:
            raise ValueError("Prnu_maps must be provided")

        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                ApplyPRNU(prnu_maps, map_augment=map_augment),
            ]
        )

    def setup(self, stage=None):
        full_ds = datasets.ImageFolder(self.data_dir, transform=self.transform)
        # Split train/val
        self.train_ds, self.val_ds = random_split(full_ds, [0.8, 0.2])
