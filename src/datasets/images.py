import os
from PIL import Image
import torch
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import v2

from .base import BaseDataModule


class LOLLightDataset(Dataset):
    """
    PyTorch Dataset for loading pairs of low-light and target images.
    Accepts explicit paths for the low-light folder and the high-light folder.
    """
    def __init__(
        self,
        low_dir: str,
        high_dir: str,
        patch_size: int = 256,
        is_train: bool = True
    ):
        super().__init__()

        self.low_dir = low_dir
        self.high_dir = high_dir
        self.is_train = is_train

        # Safety check on file extensions to avoid hidden files (.DS_Store, etc.)
        valid_exts = ('.png', '.jpg', '.jpeg')
        self.image_filenames = sorted([
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(valid_exts)
        ])

        if self.is_train:
            self.transform = v2.Compose([
                v2.RandomCrop(size=(patch_size, patch_size)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)  # Converts to [0.0, 1.0]
            ])
        else:
            self.transform = v2.Compose([
                v2.CenterCrop(size=(patch_size, patch_size)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ])

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_filenames[idx]

        low_path = os.path.join(self.low_dir, img_name)
        high_path = os.path.join(self.high_dir, img_name)

        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        low_tensor, high_tensor = self.transform(low_img, high_img)

        return low_tensor, high_tensor


class LOLLightDataModule(BaseDataModule):
    """
    Flexible DataModule that merges dynamic lists of sources (low_dir, high_dir)
    into a single data stream for PyTorch Lightning using ConcatDataset.
    """
    def __init__(
        self,
        train_sources: list[tuple[str, str]] | None = None,
        val_sources: list[tuple[str, str]] | None = None,
        batch_size: int = 8,
        patch_size: int = 256,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_sources = train_sources or []
        self.val_sources = val_sources or []
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            if not self.train_sources or not self.val_sources:
                raise ValueError(
                    "Devi passare almeno una coppia (low_dir, high_dir) in `train_sources` e `val_sources`."
                )

            # Create sub-instances of LOLLightDataset for each source
            train_subsets = [
                LOLLightDataset(
                    low_dir=low,
                    high_dir=high,
                    patch_size=self.patch_size,
                    is_train=True
                )
                for low, high in self.train_sources
            ]

            val_subsets = [
                LOLLightDataset(
                    low_dir=low,
                    high_dir=high,
                    patch_size=self.patch_size,
                    is_train=False
                )
                for low, high in self.val_sources
            ]

            # In-memory merging using ConcatDataset
            self.train_dataset = ConcatDataset(train_subsets)
            self.val_dataset = ConcatDataset(val_subsets)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )