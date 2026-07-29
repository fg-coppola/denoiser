import os
import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from .base import BaseDataModule

class LOLLightDataset(Dataset):
    """
    PyTorch Dataset for the LOL (LOw-Light) dataset.
    Applies heavy data augmentation to artificially multiply the 485 training pairs
    and prevent U-Net overfitting.
    """
    def __init__(
        self, 
        root_dir: str, 
        patch_size: int = 256, 
        is_train: bool = True
    ):
        super().__init__()
        
        # Setup directories for low-light inputs and normal-light targets
        self.low_dir = os.path.join(root_dir, 'low')
        self.high_dir = os.path.join(root_dir, 'high')
        
        # Ensure consistent ordering between input and target images
        self.image_filenames = sorted(os.listdir(self.low_dir))
        self.is_train = is_train
        
        # Modern torchvision v2 allows passing multiple images to the same transform
        # ensuring random crops and flips are applied identically to both
        if self.is_train:
            self.transform = v2.Compose([
                v2.RandomCrop(size=(patch_size, patch_size)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True) # Converts to [0.0, 1.0] range
            ])
        else:
            # During validation/testing, we use a deterministic center crop
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
        
        # Load images as standard RGB PIL Images
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        # Apply identical transformations to BOTH images simultaneously
        low_tensor, high_tensor = self.transform(low_img, high_img)
        
        return low_tensor, high_tensor

class LOLLightDataModule(BaseDataModule):
    """
    Gestisce il caricamento del dataset LOL, incapsulando i DataLoader
    e i parametri operativi per PyTorch Lightning.
    """
    def __init__(
        self,
        data_dir: str = "./LOLdataset",
        batch_size: int = 8,        # 8 o 16 sono ideali per patch 256x256 su GPU medie
        patch_size: int = 256,
        num_workers: int = 4,       # Regola in base ai core della tua CPU
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        

    def setup(self, stage: str | None = None):
        """
        Inizializza i dataset. Viene chiamato automaticamente da Lightning su ogni GPU.
        """
        # I percorsi seguono la struttura standard del dataset LOL
        train_dir = os.path.join(self.data_dir, "our485")
        val_dir = os.path.join(self.data_dir, "eval15")

        if stage == "fit" or stage is None:
            self.train_dataset = LOLLightDataset(
                root_dir=train_dir,
                patch_size=self.patch_size,
                is_train=True
            )
            self.val_dataset = LOLLightDataset(
                root_dir=val_dir,
                patch_size=self.patch_size,
                is_train=False
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,              # Fondamentale per il training
            num_workers=self.num_workers,
            pin_memory=True,           # Velocizza il trasferimento dati da CPU a GPU
            drop_last=True             # Evita crash se l'ultimo batch ha dimensioni diverse
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,             # Non serve fare shuffle in validazione
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )