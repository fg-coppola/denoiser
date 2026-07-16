import argparse
from pathlib import Path

import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.images import PRNUDataModule
from models import RestorationModule


def main(experiment_name: str, output_folder: str, data_dir: str, prnu_dir: str):
    # Setup high precision for matrix multiplication
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # Define and create output directories
    base_dir = Path(output_folder)
    logs_dir = base_dir / "logs"
    checkpoint_dir = base_dir / "checkpoints"

    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=logs_dir, name=experiment_name)

    prnu_model = RestorationModule(
        in_channels=1,
        out_channels=1,
        base_features=32,
        loss_fn=nn.L1Loss(),
        lr=1e-4,
    )

    prnu_loader = PRNUDataModule(
        data_dir=data_dir,
        prnu_dir=prnu_dir,
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=True,
        mode="min",
    )

    checkpoint_path = checkpoint_dir / experiment_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="prnu-best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,
        precision="bf16-mixed",
        max_epochs=500,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
    )

    # Check for existing checkpoint to resume training
    last_checkpoint_path = checkpoint_path / "last.ckpt"

    if last_checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        trainer.fit(prnu_model, datamodule=prnu_loader, ckpt_path=last_checkpoint_path)
    else:
        print("Starting training from scratch.")
        trainer.fit(prnu_model, datamodule=prnu_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the image restoration model")

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the training experiment",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="artifacts",
        help="Base directory where logs and checkpoints will be saved",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory where the image dataset is stored or will be downloaded to",
    )
    parser.add_argument(
        "--prnu_dir",
        type=str,
        default="prnu_maps",
        help="Directory where the PRNU maps are stored",
    )
    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        output_folder=args.output_folder,
        data_dir=args.data_dir,
        prnu_dir=args.prnu_dir,
    )
