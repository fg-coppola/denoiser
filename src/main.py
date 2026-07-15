from pathlib import Path

import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.voices import RIRDataModule
from models import RestorationModule

LOGS_PATH = "artifacts/logs"
CHECKPOINTS_PATH = "artifacts/checkpoints"
EXPERIMENT_NAME = "rir_L1-loss_dev-clean_synthetic_v1"


def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    logs_dir = Path(LOGS_PATH)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(CHECKPOINTS_PATH)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=logs_dir, name=EXPERIMENT_NAME)

    rir_model = RestorationModule(
        in_channels=1,
        out_channels=1,
        base_features=32,
        loss_fn=nn.L1Loss(),
        lr=1e-4,
    )
    rir_loader = RIRDataModule(
        data_dir="data/libriSpeech",
        target_duration_seconds=5.0,
        rir_maps=None,
        subset="dev-clean",
        download=False,
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
    checkpoint_path = checkpoint_dir / EXPERIMENT_NAME
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="rir-best-{epoch:02d}-{val_loss:.4f}",
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

    last_checkpoint_path = checkpoint_path / "last.ckpt"
    if last_checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        trainer.fit(rir_model, datamodule=rir_loader, ckpt_path=last_checkpoint_path)
    else:
        print("Starting training from scratch.")
        trainer.fit(rir_model, datamodule=rir_loader)


if __name__ == "__main__":
    main()
