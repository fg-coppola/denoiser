from pathlib import Path

import pytorch_lightning as L
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.voices import RIRDataModule
from models import RestorationModule

LOGS_PATH = "../artifacts/logs"


def main():
    logs_dir = Path(LOGS_PATH)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_dir=logs_dir, name="rir_restoration")

    rir_model = RestorationModule(
        in_channels=3,
        out_channels=3,
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
    )

    trainer = L.Trainer(
        logger=logger,
        accelerator="cuda",
        devices=1,
        max_epochs=100,
        log_every_n_steps=10,  # How often to push metrics to logger
    )

    trainer.fit(rir_model, datamodule=rir_loader)


if __name__ == "__main__":
    main()
