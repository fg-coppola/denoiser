import argparse
from pathlib import Path

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.voices import RIRDataModule
from losses.audio import WeightedDualDomainLoss
from models import RatioMaskingDenoiser, RestorationModule, UNet


def main(
    experiment_name: str,
    output_folder: str,
    training_subset: str = "dev-clean",
    batch_size: int = 24,
    base_features: int = 32,
):
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

    # criterion = CompositeSpectrogramLoss(
    #     l1_weight=1.0, sobel_weight=1.0, asymmetry_penalty=3.0
    # )
    criterion = WeightedDualDomainLoss(linear_weight=1.0, log_weight=1.0)

    unet = UNet(
        in_channels=1,
        out_channels=1,
        base_features=base_features,
        # kernel_size=(11, 5),  # Asymmetric kernel for audio spectrograms
    )
    ratio_denoiser = RatioMaskingDenoiser(base_model=unet)
    rir_model = RestorationModule(
        denoiser=ratio_denoiser,
        loss_fn=criterion,
        lr=1e-4,
    )

    rir_loader = RIRDataModule(
        data_dir="data/libriSpeech",
        target_duration_seconds=5.0,
        rir_maps=None,
        subset=training_subset,
        download=True,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
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

    # Check for existing checkpoint to resume training
    last_checkpoint_path = checkpoint_path / "last.ckpt"

    if last_checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        trainer.fit(rir_model, datamodule=rir_loader, ckpt_path=last_checkpoint_path)
    else:
        print("Starting training from scratch.")
        trainer.fit(rir_model, datamodule=rir_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the audio restoration model")

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
        "--training_subset",
        type=str,
        default="dev-clean",
        help="Subset of LibriSpeech to use for training (e.g., 'dev-clean', 'train-clean-100')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for training",
    )
    parser.add_argument(
        "--base_features",
        type=int,
        default=32,
        help="Number of base features for the model",
    )
    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        output_folder=args.output_folder,
        training_subset=args.training_subset,
        batch_size=args.batch_size,
        base_features=args.base_features,
    )
