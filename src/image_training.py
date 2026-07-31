import argparse
import math
import os
from pathlib import Path

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks.image_callback import ImageVisualizerCallback
from callbacks.image_metrics import ImageMetricsCallback
from datasets.images import LOLLightDataModule
from losses.images import CombinedLoss
from models import LowLightdenoiser, RestorationModule, UNet


def compute_accumulation_steps(
    target_effective_batch: int, micro_batch_size: int = 8
) -> int:
    accumulation_steps = math.ceil(target_effective_batch / micro_batch_size)
    actual_effective = micro_batch_size * accumulation_steps

    if actual_effective != target_effective_batch:
        print(
            f"Warning: Target batch size {target_effective_batch} is not a multiple of {micro_batch_size}."
        )
        print(
            f"Adjusted effective batch size to: {actual_effective} ({micro_batch_size} * {accumulation_steps} steps)"
        )
    else:
        print(
            f"Effective batch size: {actual_effective} ({micro_batch_size} * {accumulation_steps} steps)"
        )

    return accumulation_steps


def build_default_sources(data_dir: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Builds the default source lists for LOL v1, LOL-v2 Real, and LOL-v2 Synthetic
    based on a main data directory.
    """
    train_sources = [
        (
            os.path.join(data_dir, "lol_v1/our485/low"),
            os.path.join(data_dir, "lol_v1/our485/high"),
        ),
        (
            os.path.join(data_dir, "lol_v2/Real_captured/Train/Low"),
            os.path.join(data_dir, "lol_v2/Real_captured/Train/Normal"),
        ),
        (
            os.path.join(data_dir, "lol_v2/Synthetic/Train/Low"),
            os.path.join(data_dir, "lol_v2/Synthetic/Train/Normal"),
        ),
    ]

    val_sources = [
        (
            os.path.join(data_dir, "lol_v1/eval15/low"),
            os.path.join(data_dir, "lol_v1/eval15/high"),
        ),
        (
            os.path.join(data_dir, "lol_v2/Real_captured/Test/Low"),
            os.path.join(data_dir, "lol_v2/Real_captured/Test/Normal"),
        ),
        (
            os.path.join(data_dir, "lol_v2/Synthetic/Test/Low"),
            os.path.join(data_dir, "lol_v2/Synthetic/Test/Normal"),
        ),
    ]

    # Filter only paths that actually exist on disk to avoid errors
    valid_train = [
        (low, high) for low, high in train_sources
        if os.path.exists(low) and os.path.exists(high)
    ]
    valid_val = [
        (low, high) for low, high in val_sources
        if os.path.exists(low) and os.path.exists(high)
    ]

    return valid_train, valid_val


def main(
    experiment_name: str,
    output_folder: str,
    data_dir: str = "data",
    batch_size: int = 24,
    micro_batch_size: int = 8,
    patch_size: int = 384,
    base_features: int = 48,
):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    accum_steps = compute_accumulation_steps(
        target_effective_batch=batch_size, micro_batch_size=micro_batch_size
    )

    base_dir = Path(output_folder)
    logs_dir = base_dir / "logs"
    checkpoint_dir = base_dir / "checkpoints"

    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=logs_dir, name=experiment_name)

    criterion = CombinedLoss(
        w_mae=0.4,
        w_ssim=0.6,
    )

    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_features=base_features,
        upsample_mode="pixel_shuffle",
        decoder_dropout=0.0,
    )

    denoiser = LowLightdenoiser(base_model=unet)

    image_model = RestorationModule(
        model=denoiser,
        loss_fn=criterion,
        lr=1e-3,
    )

    # Obtain training and validation sources
    train_sources, val_sources = build_default_sources(data_dir)

    print("Sorgenti di Training caricate:")
    for low, high in train_sources:
        print(f"  - Low: {low} | High: {high}")

    print("Sorgenti di Validation caricate:")
    for low, high in val_sources:
        print(f"  - Low: {low} | High: {high}")

    lol_loader = LOLLightDataModule(
        train_sources=train_sources,
        val_sources=val_sources,
        batch_size=micro_batch_size,
        patch_size=patch_size,
        num_workers=4,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss/total",
        min_delta=1e-4,
        patience=15,
        verbose=True,
        mode="min",
    )

    checkpoint_path = checkpoint_dir / experiment_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="lowlight-best-{epoch:02d}-{val_loss/total:.4f}",
        monitor="val_loss/total",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    image_callback = ImageVisualizerCallback()
    metrics_callback = ImageMetricsCallback(num_val_batches=5)

    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,
        precision="32-true",
        max_epochs=300,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=accum_steps,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            image_callback,
            metrics_callback,
        ],
        logger=logger,
        log_every_n_steps=10,
    )

    last_checkpoint_path = checkpoint_path / "last.ckpt"

    if last_checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        trainer.fit(
            image_model, datamodule=lol_loader, ckpt_path=last_checkpoint_path
        )
    else:
        print("Starting training from scratch.")
        trainer.fit(image_model, datamodule=lol_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Low-Light Image Restoration model"
    )

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
        help="Root directory containing the dataset folders (lol_v1, lol_v2, etc.)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Target effective batch size for training",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=8,
        help="Physical batch size per step loaded in GPU memory",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=384,
        help="Size of the random square crops extracted from images",
    )
    parser.add_argument(
        "--base_features",
        type=int,
        default=48,
        help="Number of base features for the U-Net model",
    )
    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        output_folder=args.output_folder,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        patch_size=args.patch_size,
        base_features=args.base_features,
    )