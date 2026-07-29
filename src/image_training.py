import argparse
import math
from pathlib import Path

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Updated imports to point to image-specific modules
from datasets.images import LOLLightDataModule
from losses.images import MinimalLLLoss
from models import RestorationModule, UNet, LowLightdenoiser
from callbacks.image_callback import ImageVisualizerCallback


def compute_accumulation_steps(
    target_effective_batch: int, micro_batch_size: int = 8
) -> int:
    """
    Calculates the required gradient accumulation steps for a fixed dataloader batch size.
    Uses ceiling division to ensure the effective batch size is at least the target.
    """
    accumulation_steps = math.ceil(target_effective_batch / micro_batch_size)

    # Calculate what the actual effective batch size will be
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


def main(
    experiment_name: str,
    output_folder: str,
    data_dir: str = "data/LOLdataset",
    batch_size: int = 24,
    patch_size: int = 256,
    base_features: int = 64,
):
    # Setup high precision for matrix multiplication
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # This is the actual batch size that will be used in the DataLoader.
    # The effective batch size will be this multiplied by the accumulation steps.
    fixed_micro_batch = 8
    # Calculate how many steps we need to accumulate to reach the target
    accum_steps = compute_accumulation_steps(
        target_effective_batch=batch_size, micro_batch_size=fixed_micro_batch
    )

    # Define and create output directories
    base_dir = Path(output_folder)
    logs_dir = base_dir / "logs"
    checkpoint_dir = base_dir / "checkpoints"

    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=logs_dir, name=experiment_name)

    criterion = MinimalLLLoss(
        w_charb=1.0,
        w_ssim=1.0,
        w_color=0.2,
        w_grad=0.8,  # <-- da 0.5 a 0.8
    )

    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_features=64,
        upsample_mode="pixel_shuffle",
        decoder_dropout=0.0,  # <-- AGGIUNGI
    )

    denoiser = LowLightdenoiser(base_model=unet)

    image_model = RestorationModule(
        model=denoiser,
        loss_fn=criterion,
        lr=2e-4,  # <-- CAMBIA
    )

    # DataLoader setup for the LOL dataset
    lol_loader = LOLLightDataModule(
        data_dir=data_dir,
        batch_size=fixed_micro_batch,
        patch_size=patch_size,
        num_workers=4,
        persistent_workers=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=15,  # <-- CAMBIA da 5 a 15
        verbose=True,
        mode="min",
    )

    checkpoint_path = checkpoint_dir / experiment_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="lowlight-best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    
    image_callback = ImageVisualizerCallback()
    
    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,
        precision="32-true",  # <-- CAMBIA da bf16-mixed
        max_epochs=300,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=accum_steps,
        callbacks=[early_stop_callback, checkpoint_callback, image_callback],
        logger=logger,
        log_every_n_steps=10,
    )


    # Check for existing checkpoint to resume training
    last_checkpoint_path = checkpoint_path / "last.ckpt"

    if last_checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        trainer.fit(image_model, datamodule=lol_loader, ckpt_path=last_checkpoint_path)
    else:
        print("Starting training from scratch.")
        trainer.fit(image_model, datamodule=lol_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Low-Light Image Restoration model")

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
        help="Path to the root directory of the LOL dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Target effective batch size for training",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of the random square crops extracted from images during training",
    )
    parser.add_argument(
        "--base_features",
        type=int,
        default=64, # Increased to 64 compared to audio (32), standard for image-to-image translation
        help="Number of base features for the U-Net model",
    )
    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        output_folder=args.output_folder,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        base_features=args.base_features,
    )