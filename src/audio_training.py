import argparse
import math
from pathlib import Path

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks.audio_logging import AudioLoggerCallback
from callbacks.perceptual_metrics import PerceptualMetricsCallback
from callbacks.spectrogram_logging import SpectrogramVisualizerCallback
from datasets.voices import RIRDataModule
from losses.audio import CompositeDereverberationLoss
from models import ComplexIRMDenoiser, RestorationModule, UNet


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
    training_subset: str = "dev-clean",
    batch_size: int = 24,
    base_features: int = 32,
):
    # Force the cache directory into a local folder within the project
    cache_dir = os.path.abspath("./.torch_compile_cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    # Setup high precision for matrix multiplication
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # This is the actual batch size that will be used in the DataLoader.
    # The effective batch size will be this multiplied by the accumulation steps.
    max_supported_micro_batch = 64
    fixed_micro_batch = min(batch_size, max_supported_micro_batch)
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

    logger = TensorBoardLogger(
        save_dir=logs_dir, name=experiment_name, version="version_0"
    )

    criterion = CompositeDereverberationLoss(
        lambda_mr_stft=1.0, lambda_sdr=1.0, compression_factor=0.3
    )

    unet = UNet(
        in_channels=2,
        out_channels=2,
        base_features=base_features,
        upsample_mode="bilinear",
    )
    spectral_denoiser = ComplexIRMDenoiser(base_model=unet)

    spectral_denoiser = torch.compile(spectral_denoiser, mode="default", dynamic=True)

    rir_model = RestorationModule(
        model=spectral_denoiser,
        loss_fn=criterion,
        lr=2e-4,
    )

    rir_loader = RIRDataModule(
        train_data_dir="data/libriSpeech",
        val_data_dir="data/audio/validation",
        test_data_dir="data/audio/test",
        subset=training_subset,
        target_duration_seconds=3.0,
        synthetic_rir_paths=["data/rir/synthetic/train"],
        real_rir_paths=["data/rir/real/train"],
        synthetic_prob=0.5,
        download=True,
        batch_size=fixed_micro_batch,
        num_workers=12,
        persistent_workers=True,
        pin_memory=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss/total",
        min_delta=0.01,
        patience=10,
        verbose=True,
        mode="min",
    )

    checkpoint_path = checkpoint_dir / experiment_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="rir-best-{epoch:02d}",
        monitor="val_loss/total",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    spectrogram_callback = SpectrogramVisualizerCallback()
    audio_callback = AudioLoggerCallback(sample_rate=16000, compression_factor=0.3)
    perceptual_callback = PerceptualMetricsCallback(
        sample_rate=16000, num_val_batches=5
    )

    trainer = L.Trainer(
        accelerator="cuda",
        devices=1,
        precision="bf16-mixed",
        max_epochs=500,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            spectrogram_callback,
            audio_callback,
            perceptual_callback,
        ],
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=accum_steps,
        # val_check_interval=0.25,  # Validate every 25% of an epoch
        limit_train_batches=0.5,  # Use 50% of the training data
    )

    # Check for existing checkpoint to resume training
    last_checkpoint_path = checkpoint_path / "last.ckpt"

    if last_checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        trainer.fit(rir_model, datamodule=rir_loader, ckpt_path=last_checkpoint_path)
    else:
        print("Starting training from scratch.")
        trainer.fit(rir_model, datamodule=rir_loader)

    print("Training complete. Running evaluation on the Test set...")
    trainer.test(rir_model, datamodule=rir_loader, ckpt_path="best")


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
