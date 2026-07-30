from typing import Any

import pytorch_lightning as L
import torch
import torchvision


class SpectrogramVisualizerCallback(L.Callback):
    """
    Callback to log validation spectrogram visuals to TensorBoard.
    Generates a 3-panel comparison: Noisy, Reconstructed, and Clean.
    Applies power-law compression purely for visualization purposes so that
    lower energy frequencies are visible.
    """

    def __init__(self, compression_factor: float = 0.3):
        super().__init__()
        self.compression_factor = compression_factor

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict,
        batch: tuple[Any, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # Process and log visuals only for the first batch of validation
        # Skip completely during sanity checks
        if batch_idx != 0 or trainer.sanity_checking:
            return

        x, y = batch
        preds = outputs.get("preds")

        if preds is None:
            return

        try:
            # 1. Unpack noisy input magnitude (currently linear)
            noisy_mag = x[0] if isinstance(x, (tuple, list)) else x

            # Apply compression ONLY for visualization
            noisy_viz_mag = torch.clamp(noisy_mag, min=1e-8) ** self.compression_factor

            n_img = noisy_viz_mag[0:1].detach().cpu()
            if n_img.dim() == 3:
                n_img = n_img.unsqueeze(1)  # Ensure shape [1, 1, Freq, Time]

            # 2. Extract real and imaginary predictions to compute magnitude (currently linear)
            if isinstance(preds, (tuple, list)):
                pred_real, pred_imag = preds[0], preds[1]
                pred_mag = torch.abs(torch.complex(pred_real, pred_imag))
            else:
                pred_mag = preds

            # Apply compression ONLY for visualization
            pred_viz_mag = torch.clamp(pred_mag, min=1e-8) ** self.compression_factor

            r_img = pred_viz_mag[0:1].detach().cpu()
            if r_img.dim() == 3:
                r_img = r_img.unsqueeze(1)  # Ensure shape [1, 1, Freq, Time]

            # 3. Compute clean reference target spectrogram
            clean_wav = y
            c_wav_sample = clean_wav[0:1].detach()

            # Ensure 2D tensor [Batch, Time] for STFT
            if c_wav_sample.dim() == 3:
                c_wav_sample = c_wav_sample.squeeze(1)

            window = torch.hann_window(1024).to(c_wav_sample.device)
            c_stft = torch.stft(
                c_wav_sample,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                window=window,
                return_complex=True,
                center=True,
            )
            c_mag = torch.abs(c_stft)

            # Apply compression ONLY for visualization
            c_viz_mag = torch.clamp(c_mag, min=1e-8) ** self.compression_factor

            c_img = c_viz_mag.cpu()
            if c_img.dim() == 3:
                c_img = c_img.unsqueeze(1)  # Ensure shape [1, 1, Freq, Time]

            # Flip frequency axis vertically so low frequencies stay at the bottom
            n_img = torch.flip(n_img, dims=[-2])
            r_img = torch.flip(r_img, dims=[-2])
            c_img = torch.flip(c_img, dims=[-2])

            # Concatenate images horizontally into a 3-column grid
            comparison_tensor = torch.cat([n_img, r_img, c_img], dim=0)
            grid = torchvision.utils.make_grid(
                comparison_tensor, nrow=3, normalize=True
            )

            # Log grid directly to TensorBoard
            if trainer.logger and hasattr(trainer.logger.experiment, "add_image"):
                trainer.logger.experiment.add_image(
                    "Validation: 1.Noisy | 2.Reconstructed | 3.Clean",
                    grid,
                    global_step=trainer.global_step,
                )

        except Exception as e:
            print(f"[Warning] Failed to generate validation visuals in callback: {e}")
