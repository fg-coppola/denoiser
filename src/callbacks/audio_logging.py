from typing import Any

import pytorch_lightning as L
import torch


class AudioLoggerCallback(L.Callback):
    """
    Callback to log validation audio samples to TensorBoard.
    Reconstructs audio using Inverse STFT for Noisy, Reconstructed, and Clean.
    """

    def __init__(self, sample_rate: int = 16000, compression_factor: float = 0.3):
        super().__init__()
        self.sample_rate = sample_rate
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
        # Process and log audio only for the first batch of validation
        if batch_idx != 0 or trainer.sanity_checking:
            return

        x, y = batch
        preds = outputs.get("preds")

        if preds is None:
            return

        try:
            # 1. Extract Noisy Magnitude and Phase (assuming x = (mag, phase))
            noisy_comp_mag = x[0][0:1] if isinstance(x, (tuple, list)) else x[0:1]
            noisy_phase = (
                x[1][0:1]
                if isinstance(x, (tuple, list)) and len(x) > 1
                else torch.zeros_like(noisy_comp_mag)
            )

            # 2. Extract Reconstructed Magnitude and Phase
            if isinstance(preds, (tuple, list)):
                pred_real, pred_imag = preds[0][0:1], preds[1][0:1]
                pred_complex_tensor = torch.complex(pred_real, pred_imag)
                pred_comp_mag = torch.abs(pred_complex_tensor)
                pred_phase = torch.angle(pred_complex_tensor)
            else:
                pred_comp_mag = preds[0:1]
                pred_phase = (
                    noisy_phase  # Fallback to noisy phase if only mag is predicted
                )

            # 3. De-compress Magnitudes (invert ** 0.3)
            noisy_linear_mag = torch.clamp(noisy_comp_mag, min=1e-8) ** (
                1.0 / self.compression_factor
            )
            pred_linear_mag = torch.clamp(pred_comp_mag, min=1e-8) ** (
                1.0 / self.compression_factor
            )

            # 4. Build Complex Spectrograms
            noisy_complex = noisy_linear_mag * torch.exp(1j * noisy_phase)
            pred_complex = pred_linear_mag * torch.exp(1j * pred_phase)

            # Squeeze channel dim for iSTFT: [1, 1, Freq, Time] -> [1, Freq, Time]
            if noisy_complex.dim() == 4:
                noisy_complex = noisy_complex.squeeze(1)
                pred_complex = pred_complex.squeeze(1)

            # 5. Inverse STFT
            window = torch.hann_window(1024).to(noisy_complex.device)
            noisy_wav = torch.istft(
                noisy_complex,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                window=window,
                center=True,
            )
            pred_wav = torch.istft(
                pred_complex,
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                window=window,
                center=True,
            )

            # Clean waveform is directly available from target 'y'
            clean_wav = y[0:1].squeeze(1) if y[0:1].dim() == 3 else y[0:1]

            # 6. Log to TensorBoard
            if trainer.logger and hasattr(trainer.logger.experiment, "add_audio"):
                tb = trainer.logger.experiment
                global_step = trainer.global_step

                # Move to CPU and squeeze batch dimension for TensorBoard [Time]
                tb.add_audio(
                    "Audio/1_Noisy",
                    noisy_wav.squeeze(0).detach().cpu(),
                    global_step,
                    self.sample_rate,
                )
                tb.add_audio(
                    "Audio/2_Reconstructed",
                    pred_wav.squeeze(0).detach().cpu(),
                    global_step,
                    self.sample_rate,
                )
                tb.add_audio(
                    "Audio/3_Clean",
                    clean_wav.squeeze(0).detach().cpu(),
                    global_step,
                    self.sample_rate,
                )

        except Exception as e:
            print(f"[Warning] Failed to generate validation audio in callback: {e}")
