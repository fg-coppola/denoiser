from typing import Any

import pytorch_lightning as L
import torch


class AudioLoggerCallback(L.Callback):
    """
    Callback to log validation audio samples to TensorBoard.
    Reconstructs audio using Inverse STFT for Noisy, Reconstructed, and Clean.
    Also includes an "Oracle Phase" version that combines the predicted magnitude with the clean target phase.
    Assumes inputs are already in linear scale (no power-law compression applied).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def _normalize_peak(self, audio: torch.Tensor) -> torch.Tensor:
        """Peak-normalize to [-1, 1] to prevent TensorBoard clipping/distortion."""
        peak = torch.amax(torch.abs(audio), dim=-1, keepdim=True)
        return audio / (peak + 1e-8)

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
            # 1. Extract Noisy Magnitude and Phase
            # Assuming dataloader yields x as (noisy_mag, noisy_phase)
            noisy_mag = x[0][0:1]
            noisy_phase = x[1][0:1]

            # 2. Extract Reconstructed Components
            # The wrapper returns a tuple of (pred_real, pred_imag)
            pred_real, pred_imag = preds[0][0:1], preds[1][0:1]

            # Build complex tensor directly from the model's Cartesian outputs
            pred_complex_tensor = torch.complex(pred_real, pred_imag)

            # Extract magnitude strictly for the Oracle Phase reconstruction
            pred_mag = torch.abs(pred_complex_tensor)

            # 3. Extract Clean Target Waveform and its Perfect Phase
            clean_wav = y[0:1].squeeze(1) if y[0:1].dim() == 3 else y[0:1]

            # Setup the window for STFT/iSTFT operations
            window = torch.hann_window(self.win_length).to(clean_wav.device)

            clean_stft = torch.stft(
                clean_wav,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=True,
                center=True,
            )
            clean_phase = torch.angle(clean_stft)

            # Match channel dimension if pred_mag has it (e.g., [1, 1, Freq, Time])
            if pred_mag.dim() == 4:
                clean_phase = clean_phase.unsqueeze(1)

            # 4. Build Complex Spectrograms for iSTFT
            noisy_complex = torch.polar(noisy_mag, noisy_phase)
            pred_complex = pred_complex_tensor  # Already built above
            oracle_complex = torch.polar(pred_mag, clean_phase)

            # Squeeze channel dim for iSTFT: [1, 1, Freq, Time] -> [1, Freq, Time]
            if noisy_complex.dim() == 4:
                noisy_complex = noisy_complex.squeeze(1)
            if pred_complex.dim() == 4:
                pred_complex = pred_complex.squeeze(1)
            if oracle_complex.dim() == 4:
                oracle_complex = oracle_complex.squeeze(1)

            # 5. Inverse STFT
            expected_length = clean_wav.shape[-1]

            noisy_wav = torch.istft(
                noisy_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                length=expected_length,
            )
            pred_wav = torch.istft(
                pred_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                length=expected_length,
            )
            oracle_wav = torch.istft(
                oracle_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                length=expected_length,
            )

            # 6. Normalize all waveforms to prevent TensorBoard distortion
            noisy_wav = self._normalize_peak(noisy_wav)
            pred_wav = self._normalize_peak(pred_wav)
            oracle_wav = self._normalize_peak(oracle_wav)
            clean_wav = self._normalize_peak(clean_wav)

            # 7. Log to TensorBoard
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
                    "Audio/3_Oracle_Phase",
                    oracle_wav.squeeze(0).detach().cpu(),
                    global_step,
                    self.sample_rate,
                )
                tb.add_audio(
                    "Audio/4_Clean",
                    clean_wav.squeeze(0).detach().cpu(),
                    global_step,
                    self.sample_rate,
                )

        except Exception as e:
            print(f"[Warning] Failed to generate validation audio in callback: {e}")
