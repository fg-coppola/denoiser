from pathlib import Path
from typing import Any

import pytorch_lightning as L
import torch
import torchaudio


class AudioLoggerCallback(L.Callback):
    """
    Callback to log validation audio samples to TensorBoard and save
    reconstructed audio files to disk during validation and testing.
    Reconstructs audio using Inverse STFT for Noisy, Reconstructed, and Clean.
    Assumes inputs are already in linear scale (no power-law compression applied).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        save_dir: str | Path | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_peak(self, audio: torch.Tensor) -> torch.Tensor:
        """Peak-normalize to [-1, 1] to prevent TensorBoard clipping/distortion."""
        peak = torch.amax(torch.abs(audio), dim=-1, keepdim=True)
        return audio / (peak + 1e-8)

    def _process_and_save(
        self,
        trainer: L.Trainer,
        batch: tuple[Any, Any],
        preds: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        stage: str,
        dataloader_idx: int = 0,
    ):
        """Unified method to handle STFT reconstruction, TensorBoard logging, and disk saving."""
        try:
            x, y = batch

            # 1. Extract Noisy Magnitude and Phase (only the first sample of the batch)
            noisy_mag = x[0][0:1]
            noisy_phase = x[1][0:1]

            # 2. Extract Reconstructed Components
            pred_real, pred_imag = preds[0][0:1], preds[1][0:1]
            pred_complex_tensor = torch.complex(pred_real, pred_imag)
            pred_mag = torch.abs(pred_complex_tensor)

            # 3. Extract Clean Target Waveform and its Perfect Phase
            clean_wav = y[0:1].squeeze(1) if y[0:1].dim() == 3 else y[0:1]
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

            if pred_mag.dim() == 4:
                clean_phase = clean_phase.unsqueeze(1)

            # 4. Build Complex Spectrograms for iSTFT
            noisy_complex = torch.polar(noisy_mag, noisy_phase)
            pred_complex = pred_complex_tensor
            oracle_complex = torch.polar(pred_mag, clean_phase)

            # Squeeze channel dim for iSTFT
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
            clean_wav = torch.istft(
                clean_stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                length=expected_length,
            )

            # Normalize waveforms
            noisy_wav = self._normalize_peak(noisy_wav)
            pred_wav = self._normalize_peak(pred_wav)
            oracle_wav = self._normalize_peak(oracle_wav)
            clean_wav = self._normalize_peak(clean_wav)

            # 6. TensorBoard Logging (Only during Validation to avoid spam)
            if (
                stage == "val"
                and trainer.logger
                and hasattr(trainer.logger.experiment, "add_audio")
            ):
                tb = trainer.logger.experiment
                gs = trainer.global_step
                tb.add_audio(
                    "Audio/1_Noisy",
                    noisy_wav.squeeze(0).detach().cpu(),
                    gs,
                    self.sample_rate,
                )
                tb.add_audio(
                    "Audio/2_Reconstructed",
                    pred_wav.squeeze(0).detach().cpu(),
                    gs,
                    self.sample_rate,
                )
                tb.add_audio(
                    "Audio/3_Oracle_Phase",
                    oracle_wav.squeeze(0).detach().cpu(),
                    gs,
                    self.sample_rate,
                )
                tb.add_audio(
                    "Audio/4_Clean",
                    clean_wav.squeeze(0).detach().cpu(),
                    gs,
                    self.sample_rate,
                )

            # 7. Disk Saving (Only during testing)
            if self.save_dir is not None and stage == "test":
                test_loaders = trainer.test_dataloaders
                current_dataloader = (
                    test_loaders[dataloader_idx]
                    if isinstance(test_loaders, list)
                    else test_loaders
                )
                dataset = current_dataloader.dataset

                file_stem = dataset.clean_files[batch_idx].stem
                file_name = f"{file_stem}_recon.wav"

                out_path = self.save_dir / stage / file_name
                out_path.parent.mkdir(parents=True, exist_ok=True)

                wav_to_save = pred_wav.detach().cpu()
                if wav_to_save.dim() == 1:
                    wav_to_save = wav_to_save.unsqueeze(0)

                torchaudio.save(str(out_path), wav_to_save, self.sample_rate)

        except Exception as e:
            print(f"[Warning] Failed to generate {stage} audio in callback: {e}")

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict,
        batch: tuple[Any, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx != 0 or trainer.sanity_checking:
            return

        preds = outputs.get("preds")
        if preds is not None:
            self._process_and_save(trainer, batch, preds, batch_idx, stage="val")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict,
        batch: tuple[Any, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # In test phase, we process every single batch (batch_size is 1)
        # to generate a reconstructed audio file for all 2620 samples.
        preds = outputs.get("preds")
        if preds is not None:
            self._process_and_save(trainer, batch, preds, batch_idx, stage="test")
