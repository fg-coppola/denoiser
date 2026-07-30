import pytorch_lightning as L
import torch
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)


class PerceptualMetricsCallback(L.Callback):
    """
    Validation callback that computes PESQ and STOI for both model predictions
    and the noisy baseline, then logs the deltas.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_val_batches: int = 5,
        n_fft: int = 1024,
        hop_length: int = 256,
        power_law_factor: float = 0.3,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_val_batches = num_val_batches
        self.power_law_factor = power_law_factor
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Model metrics
        self.pesq = PerceptualEvaluationSpeechQuality(sample_rate, "wb")
        self.stoi = ShortTimeObjectiveIntelligibility(sample_rate, False)

        # Baseline metrics (noisy vs clean)
        self.base_pesq = PerceptualEvaluationSpeechQuality(sample_rate, "wb")
        self.base_stoi = ShortTimeObjectiveIntelligibility(sample_rate, False)

    def _reconstruct_noisy(
        self,
        comp_mag: torch.Tensor,
        phase: torch.Tensor,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Decompress magnitude and convert back to time domain via iSTFT."""
        mag = comp_mag ** (1.0 / self.power_law_factor)
        complex_spec = torch.polar(mag, phase).squeeze(1)
        window = torch.hann_window(self.n_fft, device=device)
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            length=length,
        )

    def _normalize_peak(self, audio: torch.Tensor) -> torch.Tensor:
        """Peak-normalize to [-1, 1] to keep PESQ in its valid operating range."""
        peak = torch.amax(torch.abs(audio), dim=-1, keepdim=True)
        return audio / (peak + 1e-8)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx >= self.num_val_batches:
            return

        noisy_inputs, clean_audio = batch
        clean_audio = clean_audio.squeeze(1)
        expected_length = clean_audio.shape[-1]

        # PESQ requires at least ~0.25 s of audio
        if expected_length < int(0.25 * self.sample_rate):
            return

        noisy_comp_mag, noisy_phase = noisy_inputs

        with torch.no_grad():
            preds_real, preds_imag = pl_module(*noisy_inputs)
            pred_complex = torch.complex(preds_real, preds_imag).squeeze(1)

            device = pred_complex.device
            window = torch.hann_window(self.n_fft, device=device)

            pred_audio = torch.istft(
                pred_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                center=True,
                length=expected_length,
            )

            noisy_audio = self._reconstruct_noisy(
                noisy_comp_mag, noisy_phase, expected_length, device
            )

        # Move to CPU and normalize for metric stability
        pred_audio = self._normalize_peak(pred_audio.detach().cpu().float())
        noisy_audio = self._normalize_peak(noisy_audio.detach().cpu().float())
        clean_audio = self._normalize_peak(clean_audio.detach().cpu().float())

        # Accumulate
        self.pesq(pred_audio, clean_audio)
        self.stoi(pred_audio, clean_audio)

        self.base_pesq(noisy_audio, clean_audio)
        self.base_stoi(noisy_audio, clean_audio)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.sanity_checking:
            return

        try:
            pred_pesq = self.pesq.compute()
            pred_stoi = self.stoi.compute()

            base_pesq = self.base_pesq.compute()
            base_stoi = self.base_stoi.compute()
        except RuntimeError:
            # No data accumulated (e.g., all batches too short)
            return
        finally:
            # Always reset to avoid cross-epoch leakage
            self.pesq.reset()
            self.stoi.reset()
            self.base_pesq.reset()
            self.base_stoi.reset()

        # Compute deltas
        delta_pesq = pred_pesq - base_pesq
        delta_stoi = pred_stoi - base_stoi

        # Log model metrics
        pl_module.log("val/pesq", pred_pesq, prog_bar=True, sync_dist=True)
        pl_module.log("val/stoi", pred_stoi, prog_bar=True, sync_dist=True)

        # Log baseline metrics
        pl_module.log("val/baseline_pesq", base_pesq, sync_dist=True)
        pl_module.log("val/baseline_stoi", base_stoi, sync_dist=True)

        # Log deltas
        pl_module.log("val/delta_pesq", delta_pesq, prog_bar=True, sync_dist=True)
        pl_module.log("val/delta_stoi", delta_stoi, prog_bar=True, sync_dist=True)
