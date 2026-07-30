import pytorch_lightning as L
import torch
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)


class PerceptualMetricsCallback(L.Callback):
    """
    Validation and Test callback that computes PESQ and STOI for model predictions,
    noisy baseline, and an oracle scenario (Predicted Magnitude + Clean Phase).
    Logs the metrics, deltas, and test variance/std across the entire dataset.
    Assumes inputs are in linear scale (no power-law compression).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_val_batches: int = 5,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_val_batches = num_val_batches
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Validation metrics (stateful accumulators)
        self.pesq = PerceptualEvaluationSpeechQuality(sample_rate, "wb")
        self.stoi = ShortTimeObjectiveIntelligibility(sample_rate, False)
        self.base_pesq = PerceptualEvaluationSpeechQuality(sample_rate, "wb")
        self.base_stoi = ShortTimeObjectiveIntelligibility(sample_rate, False)

        # Test metric containers for full-run tracking, mean, and variance calculation
        self.test_pesq_scores = []
        self.test_stoi_scores = []
        self.test_base_pesq_scores = []
        self.test_base_stoi_scores = []
        self.test_oracle_mag_pesq_scores = []
        self.test_oracle_mag_stoi_scores = []

        # Temporary single-sample metric calculators for testing collection
        self._test_pesq_metric = PerceptualEvaluationSpeechQuality(sample_rate, "wb")
        self._test_stoi_metric = ShortTimeObjectiveIntelligibility(sample_rate, False)
        self._test_base_pesq_metric = PerceptualEvaluationSpeechQuality(
            sample_rate, "wb"
        )
        self._test_base_stoi_metric = ShortTimeObjectiveIntelligibility(
            sample_rate, False
        )
        self._test_oracle_mag_pesq_metric = PerceptualEvaluationSpeechQuality(
            sample_rate, "wb"
        )
        self._test_oracle_mag_stoi_metric = ShortTimeObjectiveIntelligibility(
            sample_rate, False
        )

    def _reconstruct_noisy(
        self,
        mag: torch.Tensor,
        phase: torch.Tensor,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert linear magnitude and phase back to time domain via iSTFT."""
        complex_spec = torch.polar(mag, phase).squeeze(1)
        window = torch.hann_window(self.win_length, device=device)
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
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
        if trainer.sanity_checking or batch_idx >= self.num_val_batches:
            return

        noisy_inputs, clean_audio = batch
        clean_audio = clean_audio.squeeze(1)
        expected_length = clean_audio.shape[-1]

        if expected_length < int(0.25 * self.sample_rate):
            return

        noisy_mag, noisy_phase = noisy_inputs

        with torch.no_grad():
            preds_real, preds_imag = pl_module(*noisy_inputs)
            pred_complex = torch.complex(preds_real, preds_imag).squeeze(1)

            device = pred_complex.device
            window = torch.hann_window(self.win_length, device=device)

            pred_audio = torch.istft(
                pred_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                length=expected_length,
            )

            noisy_audio = self._reconstruct_noisy(
                noisy_mag, noisy_phase, expected_length, device
            )

        pred_audio = self._normalize_peak(pred_audio.detach().cpu().float())
        noisy_audio = self._normalize_peak(noisy_audio.detach().cpu().float())
        clean_audio = self._normalize_peak(clean_audio.detach().cpu().float())

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
            return
        finally:
            self.pesq.reset()
            self.stoi.reset()
            self.base_pesq.reset()
            self.base_stoi.reset()

        delta_pesq = pred_pesq - base_pesq
        delta_stoi = pred_stoi - base_stoi

        pl_module.log("val/pesq", pred_pesq, prog_bar=True, sync_dist=True)
        pl_module.log("val/stoi", pred_stoi, prog_bar=True, sync_dist=True)
        pl_module.log("val/baseline_pesq", base_pesq, sync_dist=True)
        pl_module.log("val/baseline_stoi", base_stoi, sync_dist=True)
        pl_module.log("val/delta_pesq", delta_pesq, prog_bar=True, sync_dist=True)
        pl_module.log("val/delta_stoi", delta_stoi, prog_bar=True, sync_dist=True)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Process every batch in the test dataset without batch limitations."""
        noisy_inputs, clean_audio = batch
        clean_audio = clean_audio.squeeze(1)
        expected_length = clean_audio.shape[-1]

        if expected_length < int(0.25 * self.sample_rate):
            return

        noisy_mag, noisy_phase = noisy_inputs

        with torch.no_grad():
            preds_real, preds_imag = pl_module(*noisy_inputs)
            pred_complex = torch.complex(preds_real, preds_imag).squeeze(1)
            pred_mag = torch.abs(pred_complex)

            device = pred_complex.device
            window = torch.hann_window(self.win_length, device=device)

            # 1. Standard Model Reconstruction (Pred Mag + Noisy Phase)
            pred_audio = torch.istft(
                pred_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                length=expected_length,
            )

            # 2. Noisy Baseline Reconstruction
            noisy_audio = self._reconstruct_noisy(
                noisy_mag, noisy_phase, expected_length, device
            )

            # 3. Oracle Magnitude Reconstruction (Pred Mag + Clean Phase)
            clean_stft = torch.stft(
                clean_audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=True,
                center=True,
            )
            clean_phase = torch.angle(clean_stft)

            oracle_mag_complex = torch.polar(pred_mag, clean_phase)
            oracle_mag_audio = torch.istft(
                oracle_mag_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                length=expected_length,
            )

        # Peak normalization on CPU
        pred_audio = self._normalize_peak(pred_audio.detach().cpu().float())
        noisy_audio = self._normalize_peak(noisy_audio.detach().cpu().float())
        oracle_mag_audio = self._normalize_peak(oracle_mag_audio.detach().cpu().float())
        clean_audio = self._normalize_peak(clean_audio.detach().cpu().float())

        # Compute metrics per individual sample for variance tracking
        p_score = self._test_pesq_metric(pred_audio, clean_audio).item()
        s_score = self._test_stoi_metric(pred_audio, clean_audio).item()
        bp_score = self._test_base_pesq_metric(noisy_audio, clean_audio).item()
        bs_score = self._test_base_stoi_metric(noisy_audio, clean_audio).item()
        op_score = self._test_oracle_mag_pesq_metric(
            oracle_mag_audio, clean_audio
        ).item()
        os_score = self._test_oracle_mag_stoi_metric(
            oracle_mag_audio, clean_audio
        ).item()

        self.test_pesq_scores.append(p_score)
        self.test_stoi_scores.append(s_score)
        self.test_base_pesq_scores.append(bp_score)
        self.test_base_stoi_scores.append(bs_score)
        self.test_oracle_mag_pesq_scores.append(op_score)
        self.test_oracle_mag_stoi_scores.append(os_score)

        # Reset temporary metric instances for the next sample
        self._test_pesq_metric.reset()
        self._test_stoi_metric.reset()
        self._test_base_pesq_metric.reset()
        self._test_base_stoi_metric.reset()
        self._test_oracle_mag_pesq_metric.reset()
        self._test_oracle_mag_stoi_metric.reset()

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Aggregate test scores across all samples, computing mean, variance, and std."""
        if not self.test_pesq_scores:
            return

        # Convert lists to tensors for statistical analysis
        pesq_t = torch.tensor(self.test_pesq_scores)
        stoi_t = torch.tensor(self.test_stoi_scores)
        b_pesq_t = torch.tensor(self.test_base_pesq_scores)
        b_stoi_t = torch.tensor(self.test_base_stoi_scores)
        op_pesq_t = torch.tensor(self.test_oracle_mag_pesq_scores)
        op_stoi_t = torch.tensor(self.test_oracle_mag_stoi_scores)

        # Compute means
        mean_pesq = pesq_t.mean().item()
        mean_stoi = stoi_t.mean().item()
        mean_base_pesq = b_pesq_t.mean().item()
        mean_base_stoi = b_stoi_t.mean().item()
        mean_oracle_mag_pesq = op_pesq_t.mean().item()
        mean_oracle_mag_stoi = op_stoi_t.mean().item()

        # Compute variances and standard deviations (unbiased estimator: correction=1)
        var_pesq = pesq_t.var(unbiased=True).item()
        std_pesq = pesq_t.std(unbiased=True).item()

        var_stoi = stoi_t.var(unbiased=True).item()
        std_stoi = stoi_t.std(unbiased=True).item()

        var_oracle_mag_pesq = op_pesq_t.var(unbiased=True).item()
        std_oracle_mag_pesq = op_pesq_t.std(unbiased=True).item()

        var_oracle_mag_stoi = op_stoi_t.var(unbiased=True).item()
        std_oracle_mag_stoi = op_stoi_t.std(unbiased=True).item()

        delta_pesq = mean_pesq - mean_base_pesq
        delta_stoi = mean_stoi - mean_base_stoi

        # Log final aggregated test metrics and distribution stats
        pl_module.log("test/pesq", mean_pesq, sync_dist=True)
        pl_module.log("test/pesq_var", var_pesq, sync_dist=True)
        pl_module.log("test/pesq_std", std_pesq, sync_dist=True)

        pl_module.log("test/stoi", mean_stoi, sync_dist=True)
        pl_module.log("test/stoi_var", var_stoi, sync_dist=True)
        pl_module.log("test/stoi_std", std_stoi, sync_dist=True)

        pl_module.log("test/baseline_pesq", mean_base_pesq, sync_dist=True)
        pl_module.log("test/baseline_stoi", mean_base_stoi, sync_dist=True)

        pl_module.log("test/delta_pesq", delta_pesq, sync_dist=True)
        pl_module.log("test/delta_stoi", delta_stoi, sync_dist=True)

        # Log Oracle Magnitude metrics (Pred Mag + Clean Phase)
        pl_module.log("test/oracle_mag_pesq", mean_oracle_mag_pesq, sync_dist=True)
        pl_module.log("test/oracle_mag_pesq_var", var_oracle_mag_pesq, sync_dist=True)
        pl_module.log("test/oracle_mag_pesq_std", std_oracle_mag_pesq, sync_dist=True)

        pl_module.log("test/oracle_mag_stoi", mean_oracle_mag_stoi, sync_dist=True)
        pl_module.log("test/oracle_mag_stoi_var", var_oracle_mag_stoi, sync_dist=True)
        pl_module.log("test/oracle_mag_stoi_std", std_oracle_mag_stoi, sync_dist=True)

        # Clear lists for safety
        self.test_pesq_scores.clear()
        self.test_stoi_scores.clear()
        self.test_base_pesq_scores.clear()
        self.test_base_stoi_scores.clear()
        self.test_oracle_mag_pesq_scores.clear()
        self.test_oracle_mag_stoi_scores.clear()
