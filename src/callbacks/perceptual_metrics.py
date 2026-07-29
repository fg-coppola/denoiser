import pytorch_lightning as L
import torch
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)


class PerceptualMetricsCallback(L.Callback):
    def __init__(self, sample_rate=16000, num_val_batches=5):
        super().__init__()
        self.sample_rate = sample_rate
        # We limit the validation to a few batches because PESQ/STOI on CPU are very slow
        self.num_val_batches = num_val_batches

        # Initialize metrics (wideband 'wb' is standard for 16kHz audio)
        self.pesq_metric = PerceptualEvaluationSpeechQuality(sample_rate, "wb")
        self.stoi_metric = ShortTimeObjectiveIntelligibility(sample_rate, False)

        # Lists to store batch scores during validation
        self.val_pesq_scores = []
        self.val_stoi_scores = []
        self.val_base_pesq = []
        self.val_base_stoi = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        # Process only a small subset of validation batches
        if batch_idx >= self.num_val_batches:
            return

        # Unpack the batch
        noisy_audio, clean_audio = batch

        # Ensure we don't track gradients for metric calculation
        with torch.no_grad():
            # Get model predictions
            preds = pl_module(noisy_audio)

            # Metrics must be computed on CPU in float32
            preds = preds.detach().cpu().float()
            clean = clean_audio.detach().cpu().float()
            noisy = noisy_audio.detach().cpu().float()

            # Calculate Model metrics (Prediction vs Clean)
            pred_pesq = self.pesq_metric(preds, clean)
            pred_stoi = self.stoi_metric(preds, clean)

            # Calculate Baseline metrics (Noisy vs Clean)
            base_pesq = self.pesq_metric(noisy, clean)
            base_stoi = self.stoi_metric(noisy, clean)

            # Store the mean of the current batch
            self.val_pesq_scores.append(pred_pesq.item())
            self.val_stoi_scores.append(pred_stoi.item())
            self.val_base_pesq.append(base_pesq.item())
            self.val_base_stoi.append(base_stoi.item())

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # Skip if sanity check is running or if no scores were collected
        if trainer.sanity_checking or not self.val_pesq_scores:
            return

        # 1. Calculate epoch averages
        avg_pred_pesq = sum(self.val_pesq_scores) / len(self.val_pesq_scores)
        avg_base_pesq = sum(self.val_base_pesq) / len(self.val_base_pesq)

        avg_pred_stoi = sum(self.val_stoi_scores) / len(self.val_stoi_scores)
        avg_base_stoi = sum(self.val_base_stoi) / len(self.val_base_stoi)

        # 2. Calculate the net improvement (Delta)
        delta_pesq = avg_pred_pesq - avg_base_pesq
        delta_stoi = avg_pred_stoi - avg_base_stoi

        # 3. MERGED GRAPHS (Safe for any logger)
        # We try to get the experiment object. If it supports add_scalars (like TensorBoard),
        # we plot Model and Baseline together in the same graph.
        experiment = getattr(trainer.logger, "experiment", None)

        if hasattr(experiment, "add_scalars"):
            # This creates a "Comparison" folder in TensorBoard with the dual-line graphs
            experiment.add_scalars(
                "Comparison/PESQ",
                {"Model": avg_pred_pesq, "Baseline_Noisy": avg_base_pesq},
                global_step=trainer.global_step,
            )
            experiment.add_scalars(
                "Comparison/STOI",
                {"Model": avg_pred_stoi, "Baseline_Noisy": avg_base_stoi},
                global_step=trainer.global_step,
            )

        # 4. Standard Logging
        # We log the model's absolute performance and the Delta.
        # These work with ANY logger and are saved in the checkpoint.
        pl_module.log("val/subset_pesq", avg_pred_pesq, prog_bar=True, sync_dist=True)
        pl_module.log("val/delta_pesq", delta_pesq, sync_dist=True)

        pl_module.log("val/subset_stoi", avg_pred_stoi, prog_bar=True, sync_dist=True)
        pl_module.log("val/delta_stoi", delta_stoi, sync_dist=True)

        # 5. Clear lists for the next epoch
        self.val_pesq_scores.clear()
        self.val_stoi_scores.clear()
        self.val_base_pesq.clear()
        self.val_base_stoi.clear()
