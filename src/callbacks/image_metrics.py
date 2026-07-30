import pytorch_lightning as L
import torch
import torchmetrics.functional.image as FM


class ImageMetricsCallback(L.Callback):
    """
    Callback per il calcolo di metriche oggettive (PSNR, SSIM) su un subset
    di batch di validazione. Confronta il modello con la baseline (input grezzo).
    """

    def __init__(self, num_val_batches=5):
        super().__init__()
        self.num_val_batches = num_val_batches

        self.val_pred_psnr = []
        self.val_pred_ssim = []
        self.val_base_psnr = []
        self.val_base_ssim = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # Processa solo i primi N batch per velocità
        if batch_idx >= self.num_val_batches:
            return

        x, y = batch  # x: low-light input, y: normal-light target

        with torch.no_grad():
            preds = pl_module(x)

            # Sicurezza: clamp prima delle metriche
            preds = torch.clamp(preds, 0.0, 1.0)
            x = torch.clamp(x, 0.0, 1.0)

            # Sposta su CPU float32 (come nel tuo esempio audio)
            preds = preds.detach().cpu().float()
            target = y.detach().cpu().float()
            input_img = x.detach().cpu().float()

            # Metriche Modello (predizione vs target)
            pred_psnr = FM.peak_signal_noise_ratio(preds, target, data_range=1.0)
            pred_ssim = FM.structural_similarity_index_measure(preds, target, data_range=1.0)

            # Metriche Baseline (input grezzo vs target)
            base_psnr = FM.peak_signal_noise_ratio(input_img, target, data_range=1.0)
            base_ssim = FM.structural_similarity_index_measure(input_img, target, data_range=1.0)

            self.val_pred_psnr.append(pred_psnr.item())
            self.val_pred_ssim.append(pred_ssim.item())
            self.val_base_psnr.append(base_psnr.item())
            self.val_base_ssim.append(base_ssim.item())

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.sanity_checking or not self.val_pred_psnr:
            return

        # Medie epoca
        avg_pred_psnr = sum(self.val_pred_psnr) / len(self.val_pred_psnr)
        avg_base_psnr = sum(self.val_base_psnr) / len(self.val_base_psnr)
        avg_pred_ssim = sum(self.val_pred_ssim) / len(self.val_pred_ssim)
        avg_base_ssim = sum(self.val_base_ssim) / len(self.val_base_ssim)

        # Delta (miglioramento rispetto all'input grezzo)
        delta_psnr = avg_pred_psnr - avg_base_psnr
        delta_ssim = avg_pred_ssim - avg_base_ssim

        # Logging su TensorBoard con grafici comparativi
        experiment = getattr(trainer.logger, "experiment", None)
        if hasattr(experiment, "add_scalars"):
            experiment.add_scalars(
                "Comparison/PSNR",
                {"Model": avg_pred_psnr, "Baseline_Input": avg_base_psnr},
                global_step=trainer.global_step,
            )
            experiment.add_scalars(
                "Comparison/SSIM",
                {"Model": avg_pred_ssim, "Baseline_Input": avg_base_ssim},
                global_step=trainer.global_step,
            )

        # Logging standard (salvati nel checkpoint)
        pl_module.log("val/subset_psnr", avg_pred_psnr, prog_bar=True, sync_dist=True)
        pl_module.log("val/delta_psnr", delta_psnr, sync_dist=True)
        pl_module.log("val/subset_ssim", avg_pred_ssim, prog_bar=True, sync_dist=True)
        pl_module.log("val/delta_ssim", delta_ssim, sync_dist=True)

        # Reset per la prossima epoca
        self.val_pred_psnr.clear()
        self.val_pred_ssim.clear()
        self.val_base_psnr.clear()
        self.val_base_ssim.clear()