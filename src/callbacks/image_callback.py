from typing import Any

import pytorch_lightning as L
import torch
import torchvision



class ImageVisualizerCallback(L.Callback):
    """Callback to log validation image visuals (Low-Light, Reconstructed, Clean) to TensorBoard.

    Generates a 3-panel comparison grid for image enhancement tasks.
    """

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict | Any,
        batch: tuple[Any, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # Process and log visuals only for the first batch of validation
        # Skip completely during sanity checks
        if batch_idx != 0 or trainer.sanity_checking:
            return

        x, y = batch

        # Extract predictions from validation_step outputs or perform inference if missing
        preds = outputs.get("preds") if isinstance(outputs, dict) else None
        if preds is None:
            with torch.no_grad():
                preds = (
                    pl_module(*x)
                    if isinstance(x, (tuple, list))
                    else pl_module(x)
                )

        try:
            # 1. Extract low-light input image (first sample in batch)
            low_input = x[0] if isinstance(x, (tuple, list)) else x
            low_img = low_input[0:1].detach().cpu()  # Shape: [1, C, H, W]

            # 2. Extract model output prediction and clamp RGB values to valid range [0, 1]
            pred_output = (
                preds[0] if isinstance(preds, (tuple, list)) else preds
            )
            pred_img = pred_output[0:1].detach().cpu()  # Shape: [1, C, H, W]
            pred_img = torch.clamp(pred_img, 0.0, 1.0)

            # 3. Extract clean ground truth image
            clean_img = y[0:1].detach().cpu()  # Shape: [1, C, H, W]

            # Concatenate images horizontally into a 3-column grid
            comparison_tensor = torch.cat([low_img, pred_img, clean_img], dim=0)
            grid = torchvision.utils.make_grid(
                comparison_tensor, nrow=3, normalize=False
            )

            # Log grid directly to TensorBoard / Logger
            if trainer.logger and hasattr(
                trainer.logger.experiment, "add_image"
            ):
                trainer.logger.experiment.add_image(
                    "Validation: 1.Low-Light | 2.Reconstructed | 3.Clean",
                    grid,
                    global_step=trainer.global_step,
                )

        except Exception as e:
            print(
                f"[Warning] Failed to generate validation visuals in image callback: {e}"
            )