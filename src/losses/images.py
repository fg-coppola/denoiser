import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional.image as FM


class CombinedLoss(nn.Module):
    """
    Loss dal notebook Keras/TensorFlow:
    0.4 * MAE (L1 Loss) + 0.6 * SSIM Loss (1 - SSIM)
    """
    def __init__(self, w_mae: float = 0.4, w_ssim: float = 0.6):
        super().__init__()
        self.w_mae = w_mae
        self.w_ssim = w_ssim

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        target_clamped = torch.clamp(target, 0.0, 1.0)

        mae_loss = F.l1_loss(pred, target)

        ssim_val = FM.structural_similarity_index_measure(
            pred_clamped, target_clamped, data_range=1.0
        )
        ssim_loss = 1.0 - ssim_val

        total_loss = (self.w_mae * mae_loss) + (self.w_ssim * ssim_loss)

        return total_loss, {
            "total": total_loss.detach(),
            "mae": mae_loss.detach(),
            "ssim": ssim_loss.detach(),
        }