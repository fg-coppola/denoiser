import torch
import torch.nn as nn
import torch.nn.functional as F


class DereverberationLoss(nn.Module):
    def __init__(
        self,
        gamma_over=2.0,
        gamma_under=0.5,
        lambda_sa=1.0,
        lambda_asym=0.5,
        lambda_delta=0.2,
    ):
        """
        Initializes the multi-objective loss for dereverberation.
        Assumes inputs are already power-law compressed (e.g., c=0.3 in the dataloader)
        and the predicted spectrogram is already masked by the network.

        Args:
            gamma_over (float): Penalty weight for over-suppression (speech degradation).
            gamma_under (float): Penalty weight for under-suppression (residual reverberation).
            lambda_sa (float): Weight for the Signal Approximation (L1) Loss.
            lambda_asym (float): Weight for the Asymmetric Loss.
            lambda_delta (float): Weight for the Temporal Delta Loss.
        """
        super(DereverberationLoss, self).__init__()
        self.gamma_over = gamma_over
        self.gamma_under = gamma_under
        self.lambda_sa = lambda_sa
        self.lambda_asym = lambda_asym
        self.lambda_delta = lambda_delta

    def forward(self, pred_comp, target_comp):
        """
        Computes the global loss.
        Expected tensor shape: [Batch, Channels, Frequency, Time].

        Args:
            pred_comp: Predicted power-law compressed magnitude spectrogram
                       (already masked by the model: mask * input_compressed).
            target_comp: Target (clean) power-law compressed magnitude spectrogram.
        """
        # 1. Signal Approximation Loss (L1)
        # Measures the absolute error between the predicted and target spectrograms
        loss_sa = F.l1_loss(pred_comp, target_comp)

        # 2. Asymmetric Over-Suppression Penalty Loss
        # Calculate the directional error
        error = pred_comp - target_comp

        # error < 0: network predicted less energy than target (speech over-suppression)
        # error >= 0: network predicted more energy than target (residual reverberation)
        asym_penalty = torch.where(
            error < 0,
            self.gamma_over * torch.abs(error),
            self.gamma_under * torch.abs(error),
        )

        loss_asym = torch.mean(asym_penalty)

        # 3. Delta Spectrum Temporal Continuity Loss
        # Calculate first-order differences along the time axis (last dimension)
        # This forces the predicted envelope to decay smoothly like natural acoustics,
        # avoiding abrupt cuts that cause musical noise.
        diff_pred = pred_comp[..., 1:] - pred_comp[..., :-1]
        diff_target = target_comp[..., 1:] - target_comp[..., :-1]

        loss_delta = F.l1_loss(diff_pred, diff_target)

        # Compute the total weighted loss
        total_loss = (
            (self.lambda_sa * loss_sa)
            + (self.lambda_asym * loss_asym)
            + (self.lambda_delta * loss_delta)
        )

        return total_loss
