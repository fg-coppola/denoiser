import torch
import torch.nn as nn


class SpectralConvergenceLoss(nn.Module):
    """
    Calculates the Spectral Convergence loss.
    It penalizes structural differences between the target and predicted linear magnitudes.
    This helps significantly in reducing 'metallic' or 'robotic' artifacts by forcing
    the network to maintain continuous harmonic structures.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        # Calculate Frobenius norm over the spatial dimensions (Frequency and Time)
        # dim=(-2, -1) ensures we calculate the norm for each 2D spectrogram in the batch independently
        num = torch.linalg.matrix_norm(y_mag - x_mag, ord="fro", dim=(-2, -1))
        den = torch.linalg.matrix_norm(y_mag, ord="fro", dim=(-2, -1))

        # Divide and average across the batch (added epsilon to prevent division by zero)
        return (num / (den + 1e-7)).mean()


class HybridSpectrogramLoss(nn.Module):
    """
    A hybrid loss function combining standard L1 Loss (on log magnitudes)
    with Spectral Convergence Loss (on linear magnitudes).
    """

    def __init__(self, sc_weight: float = 1.0):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.sc_loss = SpectralConvergenceLoss()

        # The weight balancing the two losses. 1.0 is a robust starting point.
        self.sc_weight = sc_weight

    def forward(self, pred_log: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
        # 1. Standard L1 loss on the log-scaled spectrograms (Great for denoising details)
        l1 = self.l1_loss(pred_log, target_log)

        # 2. Convert log-scaled spectrograms back to linear scale for SC loss
        pred_linear = torch.expm1(pred_log)
        target_linear = torch.expm1(target_log)

        # 3. Calculate Spectral Convergence on linear magnitudes (Great for harmonic structure)
        sc = self.sc_loss(pred_linear, target_linear)

        # 4. Combine them
        return l1 + (self.sc_weight * sc)
