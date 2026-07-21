import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeSpectrogramLoss(nn.Module):
    """
    Composite loss function tailored for log-spectrogram prediction.
    Uses Asymmetric L1 for magnitude alignment and Sobel for edge/formant preservation.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        sobel_weight: float = 0.5,
        asymmetry_penalty: float = 2.5,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.sobel_weight = sobel_weight
        self.asymmetry_penalty = asymmetry_penalty

        # Define Sobel kernels for edge detection (X and Y directions)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        # Reshape for F.conv2d (out_channels, in_channels, kernel_height, kernel_width)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _asymmetric_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates Log STFT Magnitude Loss with asymmetric penalization.
        Operates securely in the log domain.
        """
        diff = pred - target
        abs_diff = torch.abs(diff)

        # Create a mask where prediction is lower than target
        under_estimation = (diff < 0).float()

        # Apply the penalty weight to the under-estimated pixels
        weighted_diff = abs_diff * (
            1.0 + (self.asymmetry_penalty - 1.0) * under_estimation
        )
        return weighted_diff.mean()

    def _sobel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes L1 loss on the spatial gradients (edges) of the log-spectrograms.
        Crucial for preserving harmonic lines and high-frequency transients.
        """
        # Pad inputs to keep spatial dimensions same after convolution
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode="replicate")
        target_pad = F.pad(target, (1, 1, 1, 1), mode="replicate")

        # Apply Sobel filters
        pred_grad_x = F.conv2d(pred_pad, self.sobel_x)
        pred_grad_y = F.conv2d(pred_pad, self.sobel_y)

        target_grad_x = F.conv2d(target_pad, self.sobel_x)
        target_grad_y = F.conv2d(target_pad, self.sobel_y)

        # Calculate L1 loss on the gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def forward(
        self, pred_log_spec: torch.Tensor, target_log_spec: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        if pred_log_spec.dim() == 3:
            pred_log_spec = pred_log_spec.unsqueeze(1)
            target_log_spec = target_log_spec.unsqueeze(1)

        l1_raw = self._asymmetric_l1(pred_log_spec, target_log_spec)
        sobel_raw = self._sobel_loss(pred_log_spec, target_log_spec)

        weighted_l1 = self.l1_weight * l1_raw
        weighted_sobel = self.sobel_weight * sobel_raw

        total_loss = weighted_l1 + weighted_sobel

        loss_components = {
            "loss/l1_asymmetric": weighted_l1,
            "loss/sobel": weighted_sobel,
        }

        return total_loss, loss_components


class WeightedDualDomainLoss(nn.Module):
    """
    Computes the L1 loss in both linear and logarithmic magnitude domains,
    applying specific weights to balance the gradient magnitudes.
    """

    def __init__(self, linear_weight: float = 1.0, log_weight: float = 1.0):
        super().__init__()
        self.linear_weight = linear_weight
        self.log_weight = log_weight

    def forward(
        self, pred_log_spec: torch.Tensor, target_log_spec: torch.Tensor
    ) -> torch.Tensor:

        # 1. Calculate Linear Loss (Focuses on volume and formants)
        pred_mag = torch.expm1(pred_log_spec)
        target_mag = torch.expm1(target_log_spec)
        linear_loss = F.l1_loss(pred_mag, target_mag)

        # 2. Calculate Log Loss (Focuses on high frequencies and details)
        log_loss = F.l1_loss(pred_log_spec, target_log_spec)

        # 3. Combine with weights
        total_loss = (self.linear_weight * linear_loss) + (self.log_weight * log_loss)

        loss_components = {
            "loss/l1_log": log_loss,
            "loss/l1_linear": linear_loss,
        }

        return total_loss, loss_components
