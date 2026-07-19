import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeSpectrogramLoss(nn.Module):
    """
    Generic restoration loss.

    L =
        λ_l1     * L1
      + λ_sobel * Sobel Gradient Loss
      + λ_sc    * Spectral Convergence

    Compatible with any tensor of shape:

        (B, C, H, W)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_sobel: float = 0.2,
        lambda_sc: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_sobel = lambda_sobel
        self.lambda_sc = lambda_sc
        self.eps = eps

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )

        self.register_buffer(
            "sobel_x",
            sobel_x.view(1, 1, 3, 3),
        )

        self.register_buffer(
            "sobel_y",
            sobel_y.view(1, 1, 3, 3),
        )

    def _sobel(self, x):

        channels = x.shape[1]

        kernel_x = self.sobel_x.repeat(channels, 1, 1, 1)
        kernel_y = self.sobel_y.repeat(channels, 1, 1, 1)

        grad_x = F.conv2d(
            x,
            kernel_x,
            padding=1,
            groups=channels,
        )

        grad_y = F.conv2d(
            x,
            kernel_y,
            padding=1,
            groups=channels,
        )

        return grad_x, grad_y

    def sobel_loss(self, prediction, target):

        pred_gx, pred_gy = self._sobel(prediction)
        tgt_gx, tgt_gy = self._sobel(target)

        loss_x = F.l1_loss(pred_gx, tgt_gx)
        loss_y = F.l1_loss(pred_gy, tgt_gy)

        return loss_x + loss_y

    def spectral_convergence_loss(
        self,
        prediction,
        target,
    ):
        """
        Computed independently for each sample in the batch,
        then averaged.

        Supports arbitrary channel count.
        """

        diff = target - prediction

        diff = diff.flatten(start_dim=1)
        target = target.flatten(start_dim=1)

        numerator = torch.linalg.norm(
            diff,
            dim=1,
        )

        denominator = torch.linalg.norm(
            target,
            dim=1,
        )

        sc = numerator / (denominator + self.eps)

        return sc.mean()

    def forward(
        self,
        prediction,
        target,
        return_components: bool = False,
    ):

        l1 = F.l1_loss(
            prediction,
            target,
        )

        sobel = self.sobel_loss(
            prediction,
            target,
        )

        # Spectral Convergence must be calculated on linear scale
        # We use torch.clamp to prevent mathematical explosions during expm1
        pred_linear = torch.expm1(torch.clamp(prediction, max=20.0))
        tgt_linear = torch.expm1(torch.clamp(target, max=20.0))

        sc = self.spectral_convergence_loss(
            pred_linear,
            tgt_linear,
        )

        total = self.lambda_l1 * l1 + self.lambda_sobel * sobel + self.lambda_sc * sc

        if return_components:
            return total, {
                "total": total.detach(),
                "l1": l1.detach(),
                "sobel": sobel.detach(),
                "spectral_convergence": sc.detach(),
            }

        return total
