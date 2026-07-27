from typing import Any

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim


class DoubleConv(nn.Module):
    """
    A block consisting of two convolutional layers, each followed by
    Group Normalization and an inplace SiLU activation function.
    Supports asymmetric kernels (e.g., (5, 3)). Includes optional
    dropout for regularization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        num_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Convert integer kernel to tuple for uniform handling
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # Dynamic padding to maintain spatial dimensions for odd-sized kernels
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        padding = (pad_h, pad_w)

        # Adjust group count if out_channels is not divisible by num_groups
        # If indivisible, falls back to InstanceNorm behavior (groups == channels)
        gn_groups = min(num_groups, out_channels)
        if out_channels % gn_groups != 0:
            gn_groups = out_channels

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,  # Bias is redundant before GroupNorm
            ),
            nn.GroupNorm(gn_groups, out_channels),
            nn.SiLU(inplace=True),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        layers.extend(
            [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.GroupNorm(gn_groups, out_channels),
                nn.SiLU(inplace=True),
            ]
        )

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downscaling block using MaxPool2d followed by a DoubleConv block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        num_groups: int = 8,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, num_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upscaling block supporting both Bilinear and Pixel Shuffle techniques.
    Includes a dedicated smoothing convolution after PixelShuffle to strictly
    avoid structural checkerboard artifacts before concatenating skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_mode: str = "bilinear",
        kernel_size: int | tuple[int, int] = 3,
        num_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        if upsample_mode == "pixel_shuffle":
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
            )
        elif upsample_mode == "bilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            raise ValueError(
                f"Unsupported upsample_mode: '{upsample_mode}'. Use 'bilinear' or 'pixel_shuffle'."
            )

        self.conv = DoubleConv(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(self, x_bottom: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        # Upscale and smooth the bottom feature map
        x_up = self.up(x_bottom)

        # Calculate spatial difference between upscaled tensor and skip connection
        diff_y = x_skip.size()[2] - x_up.size()[2]
        diff_x = x_skip.size()[3] - x_up.size()[3]

        # Pad x_up if dimensions do not match perfectly
        x_up = F.pad(
            x_up, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        # Concatenate cleaned upscaled features with skip connection
        x = torch.cat([x_skip, x_up], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    Fully Convolutional U-Net supporting variable input/output channels,
    asymmetric kernels for spectrograms, and modern upsampling methods
    (Pixel Shuffle or Bilinear) to prevent checkerboard artifacts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_features: int = 64,
        upsample_mode: str = "bilinear",  # Options: 'bilinear' or 'pixel_shuffle'
        kernel_size: int | tuple[int, int] = 3,
        num_groups: int = 8,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, base_features, kernel_size, num_groups)
        self.down1 = DownBlock(
            base_features, base_features * 2, kernel_size, num_groups
        )
        self.down2 = DownBlock(
            base_features * 2, base_features * 4, kernel_size, num_groups
        )
        self.down3 = DownBlock(
            base_features * 4, base_features * 8, kernel_size, num_groups
        )

        # Bottleneck
        self.down4 = DownBlock(
            base_features * 8, base_features * 16, kernel_size, num_groups
        )

        # Expansive Path (Decoder)
        self.up1 = UpBlock(
            base_features * 16,
            base_features * 8,
            upsample_mode,
            kernel_size,
            num_groups,
            decoder_dropout,
        )
        self.up2 = UpBlock(
            base_features * 8,
            base_features * 4,
            upsample_mode,
            kernel_size,
            num_groups,
            decoder_dropout,
        )
        self.up3 = UpBlock(
            base_features * 4,
            base_features * 2,
            upsample_mode,
            kernel_size,
            num_groups,
            decoder_dropout,
        )
        self.up4 = UpBlock(
            base_features * 2,
            base_features,
            upsample_mode,
            kernel_size,
            num_groups,
            0.0,
        )

        # Output Projection Layer (1x1 convolution mapping to desired output channels)
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder passes
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder passes with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


class ComplexSpectralMappingDenoiser(nn.Module):
    """
    Wrapper for a U-Net model to perform dereverberation via Complex Spectral Mapping.
    Predicts a complex residual to be added to the noisy input.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.unet = base_model
        self._init_residual_to_zero()

    def _init_residual_to_zero(self):
        """
        Initializes the output layer to zero so the initial prediction
        acts as an identity mapping (output = input + 0).
        """
        with torch.no_grad():
            self.unet.outc.weight.zero_()
            if self.unet.outc.bias is not None:
                self.unet.outc.bias.zero_()

    def forward(
        self, noisy_comp_mag: torch.Tensor, noisy_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Isolate the Nyquist bin (the last frequency bin)
        nyquist_mag = noisy_comp_mag[:, :, -1:, :]
        nyquist_phase = noisy_phase[:, :, -1:, :]

        # Drop the Nyquist bin to get a power-of-2 dimension (e.g., 512) for the U-Net
        truncated_mag = noisy_comp_mag[:, :, :-1, :]
        truncated_phase = noisy_phase[:, :, :-1, :]

        # Convert the truncated input to Cartesian coordinates
        noisy_real = truncated_mag * torch.cos(truncated_phase)
        noisy_imag = truncated_mag * torch.sin(truncated_phase)

        # Concatenate along the channel dimension -> Shape: [B, 2, F-1, T]
        x = torch.cat([noisy_real, noisy_imag], dim=1)

        # Predict the residual using the base U-Net
        residual = self.unet(x)
        res_real = residual[:, 0:1, :, :]
        res_imag = residual[
            :, 1:2, :, :
        ].clone()  # Clone to avoid in-place operations affecting autograd

        # Enforce Hermitian Symmetry for the DC bin (f=0)
        # The imaginary part of the DC bin must be strictly zero.
        res_imag[:, :, 0, :] = 0.0

        # Add the predicted residual to the noisy input
        pred_real = noisy_real + res_real
        pred_imag = noisy_imag + res_imag

        # We set the reconstructed Nyquist bin to zero
        nyquist_real_clean = torch.zeros_like(nyquist_mag)
        nyquist_imag_clean = torch.zeros_like(nyquist_phase)

        # Re-attach the zeroed Nyquist bin along the frequency dimension (dim=2)
        pred_real = torch.cat([pred_real, nyquist_real_clean], dim=2)
        pred_imag = torch.cat([pred_imag, nyquist_imag_clean], dim=2)

        return pred_real, pred_imag


class RestorationModule(L.LightningModule):
    """
    Generic LightningModule wrapper for signal/image restoration tasks.
    Agnostic to specific input formats, allowing reuse across different models.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module = nn.L1Loss(),
        lr: float = 1e-4,
    ):
        super().__init__()
        # Ignore complex objects to prevent PyTorch Lightning serialization issues
        self.save_hyperparameters(ignore=["loss_fn", "model"])

        self.model = model
        self.criterion = loss_fn

        # Convert 4D weights to channels_last for performance
        self._convert_model_memory_format()

    def _convert_model_memory_format(self):
        """Safely applies channels_last memory format to applicable 4D layers."""
        for param in self.model.parameters():
            if param.dim() == 4:
                param.data = param.data.to(memory_format=torch.channels_last)

    def _apply_memory_format(self, x: Any) -> Any:
        """Recursively applies channels_last to 4D tensors in inputs."""
        if isinstance(x, torch.Tensor):
            return x.to(memory_format=torch.channels_last) if x.dim() == 4 else x
        if isinstance(x, (tuple, list)):
            return type(x)(self._apply_memory_format(item) for item in x)
        if isinstance(x, dict):
            return {k: self._apply_memory_format(v) for k, v in x.items()}
        return x

    def forward(self, *args, **kwargs) -> Any:
        """
        Generic forward pass that accepts any number of arguments.
        Delegates completely to the underlying model.
        """
        args = self._apply_memory_format(args)
        kwargs = self._apply_memory_format(kwargs)
        return self.model(*args, **kwargs)

    def _shared_step(
        self, batch: tuple[Any, Any], batch_idx: int, prefix: str
    ) -> torch.Tensor:
        """
        Handles data unpacking, forward pass, and loss computation generically.
        """
        x, y = batch

        # 1. Dynamic forward pass handling single tensor, tuple, or dict inputs
        if isinstance(x, (tuple, list)):
            preds = self(*x)
        elif isinstance(x, dict):
            preds = self(**x)
        else:
            preds = self(x)

        # 2. The criterion manages its own expected prediction/target structures
        criterion_output = self.criterion(preds, y)

        # 3. Handle standard vs composite losses transparently
        if isinstance(criterion_output, tuple):
            loss, loss_components = criterion_output
            for component_name, component_value in loss_components.items():
                self.log(
                    f"{prefix}_{component_name}",
                    component_value,
                    on_step=(prefix == "train"),
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
        else:
            loss = criterion_output

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss, x, y, preds

    def training_step(self, batch: tuple[Any, Any], batch_idx: int) -> torch.Tensor:
        loss, _, _, _ = self._shared_step(batch, batch_idx, prefix="train")
        return loss

    def validation_step(self, batch: tuple[Any, Any], batch_idx: int) -> dict:
        loss, _, _, preds = self._shared_step(batch, batch_idx, prefix="val")

        # Return detached predictions only for the first batch to avoid OOM
        # The callback will retrieve these via outputs.get("preds")
        if batch_idx == 0:
            return {"loss": loss, "preds": preds.detach()}

        return {"loss": loss}

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        current_val_loss = self.trainer.callback_metrics.get("val_loss")

        if current_val_loss is not None:
            current_val_loss = current_val_loss.item()
            if getattr(self, "best_val_loss", float("inf")) > current_val_loss:
                self.best_val_loss = current_val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, fused=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
