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
            nn.SiLU(inplace=False),
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
                nn.SiLU(inplace=False),
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
    Assumes inputs are in linear scale.
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
        self, noisy_mag: torch.Tensor, noisy_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Isolate the Nyquist bin (the last frequency bin)
        nyquist_mag = noisy_mag[:, :, -1:, :]
        nyquist_phase = noisy_phase[:, :, -1:, :]

        # Drop the Nyquist bin to get a power-of-2 dimension (e.g., 512) for the U-Net
        truncated_mag = noisy_mag[:, :, :-1, :]
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


class ComplexIRMDenoiser(nn.Module):
    """
    Wrapper for a U-Net model to perform dereverberation via Complex Ideal Ratio Mask (cIRM).
    Predicts a complex mask that is element-wise multiplied with the noisy input.
    Designed as a direct drop-in replacement for ComplexSpectralMappingDenoiser.
    Assumes inputs are in linear scale.
    """

    def __init__(self, base_model: nn.Module, mask_bound: float = 10.0):
        super().__init__()
        self.unet = base_model
        # Define the bounding limit for the tanh activation
        self.mask_bound = mask_bound
        self._init_mask_to_identity()

    def _init_mask_to_identity(self):
        """
        Initializes the output layer so the initial prediction acts as an identity mask
        (mask_real = 1.0, mask_imag = 0.0). This ensures the initial prediction
        returns the unaltered input (output = input * 1).
        """
        with torch.no_grad():
            self.unet.outc.weight.zero_()
            if self.unet.outc.bias is not None:
                self.unet.outc.bias.zero_()

                # Inverse of the scaled tanh to ensure the final output is 1.0
                # mask_bound * tanh(x / mask_bound) = 1.0 => x = mask_bound * atanh(1.0 / mask_bound)
                target_val = torch.tensor(1.0 / self.mask_bound)
                initial_bias = self.mask_bound * torch.atanh(target_val)
                self.unet.outc.bias[0] = initial_bias.item()

    def forward(
        self, noisy_mag: torch.Tensor, noisy_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Isolate the Nyquist bin (the last frequency bin)
        nyquist_mag = noisy_mag[:, :, -1:, :]
        nyquist_phase = noisy_phase[:, :, -1:, :]

        # Drop the Nyquist bin to get a power-of-2 dimension (e.g., 512) for the U-Net
        truncated_mag = noisy_mag[:, :, :-1, :]
        truncated_phase = noisy_phase[:, :, :-1, :]

        # Convert the truncated input to Cartesian coordinates
        noisy_real = truncated_mag * torch.cos(truncated_phase)
        noisy_imag = truncated_mag * torch.sin(truncated_phase)

        # Concatenate along the channel dimension -> Shape: [B, 2, F-1, T]
        x = torch.cat([noisy_real, noisy_imag], dim=1)

        # Predict the complex mask using the base U-Net
        mask = self.unet(x)

        # Split the mask into real and imaginary parts
        mask_real = mask[:, 0:1, :, :]
        # Clone to avoid in-place operations affecting autograd later
        mask_imag = mask[:, 1:2, :, :].clone()

        # Enforce Hermitian Symmetry for the DC bin (f=0)
        # The imaginary part of the mask at DC must be strictly zero.
        mask_imag[:, :, 0, :] = 0.0

        # Apply a scaled tanh activation to bound the mask values.
        # This prevents exploding gradients and reduces spectral artifacts.
        mask_real = self.mask_bound * torch.tanh(mask_real / self.mask_bound)
        mask_imag = self.mask_bound * torch.tanh(mask_imag / self.mask_bound)

        # Apply the complex Ideal Ratio Mask (cIRM) via complex multiplication
        # Formula: S = Y * M -> (Y_r + jY_i) * (M_r + jM_i)
        # S_real = (Y_r * M_r) - (Y_i * M_i)
        # S_imag = (Y_r * M_i) + (Y_i * M_r)
        pred_real = (noisy_real * mask_real) - (noisy_imag * mask_imag)
        pred_imag = (noisy_real * mask_imag) + (noisy_imag * mask_real)

        # We set the reconstructed Nyquist bin to polar coordinates
        nyquist_real = nyquist_mag * torch.cos(nyquist_phase)
        nyquist_imag = nyquist_mag * torch.sin(nyquist_phase)

        # Re-attach the Nyquist bin along the frequency dimension (dim=2)
        pred_real = torch.cat([pred_real, nyquist_real], dim=2)
        pred_imag = torch.cat([pred_imag, nyquist_imag], dim=2)

        return pred_real, pred_imag


class DirectSpectralDenoiser(nn.Module):
    """
    Wrapper for a U-Net model to perform direct audio dereverberation.
    Predicts the clean real and imaginary parts directly from the noisy input,
    without using any intermediate ratio masks.
    Designed as a direct drop-in replacement for ComplexIRMDenoiser.
    Assumes inputs are in linear scale.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.unet = base_model

    def forward(
        self, noisy_mag: torch.Tensor, noisy_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Isolate the Nyquist bin (the last frequency bin)
        nyquist_mag = noisy_mag[:, :, -1:, :]
        nyquist_phase = noisy_phase[:, :, -1:, :]

        # Drop the Nyquist bin to get a power-of-2 dimension (e.g., 512) for the U-Net
        truncated_mag = noisy_mag[:, :, :-1, :]
        truncated_phase = noisy_phase[:, :, :-1, :]

        # Convert the truncated input to Cartesian coordinates
        noisy_real = truncated_mag * torch.cos(truncated_phase)
        noisy_imag = truncated_mag * torch.sin(truncated_phase)

        # Concatenate along the channel dimension -> Shape: [B, 2, F-1, T]
        x = torch.cat([noisy_real, noisy_imag], dim=1)

        # The U-Net predicts the clean real and imaginary parts directly
        pred = self.unet(x)

        # Split the prediction into real and imaginary parts
        pred_real_trunc = pred[:, 0:1, :, :]
        # Clone to avoid in-place operations affecting autograd later
        pred_imag_trunc = pred[:, 1:2, :, :].clone()

        # Enforce Hermitian Symmetry for the DC bin (f=0)
        # The imaginary part of the mask at DC must be strictly zero.
        pred_imag_trunc[:, :, 0, :] = 0.0

        # We set the reconstructed Nyquist bin to polar coordinates
        nyquist_real = nyquist_mag * torch.cos(nyquist_phase)
        nyquist_imag = nyquist_mag * torch.sin(nyquist_phase)

        # Re-attach the Nyquist bin along the frequency dimension (dim=2)
        pred_real = torch.cat([pred_real_trunc, nyquist_real], dim=2)
        pred_imag = torch.cat([pred_imag_trunc, nyquist_imag], dim=2)

        return pred_real, pred_imag


class MagnitudeIRMNoisyPhaseDenoiser(nn.Module):
    """
    Wrapper for a U-Net model to perform dereverberation via Magnitude Ideal Ratio Mask (IRM).
    Predicts a magnitude mask applied to the input magnitude, then reconstructs the
    complex spectrogram using the original noisy phase.
    Designed as a direct drop-in replacement for ComplexIRMDenoiser.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.unet = base_model
        self._init_mask_to_identity()

    def _init_mask_to_identity(self):
        """
        Initializes the output layer so the initial prediction acts as an identity mask
        (mask = 1.0). Since we use a Sigmoid activation, we set a positive bias
        so the initial output is close to 1.0, returning the unaltered input.
        """
        with torch.no_grad():
            self.unet.outc.weight.zero_()
            if self.unet.outc.bias is not None:
                # Sigmoid(4.0) is approx 0.98. This ensures a safe, near-identity start
                # without saturating the gradient completely at the very first step.
                self.unet.outc.bias.fill_(4.0)

    def forward(
        self, noisy_mag: torch.Tensor, noisy_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 1. Isolate the Nyquist bin (the last frequency bin)
        nyquist_mag = noisy_mag[:, :, -1:, :]
        nyquist_phase = noisy_phase[:, :, -1:, :]

        # Drop the Nyquist bin to get a power-of-2 dimension (e.g., 512) for the U-Net
        truncated_mag = noisy_mag[:, :, :-1, :]
        truncated_phase = noisy_phase[:, :, :-1, :]

        # 2. Predict the magnitude mask using the base U-Net
        # The U-Net receives ONLY the magnitude now. Shape: [B, 1, F-1, T]
        mask_logits = self.unet(truncated_mag)

        # Apply Sigmoid to bound the mask strictly between 0 and 1 (standard IRM)
        mask = torch.sigmoid(mask_logits)

        # 3. Apply the mask to the noisy magnitude
        pred_mag = truncated_mag * mask

        # 4. Reconstruct Complex Spectrogram using NOISY PHASE
        # We convert the masked magnitude back to Cartesian coordinates
        # using the original, untouched phase of the noisy input.
        pred_real = pred_mag * torch.cos(truncated_phase)
        pred_imag = pred_mag * torch.sin(truncated_phase)

        # 5. Handle Nyquist (unaltered)
        nyquist_real = nyquist_mag * torch.cos(nyquist_phase)
        nyquist_imag = nyquist_mag * torch.sin(nyquist_phase)

        # Re-attach the Nyquist bin along the frequency dimension (dim=2)
        pred_real = torch.cat([pred_real, nyquist_real], dim=2)
        pred_imag = torch.cat([pred_imag, nyquist_imag], dim=2)

        return pred_real, pred_imag



class LowLightdenoiser(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.unet = base_model
        with torch.no_grad():
            self.unet.outc.weight.zero_()
            if self.unet.outc.bias is not None:
                self.unet.outc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gamma correction: deterministically brightens the input.
        # The network no longer needs to learn how to illuminate, only how to correct.
        x_corr = torch.pow(x.clamp(min=0.0), 0.4)
        return x_corr + self.unet(x_corr)
    
    
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

        # 1. Dynamic forward pass handling
        if isinstance(x, (tuple, list)):
            preds = self(*x)
        elif isinstance(x, dict):
            preds = self(**x)
        else:
            preds = self(x)

        # 2. Criterion output
        criterion_output = self.criterion(preds, y)

        # 3. Handle standard vs composite losses transparently with grouped TensorBoard logging
        if isinstance(criterion_output, tuple):
            loss, loss_components = criterion_output

            # Log every component dynamically grouped under {prefix}_loss/...
            for component_name, component_value in loss_components.items():
                self.log(
                    f"{prefix}_loss/{component_name}",
                    component_value,
                    on_step=(prefix == "train"),
                    on_epoch=True,
                    prog_bar=(component_name == "total"),
                    logger=True,
                    sync_dist=(prefix != "train"),  # Just in case for multi GPU
                )
        else:
            loss = criterion_output

            self.log(
                f"{prefix}_loss/total",
                loss,
                on_step=(prefix == "train"),
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=(prefix != "train"),
            )

        return loss, x, y, preds

    def training_step(self, batch: tuple[Any, Any], batch_idx: int) -> torch.Tensor:
        loss, _, _, _ = self._shared_step(batch, batch_idx, prefix="train")
        return loss

    def validation_step(self, batch: tuple[Any, Any], batch_idx: int) -> dict:
        loss, _, _, preds = self._shared_step(batch, batch_idx, prefix="val")

        # Return detached predictions only for the first batch to avoid OOM
        if batch_idx == 0:
            return {"loss": loss, "preds": preds}

        return {"loss": loss}

    def test_step(self, batch: tuple[Any, Any], batch_idx: int) -> dict:
        """
        Evaluates the model on the test set.
        Returns predictions for ALL batches so callbacks can compute metrics.
        """
        loss, _, _, preds = self._shared_step(batch, batch_idx, prefix="test")
        return {"loss": loss, "preds": preds}

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        current_val_loss = self.trainer.callback_metrics.get("val_loss/total")

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
            patience=3,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss/total",
                "interval": "epoch",
            },
        }
