import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class DoubleConv(nn.Module):
    """
    A block consisting of two convolutional layers, each followed by
    Batch Normalization and a LeakyReLU activation function.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downscaling block using MaxPool2d followed by a DoubleConv block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upscaling block that performs Transposed Convolution (or Bilinear Upsampling)
    followed by a DoubleConv block, properly handling skip connection concatenation.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        # If bilinear, use standard upsample and halve the input channels of Conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Upscale the input feature map
        x1 = self.up(x1)

        # Calculate spatial difference between upscaled tensor x1 and skip connection x2
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # Pad x1 if dimensions do not match perfectly (handles non-power-of-2 inputs)
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Fully Convolutional U-Net supporting variable input and output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_features: int = 64,
        bilinear: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = DownBlock(base_features, base_features * 2)
        self.down2 = DownBlock(base_features * 2, base_features * 4)
        self.down3 = DownBlock(base_features * 4, base_features * 8)

        # Determine channel scale factor depending on upscaling method
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(base_features * 8, (base_features * 16) // factor)

        # Expansive Path (Decoder)
        self.up1 = UpBlock(base_features * 16, (base_features * 8) // factor, bilinear)
        self.up2 = UpBlock(base_features * 8, (base_features * 4) // factor, bilinear)
        self.up3 = UpBlock(base_features * 4, (base_features * 2) // factor, bilinear)
        self.up4 = UpBlock(base_features * 2, base_features, bilinear)

        # Output Projection Layer (1x1 convolution)
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder passes and saving skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder passes utilizing saved skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


class RestorationModule(L.LightningModule):
    """
    LightningModule wrapper for training the U-Net on signal restoration tasks.
    Supports dynamic channel initialization and various loss functions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_features: int = 64,
        loss_fn=nn.L1Loss(),
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Instantiate the U-Net with parameters
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
        )

        # Use the provided loss function for training; default is L1 Loss
        self.criterion = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # batch yields: (noisy_input, clean_target)
        noisy, clean = batch
        reconstructed = self(noisy)

        loss = self.criterion(reconstructed, clean)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        reconstructed = self(noisy)

        loss = self.criterion(reconstructed, clean)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Log images only for the very first batch of validation
        if batch_idx == 0:
            # Take the first sample from the batch (index 0)
            # Add an extra dimension to keep the shape as [1, C, H, W]
            n_img = noisy[0:1].detach().cpu()
            r_img = reconstructed[0:1].detach().cpu()
            c_img = clean[0:1].detach().cpu()

            # Concatenate the three states along the batch dimension
            # Resulting shape: [3, C, H, W]
            comparison_tensor = torch.cat([n_img, r_img, c_img], dim=0)

            # Create a side-by-side grid
            # normalize=True automatically scales values to [0, 1] for correct rendering,
            # which is extremely useful for audio spectrograms with raw dB values
            grid = torchvision.utils.make_grid(
                comparison_tensor, nrow=3, normalize=True
            )

            # Add the image grid to TensorBoard
            # global_step acts as the X-axis in the dashboard slider
            self.logger.experiment.add_image(
                "Validation: 1.Noisy | 2.Reconstructed | 3.Clean",
                grid,
                global_step=self.global_step,
            )

        return loss

    def configure_optimizers(self):
        # Standard Adam optimizer for general convergence stability
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Learning rate scheduler to fine-tune final epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
