from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """
    Computes Spectral Convergence and Log-Magnitude loss for a single STFT resolution.
    """

    def __init__(self, fft_size, hop_size, win_length):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, pred_wav, target_wav):
        pred_stft = torch.stft(
            pred_wav,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        target_stft = torch.stft(
            target_wav,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

        pred_mag = torch.clamp(torch.abs(pred_stft), min=1e-7)
        target_mag = torch.clamp(torch.abs(target_stft), min=1e-7)

        # Spectral Convergence Loss
        sc_loss = torch.norm(target_mag - pred_mag, p="fro") / torch.norm(
            target_mag, p="fro"
        )

        # Log-Magnitude Loss
        log_pred_mag = torch.log(pred_mag)
        log_target_mag = torch.log(target_mag)
        mag_loss = F.l1_loss(log_pred_mag, log_target_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Computes STFT losses over multiple resolutions.
    """

    def __init__(
        self,
        fft_sizes=[256, 512, 1024, 2048],
        hop_sizes=[64, 128, 256, 512],
        win_lengths=[256, 512, 1024, 2048],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, hs, wl))

    def forward(self, pred_wav, target_wav):
        sc_loss = 0.0
        mag_loss = 0.0

        for f in self.stft_losses:
            sc_l, mag_l = f(pred_wav, target_wav)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class ComplexDereverberationLoss(nn.Module):
    """
    End-to-End Loss for Complex Spectrogram prediction.
    It takes the predicted Real and Imaginary parts, reconstructs the waveform,
    and returns a tuple containing the total loss and a dictionary of components
    for PyTorch Lightning automatic logging.
    """

    def __init__(
        self,
        original_n_fft: int = 1024,
        original_hop: int = 256,
        original_win: int = 1024,
        compression_factor: float = 0.3,
        lambda_time: float = 1.0,
        lambda_mr_stft: float = 1.0,
    ):
        super().__init__()
        self.original_n_fft = original_n_fft
        self.original_hop = original_hop
        self.original_win = original_win
        self.compression_factor = compression_factor

        self.lambda_time = lambda_time
        self.lambda_mr_stft = lambda_mr_stft

        self.register_buffer("original_window", torch.hann_window(original_win))

        # Assumes MultiResolutionSTFTLoss is already defined in your code
        self.mr_stft_loss = MultiResolutionSTFTLoss()

    def forward(
        self, preds: Tuple[torch.Tensor, torch.Tensor], target_wav: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        pred_real, pred_imag = preds
        original_length = target_wav.shape[-1]

        # Remove channel dimension if present for STFT operations
        if pred_real.dim() == 4:
            pred_real = pred_real.squeeze(1)
            pred_imag = pred_imag.squeeze(1)
        if target_wav.dim() == 3:
            target_wav = target_wav.squeeze(1)

        # 1. Combine predicted components into a PyTorch Complex Tensor
        pred_complex_comp = torch.complex(pred_real, pred_imag)

        # 2. Uncompress to linear magnitude while preserving the predicted phase
        pred_mag = torch.abs(pred_complex_comp)
        pred_phase = torch.angle(pred_complex_comp)

        linear_mag = torch.clamp(pred_mag, min=1e-8) ** (1.0 / self.compression_factor)
        linear_complex_spec = torch.polar(linear_mag, pred_phase)

        # 3. Differentiable Inverse-STFT
        pred_wav = torch.istft(
            linear_complex_spec,
            n_fft=self.original_n_fft,
            hop_length=self.original_hop,
            win_length=self.original_win,
            window=self.original_window,
            center=True,
            length=original_length,
        )

        # 4. Compute Sub-Losses
        loss_time = F.l1_loss(pred_wav, target_wav)
        sc_loss, mag_loss = self.mr_stft_loss(pred_wav, target_wav)
        loss_mr_stft = sc_loss + mag_loss

        # 5. Total Loss formulation
        total_loss = (self.lambda_time * loss_time) + (
            self.lambda_mr_stft * loss_mr_stft
        )

        # 6. Prepare dictionary for PyTorch Lightning logging
        loss_components = {
            "time_l1": loss_time,
            "mr_stft_total": loss_mr_stft,
            "mr_stft_sc": sc_loss,
            "mr_stft_mag": mag_loss,
        }

        return total_loss, loss_components
