import torch
from torch import nn


class SISDRLoss(nn.Module):
    """
    Calculates negative Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.
    SI-SDR is intrinsically phase-aware as it operates in the time domain.
    """

    def __init__(self, zero_mean: bool = True, eps: float = 1e-8):
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure dimensions match and compute over the time axis (last dimension)
        if self.zero_mean:
            preds = preds - torch.mean(preds, dim=-1, keepdim=True)
            target = target - torch.mean(target, dim=-1, keepdim=True)

        # Scale target to match the predicted signal's energy
        alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + self.eps) / (
            torch.sum(target**2, dim=-1, keepdim=True) + self.eps
        )
        target_scaled = alpha * target
        noise = preds - target_scaled

        # Compute SI-SDR
        val = (torch.sum(target_scaled**2, dim=-1) + self.eps) / (
            torch.sum(noise**2, dim=-1) + self.eps
        )
        si_sdr = 10 * torch.log10(val)

        # Return negative SI-SDR to minimize it
        return -torch.mean(si_sdr)


class STFTLoss(nn.Module):
    """
    Computes purely magnitude-based Spectral Convergence and Log-Magnitude losses.
    Robust against zero-padded regions and natural silences via energy masking.
    """

    def __init__(self, fft_size: int, hop_size: int, win_length: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(
        self, pred_wav: torch.Tensor, target_wav: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

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

        # 1. Spectral Convergence Loss
        sc_loss = torch.linalg.matrix_norm(
            target_mag - pred_mag, ord="fro", dim=(1, 2)
        ) / (torch.linalg.matrix_norm(target_mag, ord="fro", dim=(1, 2)) + 1e-8)
        sc_loss = torch.mean(sc_loss)

        # 2. Log-Magnitude Loss with Energy Masking
        active_mask = (target_mag > 1e-5).float()
        log_mag_diff = torch.abs(torch.log(pred_mag) - torch.log(target_mag))
        mag_loss = torch.sum(log_mag_diff * active_mask) / (
            torch.sum(active_mask) + 1e-8
        )

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Standard MR-STFT Loss (Magnitude only).
    Phase alignment is delegated to the SI-SDR loss.
    """

    def __init__(
        self,
        fft_sizes: list[int] = [256, 512, 1024, 2048],
        hop_sizes: list[int] = [64, 128, 256, 512],
        win_lengths: list[int] = [256, 512, 1024, 2048],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList(
            [
                STFTLoss(fs, hs, wl)
                for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(
        self, pred_wav: torch.Tensor, target_wav: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sc_loss = 0.0
        mag_loss = 0.0

        for f in self.stft_losses:
            sc_l, mag_l = f(pred_wav, target_wav)
            sc_loss += sc_l
            mag_loss += mag_l

        n = len(self.stft_losses)
        return sc_loss / n, mag_loss / n


class CompositeDereverberationLoss(nn.Module):
    """
    Complex Spectral Mapping Loss.
    Delegates phase and time-domain structure to SI-SDR, and spectral sharpness to MR-STFT.
    No direct L1 penalty is applied to the Cartesian components to avoid the compensation effect.
    """

    def __init__(
        self,
        original_n_fft: int = 1024,
        original_hop: int = 256,
        original_win: int = 1024,
        compression_factor: float = 0.3,
        lambda_sdr: float = 1.0,
        lambda_mr_stft: float = 0.5,
    ):
        super().__init__()
        self.original_n_fft = original_n_fft
        self.original_hop = original_hop
        self.original_win = original_win
        self.compression_factor = compression_factor

        self.lambda_sdr = lambda_sdr
        self.lambda_mr_stft = lambda_mr_stft

        self.register_buffer("original_window", torch.hann_window(original_win))
        self.mr_stft_loss = MultiResolutionSTFTLoss()
        self.sdr_loss = SISDRLoss()

    def forward(
        self, preds: tuple[torch.Tensor, torch.Tensor], target_wav: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        pred_real, pred_imag = preds
        original_length = target_wav.shape[-1]

        # Clean dimensions safely
        if pred_real.dim() == 4:
            pred_real = pred_real.squeeze(1)
            pred_imag = pred_imag.squeeze(1)
        if target_wav.dim() == 3:
            target_wav = target_wav.squeeze(1)

        # 1. iSTFT Reconstruction for Loss Computation
        power_spec = pred_real**2 + pred_imag**2 + 1e-8
        compressed_mag = torch.sqrt(power_spec)

        # Decompress magnitude to linear scale
        linear_mag = compressed_mag ** (1.0 / self.compression_factor)

        # Safe phase reconstruction (normalize by compressed magnitude)
        denom = compressed_mag + 1e-8
        real_norm = pred_real / denom
        imag_norm = pred_imag / denom

        linear_real = linear_mag * real_norm
        linear_imag = linear_mag * imag_norm
        linear_complex_spec = torch.complex(linear_real, linear_imag)

        # Reconstruct the waveform in a differentiable way
        pred_wav = torch.istft(
            linear_complex_spec,
            n_fft=self.original_n_fft,
            hop_length=self.original_hop,
            win_length=self.original_win,
            window=self.original_window,
            center=True,
            length=original_length,
        )

        # 2. Time-Domain Phase-Aware Loss (SI-SDR)
        loss_sdr = self.sdr_loss(pred_wav, target_wav)

        # 3. Multi-Resolution Spectral Loss (Magnitude only)
        sc_loss, mag_loss = self.mr_stft_loss(pred_wav, target_wav)
        loss_mr_stft = sc_loss + mag_loss

        # 4. Final Combination
        total_loss = self.lambda_sdr * loss_sdr + self.lambda_mr_stft * loss_mr_stft

        loss_components = {
            "total": total_loss,
            "si_sdr": loss_sdr,
            "mr_stft": loss_mr_stft,
            "stft_sc": sc_loss,
            "stft_mag": mag_loss,
        }

        return total_loss, loss_components
