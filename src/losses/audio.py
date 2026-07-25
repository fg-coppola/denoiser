import torch
import torch.nn.functional as F
from torch import nn


class ComplexSTFTLoss(nn.Module):
    """
    Computes Spectral Convergence, Log-Magnitude, and Complex Cartesian
    losses for a single STFT resolution.
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

        # 1. Spectral Convergence Loss
        sc_loss = torch.norm(target_mag - pred_mag, p="fro") / torch.norm(
            target_mag, p="fro"
        )

        # 2. Log-Magnitude Loss
        log_pred_mag = torch.log(pred_mag)
        log_target_mag = torch.log(target_mag)
        mag_loss = F.l1_loss(log_pred_mag, log_target_mag)

        # 3. Complex Cartesian Loss (L1 on Real and Imaginary parts)
        complex_loss = F.l1_loss(pred_stft.real, target_stft.real) + F.l1_loss(
            pred_stft.imag, target_stft.imag
        )

        return sc_loss, mag_loss, complex_loss


class MultiResolutionComplexSTFTLoss(nn.Module):
    """
    Computes Complex STFT losses over multiple resolutions.
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
            self.stft_losses.append(ComplexSTFTLoss(fs, hs, wl))

    def forward(self, pred_wav, target_wav):
        sc_loss = 0.0
        mag_loss = 0.0
        complex_loss = 0.0

        for f in self.stft_losses:
            sc_l, mag_l, comp_l = f(pred_wav, target_wav)
            sc_loss += sc_l
            mag_loss += mag_l
            complex_loss += comp_l

        # Average losses across all resolutions
        num_resolutions = len(self.stft_losses)
        sc_loss /= num_resolutions
        mag_loss /= num_resolutions
        complex_loss /= num_resolutions

        return sc_loss, mag_loss, complex_loss


class ComplexDereverberationLoss(nn.Module):
    """
    End-to-End Loss for Complex Spectrogram prediction utilizing cMR-STFT.
    Handles mathematically safe power uncompression to prevent exploding gradients.
    """

    def __init__(
        self,
        original_n_fft: int = 1024,
        original_hop: int = 256,
        original_win: int = 1024,
        compression_factor: float = 0.3,
        lambda_mr_stft: float = 1.0,
        lambda_complex: float = 1.0,
    ):
        super().__init__()
        self.original_n_fft = original_n_fft
        self.original_hop = original_hop
        self.original_win = original_win
        self.compression_factor = compression_factor

        self.lambda_mr_stft = lambda_mr_stft
        self.lambda_complex = lambda_complex

        self.register_buffer("original_window", torch.hann_window(original_win))

        self.cmr_stft_loss = MultiResolutionComplexSTFTLoss()

    def forward(
        self, preds: tuple[torch.Tensor, torch.Tensor], target_wav: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        pred_real, pred_imag = preds
        original_length = target_wav.shape[-1]

        # Clean dimensions for STFT
        if pred_real.dim() == 4:
            pred_real = pred_real.squeeze(1)
            pred_imag = pred_imag.squeeze(1)
        if target_wav.dim() == 3:
            target_wav = target_wav.squeeze(1)

        # 1. Safe magnitude calculation with epsilon
        power_spec = pred_real**2 + pred_imag**2 + 1e-8
        compressed_mag = torch.sqrt(power_spec)

        # 2. Uncompress to linear magnitude
        linear_mag = compressed_mag ** (1.0 / self.compression_factor)

        # 3. Reconstruct complex spectrogram safely
        real_norm = pred_real / compressed_mag
        imag_norm = pred_imag / compressed_mag

        linear_real = linear_mag * real_norm
        linear_imag = linear_mag * imag_norm
        linear_complex_spec = torch.complex(linear_real, linear_imag)

        # 4. Inverse-STFT (Needed only because we are doing Multi-Resolution)
        pred_wav = torch.istft(
            linear_complex_spec,
            n_fft=self.original_n_fft,
            hop_length=self.original_hop,
            win_length=self.original_win,
            window=self.original_window,
            center=True,
            length=original_length,
        )

        # 5. Compute only Complex MR-STFT Sub-Losses
        sc_loss, mag_loss, complex_loss = self.cmr_stft_loss(pred_wav, target_wav)
        loss_mr_stft = sc_loss + mag_loss

        # 6. Total Loss formulation
        total_loss = (self.lambda_mr_stft * loss_mr_stft) + (
            self.lambda_complex * complex_loss
        )

        # 7. Logging dictionary
        loss_components = {
            "cmr_stft_total": total_loss,
            "stft_sc": sc_loss,
            "stft_mag": mag_loss,
            "stft_complex": complex_loss,
        }

        return total_loss, loss_components
