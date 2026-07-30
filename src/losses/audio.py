import torch
from torch import nn


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


class MagnitudeOnlyDereverberationLoss(nn.Module):
    def __init__(
        self,
        original_n_fft: int = 1024,
        original_hop: int = 256,
        original_win: int = 1024,
        l1_weight: float = 10.0,  # Diamo un peso importante alla loss diretta
    ):
        super().__init__()
        self.original_n_fft = original_n_fft
        self.original_hop = original_hop
        self.original_win = original_win
        self.l1_weight = l1_weight

        self.register_buffer("original_window", torch.hann_window(original_win))
        self.mr_stft_loss = MultiResolutionSTFTLoss()

    def forward(self, preds, target_wav):
        pred_real, pred_imag = preds
        original_length = target_wav.shape[-1]

        if pred_real.dim() == 4:
            pred_real = pred_real.squeeze(1)
            pred_imag = pred_imag.squeeze(1)
        if target_wav.dim() == 3:
            target_wav = target_wav.squeeze(1)

        # Creiamo il tensore complesso
        pred_complex = torch.complex(pred_real, pred_imag)

        # 1. Magnitudo predetta
        pred_mag = torch.abs(pred_complex)

        # 2. Otteniamo la magnitudo target estraendo la STFT
        with torch.no_grad():
            target_stft = torch.stft(
                target_wav,
                n_fft=self.original_n_fft,
                hop_length=self.original_hop,
                win_length=self.original_win,
                window=self.original_window,
                return_complex=True,
                center=True,
            )
            target_mag = torch.abs(target_stft)

        # 3. Calcolo L1
        l1_loss = torch.nn.functional.l1_loss(pred_mag, target_mag)
        # ---------------------------------------------------------

        # Ricostruzione per la MR-STFT
        pred_wav = torch.istft(
            pred_complex,
            n_fft=self.original_n_fft,
            hop_length=self.original_hop,
            win_length=self.original_win,
            window=self.original_window,
            center=True,
            length=original_length,
        )

        sc_loss, mag_loss = self.mr_stft_loss(pred_wav, target_wav)

        # Sommiamo tutto (la L1 pesata aiuta la convergenza iniziale)
        total_loss = sc_loss + mag_loss + (self.l1_weight * l1_loss)

        loss_components = {
            "total": total_loss,
            "stft_sc": sc_loss,
            "stft_mag": mag_loss,
            "direct_l1": l1_loss,  # Aggiunta ai log
        }

        return total_loss, loss_components
