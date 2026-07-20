import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from models import RestorationModule, UNet


class AudioRestorer:
    """
    A class dedicated to restoring noisy audio using a trained U-Net model,
    with built-in support for saving spectrogram comparison plots.
    """

    def __init__(
        self, checkpoint_path: str, device: str = None, base_features: int = 32
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading model onto {self.device} from: {checkpoint_path}")

        # Initialize the model architecture
        unet = UNet(
            in_channels=1,
            out_channels=1,
            base_features=base_features,
            kernel_size=(11, 5),
        )

        self.model = RestorationModule.load_from_checkpoint(
            checkpoint_path, denoiser=unet, strict=False
        )
        self.model.eval()
        self.model.to(self.device)

        # STFT parameters (MUST exactly match the training configuration)
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.sample_rate = 16000

        self.window = torch.hann_window(self.win_length).to(self.device)

    def _pad_for_unet(self, log_spec: torch.Tensor) -> Tuple[torch.Tensor, int]:
        time_frames = log_spec.shape[-1]
        pad_amount = (16 - (time_frames % 16)) % 16

        if pad_amount > 0:
            log_spec = F.pad(log_spec, (0, pad_amount))

        return log_spec, pad_amount

    def _load_and_resample(self, file_path: str) -> torch.Tensor:
        """Helper method to load an audio file and format it for STFT."""
        waveform, sr = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return waveform.to(self.device)

    def _save_spectrogram_plot(
        self,
        noisy_spec: torch.Tensor,
        restored_spec: torch.Tensor,
        output_path: str,
        reference_spec: Optional[torch.Tensor] = None,
    ):
        """Generates and saves a side-by-side comparison image of the log spectrograms."""
        num_plots = 3 if reference_spec is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

        # Convert tensors to 2D numpy arrays for Matplotlib
        noisy_np = noisy_spec.cpu().numpy()
        restored_np = restored_spec.cpu().numpy()

        # Plot Noisy
        axes[0].imshow(noisy_np, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title("Noisy Input")
        axes[0].set_xlabel("Frames")
        axes[0].set_ylabel("Frequency Bins")

        # Plot Restored
        axes[1].imshow(restored_np, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_title("Restored Output")
        axes[1].set_xlabel("Frames")

        # Plot Reference if available
        if reference_spec is not None:
            ref_np = reference_spec.cpu().numpy()
            axes[2].imshow(ref_np, aspect="auto", origin="lower", cmap="magma")
            axes[2].set_title("Clean Reference (Target)")
            axes[2].set_xlabel("Frames")

        plt.tight_layout()

        # Save image with same name as audio output but .png extension
        img_path = Path(output_path).with_suffix(".png")
        plt.savefig(str(img_path), dpi=150)
        plt.close(fig)
        print(f"Spectrogram comparison saved to: {img_path.absolute()}")

    def restore_audio(
        self, input_path: str, output_path: str, reference_path: str = None
    ):
        print(f"\nProcessing file: {input_path}")

        # 1. Load noisy waveform
        waveform = self._load_and_resample(input_path)
        original_length = waveform.shape[-1]

        # 2. Extract Complex STFT
        print("Extracting Phase and Magnitude...")
        stft_complex = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

        magnitude = torch.abs(stft_complex)
        noisy_phase = torch.angle(stft_complex)

        # 3. Pre-process for the U-Net
        noisy_log_spec = torch.log1p(magnitude)
        padded_log_spec, pad_amount = self._pad_for_unet(noisy_log_spec)
        input_tensor = padded_log_spec.unsqueeze(0)

        # 4. Neural Network Inference
        print("Enhancing Magnitude using U-Net...")
        with torch.no_grad():
            output_log_spec = self.model(input_tensor)

        # 5. Post-process U-Net output
        output_log_spec = output_log_spec.squeeze(0)

        if pad_amount > 0:
            output_log_spec = output_log_spec[..., :-pad_amount]

        # 6. Generate Comparison Plot
        # We process the reference file if provided to get its log spectrogram
        ref_log_spec = None
        if reference_path:
            print(f"Loading reference clean audio: {reference_path}")
            ref_waveform = self._load_and_resample(reference_path)

            ref_stft = torch.stft(
                ref_waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                return_complex=True,
            )
            ref_log_spec = torch.log1p(torch.abs(ref_stft))

        print("Generating spectrogram plot...")
        self._save_spectrogram_plot(
            noisy_spec=noisy_log_spec.squeeze(0),
            restored_spec=output_log_spec.squeeze(0),
            output_path=output_path,
            reference_spec=ref_log_spec.squeeze(0)
            if ref_log_spec is not None
            else None,
        )

        # 7. Reconstruct the Audio
        clean_magnitude = torch.expm1(output_log_spec)
        print("Reconstructing waveform (Inverse STFT)...")
        reconstructed_complex = torch.polar(clean_magnitude, noisy_phase)

        reconstructed_waveform = torch.istft(
            reconstructed_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=original_length,
        )

        # 8. Save the final cleaned audio
        reconstructed_waveform = reconstructed_waveform.cpu()

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(
            str(out_file),
            reconstructed_waveform,
            sample_rate=self.sample_rate,
        )
        print(f"Success! Saved output audio to: {out_file.absolute()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Audio Inference (Noisy Phase)")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the noisy input .wav file"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the trained PyTorch Lightning .ckpt model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cleaned_output.wav",
        help="Path where the cleaned audio will be saved",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="[Optional] Path to the clean ground-truth audio to generate a 3-way spectrogram comparison plot",
    )
    parser.add_argument(
        "--base_features",
        type=int,
        default=32,
        help="Number of base features for the U-Net model (must match training configuration)",
    )

    args = parser.parse_args()

    restorer = AudioRestorer(
        checkpoint_path=args.ckpt, base_features=args.base_features
    )
    restorer.restore_audio(
        input_path=args.input, output_path=args.output, reference_path=args.reference
    )
