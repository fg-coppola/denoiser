import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
import torchaudio.transforms as T

from models import ComplexIRMDenoiser, RestorationModule, UNet


class AudioRestorer:
    """
    A class dedicated to restoring noisy audio using a trained U-Net model.
    Now upgraded to handle Complex Spectrogram predictions (Magnitude + Phase),
    with built-in support for saving spectrogram comparison plots and running oracle A/B tests.
    """

    def __init__(
        self, checkpoint_path: str, device: str = None, base_features: int = 32
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading model onto {self.device} from: {checkpoint_path}")

        # Initialize the model architecture
        # in_channels and out_channels are 2 to handle Real and Imaginary parts
        unet = UNet(
            in_channels=2,
            out_channels=2,
            base_features=base_features,
            # kernel_size=(11, 5),
        )
        complex_denoiser = ComplexIRMDenoiser(base_model=unet)
        self.model = RestorationModule.load_from_checkpoint(
            checkpoint_path, model=complex_denoiser, strict=False
        )
        self.model.eval()
        self.model.to(self.device)

        # STFT parameters (MUST exactly match the training configuration)
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.sample_rate = 16000

        self.compression_factor = 0.3

        self.window = torch.hann_window(self.win_length).to(self.device)

    def _pad_for_unet(self, spec: torch.Tensor) -> tuple[torch.Tensor, int]:
        time_frames = spec.shape[-1]
        pad_amount = (16 - (time_frames % 16)) % 16

        if pad_amount > 0:
            spec = F.pad(spec, (0, pad_amount))

        return spec, pad_amount

    def _load_and_resample(self, file_path: str) -> torch.Tensor:
        """
        Helper method to load an audio file and format it for STFT.
        """
        waveform, sr = torchaudio.load(file_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if sample rates do not match
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return waveform.to(self.device)

    def _save_spectrogram_plot(
        self,
        noisy_spec: torch.Tensor,
        restored_spec: torch.Tensor,
        output_path: str,
        reference_spec: torch.Tensor | None = None,
    ):
        """Generates and saves a side-by-side comparison image of the spectrograms."""
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

    def evaluate_and_restore(self, noisy_path: str, clean_path: str, output_dir: str):
        """
        Runs full evaluation by predicting both real and imaginary components,
        and producing Test A, B, C, D reconstructions alongside a comparison plot.
        All outputs are saved to the provided output directory.
        """
        print("\n--- Running Oracle Evaluation & Audio Restoration ---")
        print(f"Noisy file: {noisy_path}")
        print(f"Clean file: {clean_path}")

        # Ensure the target directory exists
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load both aligned waveforms
        noisy_waveform = self._load_and_resample(noisy_path)
        clean_waveform = self._load_and_resample(clean_path)
        original_length = noisy_waveform.shape[-1]

        # 2. Extract Complex STFT for both
        print("Extracting Phase and Magnitude components...")
        noisy_stft = torch.stft(
            noisy_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        clean_stft = torch.stft(
            clean_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )

        noisy_mag = torch.abs(noisy_stft)
        noisy_phase = torch.angle(noisy_stft)

        clean_mag = torch.abs(clean_stft)
        clean_phase = torch.angle(clean_stft)

        # 3. Pre-process for the U-Net (Power-Law Compression)
        noisy_comp_spec = torch.clamp(noisy_mag, min=1e-8) ** self.compression_factor
        clean_comp_spec = torch.clamp(clean_mag, min=1e-8) ** self.compression_factor

        padded_comp_spec, pad_amount = self._pad_for_unet(noisy_comp_spec)
        padded_noisy_phase, _ = self._pad_for_unet(noisy_phase)

        # 4. Neural Network Inference (Complex Domain)
        print("Enhancing Spectrogram using Complex U-Net...")
        with torch.no_grad():
            # Model now returns predicted Real and Imaginary components
            pred_real, pred_imag = self.model(
                padded_comp_spec.unsqueeze(0), padded_noisy_phase.unsqueeze(0)
            )
            pred_real = pred_real.squeeze(0)
            pred_imag = pred_imag.squeeze(0)

        # Remove padding if it was added
        if pad_amount > 0:
            pred_real = pred_real[..., :-pad_amount]
            pred_imag = pred_imag[..., :-pad_amount]

        # Convert Cartesian (Real/Imag) back to Polar (Mag/Phase)
        pred_complex = torch.complex(pred_real, pred_imag)
        pred_comp_mag = torch.abs(pred_complex)
        pred_phase = torch.angle(pred_complex)

        # Invert the Power-Law Compression to get the linear magnitude for iSTFT
        pred_mag = pred_comp_mag ** (1.0 / self.compression_factor)

        # 5. Generate and Save Comparison Plot
        print("Generating spectrogram comparison plot...")
        plot_path = os.path.join(output_dir, "spectrogram_comparison.png")
        self._save_spectrogram_plot(
            noisy_spec=noisy_comp_spec.squeeze(0),
            restored_spec=pred_comp_mag.squeeze(0),
            output_path=plot_path,
            reference_spec=clean_comp_spec.squeeze(0),
        )

        # 6. RECONSTRUCTION EXPERIMENTS
        print("Reconstructing waveform variations (Inverse STFT & Griffin-Lim)...")

        # Helper function for generating audio from mag and phase
        def reconstruct_audio(magnitude, phase):
            complex_spec = torch.polar(magnitude, phase)
            wav = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                length=original_length,
            )

            # DC Blocker: High-pass filter at 40 Hz.
            wav = F_audio.highpass_biquad(
                wav, sample_rate=self.sample_rate, cutoff_freq=40.0
            )

            max_val = torch.max(torch.abs(wav))
            if max_val > 1.0:
                # Scale down to 0.99 to ensure no hard clipping on playback
                wav = (wav / max_val) * 0.99
            return wav.cpu()

        # Helper function for generating audio using Griffin-Lim
        griffin_lim_transform = T.GriffinLim(
            n_fft=self.n_fft,
            n_iter=128,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=1.0,  # Linear magnitude
            momentum=0.99,
            length=original_length,
        ).to(self.device)

        def reconstruct_audio_griffin_lim(magnitude):
            wav = griffin_lim_transform(magnitude)

            # DC Blocker: High-pass filter at 40 Hz.
            wav = F_audio.highpass_biquad(
                wav, sample_rate=self.sample_rate, cutoff_freq=40.0
            )

            max_val = torch.max(torch.abs(wav))
            if max_val > 1.0:
                # Scale down to 0.99 to ensure no hard clipping on playback
                wav = (wav / max_val) * 0.99
            return wav.cpu()

        # Test A: The Griffin-Lim Baseline (Clean Mag + Griffin-Lim Phase)
        audio_a = reconstruct_audio_griffin_lim(clean_mag)

        # Test B: Oracle Target (Pred Mag + Clean Phase)
        audio_b = reconstruct_audio(pred_mag, clean_phase)

        # Test C: THE ACTUAL SYSTEM OUTPUT (Pred Mag + PRED Phase)
        audio_c = reconstruct_audio(pred_mag, pred_phase)

        # Test D: The Griffin-Lim Reality (Pred Mag + Griffin-Lim Phase)
        audio_d = reconstruct_audio_griffin_lim(pred_mag)

        # 7. Save all variations in the output directory
        torchaudio.save(
            os.path.join(output_dir, "test_A_cleanMag_griffinLim.wav"),
            audio_a,
            self.sample_rate,
        )
        torchaudio.save(
            os.path.join(output_dir, "test_B_predMag_cleanPhase.wav"),
            audio_b,
            self.sample_rate,
        )
        torchaudio.save(
            os.path.join(output_dir, "test_C_predMag_PREDPhase.wav"),
            audio_c,
            self.sample_rate,
        )
        torchaudio.save(
            os.path.join(output_dir, "test_D_predMag_griffinLim.wav"),
            audio_d,
            self.sample_rate,
        )

        print(
            f"Success! All audio tests and spectrograms saved in: {os.path.abspath(output_dir)}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complex U-Net Audio Inference")
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
        default="output",
        help="Path of the folder where the cleaned audio will be saved",
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
    restorer.evaluate_and_restore(
        noisy_path=args.input,
        clean_path=args.reference if args.reference else args.input,
        output_dir=args.output,
    )
