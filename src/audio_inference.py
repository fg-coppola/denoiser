import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# Adjust this import based on your actual project structure
from models import RestorationModule


class AudioRestorer:
    """
    A class dedicated to restoring noisy audio using a trained U-Net model.
    It utilizes the 'Noisy Phase' trick to prevent metallic/robotic artifacts
    typically introduced by Griffin-Lim estimation.
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading model onto {self.device} from: {checkpoint_path}")

        # Load the U-Net model from the checkpoint
        self.model = RestorationModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

        # STFT parameters (MUST exactly match the training configuration)
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.sample_rate = 16000

        # Pre-compute the Hann window and send it to the correct device
        self.window = torch.hann_window(self.win_length).to(self.device)

    def _pad_for_unet(self, log_spec: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Pads the time dimension to ensure it's a multiple of 16.
        This is required because the U-Net uses 4 levels of Max Pooling (2^4 = 16).
        """
        time_frames = log_spec.shape[-1]
        pad_amount = (16 - (time_frames % 16)) % 16

        if pad_amount > 0:
            # Pad the right side of the last dimension
            log_spec = F.pad(log_spec, (0, pad_amount))

        return log_spec, pad_amount

    def restore_audio(self, input_path: str, output_path: str):
        """
        Processes a single audio file through the full pipeline:
        Load -> Resample -> STFT -> U-Net -> ISTFT -> Save.
        """
        print(f"\nProcessing file: {input_path}")

        # 1. Load and prepare the audio waveform
        waveform, sr = torchaudio.load(input_path)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to target sample rate (16 kHz)
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        original_length = waveform.shape[-1]
        waveform = waveform.to(self.device)

        # 2. Extract Complex STFT (Magnitude & Phase)
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
        log_spec = torch.log1p(magnitude)
        log_spec, pad_amount = self._pad_for_unet(log_spec)

        # Add a batch dimension: [1, Channels, Freq, Time]
        input_tensor = log_spec.unsqueeze(0)

        # 4. Neural Network Inference
        print("Enhancing Magnitude using U-Net...")
        with torch.no_grad():
            output_log_spec = self.model(input_tensor)

        # 5. Post-process U-Net output
        # Remove batch dimension
        output_log_spec = output_log_spec.squeeze(0)

        # Remove the padding added earlier
        if pad_amount > 0:
            output_log_spec = output_log_spec[..., :-pad_amount]

        # Reverse the logarithmic scaling
        clean_magnitude = torch.expm1(output_log_spec)

        # 6. Reconstruct the Audio
        print("Reconstructing waveform (Inverse STFT)...")
        reconstructed_complex = torch.polar(clean_magnitude, noisy_phase)

        # The 'length' parameter ensures the output matches the original waveform exactly
        reconstructed_waveform = torch.istft(
            reconstructed_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=original_length,
        )

        # 7. Save the final cleaned audio
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

    args = parser.parse_args()

    # Initialize the restorer class and process the file
    restorer = AudioRestorer(checkpoint_path=args.ckpt)
    restorer.restore_audio(input_path=args.input, output_path=args.output)
