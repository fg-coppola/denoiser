import torch
import torchaudio
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms

from transforms.rir import ApplyRIR

from .base import BaseDataModule


class STFTFeatureExtractor:
    """Extracts both Magnitude and Phase from the waveform."""

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256):
        self.sample_rate = sample_rate

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            power=None,  # Setting power to None returns the Complex STFT
            normalized=False,
        )

    def __call__(self, waveform):
        if waveform.ndim != 1:
            raise ValueError(
                f"Input waveform must have shape [T], got {tuple(waveform.shape)}"
            )

        complex_spec = self.spectrogram(waveform)

        # Extract magnitude and phase
        mag = torch.abs(complex_spec)
        phase = torch.angle(complex_spec)

        return mag.unsqueeze(0), phase.unsqueeze(0)


class PowerLawCompression:
    """Applies compression only to the magnitude, passing the phase untouched."""

    def __init__(self, compression_factor=0.3):
        self.compression_factor = compression_factor

    def __call__(self, features):
        mag, phase = features

        # Compress magnitude
        comp_mag = torch.clamp(mag, min=1e-8) ** self.compression_factor

        return comp_mag, phase


class IdentityWaveform:
    """Passes the raw waveform untouched (used for the clean target in MR-STFT)."""

    def __call__(self, waveform):
        # Add channel dimension to match expected PyTorch conventions [1, T]
        return waveform.unsqueeze(0)


class UnlabeledVoiceDataset(Dataset):
    def __init__(
        self, source_dataset, clean_transform, noisy_transform, target_num_samples
    ):
        self.source_dataset = source_dataset
        self.clean_transform = clean_transform
        self.noisy_transform = noisy_transform
        self.target_num_samples = target_num_samples

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, *_ = self.source_dataset[idx]

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        waveform = waveform.to(torch.float32)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=16000,
            )

        current_num_samples = waveform.shape[-1]
        if current_num_samples > self.target_num_samples:
            raise ValueError(
                f"Audio clip is longer than the configured target length: {current_num_samples} > {self.target_num_samples}"
            )

        if current_num_samples < self.target_num_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.target_num_samples - current_num_samples),
            )

        # clean_target is now just the padded waveform [1, T]
        clean_target = self.clean_transform(waveform)

        # noisy_inputs is now a tuple: (noisy_comp_mag, noisy_phase)
        noisy_inputs = self.noisy_transform(waveform)

        return noisy_inputs, clean_target


class RIRDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        target_duration_seconds,
        rir_maps=None,
        subset="train-clean-100",
        download=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.subset = subset
        self.download = download
        self.target_duration_seconds = target_duration_seconds
        self.target_num_samples = int(self.target_duration_seconds * 16000)

        self.clean_transform = transforms.Compose(
            [
                IdentityWaveform(),
            ]
        )
        self.noisy_transform = transforms.Compose(
            [
                ApplyRIR(rir_maps),
                STFTFeatureExtractor(),
                PowerLawCompression(),
            ]
        )

    def setup(self, stage=None):
        source_ds = torchaudio.datasets.LIBRISPEECH(
            root=self.data_dir,
            url=self.subset,
            download=self.download,
        )

        if len(source_ds) == 0:
            raise ValueError(
                f"No audio samples found in {self.data_dir} for subset {self.subset}"
            )

        filtered_indices = []

        for idx in range(len(source_ds)):
            waveform, sample_rate, *_ = source_ds[idx]
            num_samples = waveform.shape[-1]

            if sample_rate != 16000:
                num_samples = int(round(num_samples * 16000 / sample_rate))

            if num_samples <= self.target_num_samples:
                filtered_indices.append(idx)

        if not filtered_indices:
            raise ValueError(
                f"No audio samples with duration up to {self.target_duration_seconds} seconds were found in {self.data_dir} for subset {self.subset}"
            )

        source_ds = Subset(source_ds, filtered_indices)

        full_ds = UnlabeledVoiceDataset(
            source_dataset=source_ds,
            clean_transform=self.clean_transform,
            noisy_transform=self.noisy_transform,
            target_num_samples=self.target_num_samples,
        )

        self.train_ds, self.val_ds = random_split(full_ds, [0.8, 0.2])
