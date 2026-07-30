import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from transforms.rir import ApplyRIR

from .base import BaseDataModule


class SpeedPerturbation:
    """
    Resamples the waveform to change playback speed (and pitch).
    Must be applied BEFORE the RIR convolution so the room acoustics
    respond to the correctly scaled source signal.
    """

    def __init__(self, factors=(0.9, 1.0, 1.1), p=0.5):
        self.factors = factors
        self.p = p

    def __call__(self, waveform):
        if not self.factors or random.random() > self.p:
            return waveform
        factor = random.choice(self.factors)

        # speed returns (waveform, new_sample_rate); we only need the waveform
        waveform, _ = torchaudio.functional.speed(
            waveform, orig_freq=16000, factor=factor
        )
        return waveform


class RandomGain:
    """
    Applies a random gain in dB.
    Shared between clean and noisy branches so the target remains aligned.
    """

    def __init__(self, db_range=(-6.0, 0.0), p=0.8):
        self.db_range = db_range
        self.p = p

    def __call__(self, waveform):
        if random.random() > self.p:
            return waveform
        gain_db = random.uniform(*self.db_range)
        return waveform * (10 ** (gain_db / 20.0))


class TemporalCropOrPad:
    """
    Ensures every waveform has exactly the target length.
    - If too long: takes a random crop.
    - If too short: pads with digital silence (zeros) to avoid looping artifacts.
    """

    def __init__(self, target_samples):
        self.target_samples = target_samples

    def __call__(self, waveform):
        current_len = waveform.shape[-1]

        # Pad with zeros if too short
        if current_len < self.target_samples:
            pad_amount = self.target_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Random slice if too long
        elif current_len > self.target_samples:
            start = random.randint(0, current_len - self.target_samples)
            waveform = waveform[start : start + self.target_samples]

        return waveform


class STFTFeatureExtractor:
    """Extracts both Magnitude and Phase from the waveform."""

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft

        self.window = torch.hann_window(self.win_length)

    def __call__(self, waveform):
        if waveform.ndim != 1:
            raise ValueError(
                f"Input waveform must have shape [T], got {tuple(waveform.shape)}"
            )

        window = self.window.to(waveform.device)

        complex_spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )

        mag = torch.abs(complex_spec)
        phase = torch.angle(complex_spec)

        return mag.unsqueeze(0), phase.unsqueeze(0)


class PowerLawCompression:
    """Applies compression only to the magnitude, passing the phase untouched."""

    def __init__(self, compression_factor=0.3):
        self.compression_factor = compression_factor

    def __call__(self, features):
        mag, phase = features
        comp_mag = torch.clamp(mag, min=1e-8) ** self.compression_factor
        return comp_mag, phase


class IdentityWaveform:
    """Passes the raw waveform untouched (used for the clean target in MR-STFT)."""

    def __call__(self, waveform):
        return waveform.unsqueeze(0)


class UnlabeledVoiceDataset(Dataset):
    """
    Dataset that loads clean utterances and applies on-the-fly augmentations.
    Used ONLY for training to ensure infinite variations.
    """

    def __init__(
        self,
        source_dataset,
        apply_rir,
        clean_transform,
        noisy_feature_transform,
        target_num_samples,
        speed_factors=(0.9, 1.0, 1.1),
        gain_db_range=(-6.0, 0.0),
    ):
        self.source_dataset = source_dataset
        self.apply_rir = apply_rir
        self.clean_transform = clean_transform
        self.noisy_feature_transform = noisy_feature_transform

        self.speed_aug = SpeedPerturbation(factors=speed_factors, p=0.5)
        self.gain_aug = RandomGain(db_range=gain_db_range, p=0.8)
        self.crop_or_pad = TemporalCropOrPad(target_samples=target_num_samples)

    def __len__(self):
        return len(self.source_dataset)

    def _load_waveform(self, idx):
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

        return waveform

    def __getitem__(self, idx):
        waveform = self._load_waveform(idx)
        waveform = self.speed_aug(waveform)

        # Fixed cropping ensures we can batch without collate_fn during training
        waveform = self.crop_or_pad(waveform)
        waveform = self.gain_aug(waveform)
        noisy_waveform = self.apply_rir(waveform)

        clean_target = self.clean_transform(waveform)
        noisy_inputs = self.noisy_feature_transform(noisy_waveform)

        return noisy_inputs, clean_target


class StaticPairedDataset(Dataset):
    """
    Loads pre-processed, paired clean and noisy audio files.
    Used for validation and testing to ensure deterministic, reproducible evaluation.
    """

    def __init__(self, data_dir, clean_transform, noisy_feature_transform):
        self.data_dir = Path(data_dir)
        self.clean_dir = self.data_dir / "clean"
        self.noisy_dir = self.data_dir / "noisy"

        self.clean_transform = clean_transform
        self.noisy_feature_transform = noisy_feature_transform

        # Ensure consistent ordering
        self.clean_files = sorted(list(self.clean_dir.glob("*.wav")))
        self.noisy_files = sorted(list(self.noisy_dir.glob("*.wav")))

        if not self.clean_files:
            raise RuntimeError(f"No .wav files found in {self.clean_dir}")
        if len(self.clean_files) != len(self.noisy_files):
            raise RuntimeError("Mismatch in clean and noisy file counts.")

    def __len__(self):
        return len(self.clean_files)

    def _load_and_format(self, path):
        waveform, sample_rate = torchaudio.load(path)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)

        waveform = waveform.to(torch.float32)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        return waveform

    def __getitem__(self, idx):
        clean_wav = self._load_and_format(self.clean_files[idx])
        noisy_wav = self._load_and_format(self.noisy_files[idx])

        clean_target = self.clean_transform(clean_wav)
        noisy_inputs = self.noisy_feature_transform(noisy_wav)

        return noisy_inputs, clean_target


class RIRDataModule(BaseDataModule):
    def __init__(
        self,
        train_data_dir,
        val_data_dir,
        test_data_dir,
        subset,
        target_duration_seconds,
        synthetic_rir_paths=None,
        real_rir_paths=None,
        synthetic_prob=0.5,
        download=False,
        speed_factors=(0.9, 1.0, 1.1),
        gain_db_range=(-6.0, 0.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.subset = subset

        self.download = download
        self.target_duration_seconds = target_duration_seconds
        self.target_num_samples = int(self.target_duration_seconds * 16000)

        self.speed_factors = speed_factors
        self.gain_db_range = gain_db_range

        self.synthetic_rir_paths = synthetic_rir_paths
        self.real_rir_paths = real_rir_paths
        self.synthetic_prob = synthetic_prob

        # ApplyRIR remains active for the dynamic train loop
        self.apply_rir = ApplyRIR(
            synthetic_rir_paths=self.synthetic_rir_paths,
            real_rir_paths=self.real_rir_paths,
            synthetic_prob=self.synthetic_prob,
            target_sr=16000,
        )

        self.clean_transform = IdentityWaveform()
        self.noisy_feature_transform = transforms.Compose(
            [
                STFTFeatureExtractor(),
                PowerLawCompression(),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # 1. Dynamic Training Set (train-clean-100)
            train_source_ds = torchaudio.datasets.LIBRISPEECH(
                root=self.train_data_dir,
                url=self.subset,
                download=self.download,
            )

            if len(train_source_ds) == 0:
                raise ValueError(
                    f"No audio samples found in {self.train_data_dir} for {self.subset}"
                )

            self.train_ds = UnlabeledVoiceDataset(
                source_dataset=train_source_ds,
                apply_rir=self.apply_rir,
                clean_transform=self.clean_transform,
                noisy_feature_transform=self.noisy_feature_transform,
                target_num_samples=self.target_num_samples,
                speed_factors=self.speed_factors,
                gain_db_range=self.gain_db_range,
            )

            # 2. Static Validation Set (already cropped to target duration during offline generation)
            self.val_ds = StaticPairedDataset(
                data_dir=self.val_data_dir,
                clean_transform=self.clean_transform,
                noisy_feature_transform=self.noisy_feature_transform,
            )

        if stage == "test" or stage is None:
            # 3. Static Test Set (variable lengths, uncropped)
            self.test_ds = StaticPairedDataset(
                data_dir=self.test_data_dir,
                clean_transform=self.clean_transform,
                noisy_feature_transform=self.noisy_feature_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        # Default collate is fine because val files were pre-cropped strictly to target length
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        # Forced batch_size=1 handles variable-length test files automatically
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
