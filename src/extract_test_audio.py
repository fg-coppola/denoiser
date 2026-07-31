import os
import random

import torch
import torchaudio
from torchvision import transforms

from datasets.voices import UnlabeledVoiceDataset
from transforms.rir import ApplyRIR

# Configuration variables
DATA_DIR = "data/libriSpeech"
TARGET_DURATION = 5.0  # Seconds
SAMPLE_RATE = 16000
TARGET_SAMPLES = int(TARGET_DURATION * SAMPLE_RATE)


def extract_random_audio():
    """
    Extracts a random audio sample <= 5 seconds from Librispeech,
    applies padding to reach exactly 5 seconds, corrupts it with RIR,
    and saves both versions as .wav files.
    """

    # 1. Load the raw Librispeech dataset
    source_ds = torchaudio.datasets.LIBRISPEECH(
        root=DATA_DIR,
        url="train-clean-360",
        download=True,
    )

    # 2. Define transforms that ONLY apply RIR (bypassing spectrograms)
    clean_transform = transforms.Compose([])
    noisy_transform = transforms.Compose([ApplyRIR(rir_maps=None)])

    # 3. Find a valid short clip BEFORE using the dataset wrapper
    valid_idx = -1
    print("Searching for an audio clip under 5 seconds...")

    while True:
        random_idx = random.randint(0, len(source_ds) - 1)
        waveform, sample_rate, *_ = source_ds[random_idx]

        # Estimate the number of samples after potential resampling
        num_samples = waveform.shape[-1]
        if sample_rate != SAMPLE_RATE:
            num_samples = int(round(num_samples * SAMPLE_RATE / sample_rate))

        # If the clip is 5 seconds or shorter, we keep it and break the loop
        if num_samples <= TARGET_SAMPLES:
            valid_idx = random_idx
            break

    # 4. Initialize the dataset wrapper
    dataset = UnlabeledVoiceDataset(
        source_dataset=source_ds,
        clean_transform=clean_transform,
        noisy_transform=noisy_transform,
        target_num_samples=TARGET_SAMPLES,
    )

    # 5. Extract the samples (this is guaranteed to work now)
    noisy_waveform, clean_waveform = dataset[valid_idx]

    # 6. Fallback padding: ensure EXACTLY 5 seconds (80,000 samples)
    current_len = clean_waveform.shape[-1]
    if current_len < TARGET_SAMPLES:
        padding_needed = TARGET_SAMPLES - current_len

        clean_waveform = torch.nn.functional.pad(
            clean_waveform, (0, padding_needed), mode="constant", value=0.0
        )
        noisy_waveform = torch.nn.functional.pad(
            noisy_waveform, (0, padding_needed), mode="constant", value=0.0
        )

    # torchaudio.save expects a 2D tensor of shape [Channels, Time]
    noisy_waveform = noisy_waveform.unsqueeze(0)
    clean_waveform = clean_waveform.unsqueeze(0)

    # 7. Save the files to disk
    output_dir = "artifacts/extracted_audio"
    os.makedirs(output_dir, exist_ok=True)

    clean_path = os.path.join(output_dir, f"clean_{valid_idx}.wav")
    noisy_path = os.path.join(output_dir, f"noisy_{valid_idx}.wav")

    torchaudio.save(clean_path, clean_waveform, SAMPLE_RATE)
    torchaudio.save(noisy_path, noisy_waveform, SAMPLE_RATE)

    print(f"Successfully processed sample index: {valid_idx}")
    print(f"Clean audio saved to: {clean_path}")
    print(f"Noisy audio saved to: {noisy_path}")


if __name__ == "__main__":
    extract_random_audio()
