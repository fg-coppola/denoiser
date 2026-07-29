import argparse
import random
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from datasets.voices import RandomGain, TemporalCropOrPad
from transforms.rir import ApplyRIR


def process_and_save_split(
    split_name, ls_subset, source_dir, output_dir, apply_rir, crop_or_pad, gain_aug
):
    print(f"\n--- Generating static dataset for '{split_name}' ---")
    print(f"Source LibriSpeech subset: {ls_subset}")

    # Load LibriSpeech subset
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=source_dir, url=ls_subset, download=True
    )

    # Create output directories (e.g., ./data/static_eval/validation/clean)
    clean_dir = Path(output_dir) / split_name / "clean"
    noisy_dir = Path(output_dir) / split_name / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    # Set a fixed global seed for reproducibility based on the split name
    seed = 42 if split_name == "validation" else 999
    random.seed(seed)
    torch.manual_seed(seed)

    for i in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
        # LibriSpeech returns: waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id
        waveform, sample_rate, _, speaker_id, chapter_id, utterance_id = dataset[i]

        # Create a unique filename
        filename = f"{speaker_id}_{chapter_id}_{utterance_id}.wav"
        clean_path = clean_dir / filename
        noisy_path = noisy_dir / filename

        # Skip if files already exist (useful if the script is interrupted)
        if clean_path.exists() and noisy_path.exists():
            continue

        # 1. Base Normalization (Mono, 16kHz, Float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        waveform = waveform.to(torch.float32)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        # 2. Apply Random Gain (Applied to BOTH validation and test)
        waveform = gain_aug(waveform)

        # 3. Apply Temporal Crop/Pad (Applied ONLY to validation for batching efficiency)
        if split_name == "validation":
            waveform = crop_or_pad(waveform)
        # If split_name == "test", the waveform keeps its original full length for accurate metrics

        # The clean target is the anechoic audio with gain (and crop, if applicable)
        clean_waveform = waveform.clone()

        # 4. RIR Convolution
        noisy_waveform = apply_rir(waveform)

        # 5. Save to disk (unsqueeze adds the channel dimension back: [1, Time])
        torchaudio.save(clean_path, clean_waveform.unsqueeze(0), 16000)
        torchaudio.save(noisy_path, noisy_waveform.unsqueeze(0), 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate offline static datasets for validation or test"
    )

    # Split vs Subset configuration
    parser.add_argument(
        "--split_name",
        type=str,
        choices=["validation", "test"],
        required=True,
        help="Logical name of the split (validation or test)",
    )
    parser.add_argument(
        "--ls_subset",
        type=str,
        required=True,
        help="LibriSpeech subset to download/use (e.g., dev-clean, test-clean, dev-other)",
    )

    # Paths and Params
    parser.add_argument(
        "--source_dir",
        type=str,
        default="./data",
        help="Directory for downloading LibriSpeech",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/static_eval",
        help="Directory to save final .wav files",
    )
    parser.add_argument(
        "--target_duration",
        type=float,
        default=4.0,
        help="Target duration in seconds (for validation crop only)",
    )
    parser.add_argument(
        "--synth_rir_dir",
        type=str,
        required=True,
        help="Path to synthetic RIRs directory",
    )
    parser.add_argument(
        "--real_rir_dir",
        type=str,
        default=None,
        help="Path to real RIRs directory (optional)",
    )

    args = parser.parse_args()

    # Initialize Transformations
    target_samples = int(args.target_duration * 16000)
    crop_or_pad = TemporalCropOrPad(target_samples)
    gain_aug = RandomGain(db_range=(-6.0, 0.0), p=0.8)

    apply_rir = ApplyRIR(
        synthetic_rir_paths=[args.synth_rir_dir],
        real_rir_paths=[args.real_rir_dir] if args.real_rir_dir else None,
        synthetic_prob=0.5,
        target_sr=16000,
    )

    # Execute generation for the requested split
    process_and_save_split(
        split_name=args.split_name,
        ls_subset=args.ls_subset,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        apply_rir=apply_rir,
        crop_or_pad=crop_or_pad,
        gain_aug=gain_aug,
    )

    print(f"\nSuccessfully generated {args.split_name} dataset!")
    print(f"Output saved in: {Path(args.output_dir) / args.split_name}")
