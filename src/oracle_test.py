import argparse

import torch
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from tqdm import tqdm

from datasets.voices import RIRDataModule


def _normalize_peak(audio: torch.Tensor) -> torch.Tensor:
    """Peak-normalize to [-1, 1] exactly as in the callback."""
    peak = torch.amax(torch.abs(audio), dim=-1, keepdim=True)
    return audio / (peak + 1e-8)


def run_offline_evaluation(
    test_dataloader,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Running Offline Baseline & Oracle Evaluation on {device}...")

    # Initialize metrics
    pesq_metric = PerceptualEvaluationSpeechQuality(sample_rate, "wb")
    stoi_metric = ShortTimeObjectiveIntelligibility(sample_rate, False)

    # Storage for Baseline (Noisy Audio)
    base_pesq_scores = []
    base_stoi_scores = []

    # Storage for Oracle (Clean Mag + Noisy Phase)
    oracle_pesq_scores = []
    oracle_stoi_scores = []

    # Hann window for STFT/iSTFT
    window = torch.hann_window(win_length, device=device)

    for batch in tqdm(test_dataloader, desc="Evaluating Baseline & Oracle"):
        noisy_inputs, clean_audio = batch

        # Unpack noisy inputs and clean audio
        noisy_mag, noisy_phase = noisy_inputs
        noisy_mag = noisy_mag.to(device)
        noisy_phase = noisy_phase.to(device)
        clean_audio = clean_audio.squeeze(1).to(device)

        expected_length = clean_audio.shape[-1]

        # Skip short audio exactly as in callback
        if expected_length < int(0.25 * sample_rate):
            continue

        with torch.no_grad():
            # 1. NOISY BASELINE RECONSTRUCTION (Noisy Mag + Noisy Phase)
            noisy_complex = torch.polar(noisy_mag, noisy_phase).squeeze(1)
            noisy_audio = torch.istft(
                noisy_complex,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                length=expected_length,
            )

            # 2. CLEAN STFT FOR ORACLE MAGNITUDE
            clean_stft = torch.stft(
                clean_audio,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
                center=True,
            )
            clean_mag = torch.abs(clean_stft)

            if clean_mag.dim() < noisy_phase.dim():
                clean_mag = clean_mag.unsqueeze(1)

            # 3. ORACLE COMBINATION (Clean Mag + Noisy Phase)
            oracle_complex = torch.polar(clean_mag, noisy_phase).squeeze(1)
            oracle_audio = torch.istft(
                oracle_complex,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                length=expected_length,
            )

        # 4. Normalize peak on CPU
        noisy_audio = _normalize_peak(noisy_audio.detach().cpu().float())
        oracle_audio = _normalize_peak(oracle_audio.detach().cpu().float())
        clean_audio_cpu = _normalize_peak(clean_audio.detach().cpu().float())

        # 5. Compute Baseline Metrics
        bp_score = pesq_metric(noisy_audio, clean_audio_cpu).item()
        bs_score = stoi_metric(noisy_audio, clean_audio_cpu).item()
        base_pesq_scores.append(bp_score)
        base_stoi_scores.append(bs_score)

        pesq_metric.reset()
        stoi_metric.reset()

        # 6. Compute Oracle Metrics
        op_score = pesq_metric(oracle_audio, clean_audio_cpu).item()
        os_score = stoi_metric(oracle_audio, clean_audio_cpu).item()
        oracle_pesq_scores.append(op_score)
        oracle_stoi_scores.append(os_score)

        pesq_metric.reset()
        stoi_metric.reset()

    # --- Statistical Aggregation ---
    b_pesq_t = torch.tensor(base_pesq_scores)
    b_stoi_t = torch.tensor(base_stoi_scores)
    o_pesq_t = torch.tensor(oracle_pesq_scores)
    o_stoi_t = torch.tensor(oracle_stoi_scores)

    # Baseline Stats
    b_mean_pesq, b_std_pesq, b_var_pesq = (
        b_pesq_t.mean().item(),
        b_pesq_t.std(unbiased=True).item(),
        b_pesq_t.var(unbiased=True).item(),
    )
    b_mean_stoi, b_std_stoi, b_var_stoi = (
        b_stoi_t.mean().item(),
        b_stoi_t.std(unbiased=True).item(),
        b_stoi_t.var(unbiased=True).item(),
    )

    # Oracle Stats
    o_mean_pesq, o_std_pesq, o_var_pesq = (
        o_pesq_t.mean().item(),
        o_pesq_t.std(unbiased=True).item(),
        o_pesq_t.var(unbiased=True).item(),
    )
    o_mean_stoi, o_std_stoi, o_var_stoi = (
        o_stoi_t.mean().item(),
        o_stoi_t.std(unbiased=True).item(),
        o_stoi_t.var(unbiased=True).item(),
    )

    print("\n" + "=" * 50)
    print("NOISY BASELINE RESULTS (Use these in Callback)")
    print("=" * 50)
    print(f"  pesq_baseline:        {b_mean_pesq:.16f}")
    print(f"  test/baseline_pesq_std: {b_std_pesq:.16f}")
    print(f"  test/baseline_pesq_var: {b_var_pesq:.16f}")
    print("-" * 50)
    print(f"  stoi_baseline:        {b_mean_stoi:.16f}")
    print(f"  test/baseline_stoi_std: {b_std_stoi:.16f}")
    print(f"  test/baseline_stoi_var: {b_var_stoi:.16f}")

    print("\n" + "=" * 50)
    print("ORACLE TEST RESULTS (Clean Mag + Noisy Phase)")
    print("=" * 50)
    print(f"  test/oracle_pesq:     {o_mean_pesq:.16f}")
    print(f"  test/oracle_pesq_std: {o_std_pesq:.16f}")
    print(f"  test/oracle_pesq_var: {o_var_pesq:.16f}")
    print("-" * 50)
    print(f"  test/oracle_stoi:     {o_mean_stoi:.16f}")
    print(f"  test/oracle_stoi_std: {o_std_stoi:.16f}")
    print(f"  test/oracle_stoi_var: {o_var_stoi:.16f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run offline evaluation for baseline & oracle performance."
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="data/audio/test",
        help="Path to the directory containing test audio data.",
    )

    args = parser.parse_args()

    rir_loader = RIRDataModule(
        train_data_dir="data/libriSpeech",
        val_data_dir="data/audio/validation",
        test_data_dir=args.test_data_dir,
        subset="train-clean-100",
        target_duration_seconds=3.0,
        synthetic_rir_paths=["data/rir/synthetic/train"],
        real_rir_paths=["data/rir/real/train"],
        synthetic_prob=0.5,
        download=True,
        batch_size=64,
        num_workers=12,
        persistent_workers=True,
        pin_memory=True,
    )
    rir_loader.setup(stage="test")
    test_dataloader = rir_loader.test_dataloader()
    run_offline_evaluation(test_dataloader)
