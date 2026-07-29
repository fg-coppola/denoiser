import argparse
import concurrent.futures
import multiprocessing
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio
from scipy import signal
from tqdm import tqdm


def generate_improved_synthetic_rir(
    target_sr: int = 16000, normalize: str = "none"
) -> torch.Tensor:
    """
    Generates a highly realistic synthetic room impulse response using pyroomacoustics.
    Features heterogeneous wall materials, hybrid ISM/Ray-Tracing simulation,
    air absorption, and distance-constrained source-microphone placement.
    """
    # 1. Randomize room dimensions with realistic proportions
    room_dims = [
        float(np.random.uniform(3.0, 10.0)),  # Width
        float(np.random.uniform(3.0, 12.0)),  # Length
        float(np.random.uniform(2.4, 4.0)),  # Height
    ]

    # 2. Heterogeneous materials for different surfaces
    absorption_floor = float(np.random.uniform(0.1, 0.6))
    absorption_ceiling = float(np.random.uniform(0.05, 0.3))
    absorption_walls = float(np.random.uniform(0.05, 0.4))

    materials = pra.make_materials(
        floor=absorption_floor,
        ceiling=absorption_ceiling,
        east=absorption_walls,
        west=absorption_walls,
        north=absorption_walls,
        south=absorption_walls,
    )

    # 3. Hybrid Simulator (ISM + Ray Tracing)
    room = pra.ShoeBox(
        room_dims,
        fs=target_sr,
        materials=materials,
        max_order=10,
        ray_tracing=True,
        air_absorption=True,
    )

    room.set_ray_tracing(receiver_radius=0.2, n_rays=5000, energy_thres=1e-5)

    # 4. Smart positioning with distance constraints
    min_distance = 1.0
    valid_position = False
    source_pos = [0.0, 0.0, 0.0]
    mic_pos = [0.0, 0.0, 0.0]

    for _ in range(50):
        source_pos = [
            float(np.random.uniform(0.5, room_dims[0] - 0.5)),
            float(np.random.uniform(0.5, room_dims[1] - 0.5)),
            float(np.random.uniform(1.2, 1.8)),
        ]
        mic_pos = [
            float(np.random.uniform(0.5, room_dims[0] - 0.5)),
            float(np.random.uniform(0.5, room_dims[1] - 0.5)),
            float(np.random.uniform(1.0, 1.8)),
        ]
        dist = np.linalg.norm(np.array(source_pos) - np.array(mic_pos))
        if dist >= min_distance:
            valid_position = True
            break

    if not valid_position:
        source_pos = [0.5, room_dims[1] / 2, 1.5]
        mic_pos = [room_dims[0] - 0.5, room_dims[1] / 2, 1.5]

    room.add_source(source_pos)
    room.add_microphone_array(pra.MicrophoneArray(np.array([mic_pos]).T, room.fs))
    room.compute_rir()
    raw_rir = room.rir[0][0]

    # 5. Randomized Bandpass filtering
    low_cut_hz = float(np.random.uniform(20.0, 100.0))
    high_cut_hz = float(np.random.uniform(0.8, 0.95) * (target_sr / 2.0))
    b, a = signal.butter(2, [low_cut_hz, high_cut_hz], btype="bandpass", fs=target_sr)
    colored_rir = signal.lfilter(b, a, raw_rir)

    # 6. SNR-based noise floor
    rir_power = np.mean(colored_rir**2)
    target_snr_db = float(np.random.uniform(40, 80))
    noise_power = rir_power / (10 ** (target_snr_db / 10))
    noise_floor = np.random.normal(0, np.sqrt(noise_power), size=len(colored_rir))
    final_rir = colored_rir + noise_floor

    # 7. Optional normalization
    if normalize == "peak":
        max_abs = np.max(np.abs(final_rir))
        if max_abs > 0:
            final_rir = final_rir / max_abs
    elif normalize == "rms":
        rms = np.sqrt(np.mean(final_rir**2))
        if rms > 0:
            final_rir = final_rir / rms
    elif normalize == "none":
        pass  # Leave the RIR at its natural energy
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")

    return torch.tensor(final_rir, dtype=torch.float32)


def process_single_rir(task_args: tuple) -> bool:
    """Worker function for multiprocessing."""
    idx, output_dir, sample_rate, file_format, normalize = task_args
    try:
        rir_tensor = generate_improved_synthetic_rir(
            target_sr=sample_rate, normalize=normalize
        )
        output_path = Path(output_dir)

        if file_format == "wav":
            rir_2d = rir_tensor.unsqueeze(0)
            file_path = output_path / f"synthetic_rir_{idx:06d}.wav"
            torchaudio.save(str(file_path), rir_2d, sample_rate)
        else:
            file_path = output_path / f"synthetic_rir_{idx:06d}.pt"
            torch.save(rir_tensor, file_path)
        return True
    except Exception as e:
        print(f"Error generating RIR {idx}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate improved synthetic RIRs."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--num_rirs", type=int, default=5000, help="Total RIRs to generate"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Target sample rate"
    )
    parser.add_argument(
        "--format", type=str, choices=["wav", "pt"], default="pt", help="Output format"
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["none", "peak", "rms"],
        default="none",
        help="Normalization mode. Use 'none' if you handle RMS matching downstream.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="CPU cores to use",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.num_rirs} RIRs using {args.workers} workers (seed={args.seed})..."
    )

    tasks = [
        (i, args.output_dir, args.sample_rate, args.format, args.normalize)
        for i in range(args.num_rirs)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(
            tqdm(
                executor.map(process_single_rir, tasks),
                total=args.num_rirs,
                desc="Generating RIRs",
            )
        )

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
