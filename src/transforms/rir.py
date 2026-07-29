import random
from pathlib import Path

import torch
import torchaudio


class ApplyRIR:
    def __init__(
        self,
        synthetic_rir_paths=None,
        real_rir_paths=None,
        synthetic_prob=0.5,
        target_sr=16000,
        max_rir_len_sec=1.5,
    ):
        self.target_sr = target_sr
        self.synthetic_prob = synthetic_prob
        self.max_rir_len_sec = max_rir_len_sec
        self.max_samples = int(target_sr * max_rir_len_sec) if max_rir_len_sec else None

        if synthetic_rir_paths is None and real_rir_paths is None:
            raise ValueError(
                "At least one of synthetic_rir_paths or real_rir_paths must be provided."
            )

        self.synthetic_rir_pool = self._build_pool(synthetic_rir_paths)
        real_rir_paths_list = self._build_pool(real_rir_paths)

        if not self.synthetic_rir_pool and not real_rir_paths_list:
            raise ValueError("No valid RIR files found in the provided paths.")

        # Caching in RAM of real RIRs to avoid on-the-fly resampling and disk I/O
        self.real_rir_cache = []
        if real_rir_paths_list:
            print(f"Caching {len(real_rir_paths_list)} real RIRs into RAM...")
            for path in real_rir_paths_list:
                rir_tensor = self._load_rir(path)
                self.real_rir_cache.append(rir_tensor)

    def _build_pool(self, source):
        if source is None:
            return []

        if isinstance(source, (str, Path)):
            source = [source]

        paths = []
        for s in source:
            p = Path(s)
            if p.is_dir():
                for ext in ("*.pt", "*.wav", "*.flac", "*.ogg"):
                    paths.extend(p.glob(ext))
            elif p.is_file():
                paths.append(p)

        return sorted(paths)

    def _load_rir(self, path):
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".pt":
            # mmap=True acts as a zero-copy read. The tensor is mapped directly
            # from disk, bypassing standard CPU RAM allocation bottlenecks.
            rir = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        else:
            rir, sr = torchaudio.load(str(path))
            if sr != self.target_sr:
                rir = torchaudio.functional.resample(rir, sr, self.target_sr)

        if rir.ndim == 2:
            rir = rir.mean(dim=0)

        if rir.ndim != 1:
            raise ValueError(
                f"RIR must have shape [T] or [C, T], got {tuple(rir.shape)}"
            )

        rir = rir.to(torch.float32)

        if self.max_samples is not None and rir.shape[0] > self.max_samples:
            rir = rir[: self.max_samples]

        return rir

    def _sample_rir_map(self):
        use_synthetic = random.random() < self.synthetic_prob

        # Synthetic RIRs are always loaded from disk to avoid caching large datasets in RAM.
        # While the real ones are cached in RAM to avoid repeated resampling and disk I/O.
        if use_synthetic and self.synthetic_rir_pool:
            rir_path = random.choice(self.synthetic_rir_pool)
            return self._load_rir(rir_path)
        elif self.real_rir_cache:
            return random.choice(self.real_rir_cache)
        else:
            rir_path = random.choice(self.synthetic_rir_pool)
            return self._load_rir(rir_path)

    def __call__(self, waveform):
        if waveform.ndim != 1:
            raise ValueError(
                f"Input waveform must have shape [T], got {tuple(waveform.shape)}"
            )

        rir_map = self._sample_rir_map()

        delay = torch.argmax(torch.abs(rir_map)).item()

        noisy_waveform = torchaudio.functional.fftconvolve(
            waveform,
            rir_map,
            mode="full",
        )

        end = delay + waveform.shape[-1]
        noisy_waveform = noisy_waveform[delay:end]

        clean_rms = torch.sqrt(torch.mean(waveform**2))
        noisy_rms = torch.sqrt(torch.mean(noisy_waveform**2)) + 1e-8
        noisy_waveform = noisy_waveform * (clean_rms / noisy_rms)

        return noisy_waveform
