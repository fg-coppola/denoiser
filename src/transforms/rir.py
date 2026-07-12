import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio
from scipy import signal


class ApplyRIR:
    @staticmethod
    def _generate_robust_synthetic_rir(target_sr=16000):
        room_dims = [
            np.random.uniform(3.0, 8.0),
            np.random.uniform(4.0, 10.0),
            np.random.uniform(2.5, 3.5),
        ]
        absorption = np.random.uniform(0.15, 0.45)

        room = pra.ShoeBox(
            room_dims,
            fs=target_sr,
            materials=pra.Material(absorption),
            max_order=15,
        )
        source_pos = [
            np.random.uniform(0.5, room_dims[0] - 0.5),
            np.random.uniform(0.5, room_dims[1] - 0.5),
            1.5,
        ]
        mic_pos = [
            np.random.uniform(0.5, room_dims[0] - 0.5),
            np.random.uniform(0.5, room_dims[1] - 0.5),
            1.5,
        ]

        room.add_source(source_pos)
        room.add_microphone_array(pra.MicrophoneArray(np.array([mic_pos]).T, room.fs))
        room.compute_rir()
        raw_rir = room.rir[0][0]

        low_cut_hz = 40.0
        high_cut_hz = 0.85 * (target_sr / 2.0)
        b, a = signal.butter(
            2,
            [low_cut_hz, high_cut_hz],
            btype="bandpass",
            fs=target_sr,
        )
        colored_rir = signal.lfilter(b, a, raw_rir)
        noise_floor = np.random.normal(0, 1e-4, size=len(colored_rir))
        final_rir = colored_rir + noise_floor

        max_abs = np.max(np.abs(final_rir))
        if max_abs > 0:
            final_rir = final_rir / max_abs

        return torch.tensor(final_rir, dtype=torch.float32)

    def __init__(self, rir_maps=None, target_sr=16000):
        self.target_sr = target_sr
        self.use_synthetic_rir = rir_maps is None

        if self.use_synthetic_rir:
            self.rir_maps = []
            return

        if isinstance(rir_maps, (torch.Tensor, np.ndarray, str, Path)):
            rir_maps = [rir_maps]
        else:
            rir_maps = list(rir_maps)

        if not rir_maps:
            raise ValueError("rir_maps must contain at least one room impulse response")

        self.rir_maps = [self._load_rir_map(rir_map) for rir_map in rir_maps]

    def _load_rir_map(self, rir_map):
        if isinstance(rir_map, torch.Tensor):
            return rir_map

        if isinstance(rir_map, np.ndarray):
            return torch.from_numpy(rir_map)

        if isinstance(rir_map, (str, Path)):
            path = Path(rir_map)
            raise NotImplementedError(
                f"RIR loading from file is intentionally left as a placeholder for {path}."
            )

        raise TypeError(
            "rir_maps must contain tensors, numpy arrays, or file placeholders to be wired in later"
        )

    def _normalize_rir_map(self, rir_map):
        if rir_map.ndim == 2:
            rir_map = rir_map.mean(dim=0)

        if rir_map.ndim != 1:
            raise ValueError(
                f"RIR must have shape [T] or [C, T], got {tuple(rir_map.shape)}"
            )

        return rir_map

    def _sample_rir_map(self):
        if self.use_synthetic_rir:
            return self._generate_robust_synthetic_rir(target_sr=self.target_sr)

        rir_map = self._normalize_rir_map(random.choice(self.rir_maps))
        return rir_map.to(torch.float32)

    def __call__(self, waveform):
        if waveform.ndim != 1:
            raise ValueError(
                f"Input waveform must have shape [T], got {tuple(waveform.shape)}"
            )

        rir_map = self._sample_rir_map()
        noisy_waveform = torchaudio.functional.fftconvolve(
            waveform,
            rir_map,
            mode="same",
        )
        return torch.clamp(noisy_waveform, -1.0, 1.0)
