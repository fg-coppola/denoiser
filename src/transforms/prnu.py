import random

import torch


class ApplyPRNU:
    def __init__(self, prnu_maps, intensity=0.01, map_augment=False):
        if prnu_maps is None:
            raise ValueError("prnu_maps must not be None")

        if isinstance(prnu_maps, torch.Tensor):
            prnu_maps = [prnu_maps]
        else:
            prnu_maps = list(prnu_maps)

        if not prnu_maps:
            raise ValueError("prnu_maps must contain at least one PRNU map")

        self.prnu_maps = prnu_maps
        self.intensity = intensity
        self.map_augment = map_augment

    def _normalize_prnu_map(self, prnu_map):
        if prnu_map.ndim == 2:
            prnu_map = prnu_map.unsqueeze(0)

        if prnu_map.ndim != 3:
            raise ValueError(
                f"PRNU map must have shape [H, W] or [1, H, W] or [3, H, W], got {tuple(prnu_map.shape)}"
            )

        if prnu_map.shape[0] not in (1, 3):
            raise ValueError(
                f"PRNU map must have 1 or 3 channels, got {prnu_map.shape[0]} channels"
            )

        return prnu_map

    def _map_augment(self, prnu_map):  # TODO: scegliere gli augment giusti
        # Random rotation by multiples of 90 degrees
        if random.random() > 0.5:
            prnu_map = torch.rot90(prnu_map, k=random.randint(1, 3), dims=[1, 2])

        # Random horizontal flip
        if random.random() > 0.5:
            prnu_map = torch.flip(prnu_map, dims=[2])  # Flip width dimension

        # Random vertical flip
        if random.random() > 0.5:
            prnu_map = torch.flip(prnu_map, dims=[1])  # Flip height dimension

        return prnu_map

    def _sample_prnu_map(self):
        prnu_map = self._normalize_prnu_map(random.choice(self.prnu_maps))
        if self.map_augment:
            prnu_map = self._map_augment(prnu_map)
        return prnu_map

    def __call__(self, img):
        if img.ndim != 3:
            raise ValueError(
                f"Input image must have shape [C, H, W], got {tuple(img.shape)}"
            )

        prnu_map = self._sample_prnu_map()

        if prnu_map.shape[1:] != img.shape[1:]:
            raise ValueError(
                f"PRNU map shape {tuple(prnu_map.shape)} does not match image shape {tuple(img.shape)}"
            )

        if prnu_map.shape[0] == 1:
            prnu_map = prnu_map.expand(img.shape[0], -1, -1)

        noise = img * prnu_map * self.intensity
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0, 1)
