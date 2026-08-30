"""
Upsampling preprocessor for time-domain sample dicts.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from . import register_preprocessor
from .base_preprocessor import BasePreprocessor


@register_preprocessor("upsampler")
class UpsamplerPreprocessor(BasePreprocessor):
    """Resample sample['x'] from a source rate to a target rate."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.source_rate = int(self.cfg.get("source_rate", 1000))
        self.target_rate = int(self.cfg.get("target_rate", 2048))

        ratio = Fraction(self.target_rate, self.source_rate)
        self._up = ratio.numerator
        self._down = ratio.denominator

    def _resample_array(self, array: np.ndarray) -> np.ndarray:
        x = np.asarray(array, dtype=np.float32)
        if x.ndim < 2:
            raise ValueError(
                "upsampler expects sample['x'] to be at least 2D "
                f"(channels, time), got {x.shape}."
            )

        resampled = signal.resample_poly(x, up=self._up, down=self._down, axis=-1)
        expected_length = int(round(x.shape[-1] * self.target_rate / self.source_rate))
        current_length = resampled.shape[-1]
        if current_length > expected_length:
            resampled = resampled[..., :expected_length]
        elif current_length < expected_length:
            pad_width = [(0, 0)] * resampled.ndim
            pad_width[-1] = (0, expected_length - current_length)
            resampled = np.pad(resampled, pad_width, mode="edge")
        return resampled.astype(np.float32, copy=False)

    def _transform_one(self, sample):
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("upsampler requires sample['x'].")

        out = dict(sample)
        out["x"] = self._resample_array(sample["x"])
        if "sampling_rate" in out:
            out["sampling_rate"] = self.target_rate
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
