"""Polyphase waveform resampling with one contract for either direction."""

from math import gcd

import numpy as np
import torch
from scipy.signal import resample_poly

from . import register_preprocessor
from .base_preprocessor import BasePreprocessor


@register_preprocessor("resample")
class ResamplePreprocessor(BasePreprocessor):
    """Resample (channels, time) waveforms to float32 NumPy arrays.

    Output length is ceil(input_length * target_rate / source_rate), matching
    scipy.signal.resample_poly. Crop context windows before resampling so their
    sample-index metadata is consumed at the original sampling rate.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        for key in ("source_rate", "target_rate"):
            if key not in cfg:
                raise ValueError(f"resample requires {key}.")
            value = cfg[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"resample {key} must be a positive integer in Hz.")
            if value <= 0:
                raise ValueError(f"resample {key} must be positive, got {value}.")

        self.source_rate = cfg.source_rate
        self.target_rate = cfg.target_rate
        divisor = gcd(self.source_rate, self.target_rate)
        self._up = self.target_rate // divisor
        self._down = self.source_rate // divisor

    def _transform_one(self, sample):
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("resample requires sample['x'].")
        x = sample["x"]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[-1] == 0:
            raise ValueError(
                "resample expects sample['x'] with shape (channels, time) "
                f"and at least one time sample, got {x.shape}."
            )

        out = dict(sample)
        # Keep SciPy's native length; direction-specific trimming caused drift.
        out["x"] = resample_poly(x, self._up, self._down, axis=-1).astype(
            np.float32, copy=False
        )
        out["sampling_rate"] = self.target_rate
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
