"""
Downsample-then-pad preprocessor for time-domain sample dicts.

Unlike `upsampler` (which resamples straight to the target sample count and
so time-stretches a short window, shifting all frequencies down), this op
resamples to the model's true native rate first -- preserving correct
spectral scaling -- and only pads the remaining samples needed to reach a
fixed target length.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from . import register_preprocessor
from .base_preprocessor import BasePreprocessor


@register_preprocessor("downsample_pad")
class DownsamplePadPreprocessor(BasePreprocessor):
    """Resample sample['x'] to native_rate, then pad/crop to target_len samples."""

    VALID_PAD_MODES = {"reflect", "zero"}
    VALID_PAD_SIDES = {"end", "start", "both"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.source_rate = int(self.cfg.get("source_rate"))
        self.native_rate = int(self.cfg.get("native_rate", 512))
        self.target_len = int(self.cfg.get("target_len", 2560))
        self.pad_mode = str(self.cfg.get("pad_mode", "reflect"))
        self.pad_side = str(self.cfg.get("pad_side", "end"))

        if self.pad_mode not in self.VALID_PAD_MODES:
            raise ValueError(
                f"Invalid pad_mode '{self.pad_mode}'. "
                f"Must be one of {sorted(self.VALID_PAD_MODES)}"
            )
        if self.pad_side not in self.VALID_PAD_SIDES:
            raise ValueError(
                f"Invalid pad_side '{self.pad_side}'. "
                f"Must be one of {sorted(self.VALID_PAD_SIDES)}"
            )

        # Same resampling method as `upsampler`: exact up/down ratio via
        # scipy.signal.resample_poly, so the two ops are directly comparable.
        ratio = Fraction(self.native_rate, self.source_rate)
        self._up = ratio.numerator
        self._down = ratio.denominator

    def _resample_to_native(self, x: np.ndarray) -> np.ndarray:
        resampled = signal.resample_poly(x, up=self._up, down=self._down, axis=-1)
        expected_length = int(round(x.shape[-1] * self.native_rate / self.source_rate))
        current_length = resampled.shape[-1]
        if current_length > expected_length:
            resampled = resampled[..., :expected_length]
        elif current_length < expected_length:
            pad_width = [(0, 0)] * resampled.ndim
            pad_width[-1] = (0, expected_length - current_length)
            resampled = np.pad(resampled, pad_width, mode="edge")
        return resampled

    def _pad_or_crop(self, x: np.ndarray) -> np.ndarray:
        current_length = x.shape[-1]
        if current_length == self.target_len:
            return x
        if current_length > self.target_len:
            # More samples than needed (e.g. a longer native-rate window):
            # center-crop back down to target_len rather than stretching.
            start = (current_length - self.target_len) // 2
            return x[..., start : start + self.target_len]

        pad_total = self.target_len - current_length
        if self.pad_side == "end":
            pad_before, pad_after = 0, pad_total
        elif self.pad_side == "start":
            pad_before, pad_after = pad_total, 0
        else:  # "both"
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before

        np_pad_mode = "reflect" if self.pad_mode == "reflect" else "constant"
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (pad_before, pad_after)
        return np.pad(x, pad_width, mode=np_pad_mode)

    def _transform_one(self, sample):
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("downsample_pad requires sample['x'].")

        x = np.asarray(sample["x"], dtype=np.float32)
        if x.ndim < 2:
            raise ValueError(
                "downsample_pad expects sample['x'] to be at least 2D "
                f"(channels, time), got {x.shape}."
            )

        resampled = self._resample_to_native(x)
        out_x = self._pad_or_crop(resampled).astype(np.float32, copy=False)

        out = dict(sample)
        out["x"] = out_x
        if "sampling_rate" in out:
            out["sampling_rate"] = self.native_rate
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
