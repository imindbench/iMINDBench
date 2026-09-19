"""
Time-domain filtering preprocessor (notch + optional high-pass/bandpass).
"""

from __future__ import annotations

import math

import numpy as np

from . import register_preprocessor
from .base_preprocessor import BasePreprocessor


@register_preprocessor("time_domain_filter_diver_style")
class TimeDomainFilterDIVERstylePreprocessor(BasePreprocessor):
    """Apply MNE high-pass and notch filtering with DIVER's historical defaults."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sampling_rate = self.cfg.get("sampling_rate", 2048)
        if isinstance(self.sampling_rate, bool) or not isinstance(
            self.sampling_rate, (int, float)
        ):
            raise TypeError("sampling_rate must be a positive number.")
        if not math.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be finite and positive.")
        # DIVER uses 0.5 Hz when high_pass_hz is omitted or explicitly None.
        # Use 0.0 to disable it; the standard filter has a different default.
        self.high_pass_hz = self.cfg.get("high_pass_hz", None)
        if self.high_pass_hz is None:
            self.high_pass_hz = 0.5
        if isinstance(self.high_pass_hz, bool) or not isinstance(
            self.high_pass_hz, (int, float)
        ):
            raise TypeError("high_pass_hz must be a number or None (0.5 Hz).")
        if (
            not math.isfinite(self.high_pass_hz)
            or not 0 <= self.high_pass_hz < self.sampling_rate / 2
        ):
            raise ValueError("high_pass_hz must be finite and in [0, Nyquist).")

    def _transform_one(self, sample):
        """Apply filtering to one sample dict while preserving aligned metadata."""
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("time_domain_filter requires sample['x'].")

        x = np.asarray(sample["x"], dtype=np.float32)
        if x.ndim < 2:
            raise ValueError(
                "time_domain_filter expects sample['x'] to be at least 2D "
                f"(channels, time), got {x.shape}."
            )
        out = dict(sample)
        x = np.asarray(x, dtype=np.float64)
        freqs_to_notch = self.cfg.get("notch_freqs", [60, 120, 180])
        try:
            from mne.filter import filter_data, notch_filter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "time_domain_filter_diver_style requires mne. Install dependencies "
                "from the repository-root environment.yml before using DIVER "
                "preprocessors."
            ) from exc

        filtered_data = filter_data(
            x,
            sfreq=self.sampling_rate,
            l_freq=self.high_pass_hz or None,
            h_freq=None,
            verbose=False,
        )
        notched_data = notch_filter(
            filtered_data, Fs=self.sampling_rate, freqs=freqs_to_notch, verbose=False
        )
        out["x"] = notched_data.astype(np.float32, copy=False)
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
