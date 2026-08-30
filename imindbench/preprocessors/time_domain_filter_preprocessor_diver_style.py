"""
Time-domain filtering preprocessor (notch + optional high-pass/bandpass).
"""

from __future__ import annotations

import numpy as np

from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("time_domain_filter_diver_style")
class TimeDomainFilterDIVERstylePreprocessor(BasePreprocessor):
    """Apply notch filtering and optional high-gamma bandpass."""

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
        orig_sr = self.cfg.get("sampling_rate", 2048)
        out = dict(sample)
        x = np.asarray(x, dtype=np.float64)
        freqs_to_notch = self.cfg.get("notch_freqs", [60, 120, 180])
        try:
            from mne.filter import filter_data, notch_filter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "time_domain_filter_diver_style requires mne. Install the "
                "examples/imindbench/environment.yml environment before "
                "using DIVER preprocessors."
            ) from exc

        filtered_data = filter_data(
            x, sfreq=orig_sr, l_freq=0.5, h_freq=None, verbose=False
        )
        notched_data = notch_filter(
            filtered_data, Fs=orig_sr, freqs=freqs_to_notch, verbose=False
        )
        out["x"] = notched_data.astype(np.float32, copy=False)
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
