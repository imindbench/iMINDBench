"""Crop context-expanded features back to the model-visible target window."""

from __future__ import annotations

import math

import numpy as np

from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("crop_to_target_window")
class CropToTargetWindowPreprocessor(BasePreprocessor):
    """Crop waveform or STFT context outputs using generic context metadata."""

    is_crop_to_target_window_stage = True

    @staticmethod
    def _require_context(sample):
        required = ("target_start_sample_in_context", "target_num_samples")
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(
                "crop_to_target_window requires context metadata keys: " f"{missing}."
            )
        return (
            int(sample["target_start_sample_in_context"]),
            int(sample["target_num_samples"]),
        )

    @staticmethod
    def _crop_waveform(x: np.ndarray, *, start: int, length: int) -> np.ndarray:
        end = start + length
        if start < 0 or end > x.shape[-1]:
            raise ValueError(
                "Target waveform crop is out of bounds: "
                f"crop=({start}, {end}), shape={x.shape}."
            )
        return x[:, start:end]

    @staticmethod
    def _stft_target_frame_count(sample) -> int:
        target_samples = int(sample["target_num_samples"])
        hop = int(sample["context_stft_hop_samples"])
        if hop <= 0:
            raise ValueError(f"context_stft_hop_samples must be positive, got {hop}.")
        padded = bool(sample.get("context_stft_padded", False))
        if (
            bool(sample.get("context_stft_use_scipy", False))
            and sample.get("context_stft_boundary") is None
        ):
            nperseg = int(sample["context_stft_nperseg"])
            if nperseg <= 0:
                raise ValueError(
                    f"context_stft_nperseg must be positive, got {nperseg}."
                )
            if target_samples < nperseg:
                n_frames = 1
            elif padded:
                n_frames = int(math.ceil((target_samples - nperseg) / hop)) + 1
            else:
                n_frames = int((target_samples - nperseg) // hop) + 1
        elif padded:
            n_frames = int(math.ceil(target_samples / hop)) + 1
        else:
            n_frames = int(target_samples // hop) + 1
        clip_k = int(sample.get("context_stft_clip_k", 0) or 0)
        if clip_k < 0:
            raise ValueError(f"context_stft_clip_k must be non-negative, got {clip_k}.")
        return max(0, n_frames - (2 * clip_k))

    def _crop_stft(self, x: np.ndarray, *, sample, start_samples: int) -> np.ndarray:
        hop = int(sample["context_stft_hop_samples"])
        if start_samples % hop != 0:
            raise ValueError(
                "STFT target crop is not aligned to hop length: "
                f"start_samples={start_samples}, hop={hop}."
            )
        clip_k = int(sample.get("context_stft_clip_k", 0) or 0)
        if clip_k < 0:
            raise ValueError(f"context_stft_clip_k must be non-negative, got {clip_k}.")
        start_frame = start_samples // hop
        n_frames = self._stft_target_frame_count(sample)
        end_frame = start_frame + n_frames
        if start_frame < 0 or end_frame > x.shape[1]:
            raise ValueError(
                "Target STFT crop is out of bounds: "
                f"frames=({start_frame}, {end_frame}), shape={x.shape}."
            )
        return x[:, start_frame:end_frame, :]

    def _transform_one(self, sample):
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("crop_to_target_window requires sample['x'].")
        start, length = self._require_context(sample)
        x = np.asarray(sample["x"], dtype=np.float32)
        out = dict(sample)
        if sample.get("context_time_axis") == "stft":
            if x.ndim != 3:
                raise ValueError(
                    "STFT crop expects sample['x'] with shape "
                    f"(channels, time_bins, freqs), got {x.shape}."
                )
            out["x"] = self._crop_stft(x, sample=sample, start_samples=start)
        else:
            if x.ndim != 2:
                raise ValueError(
                    "Waveform crop expects sample['x'] with shape "
                    f"(channels, time), got {x.shape}."
                )
            out["x"] = self._crop_waveform(x, start=start, length=length)
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
