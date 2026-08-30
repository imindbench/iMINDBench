"""Load a larger raw waveform context window for time-aware preprocessors."""

from __future__ import annotations

import numpy as np

from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("context_window")
class ContextWindowPreprocessor(BasePreprocessor):
    """Replace sample['x'] with provider-loaded raw context around the target."""

    is_context_window_stage = True

    def __init__(self, cfg):
        super().__init__(cfg)
        context_window_sec = float(self.cfg.get("context_window_sec"))
        if context_window_sec <= 0.0:
            raise ValueError(
                f"context_window_sec must be positive, got {context_window_sec}."
            )
        if self.cfg.get("sampling_rate", None) is None:
            raise ValueError("context_window requires explicit sampling_rate.")
        sampling_rate = float(self.cfg.get("sampling_rate"))
        if sampling_rate <= 0.0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}.")
        pad_mode = str(self.cfg.get("pad_mode", "reflect"))
        if pad_mode != "reflect":
            raise ValueError(
                f"Unsupported context pad_mode '{pad_mode}'. Only 'reflect' is supported."
            )
        self.context_window_sec = context_window_sec
        self.sampling_rate = sampling_rate
        self.pad_mode = pad_mode
        # Require a downstream crop stage unless the model consumes full context.
        self.crop_back = bool(self.cfg.get("crop_back", True))
        # Right alignment centers the target in the final trailing segment.
        self.alignment = str(self.cfg.get("alignment", "center"))
        if self.alignment not in ("center", "right"):
            raise ValueError(
                f"context_window alignment must be 'center' or 'right', got "
                f"'{self.alignment}'."
            )
        self.trailing_segment_sec = float(
            self.cfg.get("trailing_segment_sec", context_window_sec)
        )
        self._recording_provider = None
        self._alignment_samples = 1

    def set_fold_context(self, context):
        context = dict(context or {})
        self._recording_provider = context.get("recording_provider")
        self._alignment_samples = int(context.get("context_alignment_samples", 1))
        if self._alignment_samples <= 0:
            raise ValueError(
                "context_alignment_samples must be positive, got "
                f"{self._alignment_samples}."
            )

    @staticmethod
    def _reflect_pad_time(array: np.ndarray, left: int, right: int) -> np.ndarray:
        if left <= 0 and right <= 0:
            return array
        if array.shape[-1] < 2:
            raise ValueError(
                "Cannot reflect-pad context with fewer than 2 time samples; "
                f"got shape {array.shape}."
            )
        return np.pad(array, ((0, 0), (left, right)), mode="reflect")

    def _resolve_recording_sampling_rate(self, recording, recording_id: str) -> float:
        recording_sampling_rate = getattr(recording, "sampling_rate", None)
        if recording_sampling_rate is None:
            seeg_data = getattr(recording, "seeg_data", None)
            recording_sampling_rate = getattr(seeg_data, "sampling_rate", None)
        if recording_sampling_rate is None:
            get_sampling_rate = getattr(
                self._recording_provider, "get_sampling_rate", None
            )
            if callable(get_sampling_rate):
                recording_sampling_rate = get_sampling_rate(recording_id)
        if recording_sampling_rate is None:
            recording_sampling_rate = getattr(
                self._recording_provider, "sampling_rate", None
            )
        if recording_sampling_rate is None:
            raise ValueError(
                "context_window requires recording sampling rate from "
                "recording.sampling_rate, recording.seeg_data.sampling_rate, "
                "provider.get_sampling_rate(recording_id), or "
                "provider.sampling_rate."
            )
        recording_sampling_rate = float(recording_sampling_rate)
        if not np.isfinite(recording_sampling_rate) or recording_sampling_rate <= 0.0:
            raise ValueError(
                "context_window recording sampling rate must be positive and finite, "
                f"got {recording_sampling_rate}."
            )
        return recording_sampling_rate

    def _window_indices(self, target_start: float, target_len: int):
        target_start_i = int(np.ceil(float(target_start) * self.sampling_rate - 1e-9))
        context_samples = int(round(self.context_window_sec * self.sampling_rate))
        if context_samples < target_len:
            raise ValueError(
                "context_window_sec must be at least the target sample duration: "
                f"context_window_sec={self.context_window_sec}, "
                f"target_samples={target_len}."
            )
        midpoint = target_start_i + target_len // 2
        if self.alignment == "right":
            trailing_samples = int(
                round(self.trailing_segment_sec * self.sampling_rate)
            )
            context_start_i = midpoint + trailing_samples // 2 - context_samples
        else:
            context_start_i = midpoint - context_samples // 2
        remainder = (target_start_i - context_start_i) % self._alignment_samples
        if remainder:
            context_start_i -= self._alignment_samples - remainder
        context_end_i = context_start_i + context_samples
        return target_start_i, context_start_i, context_end_i

    def _transform_one(self, sample):
        if self._recording_provider is None:
            raise RuntimeError(
                "context_window requires fold context key 'recording_provider'."
            )
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        for key in (
            "x",
            "recording_id",
            "window_start_sec",
            "window_end_sec",
            "channel_indices",
        ):
            if key not in sample:
                raise KeyError(f"context_window requires sample[{key!r}].")

        x = np.asarray(sample["x"], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(
                "context_window expects raw waveform sample['x'] with shape "
                f"(channels, time), got {x.shape}."
            )
        target_len = int(x.shape[-1])
        target_start_i, context_start_i, context_end_i = self._window_indices(
            float(sample["window_start_sec"]),
            target_len,
        )
        loaded_start_i = max(0, context_start_i)
        loaded_start_sec = loaded_start_i / self.sampling_rate
        requested_start_sec = context_start_i / self.sampling_rate
        requested_end_sec = context_end_i / self.sampling_rate

        recording_id = str(sample["recording_id"])
        recording = self._recording_provider.get_recording(recording_id)
        recording_sampling_rate = self._resolve_recording_sampling_rate(
            recording,
            recording_id,
        )
        if not np.isclose(float(recording_sampling_rate), self.sampling_rate):
            raise ValueError(
                "context_window sampling_rate does not match recording.sampling_rate: "
                f"{self.sampling_rate} vs {float(recording_sampling_rate)}."
            )
        window = recording.slice(loaded_start_sec, requested_end_sec)
        window_data = np.asarray(window.seeg_data.data)
        if window_data.ndim != 2:
            raise ValueError(
                "Context recording slice must return 2D seeg_data.data "
                f"(time, channels), got {window_data.shape}."
            )

        channel_indices_raw = np.asarray(sample["channel_indices"])
        if not np.issubdtype(channel_indices_raw.dtype, np.integer):
            raise TypeError("sample['channel_indices'] must contain integer values.")
        channel_indices = channel_indices_raw.astype(np.int64, copy=False).reshape(-1)
        if channel_indices.shape[0] != x.shape[0]:
            raise ValueError(
                "sample['channel_indices'] length must match sample['x'] channels: "
                f"{channel_indices.shape[0]} vs {x.shape[0]}."
            )
        if np.any(channel_indices < 0) or np.any(
            channel_indices >= window_data.shape[1]
        ):
            raise ValueError(
                "sample['channel_indices'] values must be in raw channel bounds "
                f"[0, {window_data.shape[1]}), got {channel_indices.tolist()}."
            )
        context = np.asarray(window_data[:, channel_indices].T, dtype=np.float32)

        context_samples = context_end_i - context_start_i
        left_pad = loaded_start_i - context_start_i
        loaded_samples = int(context.shape[-1])
        max_loaded_samples = context_samples - left_pad
        if loaded_samples > max_loaded_samples:
            context = context[:, :max_loaded_samples]
            loaded_samples = int(context.shape[-1])
        right_pad = max(0, context_samples - left_pad - loaded_samples)
        padded_context = self._reflect_pad_time(
            context,
            int(left_pad),
            int(right_pad),
        )

        out = dict(sample)
        out["x"] = padded_context
        out["context_requested_start_sec"] = requested_start_sec
        out["context_requested_end_sec"] = requested_end_sec
        out["context_loaded_start_sec"] = loaded_start_sec
        out["context_loaded_end_sec"] = (
            loaded_start_sec + loaded_samples / self.sampling_rate
        )
        out["context_pad_left_samples"] = int(left_pad)
        out["context_pad_right_samples"] = int(right_pad)
        out["context_pad_mode"] = self.pad_mode
        out["target_start_sample_in_context"] = int(target_start_i - context_start_i)
        out["target_num_samples"] = target_len
        out["context_alignment_samples"] = int(self._alignment_samples)
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
