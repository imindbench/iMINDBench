"""
Time-domain filtering preprocessor (notch + optional high-pass/bandpass).
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy import signal

from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("time_domain_filter")
class TimeDomainFilterPreprocessor(BasePreprocessor):
    """Apply notch filtering and optional high-gamma bandpass."""

    NOTCH_FREQS = (60, 120, 180, 240, 300, 360)
    FILTER_CACHE_VERSION = 1
    FILTER_CACHE_MODES = {"read_only", "read_write", "refresh"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.session_wise = bool(self.cfg.get("session_wise", False))
        sampling_rate = float(self.cfg.get("sampling_rate", 2048))
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
        nyquist = sampling_rate / 2.0
        high_gamma = bool(self.cfg.get("high_gamma", False))
        self.notch_zero_phase = bool(self.cfg.get("notch_zero_phase", False))
        high_pass_hz = float(self.cfg.get("high_pass_hz", 0.0))
        self.high_pass_zero_phase = bool(
            self.cfg.get("high_pass_zero_phase", self.notch_zero_phase)
        )
        self.use_sos_notch_cascade = bool(self.cfg.get("use_sos_notch_cascade", False))
        self._filter_cache_enabled = bool(self.cfg.get("filter_cache_enabled", False))
        self._filter_cache_mode = str(self.cfg.get("filter_cache_mode", "read_write"))
        if self._filter_cache_mode not in self.FILTER_CACHE_MODES:
            allowed = ", ".join(sorted(self.FILTER_CACHE_MODES))
            raise ValueError(
                f"filter_cache_mode must be one of {allowed}; got "
                f"'{self._filter_cache_mode}'."
            )
        self._filter_cache_dir = self._resolve_filter_cache_dir(
            self.cfg.get("filter_cache_dir", None)
        )

        notch_q = float(self.cfg.get("notch_q", 30))
        self._notch_filters = []
        notch_sos_sections = []
        for freq in self.NOTCH_FREQS:
            w0 = freq / nyquist
            if w0 >= 1.0:
                continue
            # Filter design depends only on the config, so compute it once and
            # reuse the coefficients for every sample transform.
            notch_filter = signal.iirnotch(w0, notch_q)
            self._notch_filters.append(notch_filter)
            if self.use_sos_notch_cascade:
                notch_sos_sections.append(signal.tf2sos(*notch_filter))
        self._notch_sos = (
            None if not notch_sos_sections else np.vstack(notch_sos_sections)
        )

        self._high_pass_sos = None
        if high_pass_hz > 0.0:
            if high_pass_hz >= nyquist:
                raise ValueError(
                    f"high_pass_hz must be < Nyquist ({nyquist}), got {high_pass_hz}"
                )
            self._high_pass_sos = signal.butter(
                int(self.cfg.get("high_pass_order", 4)),
                high_pass_hz,
                btype="highpass",
                analog=False,
                fs=sampling_rate,
                output="sos",
            )
        self._filter_cache_identity = self._build_filter_cache_identity(
            sampling_rate=sampling_rate,
            high_gamma=high_gamma,
            high_pass_hz=high_pass_hz,
            high_pass_order=int(self.cfg.get("high_pass_order", 4)),
            high_pass_zero_phase=self.high_pass_zero_phase,
            notch_q=notch_q,
            notch_zero_phase=self.notch_zero_phase,
            bandpass_q=int(self.cfg.get("bandpass_q", 5)),
            bandpass_low=float(self.cfg.get("bandpass_low", 70)),
            bandpass_high=float(self.cfg.get("bandpass_high", 250)),
            use_sos_notch_cascade=self.use_sos_notch_cascade,
        )

        self._bandpass_sos = None
        if high_gamma:
            self._bandpass_sos = signal.butter(
                int(self.cfg.get("bandpass_q", 5)),
                [
                    float(self.cfg.get("bandpass_low", 70)),
                    float(self.cfg.get("bandpass_high", 250)),
                ],
                btype="bandpass",
                analog=False,
                fs=sampling_rate,
                output="sos",
            )

        if self.session_wise:
            if not self.notch_zero_phase:
                raise ValueError(
                    "session_wise=true requires notch_zero_phase=true to run "
                    "full-session zero-phase notch filtering."
                )
            if self._high_pass_sos is not None and not self.high_pass_zero_phase:
                raise ValueError(
                    "session_wise=true requires high_pass_zero_phase=true when "
                    "high_pass_hz is configured."
                )

    def _resolve_filter_cache_dir(self, value: Any) -> Path | None:
        if not self._filter_cache_enabled:
            return None
        if value is None:
            raise ValueError(
                "filter_cache_dir must be set when filter_cache_enabled=true."
            )
        if not isinstance(value, str):
            raise TypeError(
                "filter_cache_dir must be a str when set, got "
                f"{type(value).__name__}."
            )
        normalized = value.strip()
        if not normalized:
            raise ValueError("filter_cache_dir must be non-empty when set.")
        return Path(normalized)

    def _build_filter_cache_identity(
        self,
        *,
        sampling_rate: float,
        high_gamma: bool,
        high_pass_hz: float,
        high_pass_order: int,
        high_pass_zero_phase: bool,
        notch_q: float,
        notch_zero_phase: bool,
        bandpass_q: int,
        bandpass_low: float,
        bandpass_high: float,
        use_sos_notch_cascade: bool,
    ) -> dict[str, Any]:
        return {
            "cache_version": self.FILTER_CACHE_VERSION,
            "preprocessor": "time_domain_filter",
            "notch_freqs": list(self.NOTCH_FREQS),
            "sampling_rate": sampling_rate,
            "high_gamma": high_gamma,
            "high_pass_hz": high_pass_hz,
            "high_pass_order": high_pass_order,
            "high_pass_zero_phase": high_pass_zero_phase,
            "notch_q": notch_q,
            "notch_zero_phase": notch_zero_phase,
            "bandpass_q": bandpass_q,
            "bandpass_low": bandpass_low,
            "bandpass_high": bandpass_high,
            "use_sos_notch_cascade": use_sos_notch_cascade,
        }

    def _filter_cache_path(self, x: np.ndarray) -> Path | None:
        if self._filter_cache_dir is None:
            return None
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        identity = {
            "filter": self._filter_cache_identity,
            "input_dtype": str(x_c.dtype),
            "input_shape": [int(dim) for dim in x_c.shape],
            "input_sha256": hashlib.sha256(x_c.tobytes(order="C")).hexdigest(),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self._filter_cache_dir / digest[:2] / f"{digest}.npy"

    def _load_filter_cache(self, cache_path: Path) -> np.ndarray | None:
        if not cache_path.exists() or self._filter_cache_mode == "refresh":
            return None
        try:
            cached = np.load(cache_path, allow_pickle=False)
        except Exception:
            return None
        if cached.dtype != np.float32:
            return None
        return cached

    def _save_filter_cache(self, cache_path: Path, value: np.ndarray) -> None:
        if self._filter_cache_mode == "read_only":
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f"{cache_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
            tmp_path = Path(handle.name)
        tmp_path.replace(cache_path)

    def _filter_array(self, array: np.ndarray) -> np.ndarray:
        # Match legacy processed-mode filtering semantics used by the old BYD
        # pipeline: filter in float64, then cast back to float32.
        filtered = np.asarray(array, dtype=np.float64, order="C")
        if self.use_sos_notch_cascade and self._notch_sos is not None:
            if self.notch_zero_phase:
                filtered = signal.sosfiltfilt(self._notch_sos, filtered, axis=-1)
            else:
                filtered = signal.sosfilt(self._notch_sos, filtered, axis=-1)
        else:
            for b, a in self._notch_filters:
                if self.notch_zero_phase:
                    filtered = signal.filtfilt(b, a, filtered, axis=-1)
                else:
                    filtered = signal.lfilter(b, a, filtered, axis=-1)

        if self._high_pass_sos is not None:
            if self.high_pass_zero_phase:
                filtered = signal.sosfiltfilt(self._high_pass_sos, filtered, axis=-1)
            else:
                filtered = signal.sosfilt(self._high_pass_sos, filtered, axis=-1)

        if self._bandpass_sos is not None:
            filtered = signal.sosfilt(self._bandpass_sos, filtered, axis=-1)

        return filtered.astype(np.float32, copy=False)

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
        cache_path = self._filter_cache_path(x)
        if cache_path is not None:
            cached = self._load_filter_cache(cache_path)
            if cached is not None:
                out = dict(sample)
                out["x"] = cached
                return out
            if self._filter_cache_mode == "read_only":
                raise RuntimeError(
                    "filter_cache_mode='read_only' requires a filter cache hit, "
                    f"but no valid cache entry was found at '{cache_path}'."
                )
        out = dict(sample)
        out["x"] = self._filter_array(x)
        if cache_path is not None:
            self._save_filter_cache(cache_path, out["x"])
        return out

    def _extract_signal(self, sample) -> np.ndarray:
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
        return x

    def _channel_signature(self, sample: dict, x: np.ndarray) -> tuple:
        channel_ids = sample.get("channel_ids")
        if isinstance(channel_ids, list):
            return ("channel_ids", tuple(channel_ids))
        channel_names = sample.get("channel_names")
        if isinstance(channel_names, list):
            return ("channel_names", tuple(channel_names))
        return ("n_channels", int(x.shape[0]))

    def _transform_session_wise(self, samples: list[dict]) -> list[dict]:
        grouped: dict[str, list[tuple[int, dict, np.ndarray, float, int]]] = (
            defaultdict(list)
        )
        for original_idx, sample in enumerate(samples):
            x = self._extract_signal(sample)
            recording_id = sample.get("recording_id")
            if not isinstance(recording_id, str):
                raise TypeError(
                    "session_wise time_domain_filter requires sample['recording_id'] "
                    f"to be a str, got {type(recording_id).__name__}."
                )
            grouped[recording_id].append(
                (
                    original_idx,
                    sample,
                    x,
                    float(sample.get("window_start_sec", 0.0)),
                    int(sample.get("sample_idx", original_idx)),
                )
            )

        out_samples: list[dict | None] = [None] * len(samples)
        for recording_id, items in grouped.items():
            items.sort(key=lambda item: (item[3], item[4], item[0]))
            reference_signature = None
            chunk_lengths: list[int] = []
            chunks: list[np.ndarray] = []
            for _, sample, x, _, _ in items:
                signature = self._channel_signature(sample, x)
                if reference_signature is None:
                    reference_signature = signature
                elif signature != reference_signature:
                    raise ValueError(
                        "session_wise time_domain_filter requires consistent channel "
                        f"layout within recording '{recording_id}'. Expected "
                        f"{reference_signature}, got {signature}."
                    )
                chunk_lengths.append(int(x.shape[-1]))
                chunks.append(np.asarray(x, dtype=np.float32, order="C"))

            filtered = self._filter_array(np.concatenate(chunks, axis=-1))
            split_points = np.cumsum(chunk_lengths[:-1], dtype=np.int64)
            for item, filtered_chunk in zip(
                items, np.split(filtered, split_points, axis=-1)
            ):
                original_idx, sample, *_ = item
                out = dict(sample)
                out["x"] = filtered_chunk.astype(np.float32, copy=False)
                out_samples[original_idx] = out

        return [sample for sample in out_samples if sample is not None]

    def transform_samples(self, samples):
        sample_list = list(samples)
        if self.session_wise:
            return self._transform_session_wise(sample_list)
        return [self._transform_one(sample) for sample in sample_list]
