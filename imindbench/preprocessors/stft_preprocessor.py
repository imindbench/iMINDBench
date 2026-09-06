"""
STFT (Short-Time Fourier Transform) preprocessor.

Canonical sample axis convention is ``(channels, time, features...)``. Raw
time-domain inputs are ``(channels, time)`` and STFT outputs are
``(channels, timebins, freqs)``.
"""

from numbers import Integral

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from . import register_preprocessor
from .base_preprocessor import BasePreprocessor


def _optional_int(cfg, key):
    value = cfg.get(key, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"STFT {key} must be an int, got {type(value).__name__}.")
    return int(value)


def resolve_stft_overlap(cfg):
    """Resolve STFT window and overlap using one canonical stride knob."""
    nperseg = _optional_int(cfg, "nperseg")
    if nperseg is None:
        nperseg = 512
    if nperseg <= 0:
        raise ValueError(f"STFT nperseg must be > 0, got {nperseg}.")

    hop_length = _optional_int(cfg, "hop_length")
    noverlap = _optional_int(cfg, "noverlap")
    poverlap = cfg.get("poverlap", None)
    if hop_length is not None and noverlap is not None:
        raise ValueError("Specify only one of STFT hop_length or noverlap.")
    if poverlap is not None and (hop_length is not None or noverlap is not None):
        raise ValueError("Specify only one of STFT hop_length, noverlap, or poverlap.")

    if hop_length is not None:
        noverlap = nperseg - hop_length
    elif noverlap is None:
        poverlap = 0.75 if poverlap is None else poverlap
        if isinstance(poverlap, bool) or not isinstance(poverlap, (int, float)):
            raise TypeError(
                f"STFT poverlap must be numeric, got {type(poverlap).__name__}."
            )
        noverlap = int(nperseg * float(poverlap))
        hop_length = nperseg - noverlap
    else:
        hop_length = nperseg - noverlap

    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError(
            "Invalid STFT overlap; expected 0 <= noverlap < nperseg, got "
            f"noverlap={noverlap}, nperseg={nperseg}."
        )
    if hop_length <= 0:
        raise ValueError(
            f"Invalid STFT hop_length={hop_length}; check nperseg/overlap settings."
        )

    return nperseg, noverlap, hop_length


def zscore(a, dim):
    """Compute a z-scored version of ``a`` across ``dim`` for torch or numpy."""
    if isinstance(a, torch.Tensor):
        mean = a.mean(dim=dim, keepdim=True)
        std = a.std(dim=dim, keepdim=True, unbiased=False)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (a - mean) / std

    arr = np.asarray(a)
    mean = arr.mean(axis=dim, keepdims=True)
    std = arr.std(axis=dim, keepdims=True)
    std[std == 0] = 1.0
    return (arr - mean) / std


@register_preprocessor("stft")
class STFTPreprocessor(BasePreprocessor):
    """
    Preprocessor that converts raw ``(channels, time)`` samples to
    ``(channels, timebins, freqs)`` magnitude features.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.clip_k = cfg.get(
            "clip_k", 0
        )  # Default 0 = no clipping (backwards compatible)

    def get_context_alignment_samples(self) -> int:
        """Return the sample stride required to align context STFT frames."""
        _, _, hop_length = resolve_stft_overlap(self.cfg)
        return int(hop_length)

    def _run_stft_tensor(self, x: torch.Tensor) -> torch.Tensor:
        # Batch input is (batch_size, channels, time). Output is
        # (batch_size, channels, timebins, freqs), matching the canonical
        # sample convention (channels, time, features...).
        batch_size, n_electrodes, _ = x.shape

        # convert to float32 and reshape for STFT
        x = x.to(dtype=torch.float32)
        x = x.reshape(batch_size * n_electrodes, -1)

        # STFT parameters
        nperseg, noverlap, hop_length = resolve_stft_overlap(self.cfg)
        window_type = self.cfg.get("window", "hann")
        sampling_rate = self.cfg.get("sampling_rate", 2048)
        max_frequency = self.cfg.get("max_frequency", 150)
        min_frequency = self.cfg.get("min_frequency", 0)
        normalizing = self.cfg.get("normalizing", "none")
        boundary = self.cfg.get("boundary", None)
        padded = bool(self.cfg.get("padded", False))
        use_scipy = bool(self.cfg.get("use_scipy", False))

        if boundary is not None and boundary != "zeros":
            raise ValueError(f"Unsupported STFT boundary mode: {boundary}")

        if use_scipy:
            x_np = x.detach().cpu().numpy()
            f, _, Zxx = signal.stft(
                x_np,
                fs=sampling_rate,
                window=window_type,
                nperseg=nperseg,
                noverlap=noverlap,
                boundary=boundary,
                padded=padded,
                return_onesided=True,
            )
            freq_channel_cutoff = int(self.cfg.get("freq_channel_cutoff", 0) or 0)
            if freq_channel_cutoff > 0:
                Zxx = Zxx[:, :freq_channel_cutoff, :]
                f = f[:freq_channel_cutoff]
            else:
                freq_mask = (f >= min_frequency) & (f <= max_frequency)
                Zxx = Zxx[:, freq_mask, :]
            x_np = np.abs(Zxx)
            if normalizing == "zscore":
                x_np = zscore(x_np, dim=-1)
            elif normalizing not in ("none", None):
                raise ValueError(f"Unsupported STFT normalizing mode: {normalizing}")
            x_np = x_np.transpose(0, 2, 1)
            _, n_times, n_freqs = x_np.shape
            x = torch.from_numpy(
                x_np.reshape(batch_size, n_electrodes, n_times, n_freqs)
            )
        else:
            torch_dtype = self.cfg.get("torch_dtype", "float32")
            if torch_dtype == "float64":
                x = x.to(dtype=torch.float64)
                window_dtype = torch.float64
            elif torch_dtype == "float32" or torch_dtype is None:
                window_dtype = torch.float32
            else:
                raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

            if window_type == "hann":
                window = torch.hann_window(nperseg, device=x.device, dtype=window_dtype)
            elif window_type == "boxcar":
                window = torch.ones(nperseg, device=x.device, dtype=window_dtype)
            else:
                raise ValueError(f"Invalid window type: {window_type}")

            # Match SciPy: center=True provides symmetric zero padding
            # (boundary='zeros'). padded=True adds right-end padding so the
            # number of frames is an integer.
            if padded:
                pad = nperseg // 2
                base_len = x.shape[-1] + 2 * pad
                remainder = (base_len - nperseg) % hop_length
                if remainder != 0:
                    pad_end = hop_length - remainder
                    x = F.pad(x, (0, pad_end))

            # Compute STFT
            pad_mode = self.cfg.get("pad_mode", "reflect")
            x = torch.stft(
                x,
                n_fft=nperseg,
                hop_length=hop_length,
                win_length=nperseg,
                window=window,
                return_complex=True,
                normalized=False,
                center=True,
                pad_mode=pad_mode,
            )

            # Frequency filtering: use freq_channel_cutoff if specified;
            # otherwise use Hz-based filtering.
            freq_channel_cutoff = self.cfg.get("freq_channel_cutoff", 0)
            if freq_channel_cutoff > 0:
                # Bin-based cutoff: keep first N frequency bins (PopT-style)
                x = x[:, :freq_channel_cutoff, :]
            else:
                # Hz-based filtering: filter by frequency range (existing behavior)
                freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sampling_rate)
                x = x[:, (freqs >= min_frequency) & (freqs <= max_frequency)]

            # Use magnitude (abs) for stft
            x = torch.abs(x)

            # Reshape back
            _, n_freqs, n_times = x.shape
            x = x.reshape(batch_size, n_electrodes, n_freqs, n_times)

            if normalizing == "zscore":
                x = zscore(x, dim=-1)
            elif normalizing not in ("none", None):
                raise ValueError(f"Unsupported STFT normalizing mode: {normalizing}")

            x = x.transpose(2, 3)  # (batch_size, n_electrodes, n_timebins, n_freqs)

        # Apply edge clipping if specified (for PopT compatibility)
        if self.clip_k > 0:
            x = x[:, :, self.clip_k : -self.clip_k, :]
        # If clip_k == 0, no clipping (backwards compatible)
        return x

    def _transform_one(self, sample):
        """Apply STFT to one sample dict while preserving aligned metadata."""
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("stft requires sample['x'].")
        if sample.get("context_requested_start_sec") is not None:
            normalizing = self.cfg.get("normalizing", "none")
            if normalizing == "zscore":
                raise ValueError(
                    "context_window + stft.normalizing=zscore is not supported; "
                    "crop to the target window before normalization."
                )

        x = np.asarray(sample["x"], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(
                f"stft expects sample['x'] with shape (channels, time), got {x.shape}."
            )
        x_out = self._run_stft_tensor(torch.from_numpy(x).unsqueeze(0))
        out = dict(sample)
        # Squeezed sample output is (channels, timebins, freqs).
        out["x"] = (
            x_out.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        )
        if sample.get("context_requested_start_sec") is not None:
            nperseg, _, hop_length = resolve_stft_overlap(self.cfg)
            out["context_time_axis"] = "stft"
            out["context_stft_hop_samples"] = int(hop_length)
            out["context_stft_nperseg"] = int(nperseg)
            out["context_stft_padded"] = bool(self.cfg.get("padded", False))
            out["context_stft_clip_k"] = int(self.clip_k or 0)
            out["context_stft_use_scipy"] = bool(self.cfg.get("use_scipy", False))
            out["context_stft_boundary"] = self.cfg.get("boundary", None)
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
