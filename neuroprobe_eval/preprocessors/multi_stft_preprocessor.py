"""Multi-window STFT preprocessor."""

from copy import deepcopy

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from .base_preprocessor import BasePreprocessor
from .stft_preprocessor import STFTPreprocessor, resolve_stft_overlap
from . import register_preprocessor


_STFT_DEFAULT_KEYS = {
    "boundary",
    "clip_k",
    "freq_channel_cutoff",
    "hop_length",
    "max_frequency",
    "min_frequency",
    "normalizing",
    "noverlap",
    "nperseg",
    "pad_mode",
    "padded",
    "poverlap",
    "sampling_rate",
    "torch_dtype",
    "use_scipy",
    "window",
}
_STFT_STRIDE_KEYS = {"hop_length", "noverlap", "poverlap"}


def _to_plain_dict(value, *, context):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return deepcopy(value)
    raise TypeError(f"{context} must be a mapping, got {type(value).__name__}.")


@register_preprocessor("multi_stft")
class MultiSTFTPreprocessor(BasePreprocessor):
    """Run multiple STFT windows and concatenate their frequency bins."""

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.get("hop_length", None) is None:
            raise ValueError("multi_stft requires a shared top-level hop_length.")
        if (
            cfg.get("noverlap", None) is not None
            or cfg.get("poverlap", None) is not None
        ):
            raise ValueError(
                "multi_stft requires hop_length for the shared stride; "
                "do not set top-level noverlap or poverlap."
            )

        windows = cfg.get("windows", None)
        if not isinstance(windows, (list, ListConfig)) or len(windows) == 0:
            raise ValueError("multi_stft requires a non-empty 'windows' list.")

        defaults = {
            key: deepcopy(cfg.get(key))
            for key in _STFT_DEFAULT_KEYS
            if cfg.get(key, None) is not None
        }
        self.window_labels = []
        self.window_preprocessors = []
        expected_timing_signature = None
        for idx, raw_window_cfg in enumerate(windows):
            window_cfg = _to_plain_dict(
                raw_window_cfg, context=f"multi_stft windows[{idx}]"
            )
            label = window_cfg.pop("label", f"window_{idx}")
            if not isinstance(label, str) or label == "":
                raise ValueError("multi_stft window labels must be non-empty strings.")
            for key in _STFT_STRIDE_KEYS:
                if key in window_cfg:
                    raise ValueError(
                        "multi_stft window "
                        f"'{label}' must not set {key}; set the shared "
                        "top-level hop_length instead."
                    )

            stft_cfg = deepcopy(defaults)
            stft_cfg.update(window_cfg)
            stft_cfg["name"] = "stft"
            timing_signature = self._timing_signature(stft_cfg)
            if expected_timing_signature is None:
                expected_timing_signature = timing_signature
            elif timing_signature != expected_timing_signature:
                raise ValueError(
                    "multi_stft windows must share timing settings; "
                    f"window '{label}' has {timing_signature}, expected "
                    f"{expected_timing_signature}."
                )
            self.window_labels.append(label)
            self.window_preprocessors.append(
                STFTPreprocessor(OmegaConf.create(stft_cfg))
            )

    def get_context_alignment_samples(self) -> int:
        """Return the shared STFT stride needed by upstream context_window."""
        return self.window_preprocessors[0].get_context_alignment_samples()

    @staticmethod
    def _timing_signature(stft_cfg):
        """Return fields that define frame positions before frequency concat."""
        cfg = OmegaConf.create(stft_cfg)
        nperseg, _, hop_length = resolve_stft_overlap(cfg)
        use_scipy = bool(stft_cfg.get("use_scipy", False))
        boundary = stft_cfg.get("boundary", None)
        signature = {
            "hop_length": hop_length,
            "boundary": boundary,
            "padded": bool(stft_cfg.get("padded", False)),
            "use_scipy": use_scipy,
            "clip_k": int(stft_cfg.get("clip_k", 0) or 0),
            "pad_mode": stft_cfg.get("pad_mode", "reflect"),
        }
        # SciPy uses nperseg in the time origin when no boundary extension is
        # requested, so differing nperseg values would not be aligned in time.
        if use_scipy and boundary is None:
            signature["nperseg"] = nperseg
        return tuple(sorted(signature.items()))

    def _transform_one(self, sample):
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("multi_stft requires sample['x'].")

        outputs = []
        metadata_source = None
        expected_channels = None
        expected_time_bins = None
        for label, preprocessor in zip(self.window_labels, self.window_preprocessors):
            transformed = preprocessor.transform_samples([sample])[0]
            x_window = np.asarray(transformed["x"], dtype=np.float32)
            if metadata_source is None:
                metadata_source = transformed
            if x_window.ndim != 3:
                raise ValueError(
                    "multi_stft expects each STFT window to produce "
                    f"(channels, time, freq), got {x_window.shape} for '{label}'."
                )

            channels, time_bins, _ = x_window.shape
            if expected_channels is None:
                expected_channels = channels
                expected_time_bins = time_bins
            elif channels != expected_channels:
                raise ValueError(
                    "multi_stft windows must preserve channel count; "
                    f"window '{label}' produced {channels}, expected "
                    f"{expected_channels}."
                )
            elif time_bins != expected_time_bins:
                raise ValueError(
                    "multi_stft requires matching time bins across windows; "
                    f"window '{label}' produced {time_bins}, expected "
                    f"{expected_time_bins}. Set a shared top-level hop_length "
                    "or adjust per-window STFT parameters."
                )

            outputs.append(x_window)

        out = dict(metadata_source if metadata_source is not None else sample)
        out["x"] = np.concatenate(outputs, axis=2).astype(np.float32, copy=False)
        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
