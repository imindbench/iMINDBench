"""
HTNet-style neural decoder for time-domain neuroprobe inputs.

This is a PyTorch implementation of the single-subject/within-session HTNet
path: EEGNet-style temporal/spatial convolutions with an optional Hilbert
spectral feature layer. ROI projection is intentionally left out here because
within-session runs already have a stable channel axis.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model
from .torch_base_model import TorchBaseModel


def _get_cfg_value(cfg, *names, default=None):
    for name in names:
        if name in cfg:
            return cfg.get(name)
    return default


def _as_positive_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"model.{name} must be an int, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"model.{name} must be positive, got {value}.")
    return int(value)


def _as_probability(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"model.{name} must be a float, got {type(value).__name__}.")
    out = float(value)
    if out < 0.0 or out >= 1.0:
        raise ValueError(f"model.{name} must be in [0, 1), got {out}.")
    return out


def _as_positive_float(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"model.{name} must be a float, got {type(value).__name__}.")
    out = float(value)
    if out <= 0.0:
        raise ValueError(f"model.{name} must be positive, got {out}.")
    return out


class SamePadConv2d(nn.Module):
    """2D convolution with TensorFlow/Keras-style SAME padding."""

    def __init__(self, in_channels, out_channels, kernel_size, *, groups=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size_ = tuple(int(v) for v in kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size_,
            groups=groups,
            bias=False,
        )

    def forward(self, x):
        pad_h = max(self.kernel_size_[0] - 1, 0)
        pad_w = max(self.kernel_size_[1] - 1, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h or pad_w:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


def _hilbert(x: torch.Tensor) -> torch.Tensor:
    """Compute analytic signal along the final/time dimension."""
    if torch.is_complex(x):
        raise ValueError("Hilbert input must be real-valued.")
    n_times = int(x.shape[-1])
    if n_times <= 0:
        raise ValueError("Hilbert input must have a non-empty time dimension.")

    spectrum = torch.fft.fft(x.to(dtype=torch.float32), dim=-1)
    h = torch.zeros(n_times, device=x.device, dtype=x.dtype)
    if n_times % 2 == 0:
        h[0] = 1.0
        h[n_times // 2] = 1.0
        h[1 : n_times // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n_times + 1) // 2] = 2.0
    shape = [1] * x.ndim
    shape[-1] = n_times
    return torch.fft.ifft(spectrum * h.reshape(shape), dim=-1)


def _unwrap_phase(phase: torch.Tensor, *, discont: float = math.pi) -> torch.Tensor:
    """Unwrap phase along the final/time dimension."""
    if phase.shape[-1] <= 1:
        return phase
    delta = phase[..., 1:] - phase[..., :-1]
    delta_mod = torch.remainder(delta + math.pi, 2.0 * math.pi) - math.pi
    delta_mod = torch.where(
        (delta_mod == -math.pi) & (delta > 0.0),
        torch.full_like(delta_mod, math.pi),
        delta_mod,
    )
    correction = delta_mod - delta
    correction = torch.where(
        torch.abs(delta) < discont,
        torch.zeros_like(correction),
        correction,
    )
    correction = torch.cumsum(correction, dim=-1)
    correction = F.pad(correction, (1, 0))
    return phase + correction


class HilbertFeatureLayer(nn.Module):
    """Convert filtered time-domain activations to HTNet spectral features."""

    VALID_COMPUTE_VALUES = {"power", "relative_power", "phase", "freqslide"}

    def __init__(
        self,
        *,
        compute_val: str = "power",
        do_log: bool = False,
        data_srate: float = 1000.0,
        base_split: int = 4,
    ):
        super().__init__()
        if compute_val not in self.VALID_COMPUTE_VALUES:
            raise ValueError(
                "model.compute_val must be one of "
                f"{sorted(self.VALID_COMPUTE_VALUES)}, got '{compute_val}'."
            )
        if isinstance(do_log, bool):
            self.do_log = bool(do_log)
        else:
            raise TypeError(
                f"model.do_log must be a bool, got {type(do_log).__name__}."
            )
        if isinstance(base_split, bool) or not isinstance(base_split, int):
            raise TypeError(
                f"model.base_split must be an int, got {type(base_split).__name__}."
            )
        if base_split <= 0:
            raise ValueError(f"model.base_split must be positive, got {base_split}.")
        if isinstance(data_srate, bool) or not isinstance(data_srate, (int, float)):
            raise TypeError(
                f"model.data_srate must be a number, got {type(data_srate).__name__}."
            )
        if float(data_srate) <= 0.0:
            raise ValueError(f"model.data_srate must be positive, got {data_srate}.")
        self.compute_val = compute_val
        self.data_srate = float(data_srate)
        self.base_split = int(base_split)

    def forward(self, x):
        analytic = _hilbert(x)
        if self.compute_val == "power":
            out = torch.abs(analytic)
            if self.do_log:
                out = torch.log1p(out)
            return out

        if self.compute_val == "relative_power":
            out = torch.log1p(torch.abs(analytic))
            base_width = int(out.shape[-1]) // self.base_split
            if base_width < 1:
                raise ValueError(
                    "relative_power requires time dimension >= base_split, got "
                    f"time={out.shape[-1]}, base_split={self.base_split}."
                )
            baseline = out[..., :base_width].mean(dim=-1, keepdim=True)
            return out - baseline

        phase = _unwrap_phase(torch.angle(analytic))
        if self.compute_val == "phase":
            return phase

        freq = self.data_srate * (phase[..., 1:] - phase[..., :-1]) / (2.0 * math.pi)
        return F.pad(freq, (0, 1))


class SeparableTemporalConv2d(nn.Module):
    """Depthwise temporal convolution followed by a pointwise projection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_length: int):
        super().__init__()
        self.depthwise = SamePadConv2d(
            in_channels,
            in_channels,
            kernel_size=(1, kernel_length),
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class HTNetNetwork(nn.Module):
    """Single-subject HTNet architecture."""

    def __init__(
        self,
        *,
        input_shape: tuple[int, int],
        n_classes: int,
        f1: int,
        d: int,
        f2: int,
        kernel_length: int,
        separable_kernel_length: int,
        dropout_rate: float,
        dropout_type: str,
        compute_val: str,
        do_log: bool,
        data_srate: float,
        base_split: int,
        norm_rate: float,
        depthwise_max_norm: float,
    ):
        super().__init__()
        n_channels, _ = input_shape
        self.temporal_conv = SamePadConv2d(1, f1, kernel_size=(1, kernel_length))
        self.hilbert = HilbertFeatureLayer(
            compute_val=compute_val,
            do_log=do_log,
            data_srate=data_srate,
            base_split=base_split,
        )
        self.bn_temporal = nn.BatchNorm2d(f1, eps=0.001, momentum=0.01)
        self.spatial_depthwise = nn.Conv2d(
            f1,
            f1 * d,
            kernel_size=(n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.depthwise_max_norm = float(depthwise_max_norm)
        self.bn_spatial = nn.BatchNorm2d(f1 * d, eps=0.001, momentum=0.01)
        self.activation = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = self._make_dropout(dropout_type, dropout_rate)
        self.separable_conv = SeparableTemporalConv2d(
            f1 * d,
            f2,
            kernel_length=separable_kernel_length,
        )
        self.bn_separable = nn.BatchNorm2d(f2, eps=0.001, momentum=0.01)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = self._make_dropout(dropout_type, dropout_rate)

        self._reset_keras_style_parameters()
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape, dtype=torch.float32)
            feature_dim = int(
                self._forward_features(dummy).flatten(start_dim=1).shape[1]
            )
        self.train(was_training)
        self.classifier = nn.Linear(feature_dim, n_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.norm_rate = float(norm_rate)

    @staticmethod
    def _make_dropout(dropout_type: str, dropout_rate: float) -> nn.Module:
        if dropout_type == "Dropout":
            return nn.Dropout(dropout_rate)
        if dropout_type == "SpatialDropout2D":
            return nn.Dropout2d(dropout_rate)
        raise ValueError(
            "model.dropout_type must be 'Dropout' or 'SpatialDropout2D', "
            f"got '{dropout_type}'."
        )

    def _reset_keras_style_parameters(self) -> None:
        """Use Glorot/Xavier init to match Keras Conv/Dense defaults."""
        modules = [
            self.temporal_conv.conv,
            self.spatial_depthwise,
            self.separable_conv.depthwise.conv,
            self.separable_conv.pointwise,
        ]
        for module in modules:
            nn.init.xavier_uniform_(module.weight)

    @staticmethod
    def _constrain_rows_(weight: torch.Tensor, max_norm: float) -> None:
        flattened = weight.reshape(weight.shape[0], -1)
        norms = flattened.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        desired = torch.clamp(norms, max=max_norm)
        flattened.mul_(desired / norms)

    def apply_max_norm_constraints(self) -> None:
        """Apply Keras-style max-norm constraints after optimizer updates."""
        with torch.no_grad():
            self._constrain_rows_(
                self.spatial_depthwise.weight, self.depthwise_max_norm
            )
            self._constrain_rows_(self.classifier.weight, self.norm_rate)

    def _forward_features(self, x):
        if x.ndim != 3:
            raise ValueError(
                "HTNet expects input shape (batch, channels, time), got "
                f"{tuple(x.shape)}."
            )
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.hilbert(x)
        x = self.bn_temporal(x)
        x = self.spatial_depthwise(x)
        x = self.bn_spatial(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.separable_conv(x)
        x = self.bn_separable(x)
        x = self.activation(x)
        x = self.pool2(x)
        return self.dropout2(x)

    def forward(self, x):
        x = self._forward_features(x)
        return self.classifier(x.flatten(start_dim=1))


@register_model("htnet")
class HTNetModel(TorchBaseModel):
    """HTNet classifier for aligned, time-domain within-session inputs."""

    def prepare_batch(self, batch, **kwargs):
        """Validate and prepare collated batches for HTNet."""
        _ = kwargs
        if not isinstance(batch, dict):
            raise TypeError(f"prepare_batch expected dict, got {type(batch).__name__}.")
        if "x" not in batch or "y" not in batch:
            raise KeyError("Batch must include 'x' and 'y' keys.")

        x = batch["x"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        if x.ndim != 3:
            raise ValueError(
                "HTNet expects time-domain input with shape "
                f"(batch, channels, time), got {tuple(x.shape)}. "
                "Use raw/laplacian_wav preprocessors, not STFT preprocessors."
            )

        out = dict(batch)
        out["x"] = x
        return out

    def _create_network(self, input_shape, n_classes):
        """Create HTNet for an aligned channel x time input shape."""
        if len(input_shape) != 2:
            raise ValueError(
                "HTNet expects input_shape=(channels, time), got "
                f"{tuple(input_shape)}. Use time-domain preprocessors."
            )
        n_channels, n_times = (int(input_shape[0]), int(input_shape[1]))
        if n_channels < 1:
            raise ValueError(f"HTNet requires at least one channel, got {n_channels}.")
        if n_times < 32:
            raise ValueError(
                f"HTNet input time dimension is too small for two pooling stages: {n_times}."
            )

        f1 = _as_positive_int(
            _get_cfg_value(self.cfg, "f1", "F1", default=8),
            name="f1",
        )
        d = _as_positive_int(_get_cfg_value(self.cfg, "d", "D", default=2), name="d")
        f2 = _get_cfg_value(self.cfg, "f2", "F2", default=f1 * d)
        f2 = _as_positive_int(f2, name="f2")
        kernel_length = _as_positive_int(
            _get_cfg_value(self.cfg, "kernel_length", "kernLength", default=64),
            name="kernel_length",
        )
        separable_kernel_length = _as_positive_int(
            _get_cfg_value(
                self.cfg,
                "separable_kernel_length",
                "kernLength_sep",
                default=16,
            ),
            name="separable_kernel_length",
        )
        dropout_rate = _as_probability(
            _get_cfg_value(self.cfg, "dropout_rate", "dropoutRate", default=0.5),
            name="dropout_rate",
        )
        dropout_type = str(
            _get_cfg_value(self.cfg, "dropout_type", "dropoutType", default="Dropout")
        )
        compute_val = str(self.cfg.get("compute_val", "power"))
        do_log = self.cfg.get("do_log", False)
        data_srate = self.cfg.get("data_srate", 1000.0)
        base_split = self.cfg.get("base_split", 4)
        norm_rate = _as_positive_float(
            self.cfg.get("norm_rate", 0.25), name="norm_rate"
        )
        depthwise_max_norm = _as_positive_float(
            self.cfg.get("depthwise_max_norm", 1.0),
            name="depthwise_max_norm",
        )

        return HTNetNetwork(
            input_shape=(n_channels, n_times),
            n_classes=n_classes,
            f1=f1,
            d=d,
            f2=f2,
            kernel_length=kernel_length,
            separable_kernel_length=separable_kernel_length,
            dropout_rate=dropout_rate,
            dropout_type=dropout_type,
            compute_val=compute_val,
            do_log=do_log,
            data_srate=data_srate,
            base_split=base_split,
            norm_rate=norm_rate,
            depthwise_max_norm=depthwise_max_norm,
        )

    def build_model(self, input_shape, n_classes, device=None):
        """Build HTNet for an aligned channel x time input shape."""
        self._resolve_device(device)
        self.model = self._create_network(input_shape, n_classes)
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model

    def apply_constraints(self) -> None:
        """Hook used by TorchRunner after optimizer steps."""
        if self.model is not None:
            self.model.apply_max_norm_constraints()
