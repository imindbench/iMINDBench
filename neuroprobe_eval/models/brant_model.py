"""
BaRISTA-style BRANT adaptation for short-window neuroprobe evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from . import register_model
from .mlp_model import TorchBaseModel


def _weights_init(module: nn.Module) -> None:
    """Match the archived BRANT initialization for compatible modules."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1.0)
        nn.init.zeros_(module.bias)


def _unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract a raw state_dict from common checkpoint wrappers."""
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(checkpoint).__name__}.")

    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return {
                key.removeprefix("module."): value
                for key, value in checkpoint.items()
                if torch.is_tensor(value)
            }

    for key in ("state_dict", "model", "model_state"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and all(
            isinstance(name, str) and torch.is_tensor(tensor)
            for name, tensor in value.items()
        ):
            return {
                name.removeprefix("module."): tensor for name, tensor in value.items()
            }

    raise ValueError("Unable to extract a tensor state_dict from checkpoint.")


def _matching_state_dict(
    module: nn.Module, checkpoint_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Keep only checkpoint tensors that still match the target module contract."""
    current_state = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, value in checkpoint_state.items():
        # The short-window adaptation intentionally changes seq_len from 15 -> 1,
        # so positional_encoding is expected to fail this shape check.
        current_value = current_state.get(key)
        if current_value is not None and tuple(current_value.shape) == tuple(
            value.shape
        ):
            matched[key] = value
    return matched


def _compute_band_power(
    data: torch.Tensor,
    *,
    sampling_rate: float,
    band_edges: tuple[float, ...],
) -> torch.Tensor:
    """Approximate BRANT's released periodogram-based band-power features in torch."""
    if data.ndim != 3:
        raise ValueError(
            f"Expected data with shape (batch, channels, time), got {tuple(data.shape)}."
        )
    if len(band_edges) < 2:
        raise ValueError("band_edges must contain at least two values.")

    n_samples = int(data.shape[-1])
    spectrum = torch.fft.rfft(data, dim=-1)
    power = spectrum.abs().square() / max(n_samples, 1)
    freqs = torch.fft.rfftfreq(n_samples, d=1.0 / float(sampling_rate)).to(data.device)

    band_sums = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        band_mask = (freqs > float(low)) & (freqs <= float(high))
        if bool(band_mask.any().item()):
            band_power = power[..., band_mask].sum(dim=-1)
        else:
            band_power = torch.zeros_like(power[..., 0])
        band_sums.append(torch.log10(band_power + 1.0))
    return torch.stack(band_sums, dim=-1)


def _build_mlp_head(
    *,
    in_dim: int,
    out_dim: int,
    hidden_dims: tuple[int, ...],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = int(in_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, int(hidden_dim)))
        layers.append(nn.ReLU(inplace=True))
        previous_dim = int(hidden_dim)
    layers.append(nn.Linear(previous_dim, int(out_dim)))
    head = nn.Sequential(*layers)
    head.apply(_weights_init)
    return head


def _masked_mean(values: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool channel-aligned values while ignoring padded channels."""
    valid = channel_mask.unsqueeze(-1).to(dtype=values.dtype)
    return (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class BrantInputEmbedding(nn.Module):
    """Input embedding with BRANT-compatible parameter names and state_dict layout."""

    def __init__(
        self,
        *,
        in_dim: int,
        d_model: int,
        seq_len: int,
        band_num: int,
        project_mode: str,
        learnable_mask: bool,
    ) -> None:
        super().__init__()
        self.mode = str(project_mode)
        self.band_num = int(band_num)
        self.band_encoding = nn.Parameter(torch.randn(band_num, d_model))
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        if learnable_mask:
            self.mask_encoding = nn.Parameter(torch.randn(in_dim))
            self.power_mask_encoding = nn.Parameter(torch.randn(d_model))
        else:
            self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
            self.power_mask_encoding = nn.Parameter(
                torch.zeros(d_model), requires_grad=False
            )

        self.softmax = nn.Softmax(dim=-1)
        if self.mode == "linear":
            self.proj = nn.Sequential(nn.Linear(in_dim, d_model))
        else:
            raise NotImplementedError(
                f"Unsupported BRANT project_mode '{self.mode}'. Expected 'linear'."
            )
        self.apply(_weights_init)

    def forward(self, data: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        """Project one pseudo-patch per channel into BRANT token space."""
        batch_size, n_channels, seq_len, seg_len = data.shape
        if tuple(power.shape[:3]) != (batch_size, n_channels, seq_len):
            raise ValueError(
                "power must match data batch/channel/sequence dimensions, got "
                f"{tuple(power.shape)} vs {tuple(data.shape)}."
            )
        if power.shape[-1] != self.band_num:
            raise ValueError(
                f"Expected {self.band_num} power bands, got {int(power.shape[-1])}."
            )
        if seg_len != int(self.mask_encoding.shape[0]):
            raise ValueError(
                f"Expected segment length {int(self.mask_encoding.shape[0])}, got {seg_len}."
            )
        if seq_len != int(self.positional_encoding.shape[0]):
            raise ValueError(
                "Input sequence length must match positional_encoding, got "
                f"{seq_len} vs {int(self.positional_encoding.shape[0])}."
            )

        power = self.softmax(power)
        power_emb = torch.einsum("bcsf,fd->bcsd", power, self.band_encoding)
        projected = self.proj(data.reshape(batch_size * n_channels, seq_len, seg_len))
        power_emb = power_emb.reshape(batch_size * n_channels, seq_len, -1)

        projected = projected + power_emb
        projected = projected + self.positional_encoding.unsqueeze(0)
        return projected


class BrantTimeEncoder(nn.Module):
    """Temporal encoder with BRANT-compatible checkpoint layout."""

    def __init__(
        self,
        *,
        in_dim: int,
        d_model: int,
        dim_feedforward: int,
        seq_len: int,
        n_layer: int,
        nhead: int,
        band_num: int,
        project_mode: str,
        learnable_mask: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_embedding = BrantInputEmbedding(
            in_dim=in_dim,
            d_model=d_model,
            seq_len=seq_len,
            band_num=band_num,
            project_mode=project_mode,
            learnable_mask=learnable_mask,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.apply(_weights_init)

    def forward(self, data: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        input_emb = self.input_embedding(data, power)
        return self.trans_enc(input_emb)


class BrantChannelEncoder(nn.Module):
    """Channel encoder with BRANT-compatible checkpoint layout."""

    def __init__(
        self,
        *,
        out_dim: int,
        d_model: int,
        dim_feedforward: int,
        n_layer: int,
        nhead: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.apply(_weights_init)

    def forward(
        self, time_z: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ch_z = self.trans_enc(time_z, src_key_padding_mask=src_key_padding_mask)
        rec = self.proj_out(ch_z)
        return ch_z, rec


class BrantClassifier(nn.Module):
    """Whole-window BRANT backbone plus configurable classification head."""

    def __init__(
        self,
        *,
        time_encoder: BrantTimeEncoder,
        channel_encoder: BrantChannelEncoder,
        n_classes: int,
        d_model: int,
        seg_len: int,
        seq_len: int,
        power_sampling_rate: float,
        band_edges: tuple[float, ...],
        head_type: str,
        head_hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.time_encoder = time_encoder
        self.channel_encoder = channel_encoder
        self.seg_len = int(seg_len)
        self.seq_len = int(seq_len)
        self.power_sampling_rate = float(power_sampling_rate)
        self.band_edges = tuple(float(edge) for edge in band_edges)
        self.head_type = str(head_type)
        self._last_token_logits: torch.Tensor | None = None
        self._last_channel_mask: torch.Tensor | None = None
        if self.head_type == "linear_pooled":
            self.classification_head = nn.Linear(d_model, n_classes)
            _weights_init(self.classification_head)
        elif self.head_type in {"mlp_pooled", "mlp_token"}:
            self.classification_head = _build_mlp_head(
                in_dim=d_model,
                out_dim=n_classes,
                hidden_dims=head_hidden_dims,
            )
        else:
            raise ValueError(
                "Unsupported BRANT head_type "
                f"'{self.head_type}'. Expected one of "
                "'linear_pooled', 'mlp_pooled', or 'mlp_token'."
            )

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        band_power: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected BRANT input with shape (batch, channels, time), got {tuple(x.shape)}."
            )
        if int(x.shape[-1]) != self.seg_len:
            raise ValueError(
                f"Expected time dimension {self.seg_len}, got {int(x.shape[-1])}."
            )
        if self.seq_len != 1:
            raise ValueError(
                f"BaRISTA-style BRANT adaptation expects seq_len=1, got {self.seq_len}."
            )

        batch_size, n_channels, _ = x.shape
        if channel_mask is None:
            channel_mask = torch.ones(
                (batch_size, n_channels), dtype=torch.bool, device=x.device
            )
        else:
            channel_mask = torch.as_tensor(
                channel_mask, dtype=torch.bool, device=x.device
            )
        if tuple(channel_mask.shape) != (batch_size, n_channels):
            raise ValueError(
                "channel_mask shape must match (batch, channels), got "
                f"{tuple(channel_mask.shape)} vs {(batch_size, n_channels)}."
            )
        if bool((channel_mask.sum(dim=1) < 1).any().item()):
            raise ValueError(
                "Each BRANT sample must contain at least one valid channel."
            )

        # BaRISTA-style adaptation: each short window becomes one BRANT pseudo-patch
        # per channel, so the temporal encoder runs with seq_len == 1.
        patch_data = x.unsqueeze(2)
        if band_power is None:
            band_power = _compute_band_power(
                x, sampling_rate=self.power_sampling_rate, band_edges=self.band_edges
            )
        else:
            band_power = torch.as_tensor(band_power, dtype=x.dtype, device=x.device)
            expected_power_shape = (
                batch_size,
                n_channels,
                len(self.band_edges) - 1,
            )
            if tuple(band_power.shape) != expected_power_shape:
                raise ValueError(
                    "band_power shape must match (batch, channels, bands), got "
                    f"{tuple(band_power.shape)} vs "
                    f"{expected_power_shape}."
                )
        band_power = band_power.unsqueeze(2)

        time_z = self.time_encoder(patch_data, band_power)
        time_z = time_z.squeeze(1).reshape(batch_size, n_channels, -1)

        # Padding is only a batching detail. Padded channels are masked out in
        # the channel transformer and excluded from the final pooled embedding.
        channel_pad_mask = ~channel_mask
        ch_z, _ = self.channel_encoder(time_z, src_key_padding_mask=channel_pad_mask)

        if self.head_type == "mlp_token":
            # Match the author fine-tuning head more closely by classifying each
            # channel token first, then aggregating sample-level logits.
            token_logits = self.classification_head(ch_z)
            self._last_token_logits = token_logits
            self._last_channel_mask = channel_mask
            return _masked_mean(token_logits, channel_mask)

        self._last_token_logits = None
        self._last_channel_mask = channel_mask
        pooled = _masked_mean(ch_z, channel_mask)
        return self.classification_head(pooled)


@register_model("brant")
class BrantModel(TorchBaseModel):
    """Approximate BRANT integration for short raw neuroprobe windows."""

    _DEFAULT_BAND_EDGES = (4.0, 8.0, 13.0, 30.0, 50.0, 70.0, 90.0, 110.0, 128.0)
    _VALID_HEAD_TYPES = {"linear_pooled", "mlp_pooled", "mlp_token"}

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.accepts_coords = False

        self.seg_len = int(cfg.get("seg_len", 1500))
        self.seq_len = int(cfg.get("seq_len", 1))
        self.d_model = int(cfg.get("d_model", 2048))
        self.dim_feedforward = int(cfg.get("dim_feedforward", 3072))
        self.time_ar_layer = int(cfg.get("time_ar_layer", 12))
        self.time_ar_head = int(cfg.get("time_ar_head", 16))
        self.ch_ar_layer = int(cfg.get("ch_ar_layer", 5))
        self.ch_ar_head = int(cfg.get("ch_ar_head", 16))
        self.band_num = int(cfg.get("band_num", 8))
        self.project_mode = str(cfg.get("project_mode", "linear"))
        self.learnable_mask = bool(cfg.get("learnable_mask", True))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.power_sampling_rate = float(cfg.get("power_sampling_rate", 256.0))
        self.band_edges = tuple(
            float(edge) for edge in cfg.get("band_edges", self._DEFAULT_BAND_EDGES)
        )
        self.cache_inputs = bool(cfg.get("cache_inputs", True))
        self.head_type = str(cfg.get("head_type", "mlp_token"))
        default_head_hidden_dims = (
            [self.d_model // 2, self.d_model // 4]
            if self.head_type in {"mlp_pooled", "mlp_token"}
            else []
        )
        raw_head_hidden_dims = cfg.get("head_hidden_dims", default_head_hidden_dims)
        if raw_head_hidden_dims is None:
            raw_head_hidden_dims = default_head_hidden_dims
        if isinstance(raw_head_hidden_dims, str):
            raise ValueError("model.head_hidden_dims must be a list of integers.")
        self.head_hidden_dims = tuple(int(dim) for dim in raw_head_hidden_dims)

        if self.seq_len != 1:
            raise ValueError(
                "BaRISTA-style short-window BRANT requires model.seq_len=1, got "
                f"{self.seq_len}."
            )
        if len(self.band_edges) != self.band_num + 1:
            raise ValueError(
                "band_edges length must equal band_num + 1, got "
                f"{len(self.band_edges)} vs {self.band_num + 1}."
            )
        if self.project_mode != "linear":
            raise ValueError(
                f"Unsupported BRANT project_mode '{self.project_mode}'. Expected 'linear'."
            )
        if self.head_type not in self._VALID_HEAD_TYPES:
            raise ValueError(
                f"model.head_type must be one of {sorted(self._VALID_HEAD_TYPES)}, "
                f"got '{self.head_type}'."
            )
        if self.head_type in {"mlp_pooled", "mlp_token"} and not self.head_hidden_dims:
            raise ValueError(
                f"model.head_hidden_dims must be non-empty for head_type='{self.head_type}'."
            )
        if any(dim <= 0 for dim in self.head_hidden_dims):
            raise ValueError("model.head_hidden_dims values must be positive integers.")

        upstream_ckpt_dir = cfg.get("upstream_ckpt_dir", None)
        self.upstream_ckpt_dir = (
            None
            if upstream_ckpt_dir in (None, "null")
            else Path(str(upstream_ckpt_dir))
        )
        self.random_init = bool(cfg.get("random_init", False))
        self.frozen_upstream = bool(cfg.get("frozen_upstream", False))
        self.upstream_lr = float(cfg.get("upstream_lr", 1e-7))
        self.head_lr = float(cfg.get("head_lr", 1e-3))

        self.time_encoder: BrantTimeEncoder | None = None
        self.channel_encoder: BrantChannelEncoder | None = None
        self.classification_head: nn.Module | None = None
        self._input_cache: dict[
            tuple[object, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _cache_key(
        self,
        batch: dict,
        *,
        batch_idx: int,
        time_dim: int,
        valid_mask: torch.Tensor,
    ) -> tuple[object, ...] | None:
        if not self.cache_inputs:
            return None

        required = (
            "recording_ids",
            "channel_ids",
            "sample_indices",
            "window_start_sec",
            "window_end_sec",
            "split",
        )
        if any(key not in batch for key in required):
            return None

        valid_mask_cpu = valid_mask.detach().cpu().bool().tolist()
        valid_channel_ids = tuple(
            str(channel_id)
            for channel_id, is_valid in zip(
                batch["channel_ids"][batch_idx], valid_mask_cpu
            )
            if is_valid
        )
        n_valid_channels = len(valid_channel_ids)

        return (
            str(batch["split"]),
            str(batch["recording_ids"][batch_idx]),
            valid_channel_ids,
            int(batch["sample_indices"][batch_idx]),
            float(batch["window_start_sec"][batch_idx]),
            float(batch["window_end_sec"][batch_idx]),
            int(n_valid_channels),
            int(time_dim),
            int(self.seg_len),
            float(self.power_sampling_rate),
            tuple(self.band_edges),
        )

    def _build_brant_inputs(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return BRANT-resampled waveforms and band powers, reusing CPU cache."""
        batch_size, n_channels, time_dim = map(int, x.shape)
        x_out = torch.zeros(
            (batch_size, n_channels, self.seg_len),
            dtype=x.dtype,
            device=x.device,
        )
        band_power_out = torch.zeros(
            (batch_size, n_channels, self.band_num),
            dtype=x.dtype,
            device=x.device,
        )

        valid_counts = channel_mask.sum(dim=1).tolist()
        for batch_idx, n_valid_raw in enumerate(valid_counts):
            n_valid = int(n_valid_raw)
            if n_valid < 1:
                raise ValueError(
                    "Each BRANT sample must contain at least one valid channel."
                )
            valid_mask = channel_mask[batch_idx].bool()

            cache_key = self._cache_key(
                batch,
                batch_idx=batch_idx,
                time_dim=time_dim,
                valid_mask=valid_mask,
            )
            cached = self._input_cache.get(cache_key) if cache_key is not None else None
            if cached is None:
                sample_x = x[batch_idx, valid_mask]
                if time_dim != self.seg_len:
                    sample_x = F.interpolate(
                        sample_x.reshape(n_valid, 1, time_dim),
                        size=self.seg_len,
                        mode="linear",
                        align_corners=False,
                    ).reshape(n_valid, self.seg_len)
                else:
                    sample_x = sample_x.contiguous()
                sample_power = _compute_band_power(
                    sample_x.unsqueeze(0),
                    sampling_rate=self.power_sampling_rate,
                    band_edges=self.band_edges,
                ).squeeze(0)
                if cache_key is not None:
                    self._input_cache[cache_key] = (
                        sample_x.detach().cpu(),
                        sample_power.detach().cpu(),
                    )
            else:
                sample_x, sample_power = cached
                sample_x = sample_x.to(device=x.device, dtype=x.dtype)
                sample_power = sample_power.to(device=x.device, dtype=x.dtype)

            x_out[batch_idx, valid_mask] = sample_x
            band_power_out[batch_idx, valid_mask] = sample_power

        return x_out, band_power_out

    def prepare_batch(self, batch, **kwargs):
        """Resample each whole window to one BRANT-compatible pseudo-patch."""
        if not isinstance(batch, dict):
            raise TypeError(f"prepare_batch expected dict, got {type(batch).__name__}.")
        if "x" not in batch:
            raise KeyError("prepare_batch expects key 'x'.")

        x = batch["x"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        if x.ndim != 3:
            raise ValueError(
                "BRANT requires raw waveform input with shape (batch, channels, time), got "
                f"{tuple(x.shape)}."
            )

        batch_size, n_channels, _ = map(int, x.shape)
        channel_mask = batch.get("channel_mask")
        if channel_mask is None:
            raise KeyError("Batch is missing required key 'channel_mask'.")
        if not torch.is_tensor(channel_mask):
            channel_mask = torch.as_tensor(channel_mask, dtype=torch.bool)
        else:
            channel_mask = channel_mask.bool()
        if tuple(channel_mask.shape) != (batch_size, n_channels):
            raise ValueError(
                "channel_mask shape must match (batch, channels), got "
                f"{tuple(channel_mask.shape)} vs {(batch_size, n_channels)}."
            )

        x, band_power = self._build_brant_inputs(x, channel_mask, batch)

        out = dict(batch)
        out["x"] = x.contiguous()
        model_kwargs = {
            "channel_mask": channel_mask,
            "band_power": band_power,
        }
        out["model_kwargs"] = model_kwargs
        return out

    def _build_time_encoder(self) -> BrantTimeEncoder:
        return BrantTimeEncoder(
            in_dim=self.seg_len,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            seq_len=self.seq_len,
            n_layer=self.time_ar_layer,
            nhead=self.time_ar_head,
            band_num=self.band_num,
            project_mode=self.project_mode,
            learnable_mask=self.learnable_mask,
            dropout=self.dropout,
        )

    def _build_channel_encoder(self) -> BrantChannelEncoder:
        return BrantChannelEncoder(
            out_dim=self.seg_len,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            n_layer=self.ch_ar_layer,
            nhead=self.ch_ar_head,
            dropout=self.dropout,
        )

    def _load_upstream_weights(self) -> None:
        if self.time_encoder is None or self.channel_encoder is None:
            raise RuntimeError(
                "Upstream encoders must be created before loading weights."
            )
        if self.random_init:
            return
        if self.upstream_ckpt_dir is None:
            raise ValueError(
                "model.upstream_ckpt_dir must be provided when random_init is false."
            )

        time_path = self.upstream_ckpt_dir / "time_encoder.pt"
        channel_path = self.upstream_ckpt_dir / "channel_encoder.pt"
        for path in (time_path, channel_path):
            if not path.is_file():
                raise FileNotFoundError(f"BRANT checkpoint not found: {path}")

        time_state = _unwrap_state_dict(
            torch.load(time_path, map_location="cpu", weights_only=False)
        )
        channel_state = _unwrap_state_dict(
            torch.load(channel_path, map_location="cpu", weights_only=False)
        )

        # Partial loads are intentional: we keep released encoder weights where
        # shapes still match and reinitialize only tensors changed by adaptation.
        self.time_encoder.load_state_dict(
            _matching_state_dict(self.time_encoder, time_state), strict=False
        )
        self.channel_encoder.load_state_dict(
            _matching_state_dict(self.channel_encoder, channel_state), strict=False
        )

    def _create_network(self, input_shape, n_classes):
        if len(input_shape) != 2:
            raise ValueError(
                f"BRANT expects input_shape=(channels, time), got {tuple(input_shape)}."
            )
        if int(input_shape[-1]) != self.seg_len:
            raise ValueError(
                f"BRANT expects resampled time dimension {self.seg_len}, got {input_shape[-1]}."
            )

        self.time_encoder = self._build_time_encoder()
        self.channel_encoder = self._build_channel_encoder()
        self._load_upstream_weights()

        if self.frozen_upstream:
            for param in self.time_encoder.parameters():
                param.requires_grad = False
            for param in self.channel_encoder.parameters():
                param.requires_grad = False

        network = BrantClassifier(
            time_encoder=self.time_encoder,
            channel_encoder=self.channel_encoder,
            n_classes=n_classes,
            d_model=self.d_model,
            seg_len=self.seg_len,
            seq_len=self.seq_len,
            power_sampling_rate=self.power_sampling_rate,
            band_edges=self.band_edges,
            head_type=self.head_type,
            head_hidden_dims=self.head_hidden_dims,
        )
        self.classification_head = network.classification_head
        return network

    def build_model(self, input_shape, n_classes, device=None):
        """Build the BRANT model with the configured backbone/head."""
        self._resolve_device(device)
        self.model = self._create_network(input_shape, n_classes)
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model

    def compute_loss(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        *,
        target_encoded: torch.Tensor | None = None,
        criterion: nn.Module | None = None,
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Apply archive-style token supervision for the BaRISTA token head."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if target_encoded is None:
            target_encoded = target

        if self.model is None or self.head_type != "mlp_token":
            return criterion(outputs, target_encoded)

        token_logits = getattr(self.model, "_last_token_logits", None)
        channel_mask = (
            None if model_kwargs is None else model_kwargs.get("channel_mask")
        )
        if channel_mask is None:
            channel_mask = getattr(self.model, "_last_channel_mask", None)
        if token_logits is None or channel_mask is None:
            return criterion(outputs, target_encoded)

        token_targets = torch.as_tensor(
            target_encoded, dtype=torch.long, device=token_logits.device
        )
        channel_mask = torch.as_tensor(
            channel_mask, dtype=torch.bool, device=token_logits.device
        )
        expanded_targets = token_targets.unsqueeze(1).expand(-1, channel_mask.shape[1])[
            channel_mask
        ]
        if expanded_targets.numel() == 0:
            raise ValueError(
                "BRANT token supervision requires at least one valid token."
            )
        return criterion(token_logits[channel_mask], expanded_targets)

    def get_parameter_groups(self):
        """Expose separate optimizer groups for upstream and classification head."""
        if self.model is None:
            raise RuntimeError(
                "Model must be built before requesting parameter groups."
            )

        upstream_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Match the original BRANT fine-tuning pattern: pretrained encoders
            # use ft_lr, while the randomly initialized downstream head uses lr.
            if name.startswith("classification_head."):
                head_params.append(param)
            else:
                upstream_params.append(param)

        param_groups = []
        if upstream_params:
            param_groups.append({"params": upstream_params, "lr": self.upstream_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr})
        return param_groups

    def forward(self, x, channel_mask=None, band_power=None):
        """Forward pass for direct BRANT module access."""
        return self.model(x, channel_mask=channel_mask, band_power=band_power)

    def predict_proba(self, X):
        """Predict class probabilities while honoring BRANT-specific batch inputs."""
        if self.model is None:
            raise RuntimeError("Model must be built before calling predict_proba.")
        if self.device is None:
            self._resolve_device()
        self.model.eval()

        with torch.no_grad():
            if isinstance(X, dict):
                batch = X if "model_kwargs" in X else self.prepare_batch(X)
                x = batch["x"]
                model_kwargs = dict(batch.get("model_kwargs", {}))
                if not torch.is_tensor(x):
                    x = torch.as_tensor(x, dtype=torch.float32)
                else:
                    x = x.float()
            else:
                x = torch.as_tensor(X, dtype=torch.float32)
                if x.ndim != 3:
                    raise ValueError(
                        "BRANT predict_proba expects either a prepared batch dict or "
                        f"a dense tensor/array with shape (batch, channels, time), got {tuple(x.shape)}."
                    )
                if int(x.shape[-1]) != self.seg_len:
                    raise ValueError(
                        "Dense BRANT predict_proba inputs must be already resampled "
                        f"to seg_len={self.seg_len}; pass a batch dict to use BRANT preprocessing."
                    )
                model_kwargs = {
                    "channel_mask": torch.ones(
                        (x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device
                    ),
                    "band_power": _compute_band_power(
                        x,
                        sampling_rate=self.power_sampling_rate,
                        band_edges=self.band_edges,
                    ),
                }

            x = x.to(self.device)
            for key, value in list(model_kwargs.items()):
                if torch.is_tensor(value):
                    model_kwargs[key] = value.to(self.device)
            outputs = self.model(x, **model_kwargs)
            return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
