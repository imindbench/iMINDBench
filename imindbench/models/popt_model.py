"""
PopT fine-tuning model for imindbench.

Loads pretrained PopT transformer weights and performs fine-tuning
with optional freezing of upstream encoder.
"""

import warnings

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf
from . import register_model
from imindbench.utils.logging_utils import log
from .torch_base_model import TorchBaseModel

from imindbench.models.popt_components.transformer_input import (
    TransformerEncoderInput,
)
from imindbench.models.popt_components.transformer_layers import (
    TransformerEncoderLayer,
    TransformerEncoder,
)


class PopTUpstreamTransformer(nn.Module):
    """
    Upstream PopT transformer for fine-tuning.
    Compatible with PopT checkpoint format.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_encoding = TransformerEncoderInput(cfg, dropout=0.1)
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_head,
            dim_feedforward=cfg.get("layer_dim_feedforward", cfg.hidden_dim * 4),
            activation=cfg.get("layer_activation", "gelu"),
            dropout=0.1,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers
        )

    def forward(self, inputs, src_key_mask, positions, intermediate_rep=False):
        """
        Forward pass through transformer.

        Args:
            inputs: (batch, seq_len, input_dim) input embeddings
            src_key_mask: (batch, seq_len) padding mask (False = valid, True = pad)
            positions: Tuple of (coords, seq_id) for positional encoding
            intermediate_rep: If True, return intermediate representations

        Returns:
            If intermediate_rep: (batch, seq_len, hidden_dim)
            Otherwise: (batch, seq_len, hidden_dim), attention_weights
        """
        input_specs, _ = self.input_encoding(inputs, positions)
        input_specs = input_specs.transpose(0, 1)  # [seq, batch, dim]
        output_specs, weights = self.transformer_encoder(
            input_specs, src_key_padding_mask=src_key_mask
        )
        output_specs = output_specs.transpose(0, 1)  # [batch, seq, dim]

        if intermediate_rep:
            return output_specs
        return output_specs, weights


@register_model("popt")
class PopTModel(TorchBaseModel):
    """
    PopT fine-tuning model for neuroprobe classification.

    Loads pretrained PopT transformer weights and adds a classification head.
    Supports both frozen encoder (feature extraction) and full fine-tuning modes.
    Can use randomly initialized weights instead of pretrained weights via random_init flag.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.accepts_coords = True
        self.upstream_ckpt = cfg.get("upstream_ckpt", None)
        if self.upstream_ckpt is None:
            raise ValueError("upstream_ckpt must be provided in model config")

        self.random_init = cfg.get("random_init", False)
        self.frozen_upstream = cfg.get("frozen_upstream", False)
        self.upstream_lr = cfg.get("upstream_lr", 1e-5)
        self.head_lr = cfg.get("head_lr", 1e-3)
        self.head_input_mode = self._resolve_head_input_mode(cfg)

        self.upstream = None
        self.classification_head = None

    def prepare_batch(self, batch, **kwargs):
        """Prepare collated batch into PopT-compatible inputs."""
        if not isinstance(batch, dict):
            raise TypeError(f"prepare_batch expected dict, got {type(batch).__name__}.")
        if "x" not in batch or "y" not in batch:
            raise KeyError("Batch must include 'x' and 'y' keys.")

        x = batch["x"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        if x.ndim < 3:
            raise ValueError(
                f"Expected x with shape (batch, channels, ...), got {tuple(x.shape)}."
            )
        batch_size, n_channels = int(x.shape[0]), int(x.shape[1])
        x = x.reshape(batch_size, n_channels, -1)

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
                f"{tuple(channel_mask.shape)} vs ({batch_size}, {n_channels})."
            )
        channel_mask = channel_mask.to(device=x.device)
        pad_mask = ~channel_mask

        coords = batch.get("channel_coords")
        if coords is None:
            raise ValueError("PopT variable-channel mode requires channel_coords.")
        if not torch.is_tensor(coords):
            coords = torch.as_tensor(coords, dtype=torch.float32)
        else:
            coords = coords.float()
        if tuple(coords.shape) != (batch_size, n_channels, 3):
            raise ValueError(
                "channel_coords shape must match (batch, channels, 3), got "
                f"{tuple(coords.shape)} vs ({batch_size}, {n_channels}, 3)."
            )
        coords = coords.to(device=x.device)

        seq_id = batch.get("seq_id")
        if seq_id is None:
            raise ValueError("PopT variable-channel mode requires seq_id.")
        if not torch.is_tensor(seq_id):
            seq_id = torch.as_tensor(seq_id, dtype=torch.long)
        else:
            seq_id = seq_id.long()
        if tuple(seq_id.shape) != (batch_size, n_channels):
            raise ValueError(
                "seq_id shape must match (batch, channels), got "
                f"{tuple(seq_id.shape)} vs ({batch_size}, {n_channels})."
            )
        seq_id = seq_id.to(device=x.device)

        runner_cfg = kwargs.get("runner_cfg", {}) or {}
        coord_index_policy = str(runner_cfg.get("coord_index_policy", "round_clamp"))
        valid_coord_policies = {"round_clamp", "legacy_truncation"}
        if coord_index_policy not in valid_coord_policies:
            raise ValueError(
                "runner.coord_index_policy must be one of "
                f"{sorted(valid_coord_policies)}, got '{coord_index_policy}'."
            )

        position_table_max_len = int(self.cfg.get("position_table_max_len", 5000))
        if position_table_max_len < 1:
            raise ValueError(
                "model.position_table_max_len must be >= 1, got "
                f"{position_table_max_len}."
            )

        if coord_index_policy == "legacy_truncation":
            coords_idx = coords.long()
        else:
            rounded_coords = torch.round(coords)
            max_idx = position_table_max_len - 1
            out_of_bounds_mask = (rounded_coords < 0) | (rounded_coords > max_idx)
            if bool(torch.any(out_of_bounds_mask).item()):
                n_oob = int(out_of_bounds_mask.sum().item())
                warnings.warn(
                    f"Clamping {n_oob} rounded coordinate values to [0, {max_idx}] "
                    "for PopT positional indexing.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            coords_idx = torch.clamp(rounded_coords, min=0, max=max_idx).long()

        # PopT expects a learned-style CLS slot at position 0, so evaluation
        # builds that token explicitly before the upstream encoder.
        cls_token = torch.ones(
            (batch_size, 1, int(x.shape[-1])), dtype=x.dtype, device=x.device
        )
        x_with_cls = torch.cat([cls_token, x], dim=1)
        cls_pad_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        pad_mask_with_cls = torch.cat([cls_pad_mask, pad_mask], dim=1)

        out = dict(batch)
        out["x"] = x_with_cls
        out["channel_coords"] = coords_idx
        out["seq_id"] = seq_id
        out["model_kwargs"] = {"pad_mask": pad_mask_with_cls}
        return out

    @staticmethod
    def _resolve_head_input_mode(cfg: DictConfig) -> str:
        """Resolve and validate PopT head input mode."""
        if "flatten_all_output_tokens" in cfg:
            raise ValueError(
                "flatten_all_output_tokens has been removed. "
                "Use head_input_mode (cls_only | flatten_output_tokens | cls_plus_input_residual)."
            )

        valid_modes = {
            "cls_only",
            "flatten_output_tokens",
            "cls_plus_input_residual",
        }
        mode = str(cfg.get("head_input_mode", "cls_only"))
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid head_input_mode: {mode}. "
                f"Expected one of: {sorted(valid_modes)}"
            )
        return mode

    def _load_upstream_from_checkpoint(self, checkpoint_path):
        """Load the upstream transformer from PopT or torch_brain checkpoints."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Checkpoint must be a dict, got {type(checkpoint)} from {checkpoint_path}"
            )

        if "model_cfg" in checkpoint and "model" in checkpoint:
            # Native PopT checkpoint format (PopT-BYD-BTB compatible).
            model_cfg = checkpoint["model_cfg"]
            state_dict = checkpoint["model"]
        elif "model_state" in checkpoint and "config" in checkpoint:
            # torch_brain phase6 pretraining checkpoint format.
            config = checkpoint["config"]
            if "model" not in config:
                raise ValueError(
                    "Checkpoint has 'model_state' but missing 'config.model' for model_cfg"
                )
            model_cfg = OmegaConf.create(config["model"])
            state_dict = self._remap_torch_brain_pretrain_keys(
                checkpoint["model_state"]
            )
            log(
                "[PopTModel] Auto-converted torch_brain pretraining checkpoint "
                "format (model_state/config -> model/model_cfg) for imindbench",
                priority=1,
            )
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected either keys "
                "('model_cfg' and 'model') or ('model_state' and 'config')."
            )

        # Rebuild the upstream module from checkpoint config so the classifier
        # head can be attached on top of the exact pretrained token encoder.
        upstream = PopTUpstreamTransformer(model_cfg)

        if not self.random_init:
            # strict=False preserves compatibility across checkpoint variants
            # that may carry extra keys outside the reused transformer stack.
            missing_keys, unexpected_keys = upstream.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                log(
                    f"[PopTModel] Warning: Missing keys when loading checkpoint: {missing_keys[:5]}...",
                    priority=1,
                )
            if unexpected_keys:
                log(
                    f"[PopTModel] Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...",
                    priority=1,
                )
            log("[PopTModel] Loaded pretrained checkpoint weights", priority=1)
        else:
            log("[PopTModel] Using RANDOMLY INITIALIZED weights", priority=1)

        return upstream, model_cfg

    @staticmethod
    def _remap_torch_brain_pretrain_keys(state_dict):
        """Map torch_brain phase6 key namespace to PopT namespace expected here."""
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Expected checkpoint 'model_state' to be a dict, got {type(state_dict)}"
            )

        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("input_layer."):
                new_key = "input_encoding." + new_key[len("input_layer.") :]
            elif new_key.startswith("encoder."):
                new_key = "transformer_encoder." + new_key[len("encoder.") :]

            if new_key in remapped:
                raise ValueError(
                    f"Checkpoint key collision while remapping: {key} -> {new_key}"
                )
            remapped[new_key] = value

        return remapped

    def _create_network(self, input_shape, n_classes):
        """Create the PopT model architecture."""
        self.upstream, upstream_cfg = self._load_upstream_from_checkpoint(
            self.upstream_ckpt
        )

        if self.frozen_upstream:
            for param in self.upstream.parameters():
                param.requires_grad = False
            log("[PopTModel] Upstream transformer frozen", priority=1)
        else:
            log(
                f"[PopTModel] Upstream transformer trainable (lr={self.upstream_lr})",
                priority=1,
            )

        hidden_dim = upstream_cfg.hidden_dim
        total_seq_len = int(input_shape[0])
        input_token_dim = int(input_shape[-1])
        expected_non_cls_tokens = max(0, total_seq_len - 1)

        if self.head_input_mode == "flatten_output_tokens":
            head_in_features = total_seq_len * hidden_dim
            log(
                "[PopTModel] Classification head input mode: flatten_output_tokens "
                f"(seq_len={total_seq_len}, hidden_dim={hidden_dim}, in_features={head_in_features})",
                priority=1,
            )
        elif self.head_input_mode == "cls_plus_input_residual":
            residual_dim = expected_non_cls_tokens * input_token_dim
            head_in_features = hidden_dim + residual_dim
            log(
                "[PopTModel] Classification head input mode: cls_plus_input_residual "
                "(cls from PopT output + flattened original non-CLS input tokens) "
                f"(seq_len={total_seq_len}, hidden_dim={hidden_dim}, "
                f"input_token_dim={input_token_dim}, residual_dim={residual_dim}, "
                f"in_features={head_in_features})",
                priority=1,
            )
        else:
            head_in_features = hidden_dim
            log(
                "[PopTModel] Classification head input mode: cls_only "
                f"(hidden_dim={hidden_dim})",
                priority=1,
            )

        self.classification_head = nn.Linear(head_in_features, n_classes)

        class PopTNetwork(nn.Module):
            def __init__(
                self,
                upstream,
                classification_head,
                head_input_mode,
                expected_non_cls_tokens,
                input_token_dim,
            ):
                super().__init__()
                self.upstream = upstream
                self.classification_head = classification_head
                self.head_input_mode = head_input_mode
                self.expected_non_cls_tokens = expected_non_cls_tokens
                self.input_token_dim = input_token_dim

            def forward(self, x, positions=None, pad_mask=None):
                """
                Forward pass.

                Args:
                    x: (batch, n_electrodes+1, hidden_dim) with CLS at index 0
                    positions: Optional tuple of (coords, seq_id) for positional encoding
                    pad_mask: Optional (batch, seq_len) bool mask (True = padding)

                Returns:
                    logits: (batch, n_classes)
                """
                batch_size, seq_len = x.shape[0], x.shape[1]

                if pad_mask is None:
                    pad_mask = torch.zeros(
                        batch_size, seq_len, dtype=torch.bool, device=x.device
                    )
                else:
                    pad_mask = torch.as_tensor(
                        pad_mask, dtype=torch.bool, device=x.device
                    )
                    if tuple(pad_mask.shape) != (batch_size, seq_len):
                        raise ValueError(
                            "pad_mask shape mismatch: expected "
                            f"({batch_size}, {seq_len}), got {tuple(pad_mask.shape)}."
                        )

                if positions is not None:
                    outputs = self.upstream(
                        x, pad_mask, positions, intermediate_rep=True
                    )
                else:
                    # Fallback if positions not provided (shouldn't happen for PopT)
                    outputs = self.upstream(x, pad_mask, None, intermediate_rep=True)

                if outputs.ndim != 3:
                    raise ValueError(
                        f"Expected upstream output with shape (batch, seq_len, hidden_dim), got {tuple(outputs.shape)}"
                    )
                if outputs.shape[1] == 0:
                    raise ValueError("Upstream output sequence length must be >= 1")

                if self.head_input_mode == "flatten_output_tokens":
                    features = outputs.reshape(outputs.shape[0], -1)
                    expected_features = self.classification_head.in_features
                    actual_features = features.shape[1]
                    if actual_features != expected_features:
                        raise ValueError(
                            "Flattened output feature size mismatch: "
                            f"expected {expected_features}, got {actual_features}. "
                            "Ensure a consistent token count across train/val/test when "
                            "head_input_mode=flatten_output_tokens."
                        )
                elif self.head_input_mode == "cls_only":
                    # Extract CLS token (index 0)
                    features = outputs[:, 0, :]
                elif self.head_input_mode == "cls_plus_input_residual":
                    input_tokens = x[:, 1:, :]
                    if input_tokens.shape[1] != self.expected_non_cls_tokens:
                        raise ValueError(
                            "Input residual token count mismatch: "
                            f"expected {self.expected_non_cls_tokens}, got {input_tokens.shape[1]}. "
                            "Ensure a consistent token count across train/val/test when "
                            "head_input_mode=cls_plus_input_residual."
                        )
                    if input_tokens.shape[2] != self.input_token_dim:
                        raise ValueError(
                            "Input residual token dim mismatch: "
                            f"expected {self.input_token_dim}, got {input_tokens.shape[2]}."
                        )

                    cls_features = outputs[:, 0, :]
                    residual_features = input_tokens.reshape(input_tokens.shape[0], -1)
                    features = torch.cat([cls_features, residual_features], dim=1)
                    expected_features = self.classification_head.in_features
                    actual_features = features.shape[1]
                    if actual_features != expected_features:
                        raise ValueError(
                            "CLS+input residual feature size mismatch: "
                            f"expected {expected_features}, got {actual_features}. "
                            "Ensure consistent input token count and dimension."
                        )
                else:
                    raise ValueError(
                        f"Unsupported head_input_mode: {self.head_input_mode}"
                    )

                logits = self.classification_head(features)
                return logits

        return PopTNetwork(
            self.upstream,
            self.classification_head,
            self.head_input_mode,
            expected_non_cls_tokens,
            input_token_dim,
        )

    def build_model(self, input_shape, n_classes, device=None):
        """Build the model with given input shape and number of classes."""
        self._resolve_device(device)
        self.model = self._create_network(input_shape, n_classes)
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model

    def get_parameter_groups(self):
        """
        Return separate parameter groups for different learning rates.

        Returns:
            List of parameter group dicts for optimizer
        """
        upstream_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "upstream" in name:
                    upstream_params.append(param)
                elif "classification_head" in name:
                    head_params.append(param)

        param_groups = []
        if upstream_params:
            param_groups.append({"params": upstream_params, "lr": self.upstream_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr})

        return param_groups

    def forward(self, x, positions=None, pad_mask=None):
        """
        Forward pass (for direct model calls).

        Args:
            x: (batch, n_electrodes+1, hidden_dim) with CLS at index 0
            positions: Optional tuple of (coords, seq_id) for positional encoding
            pad_mask: Optional (batch, seq_len) bool mask (True = padding)

        Returns:
            logits: (batch, n_classes)
        """
        return self.model(x, positions=positions, pad_mask=pad_mask)
