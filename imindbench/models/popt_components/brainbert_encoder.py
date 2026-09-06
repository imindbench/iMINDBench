"""
Minimal BrainBERT encoder for loading PopT checkpoints.

Uses the shared PopT component stack without coordinate-aware positional
encoding, so it fits naturally alongside the other transformer building blocks.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from imindbench.utils.logging_utils import log

from .transformer_input import TransformerEncoderInput


class BrainBERTEncoder(nn.Module):
    """Minimal BrainBERT encoder for inference (no coordinate encoding)."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        nhead,
        num_layers,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        # No position_encoding field means TransformerEncoderInput falls back to
        # the standard sinusoidal positional encoding path.
        cfg = OmegaConf.create(
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
            }
        )
        self.input_encoding = TransformerEncoderInput(cfg, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # Standard PyTorch convention
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, inputs, src_key_mask=None):
        """Encode one batch of input sequences."""
        # inputs: [batch, seq, input_dim]
        input_specs, _ = self.input_encoding(inputs)
        input_specs = input_specs.transpose(0, 1)  # [seq, batch, dim]
        output_specs = self.transformer(input_specs, src_key_padding_mask=src_key_mask)
        output_specs = output_specs.transpose(0, 1)  # [batch, seq, dim]
        return output_specs

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cpu"):
        """Load model from PopT checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        cfg = checkpoint["model_cfg"]

        # Extract architecture params from config
        model = cls(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            nhead=cfg.nhead,
            num_layers=cfg.encoder_num_layers,
            dim_feedforward=cfg.layer_dim_feedforward,
            dropout=cfg.get("dropout", 0.1),
            activation=cfg.get("layer_activation", "relu"),
        )

        # strict=False is intentional so checkpoint variants with different
        # positional encodings can still reuse the shared transformer weights.
        state_dict = checkpoint["model"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Log what didn't load (expected: positional encoding if using coordinates)
        if missing_keys or unexpected_keys:
            log("BrainBERT checkpoint loading:", priority=1)
            if missing_keys:
                log(f"  Missing keys: {missing_keys[:5]}...", priority=1, indent=1)
            if unexpected_keys:
                log(
                    f"  Unexpected keys: {unexpected_keys[:5]}...", priority=1, indent=1
                )

        return model, cfg
