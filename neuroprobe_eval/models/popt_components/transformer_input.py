"""
Unified transformer input layer that handles both simple and coordinate-based positional encoding.

Matches PopT-BYD-BTB architecture: ONE TransformerEncoderInput class that works for both use cases.
"""

import torch.nn as nn
from .positional_encoding import PositionalEncoding, MultiSubjBrainPositionalEncoding


class TransformerEncoderInput(nn.Module):
    """
    Unified input projection and positional encoding.

    Automatically selects encoding type based on cfg.position_encoding:
    - Default: Simple PositionalEncoding (for feature extraction)
    - "multi_subj_position_encoding": MultiSubjBrainPositionalEncoding (for PopT fine-tuning)

    This matches the PopT-BYD-BTB implementation exactly.
    """

    def __init__(self, cfg, dropout=0.1):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim)

        encoding_name = getattr(cfg, "position_encoding", None)
        if encoding_name == "multi_subj_position_encoding":
            self.positional_encoding = MultiSubjBrainPositionalEncoding(cfg.hidden_dim)
        else:
            self.positional_encoding = PositionalEncoding(cfg.hidden_dim)

        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_specs, positions=None):
        """
        Forward pass with optional positions.

        Args:
            input_specs: (batch, seq, input_dim) input embeddings
            positions: Optional tuple of (coords, seq_id) for multi-subject encoding
                Only used when position_encoding == "multi_subj_position_encoding"

        Returns:
            input_specs: (batch, seq, hidden_dim) processed embeddings
            pos_enc: Positional encoding values (for debugging/visualization)
        """
        input_specs = self.in_proj(input_specs)

        if isinstance(self.positional_encoding, PositionalEncoding):
            input_specs, pos_enc = self.positional_encoding(input_specs)
        else:
            input_specs, pos_enc = self.positional_encoding(
                input_specs, positions=positions
            )

        input_specs = self.layer_norm(input_specs)
        input_specs = self.dropout(input_specs)
        return input_specs, pos_enc
