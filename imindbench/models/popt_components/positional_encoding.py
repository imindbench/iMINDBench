"""
Positional encoding variants for transformer models.

Contains both simple sinusoidal and multi-subject brain coordinate encodings.
Adapted from PopulationTransformer's models/transformer_encoder_input.py.
Source and license status: THIRD_PARTY.md (PopulationTransformer).
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    From: https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/2
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        _ = dropout
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, seq):
        """
        Apply positional encoding to sequence.

        Args:
            seq: (batch, len, dim) input sequence

        Returns:
            out: (batch, len, dim) sequence with positional encoding added
            pos_enc: (batch, len, dim) positional encoding values
        """
        # seq is [batch, len, dim]
        assert len(seq.shape) == 3
        pos_enc = self.pe[:, : seq.size(1), :]
        out = seq + pos_enc
        return out, pos_enc


class MultiSubjBrainPositionalEncoding(nn.Module):
    """
    Multi-subject brain coordinate-based positional encoding.

    Uses PopT coordinate slots [L, I, P] and sequence ID to create 4-part positional embeddings:
    - L coordinate embedding
    - I coordinate embedding
    - P coordinate embedding
    - Sequence ID embedding

    The hidden dimension must be divisible by 4 (each component gets d_model/4 dimensions).
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        _ = dropout

        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4"
        # Each component (L, I, P, seq_id) gets one quarter of the hidden space.
        pe_dim = int(d_model / 4)

        pe = torch.zeros(max_len, pe_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, pe_dim, 2).float() * (-math.log(10000.0) / pe_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.max_len = max_len

    def forward(self, seq, positions):
        """
        Apply coordinate-based positional encoding to sequence.

        Args:
            seq: (batch, len, dim) input sequence (includes CLS token)
            positions: Tuple of (coords, seq_id)
                - coords: (batch, seq_len-1, 3) PopT [L, I, P] coordinates (without CLS)
                - seq_id: (batch, seq_len-1) sequence identifiers (without CLS)

        Returns:
            out: (batch, len, dim) sequence with positional encoding added
            input_embeddings: (batch, len, dim) positional encoding values
        """
        # seq is [batch, len, dim]
        assert len(seq.shape) == 3
        coords, seq_id = positions

        p_embed = self.pe[0, coords]  # Shape: (batch, seq_len-1, 3, pe_dim)
        n_batch, seq_len, n_axes, d_p_embed = p_embed.shape
        p_embed = p_embed.reshape(n_batch, seq_len, n_axes * d_p_embed)

        seq_id_embed = self.pe[0, seq_id]  # (batch, seq_len-1, pe_dim)
        input_embeddings = torch.cat(
            [p_embed, seq_id_embed], axis=-1
        )  # (batch, seq_len-1, d_model)

        batch_size, _, d_embed = seq.shape
        cls_embed = torch.unsqueeze(self.pe[0, 0].repeat(batch_size, 4), 1)

        input_embeddings = torch.cat([cls_embed, input_embeddings], axis=1)

        out = seq + input_embeddings
        return out, input_embeddings
