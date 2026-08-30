"""
Custom transformer encoder layers that return attention weights.

A simpler implementation of nn.TransformerEncoder that allows for attention weights to be returned.
From: https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/
"""

import copy
import torch.nn as nn
import torch.nn.functional as F


def _get_activation_fn(activation):
    """Get activation function by name."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module, n_layers):
    """Create ``n_layers`` deep copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer that returns attention weights.

    Similar to nn.TransformerEncoderLayer but returns attention weights for visualization.
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        """Handle state loading for backward compatibility."""
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass through encoder layer.

        Args:
            src: (seq_len, batch, d_model) input sequence
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask (True = pad, False = valid)

        Returns:
            src: (seq_len, batch, d_model) output sequence
            weights: Attention weights for visualization
        """
        src2, weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            average_attn_weights=False,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights


class TransformerEncoder(nn.Module):
    """
    Transformer encoder that returns attention weights from all layers.

    Similar to nn.TransformerEncoder but collects attention weights from each layer.
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Forward pass through encoder.

        Args:
            src: (seq_len, batch, d_model) input sequence
            mask: Optional attention mask
            src_key_padding_mask: Optional padding mask

        Returns:
            output: (seq_len, batch, d_model) output sequence
            weights: List of attention weights from each layer
        """
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights
