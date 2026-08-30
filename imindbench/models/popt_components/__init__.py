"""
PopT transformer components for fine-tuning.

Unified transformer building blocks that work for both simple and coordinate-based models.
Matches PopT-BYD-BTB architecture.
"""

from .brainbert_encoder import BrainBERTEncoder
from .positional_encoding import (
    PositionalEncoding,
    MultiSubjBrainPositionalEncoding,
)
from .transformer_input import TransformerEncoderInput
from .transformer_layers import (
    TransformerEncoderLayer,
    TransformerEncoder,
)

__all__ = [
    "BrainBERTEncoder",
    "PositionalEncoding",
    "MultiSubjBrainPositionalEncoding",
    "TransformerEncoderInput",
    "TransformerEncoderLayer",
    "TransformerEncoder",
]
