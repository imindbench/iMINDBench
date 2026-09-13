"""
PopT transformer components for fine-tuning.

Unified transformer building blocks that work for both simple and coordinate-based models.
PopT reference: https://github.com/czlwang/PopulationTransformer
"""

from .brainbert_encoder import BrainBERTEncoder
from .positional_encoding import (
    MultiSubjBrainPositionalEncoding,
    PositionalEncoding,
)
from .transformer_input import TransformerEncoderInput
from .transformer_layers import (
    TransformerEncoder,
    TransformerEncoderLayer,
)

__all__ = [
    "BrainBERTEncoder",
    "PositionalEncoding",
    "MultiSubjBrainPositionalEncoding",
    "TransformerEncoderInput",
    "TransformerEncoderLayer",
    "TransformerEncoder",
]
