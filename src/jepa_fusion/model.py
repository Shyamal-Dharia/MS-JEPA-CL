"""JEPA model — re-exports from jepa_clean."""

from ..jepa_clean.model import (
    JEPA,
    Predictor,
    RotaryMultiheadSelfAttention,
    RotaryTransformerEncoderLayer,
    TemporalConvStem,
    TokenEmbedding,
    TransformerEncoder,
    apply_rotary_embedding,
    build_sincos_pos_embed,
)

__all__ = [
    "JEPA",
    "Predictor",
    "RotaryMultiheadSelfAttention",
    "RotaryTransformerEncoderLayer",
    "TemporalConvStem",
    "TokenEmbedding",
    "TransformerEncoder",
    "apply_rotary_embedding",
    "build_sincos_pos_embed",
]
