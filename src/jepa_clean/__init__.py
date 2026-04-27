"""Clean JEPA curriculum package with VICReg and spatio-temporal masking."""

from .data import WindowedFMRIDataset
from .masking import (
    apply_spatiotemporal_mask,
    apply_temporal_mask,
    sample_spatiotemporal_mask,
    sample_temporal_mask,
)
from .model import JEPA

__all__ = [
    "JEPA",
    "WindowedFMRIDataset",
    "apply_spatiotemporal_mask",
    "apply_temporal_mask",
    "sample_spatiotemporal_mask",
    "sample_temporal_mask",
]
