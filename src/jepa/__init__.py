# src/jepa/__init__.py

from .dataset import FMRIDataset
from .masking import generate_temporal_mask, generate_spatial_mask
from .model import JEPA
from .train_phase1 import train_one_model
