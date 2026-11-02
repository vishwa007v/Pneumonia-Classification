"""
Training package for medical image classification models.
Exports:
- Trainer: Main training and validation lifecycle engine.
- compute_classification_metrics: Multi-metric evaluation helper.
"""

from .trainer import Trainer
from .metrics import compute_classification_metrics

__all__ = ["Trainer", "compute_classification_metrics"]
