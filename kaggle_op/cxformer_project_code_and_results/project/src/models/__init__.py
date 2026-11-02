"""
Model definitions package for medical image classification.
Includes:
- CustomCNN: A convolutional neural network designed and trained from scratch.
- ResNet18: ResNet-18 transfer learning model with configurable backbone freezing.
- CXformerSmall: CXformer-Small vision foundation model with configurable backbone freezing.
"""

from .custom_cnn import CustomCNN, build_custom_cnn
from .resnet import ResNet18, build_resnet18
from .cxformer import CXformerSmall, build_cxformer_small

__all__ = [
    "CustomCNN",
    "build_custom_cnn",
    "ResNet18",
    "build_resnet18",
    "CXformerSmall",
    "build_cxformer_small",
]
