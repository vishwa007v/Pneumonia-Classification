"""
Custom Convolutional Neural Network (CNN) trained from scratch.

Designed for binary classification of medical chest X-rays (NORMAL vs PNEUMONIA).
Features:
- 4 convolutional blocks with increasing feature depth (32 -> 64 -> 128 -> 256)
- Batch Normalization after each convolution for stable gradient flow
- Global Adaptive Average Pooling for spatial dimension invariance
- Dropout regularization in the classification head to prevent overfitting
"""

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    4-stage CNN architecture optimized for 224x224 Chest X-Ray images.
    
    Architecture:
      Block 1: Conv2d(3, 32, 3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)  [112x112]
      Block 2: Conv2d(32, 64, 3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) [56x56]
      Block 3: Conv2d(64, 128, 3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) [28x28]
      Block 4: Conv2d(128, 256, 3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) [14x14]
      Pooling: AdaptiveAvgPool2d((1, 1))                                         [1x1]
      Head:    Dropout(0.4) -> Linear(256, 128) -> ReLU -> Dropout(0.2) -> Linear(128, num_classes)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

        # Weight initialization (He / Kaiming normal)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits


def build_custom_cnn(num_classes: int = 2, in_channels: int = 3, dropout_rate: float = 0.4) -> CustomCNN:
    """
    Factory helper to construct the CustomCNN model.
    """
    return CustomCNN(in_channels=in_channels, num_classes=num_classes, dropout_rate=dropout_rate)
