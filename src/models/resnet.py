"""
ResNet-18 Transfer Learning Model for Medical Image Classification.

Provides:
- ResNet-18 architecture with configurable pretrained weights (ImageNet-1k)
- Replacement of final classification head for binary classification (NORMAL vs PNEUMONIA)
- Configurable backbone parameter freezing (feature extractor vs full fine-tuning)
- Automatic fallback to native PyTorch ResNet-18 when torchvision is not installed
"""

import os
import torch
import torch.nn as nn
from typing import Optional


# ============================================================================
# Native PyTorch ResNet-18 Implementation (Zero-dependency Fallback)
# ============================================================================

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class NativeResNet18(nn.Module):
    """
    Standard PyTorch implementation of ResNet-18 (He et al., 2015).
    Exact layer naming matches the torchvision ResNet-18 state_dict.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ============================================================================
# ResNet-18 Factory and Transfer Learning Wrapper
# ============================================================================

OFFICIAL_RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


class ResNet18(nn.Module):
    """
    Wrapper around ResNet-18 for transfer learning.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # Attempt to use torchvision if installed
        use_torchvision = False
        try:
            import torchvision.models as tv_models
            use_torchvision = True
        except ImportError:
            use_torchvision = False

        if use_torchvision:
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                base_model = tv_models.resnet18(weights=weights)
            except (ImportError, AttributeError):
                base_model = tv_models.resnet18(pretrained=pretrained)
        else:
            base_model = NativeResNet18(num_classes=1000)
            if pretrained:
                try:
                    state_dict = torch.hub.load_state_dict_from_url(
                        OFFICIAL_RESNET18_URL, progress=False, map_location="cpu"
                    )
                    base_model.load_state_dict(state_dict)
                    print("Loaded official PyTorch ImageNet pretrained weights into NativeResNet18.")
                except Exception as e:
                    print(f"Notice: Pretrained weights could not be loaded ({e}). Initialized from scratch.")

        # Replace classification head
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)
        self.model = base_model

        # Optionally freeze backbone layers
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freezes all layers except the final classification head (`fc`)."""
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes all parameters for end-to-end fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> ResNet18:
    """
    Factory helper to construct the ResNet-18 model for transfer learning.
    """
    return ResNet18(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
