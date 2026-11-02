"""
CXformer-Small Transfer Learning Model Wrapper.

Wraps the pretrained CXformer-Small vision foundation model (Dinov2WithRegistersModel)
for medical chest X-ray binary classification (NORMAL vs PNEUMONIA).

Key Design Features:
- Loads official pretrained CXformer-Small checkpoint from configurable directory
- Freezes all 22,058,112 backbone parameters (requires_grad = False)
- Enforces backbone evaluation mode even when wrapper is in training mode
- Uses torch.no_grad() during forward pass to eliminate activation memory overhead
- Extracts normalized CLS token feature embedding (dim 384)
- Newly initialized classification head: LayerNorm(384) -> Dropout(0.3) -> Linear(384, 2)
- Returns unnormalized raw logits of shape (batch_size, 2) compatible with Trainer
"""

import os
import sys
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_cxformer_dir(model_dir: Optional[str] = None) -> str:
    """
    Resolves the CXformer-Small model directory across local and Kaggle environments.

    If model_dir is explicitly provided:
      Validates that exact directory. Requires config.json and model.safetensors.
      Raises FileNotFoundError identifying the supplied path and missing files if invalid.
      Does NOT fall back to other locations.

    If model_dir is None:
      Performs automatic discovery searching standard local and Kaggle mount points.
    """
    if model_dir is not None:
        # Resolve relative paths against current working directory or PROJECT_ROOT
        candidates = [
            os.path.abspath(model_dir),
            os.path.abspath(os.path.join(PROJECT_ROOT, model_dir)),
        ]

        target_dir = None
        for cand in candidates:
            if os.path.isdir(cand):
                target_dir = cand
                break

        if target_dir is None:
            raise FileNotFoundError(
                f"Explicitly provided model directory does not exist: '{model_dir}' "
                f"(checked '{candidates[0]}' and '{candidates[1]}')."
            )

        missing_files = []
        if not os.path.isfile(os.path.join(target_dir, "config.json")):
            missing_files.append("config.json")
        if not os.path.isfile(os.path.join(target_dir, "model.safetensors")):
            missing_files.append("model.safetensors")

        if missing_files:
            raise FileNotFoundError(
                f"Explicitly provided model directory '{target_dir}' is missing "
                f"required file(s): {', '.join(missing_files)}."
            )

        return target_dir

    # Automatic path discovery when model_dir is None
    candidates = [
        os.path.join(PROJECT_ROOT, "CXformer_small_files"),
        os.path.abspath("CXformer_small_files"),
        "/kaggle/input/datasets/vishwad007v/cxformer-small-files",
        "/kaggle/input/cxformer-small-files",
        "/kaggle/input/cxformer-small",
        "/kaggle/input/cxformer-small-files/CXformer_small_files",
    ]

    # Dynamic search if /kaggle/input exists
    if os.path.isdir("/kaggle/input"):
        try:
            for root, dirs, files in os.walk("/kaggle/input"):
                if "model.safetensors" in files and "config.json" in files:
                    candidates.append(root)
                    break
        except Exception:
            pass

    for cand in candidates:
        if (
            cand
            and os.path.isdir(cand)
            and os.path.isfile(os.path.join(cand, "model.safetensors"))
            and os.path.isfile(os.path.join(cand, "config.json"))
        ):
            return os.path.abspath(cand)

    searched_str = "\n  - ".join([c for c in candidates if isinstance(c, str)][:6])
    raise FileNotFoundError(
        f"Could not locate valid CXformer-Small checkpoint directory via auto-discovery.\n"
        f"Searched locations:\n  - {searched_str}\n"
        f"Please provide a valid path explicitly via model_dir argument."
    )


class CXformerSmall(nn.Module):
    """
    CXformer-Small Transfer Learning Architecture for Chest X-Ray Classification.

    Pipeline:
      Input (B, 3, 518, 518) -> Dinov2WithRegistersModel (Frozen) -> CLS Token (B, 384)
      -> LayerNorm(384) -> Dropout(0.3) -> Linear(384, num_classes) -> Logits (B, num_classes)
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate

        # 1. Resolve and validate model directory
        self.resolved_model_dir = resolve_cxformer_dir(model_dir)

        # 2. Inspect configuration
        config = AutoConfig.from_pretrained(self.resolved_model_dir)
        self.hidden_dim = getattr(config, "hidden_size", 384)
        if self.hidden_dim != 384:
            raise ValueError(
                f"Architecture mismatch: expected hidden_size=384, but got {self.hidden_dim} "
                f"from config at {self.resolved_model_dir}"
            )

        # 3. Load pretrained backbone
        self.backbone = AutoModel.from_pretrained(self.resolved_model_dir)

        # 4. Freeze backbone if configured
        if self.freeze_backbone:
            self._freeze_backbone()

        # 5. Build classification head: LayerNorm -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.hidden_dim, num_classes),
        )

        # 6. Initialize classification head
        self._init_head()

    def _init_head(self):
        """Initializes weights of the classification head."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _freeze_backbone(self):
        """Freezes all backbone parameters and switches backbone to evaluation mode."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def freeze_backbone_layers(self):
        """Freezes all backbone parameters and locks backbone into evaluation mode."""
        self.freeze_backbone = True
        self._freeze_backbone()

    def unfreeze_backbone(self):
        """
        Unfreezes all backbone parameters for end-to-end fine-tuning.
        Synchronizes backbone training/evaluation mode with the wrapper's current mode.
        """
        self.freeze_backbone = False
        for param in self.backbone.parameters():
            param.requires_grad = True

        if self.training:
            self.backbone.train(True)
        else:
            self.backbone.eval()

    def train(self, mode: bool = True):
        """
        Sets the module in training mode.
        When freeze_backbone is True, enforces evaluation mode on the backbone
        while allowing the classification head to operate in training mode.
        When freeze_backbone is False, standard training mode is propagated to
        both the backbone and the classification head.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass.

        Args:
            x: Input image tensor of shape (batch_size, 3, 518, 518).

        Returns:
            logits: Unnormalized raw logits of shape (batch_size, num_classes).
        """
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(x)
        else:
            outputs = self.backbone(x)

        # Extract CLS token at sequence index 0: shape (batch_size, 384)
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Classification head -> shape (batch_size, num_classes)
        logits = self.classifier(cls_token)
        return logits


def build_cxformer_small(
    model_dir: Optional[str] = None,
    num_classes: int = 2,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
) -> CXformerSmall:
    """
    Factory helper to construct the CXformerSmall transfer learning model.
    """
    return CXformerSmall(
        model_dir=model_dir,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
    )
