import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import streamlit as st

import config

# Ensure project root is in sys.path to access src.*
project_root = str(config.PROJECT_ROOT)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_device() -> torch.device:
    """Returns available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="Loading CXformer-Small model and checkpoint...")
def load_cxformer_model() -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    """
    Loads and caches the CXformerSmall model with restored best checkpoint weights.
    Returns: (model, error_message)
    """
    try:
        from src.models.cxformer import CXformerSmall, resolve_cxformer_dir
    except ImportError as e:
        return None, f"Could not import CXformerSmall from src.models.cxformer: {e}"

    # Verify model dir
    try:
        resolved_model_dir = resolve_cxformer_dir(config.MODEL_DIR)
    except Exception as e:
        return None, f"Model directory verification failed: {e}"

    device = get_device()

    try:
        model = CXformerSmall(
            model_dir=resolved_model_dir,
            num_classes=2,
            freeze_backbone=True,
            dropout_rate=0.3
        )
    except Exception as e:
        return None, f"Failed to instantiate CXformerSmall: {e}"

    # Load checkpoint if available
    ckpt_path = config.CHECKPOINT_PATH
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=True)
            else:
                model.load_state_dict(checkpoint, strict=True)
        except Exception as e:
            return None, f"Failed to load checkpoint from '{ckpt_path}': {e}"
    else:
        return None, f"Checkpoint file not found at '{ckpt_path}'"

    model.to(device)
    model.eval()
    return model, None
