import sys
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

import config

project_root = str(config.PROJECT_ROOT)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference.model_loader import get_device

_CACHED_TRANSFORM = None

def get_inference_transform():
    """Loads and caches the official CXformer processor transform callable."""
    global _CACHED_TRANSFORM
    if _CACHED_TRANSFORM is None:
        from src.data.dataset import get_cxformer_transform
        _CACHED_TRANSFORM = get_cxformer_transform(model_dir=config.MODEL_DIR)
    return _CACHED_TRANSFORM

def predict_chest_xray(model: torch.nn.Module, image: Image.Image) -> Dict[str, Any]:
    """
    Executes true end-to-end classification on an uploaded chest X-ray.
    Never fabricates or simulates outputs.
    """
    device = get_device()
    transform = get_inference_transform()

    # Preprocess with official CustomBitImageProcessor
    img_tensor = transform(image).unsqueeze(0).to(device)  # Shape: (1, 3, 518, 518)

    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor)  # Shape: (1, 2)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    prob_normal = float(probs[0])
    prob_pneumonia = float(probs[1])
    pred_idx = int(np.argmax(probs))
    pred_class = config.CLASS_NAMES[pred_idx]

    return {
        "predicted_index": pred_idx,
        "predicted_class": pred_class,
        "prob_normal": prob_normal,
        "prob_pneumonia": prob_pneumonia,
        "logits": logits.squeeze(0).cpu().numpy().tolist(),
        "input_tensor": img_tensor
    }
