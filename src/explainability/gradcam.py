"""
Grad-CAM utilities for the project's CXformer-Small classifier.

Important:
- CXformerSmall.forward() intentionally wraps the frozen backbone in torch.no_grad().
- For Grad-CAM, this module calls the same backbone directly with gradients enabled,
  then applies the model's existing classifier head.
- The penultimate transformer block is used because the classifier consumes the CLS
  token only; gradients at the final block's patch tokens can be zero.
- This is a qualitative attention/attribution visualization, not a validated
  localization or clinical explanation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _first_tensor(output):
    """Handle transformer blocks that return either a tensor or a tuple."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        "Unexpected transformer block output. Expected a tensor or a tuple "
        "whose first element is a tensor."
    )


def compute_cxformer_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer=None,
):
    """
    Compute token-based Grad-CAM for one CXformer-Small image.

    Args:
        model: Loaded CXformerSmall instance with trained checkpoint weights.
        input_tensor: Preprocessed image tensor shaped (1, 3, 518, 518).
        target_class: Class index to explain. If None, uses the predicted class.
        target_layer: Optional transformer block. Defaults to penultimate block.

    Returns:
        cam: Float32 NumPy heatmap in [0, 1], resized to input image resolution.
        probabilities: NumPy array of class probabilities.
        predicted_class: Integer predicted class index.
        explained_class: Integer class whose score was explained.
    """
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError(
            f"Expected input shape (1, 3, H, W); got {tuple(input_tensor.shape)}"
        )

    if not hasattr(model, "backbone") or not hasattr(model, "classifier"):
        raise AttributeError(
            "Expected CXformerSmall with 'backbone' and 'classifier' attributes."
        )

    encoder_layers = getattr(
        getattr(model.backbone, "encoder", None), "layer", None
    )
    if encoder_layers is None or len(encoder_layers) < 2:
        raise AttributeError(
            "Could not find backbone.encoder.layer with at least two blocks."
        )

    if target_layer is None:
        target_layer = encoder_layers[-2]

    device = next(model.parameters()).device
    model.eval()
    x = input_tensor.to(device).detach().requires_grad_(True)

    captured = {}

    def forward_hook(_module, _inputs, output):
        activation = _first_tensor(output)
        captured["activation"] = activation
        activation.register_hook(lambda grad: captured.__setitem__("gradient", grad))

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        # Do not call model(x): its frozen-backbone forward path uses no_grad().
        # This direct path preserves the exact backbone and classifier weights.
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            backbone_output = model.backbone(x)
            last_hidden_state = backbone_output.last_hidden_state
            cls_token = last_hidden_state[:, 0, :]
            logits = model.classifier(cls_token)

            probabilities = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
            predicted_class = int(np.argmax(probabilities))
            explained_class = (
                predicted_class if target_class is None else int(target_class)
            )
            if not 0 <= explained_class < logits.shape[1]:
                raise ValueError(f"target_class {explained_class} is out of range.")

            score = logits[:, explained_class].sum()
            score.backward()

        if "activation" not in captured or "gradient" not in captured:
            raise RuntimeError("Grad-CAM hook did not capture activations and gradients.")

        activations = captured["activation"].detach()  # (1, tokens, channels)
        gradients = captured["gradient"].detach()

        if activations.ndim != 3 or gradients.shape != activations.shape:
            raise RuntimeError(
                "Expected matching (batch, tokens, channels) activation/gradient "
                f"tensors; got {tuple(activations.shape)} and {tuple(gradients.shape)}."
            )

        # CXformer-Small token layout: CLS (1) + registers (4) + patch tokens (37*37).
        patch_activations = activations[:, 5:, :]
        patch_gradients = gradients[:, 5:, :]

        token_count = patch_activations.shape[1]
        grid_size = int(round(token_count ** 0.5))
        if grid_size * grid_size != token_count:
            raise RuntimeError(
                f"Patch token count {token_count} is not a square grid. "
                "Check the CXformer input resolution/token layout."
            )

        # Grad-CAM: channel weights are mean gradients over patch locations.
        channel_weights = patch_gradients.mean(dim=1, keepdim=True)
        cam_tokens = torch.relu(
            (patch_activations * channel_weights).sum(dim=-1)
        )
        cam = cam_tokens.reshape(1, 1, grid_size, grid_size)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        cam_np = cam.detach().cpu().numpy().astype(np.float32)

        return cam_np, probabilities, predicted_class, explained_class
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def save_gradcam_overlay(
    image: Union[str, Path, Image.Image, np.ndarray],
    cam: np.ndarray,
    output_path: Union[str, Path],
    alpha: float = 0.40,
):
    """
    Save an RGB image with a Grad-CAM heatmap overlay.

    `image` should be the original, unnormalized RGB image (not the normalized
    model tensor). The heatmap is resized to the image dimensions if necessary.
    """
    if isinstance(image, (str, Path)):
        base = Image.open(image).convert("RGB")
        base_np = np.asarray(base)
    elif isinstance(image, Image.Image):
        base_np = np.asarray(image.convert("RGB"))
    else:
        base_np = np.asarray(image)
        if base_np.ndim != 3 or base_np.shape[2] != 3:
            raise ValueError("image array must have shape (H, W, 3).")
        if base_np.dtype != np.uint8:
            base_np = np.clip(base_np, 0, 255).astype(np.uint8)

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to save a colored Grad-CAM overlay. "
            "Install opencv-python-headless."
        ) from exc

    height, width = base_np.shape[:2]
    heat = cv2.resize(cam.astype(np.float32), (width, height))
    heat = np.clip(heat, 0.0, 1.0)
    heat_uint8 = np.uint8(255 * heat)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        base_np.astype(np.uint8), 1.0 - alpha, heat_rgb, alpha, 0
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    return output_path
