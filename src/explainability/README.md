# CXformer-Small explainability module

This folder adds reusable Grad-CAM code to the project repository.

## Files

- `src/explainability/gradcam.py`: computes token-based Grad-CAM from the
  penultimate CXformer transformer block and saves an overlay.
- `src/explainability/__init__.py`: package exports.

## Why the penultimate block?

The classifier uses only the CLS token. In this project's CXformer wrapper,
the final transformer block's patch-token gradients can be zero for the
CLS-only classification score. The penultimate block is therefore used for
the patch-token heatmap, matching the method tested during the Kaggle
explainability work.

## Example integration in a Kaggle notebook

The notebook must already have:
- `model`: a CXformerSmall instance with the trained best checkpoint loaded.
- `test_loader` or an equivalent source of preprocessed image tensors.
- `image_path`: path to the original RGB image.
- `input_tensor`: the same image after the official CXformer processor,
  shaped `(1, 3, 518, 518)`.

```python
from PIL import Image
from src.explainability import (
    compute_cxformer_gradcam,
    save_gradcam_overlay,
)

cam, probabilities, predicted, explained = compute_cxformer_gradcam(
    model=model,
    input_tensor=input_tensor,
)

out = save_gradcam_overlay(
    image=Image.open(image_path).convert("RGB"),
    cam=cam,
    output_path="outputs/gradcam/example_gradcam.png",
)

print("Predicted class:", predicted)
print("Probabilities:", probabilities)
print("Explained class:", explained)
print("Saved:", out)
```

Use the project's existing official `CustomBitImageProcessor` to create
`input_tensor`. Do not substitute a generic resize/normalization pipeline.

## Important interpretation limits

Grad-CAM overlays are qualitative. They do not prove that the model localized
pneumonia, and they are not clinical explanations. Heatmaps from different
images are independently normalized and should not be compared by brightness.
