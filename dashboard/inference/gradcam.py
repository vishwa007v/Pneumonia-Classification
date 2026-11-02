import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class TransformerGradCAM:
    """
    Grad-CAM implementation tailored for the CXformer-Small Vision Transformer.
    Hooks into the penultimate transformer block (model.backbone.encoder.layer[-2])
    under torch.enable_grad() to obtain patch token gradients.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        
        # Penultimate layer (11th block of 12)
        self.target_layer = model.backbone.encoder.layer[-2]

        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output[0] if isinstance(output, tuple) else output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Computes 518x518 normalized Grad-CAM heatmap array in [0, 1].
        Temporarily enables backbone requires_grad for gradient flow through the hook,
        then strictly restores backbone frozen state.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Save previous requires_grad states
        orig_grad_states = [p.requires_grad for p in self.model.backbone.parameters()]

        try:
            # Enable gradients on backbone parameters so backward hook catches gradients
            for p in self.model.backbone.parameters():
                p.requires_grad = True

            with torch.enable_grad():
                output = self.model.backbone(input_tensor)
                tokens = output.last_hidden_state

                cls_features = tokens[:, 0, :]
                logits = self.model.classifier(cls_features)

                if target_class is None:
                    target_class = int(logits.argmax(dim=1).item())

                score = logits[:, target_class].sum()
                score.backward()

            # Discard CLS token (index 0) and 4 register tokens (indices 1-4)
            # Keep 1,369 patch tokens (indices 5 to 1373)
            activations = self.activations[:, 5:, :]
            gradients = self.gradients[:, 5:, :]

            weights = gradients.mean(dim=1, keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=-1))
            cam = cam.reshape(1, 1, 37, 37)

            # Upsample to 518x518
            cam = F.interpolate(cam, size=(518, 518), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()

            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)

            return cam

        finally:
            # Strictly restore original frozen state
            for p, state in zip(self.model.backbone.parameters(), orig_grad_states):
                p.requires_grad = state

    def overlay_heatmap(self, original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """
        Overlays the 518x518 normalized heatmap onto the original image.
        Uses a colormap from blue (low) to red (high).
        """
        import cv2

        resized_orig = original_img.resize((518, 518)).convert("RGB")
        orig_arr = np.array(resized_orig)

        # Colorize heatmap (COLORMAP_JET)
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_cam = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        # Blend
        blended = np.uint8(alpha * colored_cam + (1 - alpha) * orig_arr)
        return Image.fromarray(blended)

    def close(self):
        """Removes forward and backward hooks."""
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()
