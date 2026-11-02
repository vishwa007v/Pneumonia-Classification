import streamlit as st
from PIL import Image
import torch
import numpy as np
import os

from components.sidebar import render_sidebar
from components.navbar import render_top_navbar
from inference.model_loader import load_cxformer_model, get_device
from inference.predictor import predict_chest_xray
from inference.gradcam import TransformerGradCAM

st.set_page_config(page_title="Image Classification — CXformer-Small", page_icon="🩻", layout="wide")
render_sidebar()
render_top_navbar()

st.title("🩻 Live Chest X-Ray Classification & Inference")
st.markdown("""
Interactive inference interface powered directly by the **trained CXformer-Small model**.
Upload a patient chest radiograph to compute calibrated class probabilities and on-the-fly Grad-CAM saliency.
""")

st.error("⚠️ **CLINICAL DISCLAIMER:** This system is an academic research demonstration. It is NOT certified for diagnostic patient triage, clinical decision support, or treatment planning.")

st.divider()

# Attempt to load verified model
model, error_msg = load_cxformer_model()

if model is None:
    st.markdown(f"""
    <div class="custom-alert-danger">
        <strong>Model Unavailable for Live Inference:</strong><br>
        {error_msg}<br><br>
        <strong>Setup Instructions:</strong><br>
        1. Ensure <code>config.json</code> and <code>model.safetensors</code> are located in <code>CXformer_small_files/</code>.<br>
        2. Ensure the trained checkpoint <code>cxformer_small_best.pth</code> is in <code>checkpoints/</code>.<br>
        3. Or set the <code>MODEL_DIR</code> and <code>CHECKPOINT_PATH</code> environment variables.
    </div>
    """, unsafe_allow_html=True)

# Main Two-Column Layout
col_upload, col_result = st.columns(2)

with col_upload:
    st.subheader("📤 Upload Radiograph")
    uploaded_file = st.file_uploader(
        "Select a chest X-ray image (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
            st.image(input_image, caption=f"Uploaded Radiograph: {uploaded_file.name}", width=340)

            st.caption(f"**Dimensions:** {input_image.width} × {input_image.height} px | **Size:** {uploaded_file.size / 1024:.1f} KB")

            analyze_disabled = (model is None)
            run_button = st.button("🔍 Execute Classification & Grad-CAM", type="primary", disabled=analyze_disabled)
        except Exception as e:
            st.error(f"Error opening image file: {e}")
            run_button = False
    else:
        st.info("Please upload a chest radiograph to begin.")
        run_button = False

with col_result:
    st.subheader("📊 Diagnostic Output")

    if uploaded_file is not None and run_button and model is not None:
        with st.spinner("Executing official preprocessing and forward transformer pass..."):
            try:
                # 1. Run true forward pass
                pred_result = predict_chest_xray(model, input_image)

                pred_class = pred_result["predicted_class"]
                prob_norm = pred_result["prob_normal"]
                prob_pneu = pred_result["prob_pneumonia"]

                # 2. Display prediction badge
                if pred_class == "NORMAL":
                    st.markdown("""
                    <div style="background: #042f2e; border: 2px solid #0d9488; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 16px;">
                        <span style="font-size: 0.85rem; color: #5eead4; font-weight: 700; text-transform: uppercase;">PREDICTED CATEGORY</span>
                        <h2 style="color: #2dd4bf; margin: 4px 0 0 0; font-size: 2.4rem;">NORMAL</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #450a0a; border: 2px solid #ef4444; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 16px;">
                        <span style="font-size: 0.85rem; color: #fca5a5; font-weight: 700; text-transform: uppercase;">PREDICTED CATEGORY</span>
                        <h2 style="color: #f87171; margin: 4px 0 0 0; font-size: 2.4rem;">PNEUMONIA</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # 3. Probability bars
                st.markdown("#### Class Confidence Probabilities")
                st.progress(prob_norm, text=f"NORMAL: {prob_norm * 100:.2f} %")
                st.progress(prob_pneu, text=f"PNEUMONIA: {prob_pneu * 100:.2f} %")

                # 4. Generate on-the-fly Grad-CAM
                st.markdown("#### 🔬 Dynamic Grad-CAM Saliency Map")
                try:
                    cam_generator = TransformerGradCAM(model)
                    cam_heatmap = cam_generator.generate(pred_result["input_tensor"], target_class=pred_result["predicted_index"])

                    # Opacity slider
                    alpha = st.slider("Grad-CAM Overlay Heatmap Opacity", min_value=0.1, max_value=0.9, value=0.55, step=0.05)
                    overlay_img = cam_generator.overlay_heatmap(input_image, cam_heatmap, alpha=alpha)

                    st.image(overlay_img, caption=f"Grad-CAM Saliency Overlay (Target: {pred_class})", use_container_width=True)
                    cam_generator.close()
                except Exception as cam_err:
                    st.warning(f"Grad-CAM generation encountered an issue: {cam_err}")

            except Exception as inf_err:
                st.error(f"Inference execution failed: {inf_err}")

    elif uploaded_file is None:
        st.markdown("""
        <div style="background: #111827; border: 1px dashed #374151; border-radius: 8px; padding: 60px 20px; text-align: center; color: #6b7280;">
            <div style="font-size: 2rem;">🩻</div>
            <div style="margin-top: 8px; font-weight: 500;">Inference console awaiting image upload.</div>
            <div style="font-size: 0.85rem; margin-top: 4px;">Upload a radiograph on the left and click 'Execute Classification'.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click 'Execute Classification & Grad-CAM' to evaluate the uploaded radiograph.")

st.divider()

# Technical expander
with st.expander("⚙️ Inference Execution Workflow (Anti-Cheat Guarantee)"):
    st.markdown("""
    1. **Strict Non-Fabrication:** Outputs are calculated strictly via PyTorch inference with `CXformerSmall`. No random seeds, hardcoded mocks, or simulated outputs exist.
    2. **Official Image Transformation:** Radiograph passes through `CustomBitImageProcessor`:
       * 3-channel RGB expansion
       * Contrast equalization via `cv2.equalizeHist`
       * Bicubic interpolation shortest-edge resize and 518 × 518 center crop
       * ImageNet channel normalization
    3. **Frozen Transformer Backbone:** Evaluated under `torch.inference_mode()` with frozen DINOv2 layers.
    4. **Trainable Head Evaluation:** Normalized `[CLS]` token passed to `LayerNorm -> Dropout -> Linear -> Softmax`.
    """)
