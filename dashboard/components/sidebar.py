import streamlit as st
import os
from pathlib import Path
import config

def inject_css():
    """Injects custom dark theme styling into the active Streamlit page."""
    css_path = Path(config.DASHBOARD_DIR) / "style" / "custom.css"
    if css_path.is_file():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

def check_model_available() -> tuple[bool, str]:
    """Checks if the CXformer-Small checkpoint and model directory are accessible."""
    model_dir = config.MODEL_DIR
    ckpt_path = config.CHECKPOINT_PATH

    has_model_dir = (
        model_dir
        and os.path.isdir(model_dir)
        and os.path.isfile(os.path.join(model_dir, "config.json"))
        and os.path.isfile(os.path.join(model_dir, "model.safetensors"))
    )

    has_ckpt = ckpt_path and os.path.isfile(ckpt_path)

    if has_model_dir and has_ckpt:
        return True, "Pretrained Backbone & Best Checkpoint Ready"
    elif has_model_dir:
        return False, "Pretrained weights found; Checkpoint .pth missing"
    else:
        return False, "Model directory not found. Configure MODEL_DIR in config.py"

def render_sidebar():
    """
    Renders sidebar focused strictly on system status and clinical disclaimers.
    Page navigation is now removed from sidebar and located at the top horizontally.
    """
    inject_css()

    with st.sidebar:
        # Logo + Header
        col1, col2 = st.columns([1, 3])
        with col1:
            logo_path = Path(config.ASSETS_DIR) / "logo.png"
            if logo_path.is_file():
                st.image(str(logo_path), width=50)
            else:
                st.markdown("🩺")
        with col2:
            st.markdown("### CXformer-Small")
            st.caption("Medical Image AI System")

        st.divider()

        # Model Status Diagnostic Card
        st.markdown("#### ⚙️ Runtime Diagnostic Status")
        is_ready, status_msg = check_model_available()
        if is_ready:
            st.success("✅ Model Backbone & Weights Ready")
            st.caption(f"**Backbone:** `{os.path.basename(config.MODEL_DIR)}`\n\n**Checkpoint:** `{os.path.basename(config.CHECKPOINT_PATH)}`")
        else:
            st.warning(f"⚠️ {status_msg}")

        st.divider()

        # GPU Specs
        st.markdown("#### ⚡ Hardware Environment")
        st.markdown("""
        - **Device:** NVIDIA Tesla T4 GPU
        - **VRAM Allocated:** ~14.56 GiB
        - **Input Resolution:** 518 × 518 px
        - **Batch Size:** 8
        - **Precision:** FP32
        """)

        st.divider()

        # Persistent Clinical Disclaimer
        st.markdown("""
        <div class="custom-alert-danger">
        ⚠️ <strong>Academic Prototype</strong><br>
        This system is an engineering research demonstration. It is <strong>NOT</strong> a certified diagnostic medical tool.
        </div>
        """, unsafe_allow_html=True)

        st.caption("Foundation Weights: M42 Health (CC BY-NC 4.0)")
