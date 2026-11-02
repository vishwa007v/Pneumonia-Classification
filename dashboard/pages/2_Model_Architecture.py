import streamlit as st
import pandas as pd
from components.sidebar import render_sidebar
from components.navbar import render_top_navbar
from components.architecture_diagram import render_architecture_diagram

st.set_page_config(page_title="Model Architecture — CXformer-Small", page_icon="🧠", layout="wide")
render_sidebar()
render_top_navbar()

st.title("🧠 CXformer-Small Model Architecture")
st.markdown("""
Detailed specifications of the **CXformer-Small** transfer learning model for chest radiograph classification.
Pretrained by M42 Health on **600,000+ chest X-rays**, the architecture is founded upon **DINOv2 with register tokens**.
""")

st.warning("📜 **Model License:** Pretrained foundation model licensed under **CC BY-NC 4.0** (Attribution-NonCommercial). Respect non-commercial restrictions.")

st.divider()

# Section A: Visual Pipeline Diagram
st.subheader("📐 End-to-End Classification Pipeline")
st.caption("Visual breakdown from input radiograph to final classification probability.")

render_architecture_diagram()

st.divider()

# Section B: Model Specification Table
st.subheader("🔬 Comprehensive Architecture Specifications")

col_spec1, col_spec2 = st.columns(2)

with col_spec1:
    st.markdown("#### Foundation Backbone (DINOv2 + Registers)")
    backbone_specs = {
        "Architectural Parameter": [
            "Backbone Model Class",
            "Hidden State Dimension",
            "Transformer Encoder Layers",
            "Multi-Head Attention Heads",
            "Intermediate MLP Dimension",
            "Input Image Resolution",
            "Patch Kernel Size",
            "Spatial Patch Token Grid",
            "Patch Token Count",
            "Register Token Count",
            "Classification Token [CLS]",
            "Total Sequence Length",
            "Pretrained Backbone Parameters",
            "Backbone Training State"
        ],
        "Value": [
            "Dinov2WithRegistersModel",
            "384",
            "12 Transformer Layers",
            "6 Attention Heads",
            "1,536 (4 × hidden_dim)",
            "518 × 518 pixels (3 channels)",
            "14 × 14 pixels",
            "37 × 37 patches",
            "1,369 patch tokens",
            "4 register tokens (prevents artifact accumulation)",
            "1 token at sequence index 0",
            "1,374 tokens (1 CLS + 4 Registers + 1,369 Patches)",
            "22,058,112 parameters",
            "FROZEN (requires_grad = False, torch.no_grad())"
        ]
    }
    st.dataframe(pd.DataFrame(backbone_specs), use_container_width=True, hide_index=True)

with col_spec2:
    st.markdown("#### Classification Head & Global Parameters")
    head_specs = {
        "Component / Dimension": [
            "Head Architecture",
            "Feature Normalization",
            "Dropout Regularization",
            "Output Linear Projection",
            "Head Trainable Parameters",
            "Total Model Parameters",
            "Pretraining Corpus",
            "Downstream Task Target",
            "Decision Function",
            "Execution Mode"
        ],
        "Value": [
            "LayerNorm(384) -> Dropout(0.3) -> Linear(384, 2)",
            "LayerNorm(384): 768 parameters (384 weight + 384 bias)",
            "Dropout(p = 0.3) active during training",
            "Linear(384, 2): 770 parameters (768 weights + 2 biases)",
            "1,538 parameters (100% trainable)",
            "22,059,650 parameters",
            "600K+ chest X-rays (M42 Health)",
            "Binary Classification (0: NORMAL, 1: PNEUMONIA)",
            "Softmax with Argmax at 0.5 decision threshold",
            "Backbone in eval mode under torch.no_grad()"
        ]
    }
    st.dataframe(pd.DataFrame(head_specs), use_container_width=True, hide_index=True)

st.divider()

# Section C: Training Hyperparameters
st.subheader("⚙️ Training Hyperparameters & Optimization Protocol")

train_config = {
    "Hyperparameter / Setting": [
        "Optimization Algorithm",
        "Initial Learning Rate",
        "Weight Decay",
        "Batch Size",
        "Training Duration",
        "Learning Rate Scheduler",
        "Scheduler Configuration",
        "Loss Objective",
        "Class Loss Weights",
        "Checkpoint Metric Selection",
        "Selection Optimization Mode",
        "Compute Acceleration"
    ],
    "Configured Value": [
        "AdamW (applied strictly to requires_grad parameters)",
        "0.001 (1e-3)",
        "0.01 (L2 regularization)",
        "8 images per step",
        "15 full training epochs",
        "ReduceLROnPlateau",
        "mode='min', factor=0.5, patience=2 epochs",
        "Weighted CrossEntropyLoss",
        "NORMAL: 1.9363 · PNEUMONIA: 0.6741",
        "Validation Loss (val_loss)",
        "min (saves best checkpoint whenever val_loss reaches new minimum)",
        "NVIDIA Tesla T4 GPU (CUDA, ~14.56 GiB VRAM)"
    ]
}
st.dataframe(pd.DataFrame(train_config), use_container_width=True, hide_index=True)
