import streamlit as st
import json
from pathlib import Path
import config
from components.sidebar import render_sidebar
from components.navbar import render_top_navbar

# Configure page
st.set_page_config(
    page_title="CXformer-Small Medical AI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render standard sidebar & inject dark CSS
render_sidebar()

# Render top horizontal navigation bar
render_top_navbar()

# Page Header / Hero Banner
st.title("🩺 GPU-Accelerated Medical Image Classification")
st.markdown("""
### **Binary Chest X-Ray Classification & Explainability System**
Powered by **CXformer-Small**, a DINOv2-based Vision Transformer with register tokens,
self-supervised pretrained on **600,000+ chest X-rays** by M42 Health.
""")

# Three-column specification highlights
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.info("🧠 **Model Architecture:** CXformer-Small (12 Transformer Layers)")
with col_info2:
    st.info("⚡ **Acceleration Hardware:** NVIDIA Tesla T4 GPU (14.56 GiB)")
with col_info3:
    st.info("🎯 **Training Protocol:** 15 Epochs · AdamW (LR: 0.001, WD: 0.01)")

st.divider()

# Section 1: Summary Metric Cards (2 rows of 3 columns)
st.subheader("📁 Dataset & Architecture Overview")

row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    st.metric(label="Total Dataset Images", value="5,824", help="Chest X-ray images in prepared dataset")
with row1_col2:
    st.metric(label="Training Split (75.7%)", value="4,411", help="1,139 NORMAL + 3,272 PNEUMONIA")
with row1_col3:
    st.metric(label="Validation Split (13.7%)", value="795", help="209 NORMAL + 586 PNEUMONIA")

row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    st.metric(label="Held-Out Test Split (10.6%)", value="618", help="231 NORMAL + 387 PNEUMONIA (Evaluated once)")
with row2_col2:
    st.metric(label="Total Model Parameters", value="22.06 M", help="Backbone: 22,058,112 (Frozen) + Head: 1,538 (Trained)")
with row2_col3:
    st.metric(label="Training Duration", value="15 Epochs", help="Best checkpoint selected at Epoch 15 (val_loss: 0.0536)")

st.divider()

# Section 2: Performance Highlights (Validation vs Test)
st.subheader("📊 Performance Highlights & Verified Empirical Results")

col_val, col_test = st.columns(2)

with col_val:
    st.markdown("#### ✅ Validation Performance (Best Checkpoint, Epoch 15)")
    st.caption("Selected by minimum validation loss (0.0536).")
    st.warning("⚠️ Validation metrics reflect validation set performance during training.")

    val_data = {
        "Metric": ["Accuracy", "Sensitivity (Recall)", "Specificity", "F1-Score", "ROC-AUC", "Validation Loss"],
        "Value": ["97.99 %", "98.12 %", "97.61 %", "97.43 %", "99.72 %", "0.0536"]
    }
    st.dataframe(val_data, use_container_width=True, hide_index=True)

with col_test:
    st.markdown("#### 🎯 Held-Out Test Set Performance (Final Benchmark)")
    st.caption("618 completely held-out images evaluated strictly once using epoch 15 checkpoint.")

    test_subcol1, test_subcol2, test_subcol3 = st.columns(3)
    with test_subcol1:
        st.metric(label="Test Accuracy", value="89.00 %", delta="-8.99 pp vs Val", delta_color="inverse")
    with test_subcol2:
        st.metric(label="Pneumonia Precision", value="85.21 %")
    with test_subcol3:
        st.metric(label="Sensitivity / Recall", value="99.74 %", delta="+1.62 pp vs Val")

    test_subcol4, test_subcol5, test_subcol6 = st.columns(3)
    with test_subcol4:
        st.metric(label="Normal Specificity", value="71.00 %", delta="-26.61 pp vs Val", delta_color="inverse")
    with test_subcol5:
        st.metric(label="Pneumonia F1-Score", value="91.90 %", delta="-5.53 pp vs Val", delta_color="inverse")
    with test_subcol6:
        st.metric(label="Test ROC-AUC", value="98.52 %", delta="-1.20 pp vs Val", delta_color="inverse")

# Generalization Gap Alert
st.markdown("""
<div class="custom-alert-warning">
⚠️ <strong>Empirical Generalization Gap Notice:</strong><br>
Validation Accuracy (97.99%) dropped to Test Accuracy (89.00%), a gap of <strong>−8.99 percentage points</strong>.<br>
Validation Specificity (97.61%) dropped to Test Specificity (71.00%), a gap of <strong>−26.61 percentage points</strong>.<br>
The model captures <strong>99.74% of pneumonia cases</strong> (386 of 387 detected, only 1 missed), but produces
<strong>67 false alarms</strong> on normal chest X-rays. Root causes include potential distribution shift,
threshold behavior, and unresolved patient independence.
</div>
""", unsafe_allow_html=True)

st.divider()

# Section 3: Verified Kaggle Artifact Integrations
st.subheader("🧪 Verified Empirical Findings from Kaggle GPU Runs")

res_col1, res_col2 = st.columns(2)

with res_col1:
    st.markdown("""
    <div style="background: #111827; border: 1.5px solid #06b6d4; border-radius: 10px; padding: 18px;">
        <h4 style="color: #38bdf8; margin: 0 0 8px 0;">🔬 Grad-CAM Saliency Maps Verified</h4>
        <p style="font-size: 0.95rem; color: #cbd5e1; margin: 0 0 10px 0;">
            All 4 clinical archetypes have been successfully recovered and integrated:
        </p>
        <ul style="font-size: 0.9rem; color: #94a3b8; margin: 0 0 12px 16px;">
            <li><strong>True Normal (Test #0):</strong> P(Normal)=68.58% &bull; Diffuse baseline saliency</li>
            <li><strong>True Pneumonia (Test #231):</strong> P(Pneumonia)=99.20% &bull; Focal lung consolidation</li>
            <li><strong>False Positive (Test #3):</strong> P(Pneumonia)=53.45% &bull; Borderline clavicular alarm</li>
            <li><strong>False Negative (Test #386):</strong> P(Normal)=96.49% &bull; Single retrocardiac miss</li>
        </ul>
        <span style="background: #083344; color: #67e8f9; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;">
            View full heatmaps on the Explainability page &rarr;
        </span>
    </div>
    """, unsafe_allow_html=True)

with res_col2:
    st.markdown("""
    <div style="background: #111827; border: 1.5px solid #10b981; border-radius: 10px; padding: 18px;">
        <h4 style="color: #6ee7b7; margin: 0 0 8px 0;">🔁 5-Fold Stratified Cross-Validation Status</h4>
        <p style="font-size: 0.95rem; color: #cbd5e1; margin: 0 0 10px 0;">
            Evaluated on the 5,206-image CV pool (Test set strictly sequestered):
        </p>
        <ul style="font-size: 0.9rem; color: #94a3b8; margin: 0 0 12px 16px;">
            <li><strong>Fold 1:</strong> Acc 97.98% &bull; Sens 98.45% &bull; Spec 96.67% &bull; AUC 99.62%</li>
            <li><strong>Fold 2:</strong> Acc 96.16% &bull; Sens 97.54% &bull; Spec 92.22% &bull; AUC 99.33%</li>
            <li><strong>Fold 3:</strong> Acc 97.12% &bull; Sens 96.63% &bull; Spec 98.52% &bull; AUC 99.84%</li>
            <li><strong>Fold 4:</strong> Checkpoints saved (eval pending) &bull; <strong>Fold 5:</strong> Pending</li>
        </ul>
        <span style="background: #064e3b; color: #a7f3d0; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;">
            Partial Mean (Folds 1–3): Acc 97.09% &bull; ROC-AUC 99.60%
        </span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Section 4: Module Quick Access (Flexbox Grid)
st.subheader("🗺 System Module Directory")

st.markdown("""
<div class="flex-card-container">
    <div class="flex-card flex-card-cyan">
        <h4 style="color: #38bdf8; margin: 0;">📊 Dataset Analytics</h4>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 6px;">
            Class balance, split proportions, and CustomBitImageProcessor parameters.
        </p>
    </div>
    <div class="flex-card flex-card-teal">
        <h4 style="color: #2dd4bf; margin: 0;">🧠 Model Architecture</h4>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 6px;">
            DINOv2 backbone with 4 register tokens and linear classification head.
        </p>
    </div>
    <div class="flex-card flex-card-amber">
        <h4 style="color: #fbbf24; margin: 0;">📈 Training Dynamics</h4>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 6px;">
            Full 15-epoch history table and training/validation loss trajectories.
        </p>
    </div>
    <div class="flex-card flex-card-purple">
        <h4 style="color: #c084fc; margin: 0;">🎯 Model Evaluation</h4>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 6px;">
            Interactive 2&times;2 confusion matrix, classification report, and probability bins.
        </p>
    </div>
</div>

<div class="flex-card-container">
    <div class="flex-card flex-card-green">
        <h4 style="color: #4ade80; margin: 0;">🔬 Explainable AI</h4>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 6px;">
            Transformer Grad-CAM saliency heatmaps with genuine Kaggle outputs.
        </p>
    </div>
    <div class="flex-card flex-card-red">
        <h4 style="color: #f87171; margin: 0;">🩻 Image Classification</h4>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 6px;">
            Upload patient chest X-rays to generate live PyTorch predictions and Grad-CAM.
        </p>
    </div>
    <div class="flex-card flex-card-cyan">
        <h4 style="color: #38bdf8; margin: 0;">🔁 Cross-Validation</h4>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 6px;">
            5-fold stratified CV protocol, verified metrics, and multi-fold loss curves.
        </p>
    </div>
    <div class="flex-card flex-card-teal">
        <h4 style="color: #e2e8f0; margin: 0;">ℹ️ About the Project</h4>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 6px;">
            Clinical problem statement, research caveats, references, and roadmap.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Sticky-style academic footer
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
    CXformer-Small Medical Image Classification Dashboard &bull; Academic Research Prototype &bull;
    Pretrained Weights: M42 Health (CC BY-NC 4.0) &bull; Not for Clinical Diagnostic Use
</div>
""", unsafe_allow_html=True)
