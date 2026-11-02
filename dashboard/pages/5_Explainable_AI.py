import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import config
from components.sidebar import render_sidebar
from components.navbar import render_top_navbar
from components.gradcam_viewer import render_gradcam_case

st.set_page_config(page_title="Explainable AI — CXformer-Small", page_icon="🔬", layout="wide")
render_sidebar()
render_top_navbar()

st.title("🔬 Explainable AI: Transformer Grad-CAM")
st.markdown("""
Visualizing internal attention and gradient representations for **CXformer-Small**.
Gradient-weighted Class Activation Mapping (Grad-CAM) illuminates the spatial regions influencing the model's binary decisions.
""")

st.divider()

# Section A: Educational Banner
st.markdown("""
<div class="custom-alert-info">
🔬 <strong>What is Grad-CAM in Medical Transformers?</strong><br>
Grad-CAM calculates the gradients of the target classification score with respect to patch activations.
<strong>Important Clinical Reality:</strong> Grad-CAM highlights <em>model activation saliency</em>, not confirmed biological pathology.
A highlighted region indicates features the neural network relied upon; it does not replace radiographic anatomical review.
</div>
""", unsafe_allow_html=True)

# Section B: Technical Architecture Expander
with st.expander("🛠 Technical Implementation Details: Transformer Grad-CAM on CXformer-Small", expanded=False):
    st.markdown("""
    - **Frozen Backbone Gradient Flow:** During standard forward inference, CXformer-Small executes under `torch.no_grad()`. For Grad-CAM generation, gradients are dynamically re-enabled via `torch.enable_grad()` through the backbone.
    - **Penultimate Layer Hook:** Hooks are attached to the **penultimate transformer encoder block** (`model.backbone.encoder.layer[-2]`). In DINOv2 with CLS pooling, gradients at the final layer yield zero gradient on patch tokens; the penultimate layer maintains rich spatial patch token gradients.
    - **Token Segregation:** Out of 1,374 output tokens:
      * Token `[0]`: `[CLS]` token (used for linear classification score)
      * Tokens `[1:5]`: `4 Register Tokens` (discarded to avoid synthetic attention sinks)
      * Tokens `[5:1374]`: `1,369 Spatial Patch Tokens` (reshaped to 37 × 37 spatial grid)
    - **Upsampling & Normalization:** The 37 × 37 CAM grid is upsampled via bilinear interpolation to `518 × 518` and min-max normalized to `[0.0, 1.0]`.
    - **Independent Normalization:** Each heatmap is independently scaled. Saliency brightness cannot be quantitatively compared between different patients.
    """)

st.divider()

# Section C: Four Representative Cases (2 x 2 Grid)
st.subheader("🖼 Saliency Heatmaps Across Four Clinical Archetypes")
st.caption("Representative cases extracted from the 618-image test set covering both correct decisions and edge-case errors.")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    render_gradcam_case(
        case_title="Case 1: True Normal",
        case_type="True Normal",
        test_index=0,
        true_class="NORMAL",
        pred_class="NORMAL",
        prob_normal=0.6858,
        prob_pneumonia=0.3142,
        filename="true_normal.png",
        interpretation="Model correctly identified healthy lung parenchyma. Activations are diffused across clear lung fields without intense focal clustering.",
        is_correct=True
    )

with row1_col2:
    render_gradcam_case(
        case_title="Case 2: True Pneumonia",
        case_type="True Pneumonia",
        test_index=231,
        true_class="PNEUMONIA",
        pred_class="PNEUMONIA",
        prob_normal=0.0080,
        prob_pneumonia=0.9920,
        filename="true_pneumonia.png",
        interpretation="High-confidence correct detection (P=99.20%). Saliency focuses strongly over opacities in middle and lower lung zones corresponding to infiltrates.",
        is_correct=True
    )

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    render_gradcam_case(
        case_title="Case 3: False Positive (False Alarm)",
        case_type="False Positive",
        test_index=3,
        true_class="NORMAL",
        pred_class="PNEUMONIA",
        prob_normal=0.4660,
        prob_pneumonia=0.5340,
        filename="false_positive.png",
        interpretation="Low-confidence misclassification (P=53.40%). Model highlighted bronchovascular markings or clavicular border as anomalous infiltrates.",
        is_correct=False
    )

with row2_col2:
    render_gradcam_case(
        case_title="Case 4: False Negative (Missed Case)",
        case_type="False Negative",
        test_index=386,
        true_class="PNEUMONIA",
        pred_class="NORMAL",
        prob_normal=0.9650,
        prob_pneumonia=0.0350,
        filename="false_negative.png",
        interpretation="High-confidence false negative (P=96.50% Normal). Saliency failed to detect subtle retrocardiac infiltration, illustrating the single test-set miss.",
        is_correct=False
    )

# High-resolution composite overview if available
summary_img_path = Path(config.GRADCAM_DIR) / "four_case_gradcam_summary.png"
if summary_img_path.is_file():
    with st.expander("🖼 View High-Resolution 4-Case Comparison Matrix (Original vs Saliency)", expanded=False):
        st.image(str(summary_img_path), caption="Composite Benchmark Matrix: Original Radiographs vs Grad-CAM Saliency across True Normal, True Pneumonia, False Positive, and False Negative.", use_container_width=True)

# Case Summary Metadata Table
summary_csv_path = Path(config.GRADCAM_DIR) / "gradcam_case_summary.csv"
if summary_csv_path.is_file():
    st.subheader("📋 Empirical Grad-CAM Case Summary Table")
    df_cases = pd.read_csv(summary_csv_path)
    df_cases["P(Pneumonia)"] = df_cases["p_pneumonia"].apply(lambda v: f"{v * 100:.2f} %")
    df_cases["P(Normal)"] = df_cases["p_pneumonia"].apply(lambda v: f"{(1 - v) * 100:.2f} %")
    df_cases["True Class"] = df_cases["true_label"].apply(lambda l: "NORMAL (0)" if l == 0 else "PNEUMONIA (1)")
    df_cases["Predicted Class"] = df_cases["predicted_label"].apply(lambda l: "NORMAL (0)" if l == 0 else "PNEUMONIA (1)")
    display_df = df_cases[["case", "test_index", "True Class", "Predicted Class", "P(Normal)", "P(Pneumonia)"]].rename(columns={
        "case": "Clinical Archetype",
        "test_index": "Test Index #"
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# Section D: Qualitative Limitations
st.subheader("⚠️ Qualitative Limitations & Anomaly Flags")
st.markdown("""
<div class="custom-alert-warning">
<strong>Important Observations Regarding Model Activation Patterns:</strong><br>
&bull; <strong>Non-Pulmonary Saliency:</strong> In preliminary qualitative audits, activation was intermittently noted outside primary lung fields (e.g. cervical soft tissues, humeral heads, and subdiaphragmatic spaces).<br>
&bull; <strong>Potential Artefact Reliance:</strong> Saliency along image borders suggests the model may leverage radiograph collimation edges or device markers. While this does not prove shortcut learning, it necessitates rigorous cross-dataset validation.<br>
&bull; <strong>Diagnostic Boundary:</strong> Grad-CAM remains an exploratory interpretability tool. It does not replace board-certified radiologist assessment.
</div>
""", unsafe_allow_html=True)
