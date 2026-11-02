import streamlit as st
import os
from pathlib import Path
from PIL import Image
import numpy as np
import config

def render_gradcam_case(
    case_title: str,
    case_type: str,
    test_index: int,
    true_class: str,
    pred_class: str,
    prob_normal: float,
    prob_pneumonia: float,
    filename: str,
    interpretation: str,
    is_correct: bool
):
    """
    Renders an individual Grad-CAM case card with honest placeholder handling,
    ground truth/prediction badges, probability breakdown, and clinical explanation.
    """
    card_border = "#10b981" if is_correct else "#ef4444"
    badge_bg = "#064e3b" if is_correct else "#7f1d1d"
    badge_text = "#a7f3d0" if is_correct else "#fecaca"
    badge_label = "✅ CORRECT PREDICTION" if is_correct else "❌ MISCLASSIFICATION"

    st.markdown(f"""
    <div style="background: #111827; border: 1.5px solid {card_border}; border-radius: 10px; padding: 16px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0; font-size: 1.25rem; color: #f1f5f9;">{case_title}</h3>
            <span style="background: {badge_bg}; color: {badge_text}; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">
                {badge_label}
            </span>
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
            Test Index: <strong>#{test_index}</strong> &bull; Ground Truth: <strong>{true_class}</strong> &bull; Predicted: <strong>{pred_class}</strong>
        </div>
    """, unsafe_allow_html=True)

    # Probabilities row
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.metric("P(NORMAL)", f"{prob_normal * 100:.2f} %")
    with p_col2:
        st.metric("P(PNEUMONIA)", f"{prob_pneumonia * 100:.2f} %")

    # Image or Placeholder
    img_path = Path(config.GRADCAM_DIR) / filename
    if img_path.is_file():
        img = Image.open(img_path)
        st.image(img, caption=f"Grad-CAM Heatmap Overlay — {case_title}", use_container_width=True)
    else:
        st.markdown(f"""
        <div style="background: #1f2937; border: 2px dashed #4b5563; border-radius: 8px; padding: 40px 20px; text-align: center; margin: 15px 0;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">🔬</div>
            <div style="font-size: 1rem; font-weight: 600; color: #f3f4f6;">Grad-CAM Visualization Pending</div>
            <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 6px;">
                Image file <code>assets/gradcam/{filename}</code> not yet placed in local assets.<br>
                Extract from Kaggle directory: <code>/kaggle/working/cxformer_gradcam/</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Interpretation
    st.markdown(f"""
        <div style="background: #1e293b; border-radius: 6px; padding: 10px 14px; font-size: 0.9rem; color: #cbd5e1; margin-top: 10px;">
            <strong>Interpretation:</strong> {interpretation}
        </div>
    </div>
    """, unsafe_allow_html=True)
