import streamlit as st
import pandas as pd
from components.sidebar import render_sidebar
from components.navbar import render_top_navbar
from components.charts import (
    create_grouped_bar_split_chart,
    create_stacked_proportion_chart,
    create_overall_pie_chart
)

st.set_page_config(page_title="Dataset Analytics — CXformer-Small", page_icon="📊", layout="wide")
render_sidebar()
render_top_navbar()

st.title("📊 Dataset Analytics & Distribution")
st.markdown("""
Exploration of the prepared chest X-ray classification dataset (5,824 total images).
Classes are binary: **NORMAL (Index 0)** and **PNEUMONIA (Index 1)**.
""")

st.divider()

# Section A: Summary Stat Cards
st.subheader("📌 Key Dataset Metrics")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Total Dataset Images", "5,824")
with metric_col2:
    st.metric("NORMAL Cases", "1,579", "27.11 % of total")
with metric_col3:
    st.metric("PNEUMONIA Cases", "4,245", "72.89 % of total")
with metric_col4:
    st.metric("Class Imbalance Ratio", "1 : 2.69", "Pneumonia dominant")

st.info("⚖️ **Training Class Weights:** Calculated for weighted CrossEntropyLoss — **NORMAL:** `1.9363` &bull; **PNEUMONIA:** `0.6741`.")

st.divider()

# Section B: Dataset Split Breakdown
st.subheader("📋 Split-Wise Class Distribution")

split_data = {
    "Dataset Split": ["Training", "Validation", "Testing (Held-out)", "Total Dataset"],
    "NORMAL (0)": ["1,139", "209", "231", "1,579"],
    "PNEUMONIA (1)": ["3,272", "586", "387", "4,245"],
    "Total Split Images": ["4,411", "795", "618", "5,824"],
    "NORMAL %": ["25.82 %", "26.29 %", "37.38 %", "27.11 %"],
    "PNEUMONIA %": ["74.18 %", "73.71 %", "62.62 %", "72.89 %"]
}
st.dataframe(pd.DataFrame(split_data), use_container_width=True, hide_index=True)

st.divider()

# Section C: Interactive Charts
st.subheader("📈 Visual Class Distributions")

tab1, tab2, tab3 = st.tabs(["📊 Split Comparison Bar Chart", "📶 Class Proportion Stacked Bar", "🥧 Overall Class Pie Chart"])

with tab1:
    fig_split = create_grouped_bar_split_chart()
    st.plotly_chart(fig_split, use_container_width=True)

with tab2:
    fig_prop = create_stacked_proportion_chart()
    st.plotly_chart(fig_prop, use_container_width=True)

with tab3:
    fig_pie = create_overall_pie_chart()
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# Section D: CXformer Preprocessing Pipeline
st.subheader("🔧 CXformer-Specific Image Preprocessing Pipeline")

st.markdown("""
The pipeline utilizes the official **CustomBitImageProcessor** rather than standard PyTorch/torchvision generic transforms.
This is critical for reproducing reported foundation model performance.
""")

prep_data = {
    "Preprocessing Step": [
        "Histogram Equalization",
        "Color Space",
        "Spatial Resizing",
        "Center Crop",
        "Interpolation Mode",
        "Pixel Rescaling",
        "Standardization / Normalization",
        "Output Tensor Dimension",
        "Image Processor Class"
    ],
    "Specification / Parameter": [
        "Enabled (channel-wise cv2.equalizeHist applied to contrast-enhance chest radiographs)",
        "RGB (converts grayscale/RGBA radiographs to 3-channel RGB)",
        "Shortest edge resized to 518 pixels",
        "Strict center crop to 518 × 518 pixels",
        "Bicubic interpolation (resample=3 in loaded processor)",
        "Divided by 255.0 (rescaling factor: 0.00392157)",
        "ImageNet Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]",
        "(3, 518, 518) Float32 Tensor",
        "CustomBitImageProcessor (trust_remote_code=True)"
    ]
}
st.dataframe(pd.DataFrame(prep_data), use_container_width=True, hide_index=True)

st.warning("⚠️ **Metadata Note:** Checkpoint recorded 'bilinear' in metadata string, but actual loaded processor uses **bicubic (`resample=3`)**. The dashboard runs the verified bicubic processor.")

st.divider()

# Section E: Sample Images
st.subheader("🖼 Dataset Image Gallery")
st.caption("Representative chest radiographs from the dataset splits.")

gal_col1, gal_col2, gal_col3, gal_col4 = st.columns(4)

with gal_col1:
    st.markdown("**Sample 1: NORMAL**")
    st.markdown("""
    <div style="background: #111827; border: 1px dashed #4b5563; border-radius: 8px; height: 180px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 10px;">
        <span style="font-size: 1.5rem;">🫁</span>
        <span style="color: #0d9488; font-weight: 600; margin-top: 6px;">NORMAL Chest Radiograph</span>
        <span style="color: #6b7280; font-size: 0.75rem;">Clear lung fields, sharp costophrenic angles</span>
    </div>
    """, unsafe_allow_html=True)

with gal_col2:
    st.markdown("**Sample 2: NORMAL**")
    st.markdown("""
    <div style="background: #111827; border: 1px dashed #4b5563; border-radius: 8px; height: 180px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 10px;">
        <span style="font-size: 1.5rem;">🫁</span>
        <span style="color: #0d9488; font-weight: 600; margin-top: 6px;">NORMAL Chest Radiograph</span>
        <span style="color: #6b7280; font-size: 0.75rem;">Normal cardiac silhouette, clear parenchyma</span>
    </div>
    """, unsafe_allow_html=True)

with gal_col3:
    st.markdown("**Sample 3: PNEUMONIA**")
    st.markdown("""
    <div style="background: #111827; border: 1px dashed #4b5563; border-radius: 8px; height: 180px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 10px;">
        <span style="font-size: 1.5rem;">🦠</span>
        <span style="color: #ef4444; font-weight: 600; margin-top: 6px;">PNEUMONIA Radiograph</span>
        <span style="color: #6b7280; font-size: 0.75rem;">Focal alveolar consolidation / opacity</span>
    </div>
    """, unsafe_allow_html=True)

with gal_col4:
    st.markdown("**Sample 4: PNEUMONIA**")
    st.markdown("""
    <div style="background: #111827; border: 1px dashed #4b5563; border-radius: 8px; height: 180px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 10px;">
        <span style="font-size: 1.5rem;">🦠</span>
        <span style="color: #ef4444; font-weight: 600; margin-top: 6px;">PNEUMONIA Radiograph</span>
        <span style="color: #6b7280; font-size: 0.75rem;">Diffuse interstitial infiltration</span>
    </div>
    """, unsafe_allow_html=True)

st.caption("Note: Dataset images reside at Kaggle mount points. Connect dataset directory to browse individual radiographs.")
