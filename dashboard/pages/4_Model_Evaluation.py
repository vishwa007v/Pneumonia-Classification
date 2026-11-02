import streamlit as st
import json
import pandas as pd
from pathlib import Path
import config
from components.sidebar import render_sidebar
from components.navbar import render_top_navbar
from components.confusion_matrix import render_confusion_matrix_heatmap
from components.charts import create_probability_distribution_chart

st.set_page_config(page_title="Model Evaluation — CXformer-Small", page_icon="🎯", layout="wide")
render_sidebar()
render_top_navbar()

st.title("🎯 Held-Out Test Set Evaluation")
st.markdown("""
Comprehensive empirical assessment of the selected CXformer-Small model on the **618 held-out test chest X-rays**.
This dataset was evaluated **strictly once** without any post-hoc threshold tuning or data leakage.
""")

st.divider()

# Load verified metrics
test_metrics_file = Path(config.DATA_DIR) / "test_metrics.json"
if test_metrics_file.is_file():
    with open(test_metrics_file, "r") as f:
        metrics_data = json.load(f)
        test_data = metrics_data["test_set"]
        val_data = metrics_data["validation_best"]
else:
    st.error("Test metrics file missing.")
    st.stop()

# Section A: Primary Test Metrics Cards
st.subheader("📊 Primary Test Performance Indicators")

m_row1_1, m_row1_2, m_row1_3 = st.columns(3)
with m_row1_1:
    st.metric("Test Accuracy", f"{test_data['accuracy'] * 100:.2f} %", delta="-8.99 pp vs Val", delta_color="inverse")
with m_row1_2:
    st.metric("Pneumonia Precision", f"{test_data['precision_pneumonia'] * 100:.2f} %", help="True Pneumonia / Total Predicted Pneumonia")
with m_row1_3:
    st.metric("Sensitivity / Recall", f"{test_data['recall_sensitivity'] * 100:.2f} %", delta="+1.62 pp vs Val", help="True Pneumonia / Actual Pneumonia")

m_row2_1, m_row2_2, m_row2_3 = st.columns(3)
with m_row2_1:
    st.metric("Normal Specificity", f"{test_data['specificity'] * 100:.2f} %", delta="-26.61 pp vs Val", delta_color="inverse", help="True Normal / Actual Normal")
with m_row2_2:
    st.metric("Pneumonia F1-Score", f"{test_data['f1_pneumonia'] * 100:.2f} %", delta="-5.53 pp vs Val", delta_color="inverse")
with m_row2_3:
    st.metric("Test ROC-AUC Score", f"{test_data['roc_auc'] * 100:.2f} %", delta="-1.20 pp vs Val", delta_color="inverse", help="Area Under ROC Curve")

st.divider()

# Section B: Interactive Confusion Matrix
st.subheader("🧮 Test Confusion Matrix Heatmap")
st.markdown("Actual Ground Truth (Rows) vs Model Predictions (Columns) on 618 Test Images.")

cm = test_data["confusion_matrix"]
render_confusion_matrix_heatmap(tn=cm["tn"], fp=cm["fp"], fn=cm["fn"], tp=cm["tp"])

st.markdown("""
<div class="custom-alert-warning">
⚠️ <strong>Clinical Analysis of Decision Errors:</strong><br>
&bull; <strong>Exceptional Clinical Sensitivity (99.74%):</strong> Out of 387 actual pneumonia cases in the test set, the model correctly identified <strong>386</strong>, missing only a single radiograph (<strong>FN = 1</strong>).<br>
&bull; <strong>High False Positive Rate (29.01%):</strong> Out of 231 healthy normal radiographs, the model misclassified <strong>67</strong> as pneumonia (<strong>FP = 67</strong>).<br>
&bull; <strong>Severe Specificity Drop:</strong> While validation specificity was <code>97.61%</code>, test specificity fell to <code>71.00%</code>. In a high-throughput triage screening setting, this results in significant false alarms requiring radiologist re-reads.
</div>
""", unsafe_allow_html=True)

st.divider()

# Section C: Classification Report Table
st.subheader("📑 Per-Class Classification Report")

report = test_data["classification_report"]
report_rows = [
    {
        "Class Category": "NORMAL (Index 0)",
        "Precision": f"{report['NORMAL']['precision'] * 100:.2f} %",
        "Recall / Sensitivity": f"{report['NORMAL']['recall'] * 100:.2f} %",
        "F1-Score": f"{report['NORMAL']['f1'] * 100:.2f} %",
        "Support (Count)": f"{report['NORMAL']['support']}"
    },
    {
        "Class Category": "PNEUMONIA (Index 1)",
        "Precision": f"{report['PNEUMONIA']['precision'] * 100:.2f} %",
        "Recall / Sensitivity": f"{report['PNEUMONIA']['recall'] * 100:.2f} %",
        "F1-Score": f"{report['PNEUMONIA']['f1'] * 100:.2f} %",
        "Support (Count)": f"{report['PNEUMONIA']['support']}"
    },
    {
        "Class Category": "Macro Average",
        "Precision": f"{report['macro_avg']['precision'] * 100:.2f} %",
        "Recall / Sensitivity": f"{report['macro_avg']['recall'] * 100:.2f} %",
        "F1-Score": f"{report['macro_avg']['f1'] * 100:.2f} %",
        "Support (Count)": "618"
    },
    {
        "Class Category": "Weighted Average",
        "Precision": f"{report['weighted_avg']['precision'] * 100:.2f} %",
        "Recall / Sensitivity": f"{report['weighted_avg']['recall'] * 100:.2f} %",
        "F1-Score": f"{report['weighted_avg']['f1'] * 100:.2f} %",
        "Support (Count)": "618"
    }
]
st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

st.divider()

# Section D: Probability Distribution Analysis
st.subheader("📊 Output Probability Bins & Confidence Analysis")
st.markdown("Distribution of `P(PNEUMONIA)` probabilities segmented by actual ground truth class.")

fig_prob = create_probability_distribution_chart(test_data["probability_bins"])
st.plotly_chart(fig_prob, use_container_width=True)

st.info("""
💡 **Deep-Dive Probability Finding:**
Of the **67 false positives**, exactly **32 cases** exhibited `P(PNEUMONIA) > 0.90`.
These represent confident errors rather than borderline cases hovering around the 0.5 decision threshold.
Further investigation is required to ascertain whether patient age, radiograph contrast differences, or institutional hardware artifacts induce these high-confidence shifts.
""")

st.divider()

# Section E: Validation-to-Test Generalization Comparison
st.subheader("⚖️ Validation vs Test Set Generalization Comparison")

comp_data = {
    "Metric": [
        "Accuracy",
        "Pneumonia Sensitivity / Recall",
        "Normal Specificity",
        "F1-Score",
        "ROC-AUC",
        "Sample Size (N)"
    ],
    "Validation (Epoch 15 Checkpoint)": [
        "97.99 %",
        "98.12 %",
        "97.61 %",
        "97.43 %",
        "99.72 %",
        "795 images"
    ],
    "Held-Out Test Set": [
        "89.00 %",
        "99.74 %",
        "71.00 %",
        "91.90 %",
        "98.52 %",
        "618 images"
    ],
    "Generalization Gap (Test − Val)": [
        "−8.99 pp (Drop)",
        "+1.62 pp (Higher recall)",
        "−26.61 pp (Substantial drop) ⚠️",
        "−5.53 pp (Drop)",
        "−1.20 pp (Slight drop)",
        "—"
    ]
}
st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
