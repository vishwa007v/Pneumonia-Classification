import streamlit as st
import json
import pandas as pd
from pathlib import Path
import config
from components.sidebar import render_sidebar
from components.navbar import render_top_navbar
from components.charts import (
    create_training_loss_chart,
    create_training_accuracy_chart,
    create_val_specificity_recall_chart,
    create_val_auc_chart
)

st.set_page_config(page_title="Training Performance — CXformer-Small", page_icon="📈", layout="wide")
render_sidebar()
render_top_navbar()

st.title("📈 Training Performance & Optimization Dynamics")
st.markdown("""
Empirical per-epoch progression over the full **15-epoch training run** executed on an **NVIDIA Tesla T4 GPU**.
The best model checkpoint was saved at **Epoch 15** based on minimum validation loss (`0.0536`).
""")

st.divider()

# Load verified training history
history_file = Path(config.DATA_DIR) / "training_history.json"
if history_file.is_file():
    with open(history_file, "r") as f:
        history = json.load(f)
else:
    st.error("Training history file not found at data/training_history.json.")
    st.stop()

# Section A: Best Checkpoint Metrics
st.subheader("🏆 Selected Best Checkpoint Summary (Epoch 15)")

best_col1, best_col2, best_col3, best_col4 = st.columns(4)
with best_col1:
    st.metric("Selected Best Epoch", "15", "Final training epoch")
with best_col2:
    st.metric("Best Validation Loss", "0.0536", "New global minimum")
with best_col3:
    st.metric("Validation Accuracy", "97.99 %", "+2.02 pp vs Epoch 1")
with best_col4:
    st.metric("Validation ROC-AUC", "99.72 %", "+0.22 pp vs Epoch 1")

st.divider()

# Section B: Full 15-Epoch Training History Table
st.subheader("📋 Complete 15-Epoch Empirical Training History")

table_rows = []
for i in range(len(history["epochs"])):
    ep = history["epochs"][i]
    is_best_mark = "✅ Best Checkpoint" if ep == 15 else ("⭐ Milestone" if ep in history["best_epochs"] else "—")
    table_rows.append({
        "Epoch": ep,
        "Train Loss": f"{history['train_loss'][i]:.4f}",
        "Train Acc": f"{history['train_acc'][i] * 100:.2f} %",
        "Train F1": f"{history['train_f1'][i] * 100:.2f} %",
        "Val Loss": f"{history['val_loss'][i]:.4f}",
        "Val Acc": f"{history['val_acc'][i] * 100:.2f} %",
        "Val Recall": f"{history['val_recall'][i] * 100:.2f} %",
        "Val Specificity": f"{history['val_spec'][i] * 100:.2f} %",
        "Val F1": f"{history['val_f1'][i] * 100:.2f} %",
        "Val ROC-AUC": f"{history['val_auc'][i] * 100:.2f} %",
        "Checkpoint Status": is_best_mark
    })

history_df = pd.DataFrame(table_rows)
st.dataframe(history_df, use_container_width=True, hide_index=True)

st.divider()

# Section C: Training Curve Visualizations
st.subheader("📉 Epoch Progression Curves")

tab_loss, tab_acc, tab_spec_rec, tab_auc = st.tabs([
    "📉 Training & Validation Loss",
    "🎯 Accuracy Trajectory",
    "⚖️ Sensitivity vs Specificity",
    "🔮 ROC-AUC Progression"
])

with tab_loss:
    fig_loss = create_training_loss_chart(history)
    st.plotly_chart(fig_loss, use_container_width=True)

with tab_acc:
    fig_acc = create_training_accuracy_chart(history)
    st.plotly_chart(fig_acc, use_container_width=True)

with tab_spec_rec:
    fig_sr = create_val_specificity_recall_chart(history)
    st.plotly_chart(fig_sr, use_container_width=True)

with tab_auc:
    fig_auc = create_val_auc_chart(history)
    st.plotly_chart(fig_auc, use_container_width=True)

st.divider()

# Section D: Analytical Observations
st.subheader("📌 Key Empirical Observations")
st.markdown("""
<div class="custom-alert-info">
<strong>Training Observations & Analysis:</strong><br>
&bull; <strong>Steady Convergence:</strong> Training loss dropped progressively from <code>0.1602</code> (Epoch 1) to <code>0.0746</code> (Epoch 15), reflecting effective linear classification head convergence on frozen representations.<br>
&bull; <strong>Validation Loss Minima:</strong> Validation loss achieved intermittent improvements across epochs 1, 2, 4, 9, 12, and ultimately reached its minimum of <code>0.0536</code> at epoch 15.<br>
&bull; <strong>Absence of Severe Validation Overfitting:</strong> Both training loss and validation loss generally trended downward without divergence, maintaining validation accuracy above 97% from epoch 4 onwards.<br>
&bull; <strong>The Test Generalization Gap:</strong> While validation accuracy and specificity attained <code>97.99%</code> and <code>97.61%</code> respectively, the separate held-out test set exhibited a drop to <code>89.00%</code> accuracy and <code>71.00%</code> specificity. This discrepancy underscores the necessity of independent held-out evaluation.
</div>
""", unsafe_allow_html=True)
