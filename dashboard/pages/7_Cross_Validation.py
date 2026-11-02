import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path
import config
from components.sidebar import render_sidebar
from components.navbar import render_top_navbar
from components.charts import create_cv_folds_comparison_chart, create_cv_folds_loss_curve

st.set_page_config(page_title="Cross-Validation — CXformer-Small", page_icon="🔁", layout="wide")
render_sidebar()
render_top_navbar()

st.title("🔁 Five-Fold Stratified Cross-Validation")
st.markdown("""
Robustness evaluation conducted across **5 stratified folds** utilizing the **5,206-image training + validation pool**.
The held-out **618-image test set is strictly excluded** to preserve benchmark integrity.
""")

st.divider()

# Section A: Integrity Status Banner
st.markdown("""
<div class="custom-alert-warning">
⚠️ <strong>Empirical Execution Status & Anti-Fabrication Notice:</strong><br>
&bull; <strong>Folds 1, 2, and 3:</strong> Fully executed, evaluated, and verified with checkpoints and metric logs.<br>
&bull; <strong>Fold 4:</strong> Checkpoints saved (<code>cxformer_small_best.pth</code> and <code>cxformer_small_latest.pth</code>); final evaluation metrics pending.<br>
&bull; <strong>Fold 5:</strong> Pending execution.<br>
&bull; In strict adherence to our anti-fabrication mandate, unverified fold metrics are <strong>NOT simulated or invented</strong>.
</div>
""", unsafe_allow_html=True)

st.divider()

# Section B: Experimental Protocol Specifications
st.subheader("⚙️ Cross-Validation Protocol Specifications")

col_proto1, col_proto2 = st.columns(2)

with col_proto1:
    proto_specs = {
        "Experimental Parameter": [
            "Cross-Validation Pool",
            "Pool Class Balance",
            "Excluded Benchmark Split",
            "Splitting Strategy",
            "Random Partition Seed",
            "Number of Folds (K)",
            "Model Initialization"
        ],
        "Configured Value": [
            "5,206 images (Original Train: 4,411 + Original Val: 795)",
            "NORMAL: 1,348 (25.89%) &bull; PNEUMONIA: 3,858 (74.11%)",
            "618-image Test Split (0% overlap, strictly sequestered)",
            "StratifiedKFold (preserves exact 1:2.86 ratio in every fold)",
            "Seed = 42 (reproducible split indices)",
            "5 Folds (~1,041 validation samples per fold)",
            "Fresh classification head per fold; shared frozen pretrained backbone"
        ]
    }
    st.dataframe(pd.DataFrame(proto_specs), use_container_width=True, hide_index=True)

with col_proto2:
    train_specs = {
        "Training Parameter": [
            "Epochs per Fold",
            "Batch Size",
            "Optimization Algorithm",
            "Learning Rate / Weight Decay",
            "Dropout Regularization",
            "Learning Rate Scheduler",
            "Class Weights Allocation",
            "Checkpoint Selection"
        ],
        "Configured Value": [
            "15 epochs per fold (75 total epochs across full experiment)",
            "8 images per batch",
            "AdamW (trainable parameters only)",
            "LR: 0.001 &bull; Weight Decay: 0.01",
            "p = 0.3 (applied prior to final linear classification layer)",
            "ReduceLROnPlateau (mode='min', factor=0.5, patience=2)",
            "Independently calculated strictly from each fold's training split",
            "Minimum fold validation loss (val_loss)"
        ]
    }
    st.dataframe(pd.DataFrame(train_specs), use_container_width=True, hide_index=True)

st.divider()

# Section C: Fold Sample Distribution Table
st.subheader("📊 Fold-Wise Sample Stratification Distribution")

split_matrix = [
    {"Fold": "Fold 1", "Train Total": "4,164", "Train NORMAL": "1,078", "Train PNEUMONIA": "3,086", "Val Total": "1,042", "Val NORMAL": "270", "Val PNEUMONIA": "772"},
    {"Fold": "Fold 2", "Train Total": "4,165", "Train NORMAL": "1,078", "Train PNEUMONIA": "3,087", "Val Total": "1,041", "Val NORMAL": "270", "Val PNEUMONIA": "771"},
    {"Fold": "Fold 3", "Train Total": "4,165", "Train NORMAL": "1,078", "Train PNEUMONIA": "3,087", "Val Total": "1,041", "Val NORMAL": "270", "Val PNEUMONIA": "771"},
    {"Fold": "Fold 4", "Train Total": "4,165", "Train NORMAL": "1,079", "Train PNEUMONIA": "3,086", "Val Total": "1,041", "Val NORMAL": "269", "Val PNEUMONIA": "772"},
    {"Fold": "Fold 5", "Train Total": "4,165", "Train NORMAL": "1,079", "Train PNEUMONIA": "3,086", "Val Total": "1,041", "Val NORMAL": "269", "Val PNEUMONIA": "772"}
]
st.dataframe(pd.DataFrame(split_matrix), use_container_width=True, hide_index=True)

st.divider()

# Section D: Fold Results Table (Honest File Inspection)
st.subheader("📈 Verified Fold Performance Matrix")

cv_dir = Path(config.DATA_DIR) / "cv"
fold_results = []
completed_metrics = []

for fold_num in range(1, 6):
    fold_file = cv_dir / f"fold_{fold_num}_metrics.json"
    if fold_file.is_file():
        try:
            with open(fold_file, "r") as f:
                f_data = json.load(f)
                if f_data.get("status") == "complete":
                    acc = f_data.get("accuracy", 0)
                    prec = f_data.get("precision", 0)
                    rec = f_data.get("recall", 0)
                    spec = f_data.get("specificity", 0)
                    f1 = f_data.get("f1", 0)
                    auc = f_data.get("auc", 0)
                    val_loss = f_data.get("val_loss", 0)
                    best_ep = f_data.get("best_epoch", "—")

                    completed_metrics.append({
                        "acc": acc, "prec": prec, "rec": rec,
                        "spec": spec, "f1": f1, "auc": auc, "val_loss": val_loss
                    })

                    fold_results.append({
                        "Fold": f"Fold {fold_num}",
                        "Accuracy": f"{acc * 100:.2f} %",
                        "Precision": f"{prec * 100:.2f} %",
                        "Recall": f"{rec * 100:.2f} %",
                        "Specificity": f"{spec * 100:.2f} %",
                        "F1-Score": f"{f1 * 100:.2f} %",
                        "ROC-AUC": f"{auc * 100:.2f} %",
                        "Best Epoch": str(best_ep),
                        "Val Loss": f"{val_loss:.4f}",
                        "Status": "✅ Verified"
                    })
                elif f_data.get("status") == "checkpoint_available":
                    fold_results.append({
                        "Fold": f"Fold {fold_num}",
                        "Accuracy": "[Checkpoint Ready]",
                        "Precision": "[Checkpoint Ready]",
                        "Recall": "[Checkpoint Ready]",
                        "Specificity": "[Checkpoint Ready]",
                        "F1-Score": "[Checkpoint Ready]",
                        "ROC-AUC": "[Checkpoint Ready]",
                        "Best Epoch": "[Saved]",
                        "Val Loss": "[Saved]",
                        "Status": "💾 Checkpoint Saved (Eval Pending)"
                    })
                else:
                    fold_results.append({
                        "Fold": f"Fold {fold_num}",
                        "Accuracy": "[Pending]",
                        "Precision": "[Pending]",
                        "Recall": "[Pending]",
                        "Specificity": "[Pending]",
                        "F1-Score": "[Pending]",
                        "ROC-AUC": "[Pending]",
                        "Best Epoch": "[Pending]",
                        "Val Loss": "[Pending]",
                        "Status": "⏳ Pending Execution"
                    })
        except Exception:
            fold_results.append({"Fold": f"Fold {fold_num}", "Accuracy": "[Error reading file]", "Status": "❌ Error"})
    else:
        fold_results.append({
            "Fold": f"Fold {fold_num}",
            "Accuracy": "[Pending]",
            "Precision": "[Pending]",
            "Recall": "[Pending]",
            "Specificity": "[Pending]",
            "F1-Score": "[Pending]",
            "ROC-AUC": "[Pending]",
            "Best Epoch": "[Pending]",
            "Val Loss": "[Pending]",
            "Status": "⏳ Pending Execution"
        })

# Aggregate row
if len(completed_metrics) == 5:
    df_comp = pd.DataFrame(completed_metrics)
    fold_results.append({
        "Fold": "Mean ± Std Dev (N=5)",
        "Accuracy": f"{df_comp['acc'].mean()*100:.2f} ± {df_comp['acc'].std(ddof=1)*100:.2f} %",
        "Precision": f"{df_comp['prec'].mean()*100:.2f} ± {df_comp['prec'].std(ddof=1)*100:.2f} %",
        "Recall": f"{df_comp['rec'].mean()*100:.2f} ± {df_comp['rec'].std(ddof=1)*100:.2f} %",
        "Specificity": f"{df_comp['spec'].mean()*100:.2f} ± {df_comp['spec'].std(ddof=1)*100:.2f} %",
        "F1-Score": f"{df_comp['f1'].mean()*100:.2f} ± {df_comp['f1'].std(ddof=1)*100:.2f} %",
        "ROC-AUC": f"{df_comp['auc'].mean()*100:.2f} ± {df_comp['auc'].std(ddof=1)*100:.2f} %",
        "Best Epoch": "—",
        "Val Loss": f"{df_comp['val_loss'].mean():.4f} ± {df_comp['val_loss'].std(ddof=1):.4f}",
        "Status": "✅ Complete"
    })
else:
    df_comp = pd.DataFrame(completed_metrics)
    part_mean_acc = f"{df_comp['acc'].mean()*100:.2f}%" if len(df_comp) > 0 else "—"
    part_mean_auc = f"{df_comp['auc'].mean()*100:.2f}%" if len(df_comp) > 0 else "—"
    fold_results.append({
        "Fold": f"Partial Mean (Folds 1–{len(completed_metrics)})",
        "Accuracy": part_mean_acc,
        "Precision": f"{df_comp['prec'].mean()*100:.2f}%" if len(df_comp) > 0 else "—",
        "Recall": f"{df_comp['rec'].mean()*100:.2f}%" if len(df_comp) > 0 else "—",
        "Specificity": f"{df_comp['spec'].mean()*100:.2f}%" if len(df_comp) > 0 else "—",
        "F1-Score": f"{df_comp['f1'].mean()*100:.2f}%" if len(df_comp) > 0 else "—",
        "ROC-AUC": part_mean_auc,
        "Best Epoch": "—",
        "Val Loss": f"{df_comp['val_loss'].mean():.4f}" if len(df_comp) > 0 else "—",
        "Status": f"📊 Partial ({len(completed_metrics)}/5 Verified)"
    })

st.dataframe(pd.DataFrame(fold_results), use_container_width=True, hide_index=True)

st.divider()

# Section E: Interactive Cross-Validation Visualizations
st.subheader("📈 Multi-Fold Verified Visual Comparisons")

cv_results_file = Path(config.DATA_DIR) / "cv" / "fold_results.json"
if cv_results_file.is_file():
    with open(cv_results_file, "r") as f:
        cv_summary_data = json.load(f)

    tab_cv_bars, tab_cv_loss = st.tabs(["📊 Metric Comparison Across Folds", "📉 Validation Loss Convergence (15 Epochs)"])

    with tab_cv_bars:
        fig_cv_bars = create_cv_folds_comparison_chart(cv_summary_data)
        st.plotly_chart(fig_cv_bars, use_container_width=True)

    with tab_cv_loss:
        # Load histories for folds 1, 2, 3
        fold_histories = {}
        for f_num in range(1, 4):
            f_hist_path = Path(config.DATA_DIR) / "cv" / f"fold_{f_num}_history.json"
            if f_hist_path.is_file():
                with open(f_hist_path, "r") as hf:
                    fold_histories[f"Fold {f_num}"] = json.load(hf)

        if fold_histories:
            fig_cv_loss = create_cv_folds_loss_curve(fold_histories)
            st.plotly_chart(fig_cv_loss, use_container_width=True)
        else:
            st.info("Per-epoch fold history files will appear here when loaded.")

st.divider()

# Instructions to import completed results
with st.expander("📥 Kaggle Cross-Validation Export Instructions"):
    st.markdown("""
    When full cross-validation runs finish in Kaggle:
    1. Extract the metrics from `/kaggle/working/cxformer_small_5fold_cv/fold_results.json` or individual directories.
    2. Write each fold's verified dictionary to `dashboard/data/cv/fold_{1..5}_metrics.json`.
    3. The table above will automatically compute sample standard deviation (`ddof=1`) across all 5 folds.
    """)
