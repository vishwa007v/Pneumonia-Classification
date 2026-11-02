import plotly.graph_objects as go
import plotly.express as px

def apply_dark_layout(fig: go.Figure, title: str = "", xaxis_title: str = "", yaxis_title: str = ""):
    """Applies unified dark theme styling across all Plotly charts."""
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="sans-serif", color="#f1f5f9", size=13),
        title_font=dict(size=18, color="#f1f5f9"),
        legend=dict(bgcolor="#1f2937", bordercolor="#374151", borderwidth=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#1f2937", linecolor="#374151")
    fig.update_yaxes(gridcolor="#1f2937", linecolor="#374151")
    return fig

def create_grouped_bar_split_chart():
    """Dataset class counts per split (Train, Val, Test)."""
    splits = ["Training", "Validation", "Testing"]
    normal_counts = [1139, 209, 231]
    pneumonia_counts = [3272, 586, 387]

    fig = go.Figure(data=[
        go.Bar(
            name="NORMAL (0)",
            x=splits,
            y=normal_counts,
            marker_color="#0d9488",
            text=normal_counts,
            textposition="auto"
        ),
        go.Bar(
            name="PNEUMONIA (1)",
            x=splits,
            y=pneumonia_counts,
            marker_color="#ef4444",
            text=pneumonia_counts,
            textposition="auto"
        )
    ])

    apply_dark_layout(
        fig,
        title="Class Counts Across Dataset Splits",
        xaxis_title="Dataset Split",
        yaxis_title="Number of Images"
    )
    fig.update_layout(barmode="group")
    return fig

def create_stacked_proportion_chart():
    """Proportion percentage of each class per split."""
    splits = ["Training", "Validation", "Testing"]
    normal_pct = [25.82, 26.29, 37.38]
    pneumonia_pct = [74.18, 73.71, 62.62]

    fig = go.Figure(data=[
        go.Bar(
            name="NORMAL %",
            x=splits,
            y=normal_pct,
            marker_color="#0d9488",
            text=[f"{v:.1f}%" for v in normal_pct],
            textposition="inside"
        ),
        go.Bar(
            name="PNEUMONIA %",
            x=splits,
            y=pneumonia_pct,
            marker_color="#ef4444",
            text=[f"{v:.1f}%" for v in pneumonia_pct],
            textposition="inside"
        )
    ])

    apply_dark_layout(
        fig,
        title="Class Proportions per Split (%)",
        xaxis_title="Dataset Split",
        yaxis_title="Percentage (%)"
    )
    fig.update_layout(barmode="stack", yaxis_range=[0, 100])
    return fig

def create_overall_pie_chart():
    """Overall dataset class distribution."""
    labels = ["NORMAL (1,579)", "PNEUMONIA (4,245)"]
    values = [1579, 4245]
    colors = ["#0d9488", "#ef4444"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#111827", width=2)),
        textinfo="label+percent",
        textfont_size=14
    )])

    apply_dark_layout(fig, title="Overall Dataset Class Distribution (Total: 5,824)")
    return fig

def create_training_loss_chart(history_data: dict):
    """Training vs Validation Loss curves across epochs with best checkpoints."""
    epochs = history_data["epochs"]
    train_loss = history_data["train_loss"]
    val_loss = history_data["val_loss"]

    fig = go.Figure()

    # Train loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode="lines+markers",
        name="Training Loss",
        line=dict(color="#f59e0b", dash="dash", width=2.5),
        marker=dict(size=6)
    ))

    # Val loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_loss,
        mode="lines+markers",
        name="Validation Loss",
        line=dict(color="#06b6d4", width=3),
        marker=dict(size=7)
    ))

    # Highlight best epoch 15
    best_loss = val_loss[-1]
    fig.add_trace(go.Scatter(
        x=[15],
        y=[best_loss],
        mode="markers+text",
        name="Best Checkpoint (Epoch 15)",
        marker=dict(color="#10b981", size=14, symbol="star"),
        text=["Best (0.0536)"],
        textposition="top center"
    ))

    apply_dark_layout(
        fig,
        title="Training vs Validation Loss (15 Epochs)",
        xaxis_title="Epoch",
        yaxis_title="Weighted Cross-Entropy Loss"
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    return fig

def create_training_accuracy_chart(history_data: dict):
    """Training vs Validation Accuracy curves."""
    epochs = history_data["epochs"]
    train_acc = [v * 100 for v in history_data["train_acc"]]
    val_acc = [v * 100 for v in history_data["val_acc"]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_acc,
        mode="lines+markers",
        name="Training Accuracy",
        line=dict(color="#f59e0b", dash="dash", width=2.5),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_acc,
        mode="lines+markers",
        name="Validation Accuracy",
        line=dict(color="#10b981", width=3),
        marker=dict(size=7)
    ))

    apply_dark_layout(
        fig,
        title="Training vs Validation Accuracy (%)",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)"
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    fig.update_yaxes(range=[92, 100])
    return fig

def create_val_specificity_recall_chart(history_data: dict):
    """Validation Recall vs Specificity over epochs."""
    epochs = history_data["epochs"]
    val_recall = [v * 100 for v in history_data["val_recall"]]
    val_spec = [v * 100 for v in history_data["val_spec"]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_recall,
        mode="lines+markers",
        name="Val Sensitivity / Recall (Pneumonia)",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_spec,
        mode="lines+markers",
        name="Val Specificity (Normal)",
        line=dict(color="#0d9488", width=2.5),
        marker=dict(size=6)
    ))

    apply_dark_layout(
        fig,
        title="Validation Sensitivity vs Specificity (%)",
        xaxis_title="Epoch",
        yaxis_title="Percentage (%)"
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    fig.update_yaxes(range=[94, 100])
    return fig

def create_val_auc_chart(history_data: dict):
    """Validation ROC-AUC over epochs."""
    epochs = history_data["epochs"]
    val_auc = [v * 100 for v in history_data["val_auc"]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_auc,
        mode="lines+markers",
        name="Val ROC-AUC",
        line=dict(color="#a855f7", width=3),
        marker=dict(size=7, color="#c084fc")
    ))

    apply_dark_layout(
        fig,
        title="Validation ROC-AUC Score (%) Across Epochs",
        xaxis_title="Epoch",
        yaxis_title="ROC-AUC (%)"
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    fig.update_yaxes(range=[99.2, 100])
    return fig

def create_probability_distribution_chart(prob_data: dict):
    """Histogram/Bar distribution of P(PNEUMONIA) for Actual NORMAL vs Actual PNEUMONIA."""
    bins = prob_data["bins"]
    normal_counts = prob_data["normal_counts"]
    pneumonia_counts = prob_data["pneumonia_counts"]

    fig = go.Figure()

    fig.add_bar(
        name="Actual NORMAL (N=231)",
        x=bins,
        y=normal_counts,
        marker_color="#0d9488",
        text=normal_counts,
        textposition="auto"
    )

    fig.add_bar(
        name="Actual PNEUMONIA (N=387)",
        x=bins,
        y=pneumonia_counts,
        marker_color="#ef4444",
        text=pneumonia_counts,
        textposition="auto"
    )

    apply_dark_layout(
        fig,
        title="P(PNEUMONIA) Probability Distribution by Ground Truth Class",
        xaxis_title="Predicted Probability Interval P(PNEUMONIA)",
        yaxis_title="Image Count"
    )
    fig.update_layout(barmode="group")
    return fig

def create_cv_folds_comparison_chart(fold_results: list):
    """Grouped bar chart comparing metrics across completed cross-validation folds."""
    folds = [f"Fold {d['fold']}" for d in fold_results]
    accuracies = [d["accuracy"] * 100 for d in fold_results]
    recalls = [d["recall"] * 100 for d in fold_results]
    specificities = [d["specificity"] * 100 for d in fold_results]
    f1s = [d["f1"] * 100 for d in fold_results]
    aucs = [d["auc"] * 100 for d in fold_results]

    fig = go.Figure()

    fig.add_bar(name="Accuracy (%)", x=folds, y=accuracies, marker_color="#06b6d4")
    fig.add_bar(name="Sensitivity / Recall (%)", x=folds, y=recalls, marker_color="#10b981")
    fig.add_bar(name="Specificity (%)", x=folds, y=specificities, marker_color="#0d9488")
    fig.add_bar(name="F1-Score (%)", x=folds, y=f1s, marker_color="#3b82f6")
    fig.add_bar(name="ROC-AUC (%)", x=folds, y=aucs, marker_color="#a855f7")

    apply_dark_layout(
        fig,
        title="Cross-Validation Performance Comparison Across Verified Folds",
        xaxis_title="Cross-Validation Fold",
        yaxis_title="Score (%)"
    )
    fig.update_layout(barmode="group", yaxis_range=[90, 100])
    return fig

def create_cv_folds_loss_curve(fold_histories: dict):
    """Multi-line Plotly chart showing validation loss convergence across verified CV folds."""
    fig = go.Figure()
    colors = {"Fold 1": "#06b6d4", "Fold 2": "#f59e0b", "Fold 3": "#10b981"}

    for fold_name, history in fold_histories.items():
        epochs = [d["epoch"] for d in history]
        val_losses = [d["val_loss"] for d in history]
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_losses,
            mode="lines+markers",
            name=f"{fold_name} Val Loss",
            line=dict(color=colors.get(fold_name, "#ffffff"), width=2.5),
            marker=dict(size=6)
        ))

    apply_dark_layout(
        fig,
        title="Validation Loss Convergence Across Verified Cross-Validation Folds",
        xaxis_title="Epoch (1 to 15)",
        yaxis_title="Validation Loss"
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    return fig

