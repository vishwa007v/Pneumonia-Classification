import plotly.graph_objects as go
import streamlit as st

def render_confusion_matrix_heatmap(tn=164, fp=67, fn=1, tp=386):
    """
    Renders an interactive, properly labeled 2x2 confusion matrix heatmap using Plotly.
    Rows = Actual Class (NORMAL, PNEUMONIA)
    Cols = Predicted Class (NORMAL, PNEUMONIA)
    """
    matrix = [[tn, fp], [fn, tp]]

    labels = [
        [f"<b>TN = {tn}</b><br>71.0% of Normal", f"<b>FP = {fp}</b><br>29.0% of Normal<br>(False Alarm)"],
        [f"<b>FN = {fn}</b><br>0.26% of Pneumonia<br>(Missed Case)", f"<b>TP = {tp}</b><br>99.74% of Pneumonia"]
    ]

    # Color scaling: TN & TP are green, FP is danger red, FN is amber
    # Custom colored matrix visualization
    fig = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted NORMAL", "Predicted PNEUMONIA"],
        y=["Actual NORMAL", "Actual PNEUMONIA"],
        text=labels,
        texttemplate="%{text}",
        textfont={"size": 15, "color": "white", "family": "sans-serif"},
        colorscale=[
            [0.0, "#451a03"],   # Amber for small FN
            [0.15, "#7f1d1d"],  # Dark red for FP
            [0.5, "#065f46"],   # Deep green for TN
            [1.0, "#047857"]    # Bright green for high TP
        ],
        showscale=False,
        hoverongaps=False
    ))

    fig.update_layout(
        title="Test Set Confusion Matrix (Total: 618 Images)",
        xaxis=dict(side="bottom", tickfont=dict(size=14, color="#f1f5f9")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=14, color="#f1f5f9")),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#f1f5f9"),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)
