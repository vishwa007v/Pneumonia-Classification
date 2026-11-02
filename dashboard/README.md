# CXformer-Small Medical Image Classification & Explainability Dashboard

Interactive Streamlit web dashboard for academic demonstration of the **CXformer-Small** chest X-ray classification pipeline.

---

## ⚠️ Mandatory Research & Clinical Disclaimer

This software is an **academic engineering research prototype**.
It is **NOT** a clinically validated diagnostic tool and is **NOT** certified for medical use.
Pretrained foundation backbone weights are licensed by M42 Health under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

---

## Project Structure

```
dashboard/
├── .streamlit/
│   └── config.toml               # Custom dark theme configuration
├── app.py                        # Landing page (Overview & Executive Summary)
├── pages/
│   ├── 1_Dataset_Analytics.py     # Class counts, splits, imbalance, preprocessing
│   ├── 2_Model_Architecture.py   # Visual pipeline diagram, layer dimensions, specs
│   ├── 3_Training_Performance.py # 15-epoch history table and Plotly convergence curves
│   ├── 4_Model_Evaluation.py     # Confusion matrix, classification report, probabilities
│   ├── 5_Explainable_AI.py       # Transformer Grad-CAM heatmaps for 4 archetypes
│   ├── 6_Image_Classification.py # Live CXformer inference with on-the-fly Grad-CAM
│   ├── 7_Cross_Validation.py     # 5-fold stratified CV protocol and status tracking
│   └── 8_About.py                # Research methodology, clinical trade-offs, caveats
├── components/
│   ├── sidebar.py                # Shared navigation, model status, and disclaimer
│   ├── charts.py                 # Reusable Plotly chart builders with dark layout
│   ├── confusion_matrix.py       # 2x2 interactive heatmap with exact counts
│   ├── architecture_diagram.py   # HTML/CSS pipeline flowchart with status tags
│   └── gradcam_viewer.py         # Case card renderer with honest placeholders
├── inference/
│   ├── model_loader.py           # Cached CXformerSmall loader with checkpoint restore
│   ├── predictor.py              # Official preprocessing and forward inference pass
│   └── gradcam.py                # Penultimate layer TransformerGradCAM implementation
├── data/
│   ├── test_metrics.json         # Exact test benchmark metrics and confusion matrix
│   ├── training_history.json     # Verified 15-epoch training loss and validation logs
│   └── cv/                       # 5-fold cross-validation metrics
├── assets/
│   ├── logo.png                  # Project badge icon
│   └── gradcam/                  # Grad-CAM heatmap overlay images
├── style/
│   └── custom.css                # Dark theme overrides, larger fonts, and alert boxes
├── config.py                     # Environment path resolution
├── requirements.txt              # Python package dependencies
└── README.md
```

---

## Installation & Setup

### 1. Requirements
- Python >= 3.10 (tested on Python 3.12)
- PyTorch >= 2.2
- Streamlit >= 1.35
- Transformers >= 4.40
- Plotly >= 5.20
- Pillow, OpenCV-Python-Headless, NumPy, Pandas, Scikit-learn

### 2. Install Dependencies
```bash
cd dashboard
pip install -r requirements.txt
```

### 3. Model Configuration
The dashboard looks for model assets automatically in the following default locations:
- Model Directory: `CXformer_small_files/` (containing `config.json` and `model.safetensors`)
- Checkpoint: `checkpoints/cxformer_small_best.pth`

You can also specify explicit paths via environment variables:
```bash
# Windows PowerShell
$env:MODEL_DIR="E:\Project1\gpu-network\CXformer_small_files"
$env:CHECKPOINT_PATH="E:\Project1\gpu-network\checkpoints\cxformer_small_best.pth"

# Linux / Bash / Kaggle
export MODEL_DIR="/kaggle/input/datasets/vishwad007v/cxformer-small-files"
export CHECKPOINT_PATH="/kaggle/working/gpu-network/cxformer_small_full_training_15epochs_v1/checkpoints/cxformer_small_best.pth"
```

---

## Running the Dashboard

Launch the application using Streamlit:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Running on Kaggle

1. Copy the `dashboard/` folder to `/kaggle/working/dashboard/`.
2. Start Streamlit in the background:
```bash
!pip install streamlit plotly
!streamlit run /kaggle/working/dashboard/app.py --server.port 8501 &
```
3. Expose the port via LocalTunnel or ngrok if remote viewing is needed.

---

## Implemented Features

- [x] **Executive Overview:** 6 summary metrics, validation vs. test comparison, and generalization gap alert.
- [x] **Dataset Analytics:** Split breakdowns, grouped bar charts, stacked proportions, class balance pie chart, and preprocessing specs.
- [x] **Model Architecture:** Visual flowchart from input to softmax, layer-by-layer parameter breakdowns, and training configuration.
- [x] **Training Performance:** Full 15-epoch history table, training/validation loss curves, accuracy trajectories, and specificity/recall dynamics.
- [x] **Model Evaluation:** Interactive 2x2 confusion matrix heatmap, per-class classification report, confidence probability distribution, and generalization gap matrix.
- [x] **Explainable AI (Grad-CAM):** Four clinical archetypes (True Normal, True Pneumonia, False Positive, False Negative) with probability breakdowns and qualitative limitation notices.
- [x] **Live Inference Console:** True PyTorch forward pass using official `CustomBitImageProcessor` and dynamic on-the-fly Grad-CAM saliency generation with adjustable opacity.
- [x] **Cross-Validation Monitor:** 5-fold stratified CV protocol, sample distribution matrix, and honest pending status tracking.
- [x] **Research Summary:** Problem statement, objectives, limitations, references, and future directions.
