# GPU-Accelerated Medical Image Classification and Explainability System Using Deep Learning

An educational prototype and experimental framework for chest X-ray image classification (Normal vs. Pneumonia), explainability with Grad-CAM, and CPU-versus-GPU inference performance benchmarking.

---

## 1. System & Hardware Awareness

This project is designed to run seamlessly across two target environments:

* **Local Development Environment (Laptop)**:
  * **CPU**: AMD Ryzen 5 7520U (4 cores / 8 threads)
  * **RAM**: 16 GB
  * **Graphics**: Integrated AMD Radeon Graphics
  * **OS**: Windows
  * **Compute Mode**: PyTorch runs safely in **CPU mode** (`torch.device("cpu")`), avoiding any crashes or unsupported CUDA calls.
* **GPU Benchmarking Environment (Cloud / Kaggle)**:
  * **Compute Accelerator**: NVIDIA CUDA-capable GPU (e.g., Tesla T4 / P100)
  * **Compute Mode**: Automatically enables CUDA acceleration (`torch.device("cuda")`) for training and high-throughput latency benchmarking.

---

## 2. Project Directory Structure

```text
gpu-network/
├── configs/
│   └── dataset_config.json         # Configurable dataset paths (Windows & Kaggle)
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── device.py               # Hardware diagnostics & safe CPU/CUDA device selection
│   └── data/
│       ├── __init__.py
│       └── audit.py                # Dataset auditing engine (integrity, hash leaks, splits)
├── scripts/
│   └── run_audit.py                # CLI entry point to execute dataset audit
├── reports/
│   └── dataset_audit_report.json   # Generated structured audit report (after running audit)
├── requirements.txt                # Minimal Python dependencies
├── README.md                       # Project and auditing instructions
├── data/                           # Original dataset folder (untouched and preserved)
└── GPU_Medical_Image_Classification_Project_Documentation.docx
```

---

## 3. Configuration (Windows vs. Kaggle)

Dataset paths are defined in `configs/dataset_config.json`:

```json
{
  "dataset_paths": {
    "local_windows": "data/chest img data/chest_xray",
    "kaggle_environment": "/kaggle/input/chest-xray-pneumonia/chest_xray"
  },
  "splits": ["train", "val", "test"],
  "classes": ["NORMAL", "PNEUMONIA"]
}
```

The script automatically checks:
1. `--data-dir` CLI parameter (highest priority).
2. `DATASET_DIR` environment variable.
3. Kaggle default path (`/kaggle/input/chest-xray-pneumonia/chest_xray`).
4. Local Windows path (`data/chest img data/chest_xray`).

---

## 4. How to Run the Dataset Audit

### Step 1: Install Minimal Prerequisites
Ensure Python and core dependencies are available:
```powershell
pip install -r requirements.txt
```

### Step 2: Execute the Audit
Run the audit script from the project root:

```powershell
python scripts/run_audit.py
```

#### Optional Flags:
* Specify a custom path:
  ```powershell
  python scripts/run_audit.py --data-dir "data/chest img data/chest_xray"
  ```
* In Kaggle notebook environment:
  ```bash
  python scripts/run_audit.py --data-dir "/kaggle/input/chest-xray-pneumonia/chest_xray"
  ```
* Use SHA-256 instead of MD5 for file hashing:
  ```powershell
  python scripts/run_audit.py --hash-algo sha256
  ```

---

## 5. What the Audit Inspects

1. **Hardware & Compute Diagnostics**:
   Verifies Python, PyTorch version, CPU core count, and confirms CPU or CUDA device mode without assuming an NVIDIA GPU.
2. **Directory Integrity**:
   Confirms presence of standard splits (`train`, `val`, `test`) and classes (`NORMAL`, `PNEUMONIA`).
3. **Image Health & Decode Check**:
   Opens and decodes every image file to identify unreadable or corrupted files.
4. **Sub-Etiology Classification**:
   Detects bacterial vs. viral pneumonia designations encoded in file naming conventions.
5. **Dimensions & Color Modes**:
   Records image resolution ranges (min/max dimensions) and color channels (`L` grayscale vs. `RGB`).
6. **Hash Duplicates & Cross-Split Leakage**:
   Hashes each image to detect duplicates within splits and verifies that no training images leak into test or validation splits.
7. **Filesystem Artifacts**:
   Flags `.DS_Store` and `__MACOSX` metadata files without altering or deleting any original files.

---

## 6. Audit Output and Report

Upon completion, the script:
* Prints a formatted summary table directly in your terminal.
* Generates a complete JSON audit report saved to:
  `reports/dataset_audit_report.json`

---

## 7. How to Run the Dataset Preparation Pipeline

Once the dataset audit is verified, prepare the deduplicated and stratified dataset without altering the original files:

```powershell
python scripts/run_preparation.py
```

### Optional Custom Flags:
* Custom source and prepared output directories:
  ```powershell
  python scripts/run_preparation.py --source-dir "data/chest img data/chest_xray" --prepared-dir "data/prepared"
  ```
* Custom validation ratio and random seed:
  ```powershell
  python scripts/run_preparation.py --val-ratio 0.15 --seed 42
  ```

### What the Preparation Pipeline Performs:
1. **Source Preservation**: Reads original dataset files in strictly read-only mode (`rb`). Original files are never modified, moved, renamed, or deleted.
2. **Exact-Hash Deduplication**: Removes redundant byte-identical files within original `train` (26 duplicates excluded) and `test` (6 duplicates excluded), deterministically retaining one representative file per group.
3. **Test Split Isolation**: Keeps test images completely separate as an isolated benchmark copy (618 images), preserving original test bounds.
4. **Original Val Exclusion**: Excludes the undersized 16-image original `val` folder.
5. **Stratified 85%/15% Splitting**: Splits the cleaned training pool (5,190 images) into:
   * **Train**: 4,411 images (1,139 Normal, 3,272 Pneumonia)
   * **Validation**: 779 images (201 Normal, 578 Pneumonia)
   * **Test**: 618 images (231 Normal, 387 Pneumonia)
6. **Zero-Leakage Verification**: Cryptographically verifies zero exact-hash overlap across `train`, `val`, and `test`.
7. **Artifact Materialization**: Copies retained images into `data/prepared/` organized by split and class (`train/`, `val/`, `test/`).
8. **Manifests & Reports**: Exports full audit manifests (`reports/manifests/`), JSON report (`reports/dataset_preparation_report.json`), and Markdown summary (`reports/dataset_preparation_report.md`).
9. **Heuristic Patient Analysis**: Evaluates filename-derived patient identifiers and documents clinical generalization limitations.

---

## 9. Review Protocol

> [!IMPORTANT]
> Model implementation (baseline CNN from scratch, ResNet-18 transfer learning, Grad-CAM, and CPU/GPU benchmarks) will begin only after the prepared dataset counts and manifests are reviewed.

---

## 8. Training Pipeline, Smoke Testing & Cloud Execution

The training pipeline supports two architectures:
1. **CustomCNN**: A 4-stage convolutional neural network trained completely from scratch.
2. **ResNet-18**: Transfer learning using ImageNet-1k pretrained weights, replacing the final fully connected layer with a 2-class classifier (c = nn.Linear(512, 2)).

### Key Architectural Safeguards
* **Dynamic Class Weighting**: Calculates inverse class frequencies strictly from the training split (NORMAL: ~1.936, PNEUMONIA: ~0.674) to counteract dataset imbalance. Validation and test labels are never used for weighting.
* **Validation-Based Model Selection**: Monitors validation loss and metrics each epoch. Test data is strictly sequestered from training and tuning.
* **Atomic Checkpointing**: Checkpoints are written to a .pth.tmp buffer before atomic rename (os.replace), preventing corrupted files on interrupt. Checkpoints embed all architecture, class mapping, preprocessing parameters, and training settings.
* **Resumability**: Supports resuming from latest checkpoints via --resume <checkpoint_path>.

---

### Running the Smoke Test Locally (Windows / CPU)

To verify the data loader, tensor pipelines, forward/backward passes, dynamic class weights, and checkpointing without training:

`powershell
python scripts/run_smoke_test.py
`

Or test individual models via CLI smoke test:
`powershell
# Smoke test CustomCNN (2 train batches, 1 val batch)
python scripts/train.py --model custom_cnn --smoke_test

# Smoke test ResNet-18 (2 train batches, 1 val batch)
python scripts/train.py --model resnet18 --smoke_test
`

> [!NOTE]
> The smoke test exercises code execution mechanics and tensor integrity in ~8 seconds on CPU. It does not perform full model training or establish statistical convergence.

---

### Running Full Training on Kaggle (NVIDIA T4 GPU)

When moving to Kaggle for full training and GPU benchmarking:

1. **Environment Setup**:
   - Create a Kaggle Notebook and attach the Chest X-Ray dataset.
   - Under **Settings** -> **Accelerator**, select **GPU T4 x2** or **GPU P100**.
   - Confirm GPU detection:
     `python
     import torch
     print('CUDA Available:', torch.cuda.is_available())
     print('Device Name:', torch.cuda.get_device_name(0))
     `

2. **Execute Full Training**:
   Run training with automatic CUDA acceleration:
   `ash
   # Train CustomCNN from scratch (15 epochs, batch size 32)
   python scripts/train.py --model custom_cnn --epochs 15 --batch_size 32

   # Train ResNet-18 transfer learning (15 epochs, batch size 32)
   python scripts/train.py --model resnet18 --epochs 15 --batch_size 32
   `

3. **Resuming Interrupted Training**:
   If a training session is interrupted, resume cleanly from the latest epoch:
   `ash
   python scripts/train.py --model resnet18 --resume checkpoints/resnet18_latest.pth
   `

4. **Output Artifacts**:
   - Best model weights: checkpoints/<model_name>_best.pth
   - Resumption state: checkpoints/<model_name>_latest.pth
   - Metric progression logs: checkpoints/<model_name>_history.json
