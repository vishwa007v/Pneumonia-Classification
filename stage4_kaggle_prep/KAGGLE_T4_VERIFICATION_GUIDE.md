# Stage 4A: Kaggle Tesla T4 GPU Verification & Execution Guide

This operational guide provides step-by-step instructions to verify the **CXformer-Small (Dinov2WithRegistersModel)** transfer learning pipeline on an NVIDIA Tesla T4 GPU on Kaggle.

---

## 1. Prerequisites and Assets Prepared

In `E:\Project1\gpu-network\stage4_kaggle_prep\`:
1. **`gpu_network_source.zip`** (45.5 KB): Clean, sanitized source archive containing `src/`, `scripts/`, `configs/`, and `requirements.txt`.
   - **Safeguard Enforcement**: 100% free of credentials, API keys, `kaggle.json`, data files, checkpoints, and temporary cache directories.
2. **`cxformer_t4_verification.ipynb`**: Turnkey Jupyter Notebook ready to upload to Kaggle.
3. **`package_project_for_kaggle.py`**: Automated script to re-generate the source archive whenever code changes.

---

## 2. Step-by-Step Kaggle Setup

### Step 2.1: Create a Kaggle Notebook
1. Log in to [Kaggle](https://www.kaggle.com).
2. Click **+ Create** -> **New Notebook** (or go to `https://www.kaggle.com/code` and click **New Notebook**).
3. Set the Notebook Title to: `cxformer-small-t4-verification`.

### Step 2.2: Enable Tesla T4 GPU Accelerator
1. In the right-hand sidebar under **Notebook Settings**, locate **Accelerator**.
2. Click the dropdown and select **GPU T4 x1** (or **GPU P100** if T4 is busy, but T4 is recommended).
3. Ensure **Language** is set to `Python` and **Environment** is set to `Always use latest environment`.
4. Toggle **Internet** to `On` (only needed if downloading additional packages, but our pipeline operates zero-network).

### Step 2.3: Attach the Prepared Dataset & Model Files
1. In the right-hand sidebar, click **+ Add Input** (or the dataset icon).
2. **Attach Prepared Chest X-Ray Dataset**:
   - Search for: `chest-xray-prepared` (by `vishwad007` or `vishwad007v`).
   - Click **Add**.
   - Expected mount location: `/kaggle/input/datasets/vishwad007/chest-xray-prepared/prepared/` (or `/kaggle/input/chest-xray-prepared/prepared/`).
3. **Attach Pretrained CXformer-Small Model Files**:
   - Click **+ Add Input**.
   - Search for: `cxformer-small-files` (by `vishwad007` or `vishwad007v`).
   - Click **Add**.
   - Expected mount location: `/kaggle/input/datasets/vishwad007/cxformer-small-files/` (or `/kaggle/input/cxformer-small-files/`).

### Step 2.4: Upload Project Source Archive
Choose one of two simple methods:
- **Method A (Direct Upload as Dataset)**:
  1. Go to Kaggle -> Datasets -> **New Dataset**.
  2. Upload `stage4_kaggle_prep/gpu_network_source.zip`.
  3. Name it `gpu-network-source`.
  4. In your notebook, click **+ Add Input** and add `gpu-network-source`.
- **Method B (File Upload to Working Directory)**:
  1. In the notebook file browser pane, upload `gpu_network_source.zip` directly into `/kaggle/working/`.

---

## 3. Notebook Execution Workflow

Open or upload `cxformer_t4_verification.ipynb` in Kaggle and execute the cells sequentially:

### Cell 1: Hardware & CUDA Diagnostics
```python
import sys, os, torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Identity   : {gpu_name} ({total_mem_gb:.2f} GB VRAM)")
    assert total_mem_gb >= 14.0, "Expected Tesla T4 (>=14GB VRAM)"
else:
    raise SystemError("GPU accelerator not active!")
```
- **Expected Output**: Detects `Tesla T4` with ~15.00 GB VRAM.

---

### Cell 2: Unpack Source Code & Configure Python Path
```python
import zipfile, os, sys

WORK_DIR = "/kaggle/working/gpu-network"
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)

# Auto-locate and extract source zip
for candidate in [
    "/kaggle/working/gpu_network_source.zip",
    "/kaggle/input/gpu-network-source/gpu_network_source.zip",
    "/kaggle/input/gpu_network_source.zip",
]:
    if os.path.isfile(candidate):
        with zipfile.ZipFile(candidate, "r") as zf:
            zf.extractall(WORK_DIR)
        print(f"Extracted source from: {candidate}")
        break

if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)
```

---

### Cell 3: Verify Mounted Dataset & Model Paths
- Dynamically locates and validates the 3 splits (`train/`, `val/`, `test/`) with `NORMAL/` and `PNEUMONIA/` subdirectories.
- Validates `model.safetensors` and `config.json` inside the mounted model directory.

---

### Cell 4: Verify Official CXformer Image Processor
```python
import cv2
from src.data.dataset import get_cxformer_processor, CXformerTransform
from PIL import Image

processor = get_cxformer_processor(MODEL_DIR)
assert getattr(processor, "do_hist_equalization", False) is True

# Test single real image
sample_img = os.path.join(DATA_ROOT, "train", "NORMAL", os.listdir(os.path.join(DATA_ROOT, "train", "NORMAL"))[0])
transform = CXformerTransform(processor=processor)
with Image.open(sample_img) as img:
    tensor = transform(img)

print("Preprocessed Tensor Shape:", tensor.shape)  # Expected: (3, 518, 518)
assert tensor.shape == (3, 518, 518)
assert tensor.dtype == torch.float32
```

---

### Cell 5: Run Existing Smoke Test
```bash
python scripts/train.py --model cxformer_small --data_dir $DATA_ROOT --model_dir $MODEL_DIR --batch_size 2 --smoke_test
```
- **Validation**:
  - Validates 2 training batches and 1 validation batch on the T4 GPU.
  - Verifies atomic checkpoint saving, reloading, and cleanup.

---

### Cell 6: Systematic GPU Peak Memory & Latency Benchmark
Tests mini-batch sizes **2, 4, 8, and 16** with empirical profiling:
1. Resets CUDA peak memory statistics (`torch.cuda.reset_peak_memory_stats()`).
2. Performs forward pass -> weighted CrossEntropyLoss -> backward pass -> AdamW step on classification head.
3. Confirms backbone gradient isolation (`p.grad is None` across all 224 tensors).
4. Records maximum VRAM allocated (`torch.cuda.max_memory_allocated() / (1024**2)`).
5. Catches any `torch.cuda.OutOfMemoryError` and continues without crashing.
6. Outputs a benchmark comparison table:

| Mini-Batch Size | Status | Step Latency (s) | Throughput (img/s) | Peak VRAM (MB / GB) |
|---|---|---|---|---|
| 2 | PASS | Measured | Measured | Measured |
| 4 | PASS | Measured | Measured | Measured |
| 8 | PASS | Measured | Measured | Measured |
| 16 | PASS / OOM | Measured | Measured | Measured |

---

### Cell 7: Controlled 1-Epoch Training Trial
- Runs only after memory benchmarking completes.
- Automatically selects the largest passing batch size:
```bash
python scripts/train.py --model cxformer_small --data_dir $DATA_ROOT --model_dir $MODEL_DIR --batch_size $SELECTED_BS --epochs 1
```
- **Trial Verifications**:
  - Full epoch executes on train split (4,411 images).
  - Validation split (795 images) evaluates without gradients.
  - Generates valid Loss, Accuracy, Sensitivity, Specificity, Macro F1, and ROC-AUC.
  - Saves `cxformer_small_best.pth` and `cxformer_small_latest.pth`.

---

### Cell 8: Checkpoint Integrity & Bitwise Reload Verification
- Reloads `checkpoints/cxformer_small_best.pth` into a fresh model instance.
- Verifies bitwise forward pass agreement on a sample image:
  $$\max | \text{logits}_{\text{trained}} - \text{logits}_{\text{reloaded}} | < 10^{-5}$$
- Verifies that evaluation mode was preserved and backbone parameters remained unchanged.

---

## 4. Safety and Scope Boundaries

1. **Zero Full Training**: This preparation stage and the notebook run only 1-step memory benchmarks and a single 1-epoch controlled trial. Full 15-epoch training will only be executed after reviewing trial metrics.
2. **Zero In-Place Modification**: Dataset and model files remain strictly read-only inside `/kaggle/input`.
3. **Reproducibility**: Random seed 42 is enforced across all runs.
