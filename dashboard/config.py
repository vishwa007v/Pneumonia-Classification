import os
from pathlib import Path

# Base directories
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

# Candidate locations for pretrained CXformer model files
CANDIDATE_MODEL_DIRS = [
    os.environ.get("MODEL_DIR"),
    str(PROJECT_ROOT / "CXformer_small_files"),
    str(DASHBOARD_DIR / "CXformer_small_files"),
    "/kaggle/input/datasets/vishwad007v/cxformer-small-files",
    "/kaggle/input/cxformer-small-files",
    "/kaggle/input/cxformer-small",
]

MODEL_DIR = None
for cand in CANDIDATE_MODEL_DIRS:
    if cand and os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
        MODEL_DIR = cand
        break

if MODEL_DIR is None:
    MODEL_DIR = os.environ.get("MODEL_DIR", "/kaggle/input/datasets/vishwad007v/cxformer-small-files")

# Candidate locations for checkpoint .pth file
CANDIDATE_CHECKPOINT_PATHS = [
    os.environ.get("CHECKPOINT_PATH"),
    str(PROJECT_ROOT / "checkpoints" / "cxformer_small_best.pth"),
    str(DASHBOARD_DIR / "checkpoints" / "cxformer_small_best.pth"),
    "/kaggle/working/gpu-network/cxformer_small_full_training_15epochs_v1/checkpoints/cxformer_small_best.pth",
    "/kaggle/input/datasets/vishwad007v/cxformer-small-training-files/cxformer_small_best.pth",
]

CHECKPOINT_PATH = None
for cand in CANDIDATE_CHECKPOINT_PATHS:
    if cand and os.path.isfile(cand):
        CHECKPOINT_PATH = cand
        break

if CHECKPOINT_PATH is None:
    CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", str(PROJECT_ROOT / "checkpoints" / "cxformer_small_best.pth"))

DATA_DIR = DASHBOARD_DIR / "data"
GRADCAM_DIR = DASHBOARD_DIR / "assets" / "gradcam"
ASSETS_DIR = DASHBOARD_DIR / "assets"

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
CLASS_COLORS = {0: "#0d9488", 1: "#ef4444"}  # teal, red
