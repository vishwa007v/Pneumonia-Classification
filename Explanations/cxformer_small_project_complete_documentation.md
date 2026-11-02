# CXformer-Small: GPU-Accelerated Medical Image Classification and Explainability

## 1. Project overview

This project builds a binary chest X-ray classification and
explainability system using **CXformer-Small**. The classes are `NORMAL`
(index `0`) and `PNEUMONIA` (index `1`). It uses a pretrained
CXformer-Small transformer backbone and a trainable two-class
classification head. The backbone is frozen during the documented
training run. Training is performed in Kaggle on an NVIDIA Tesla T4 GPU.

The project includes dataset preparation, CXformer-specific
preprocessing, model training, validation, a separate test evaluation,
Grad-CAM visual explanations, and an ongoing five-fold stratified
cross-validation experiment. A Streamlit dashboard is planned; it should
not be described as implemented until it has been built and tested.

## 2. Dataset and file locations

### Kaggle input locations

``` text
Project source:
/kaggle/input/datasets/vishwad007v/gpu-network-source/

Prepared dataset:
/kaggle/input/datasets/vishwad007v/chest-xray-prepared/prepared/

Pretrained CXformer-Small files:
/kaggle/input/datasets/vishwad007v/cxformer-small-files/

Backup training package:
/kaggle/input/datasets/vishwad007v/cxformer-small-training-files/
```

### Prepared dataset counts

  Split          NORMAL   PNEUMONIA   Total
  ------------ -------- ----------- -------
  Train           1,139       3,272   4,411
  Validation        209         586     795
  Test              231         387     618
  Total           1,579       4,245   5,824

Class mapping:

``` python
class_to_idx = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}
```

The CV pool combines only original train and validation images: 5,206
images (NORMAL 1,348; PNEUMONIA 3,858). The 618-image test set is
excluded from CV.

### Dataset caveats

-   The prepared test split is a deduplicated subset of 618 images, not
    the full untouched original official benchmark split of 624.
-   An exact file-hash audit found no exact cross-split duplicate
    overlaps.
-   Patient-level independence has not been established.
    Filename-derived patient identifiers are imperfect; pneumonia
    filename-derived IDs showed 155 heuristic train/test overlaps, while
    normal identifiers were unreliable. Do not claim patient-independent
    splits or patient-grouped CV.
-   The test set has already been evaluated. It remains separate from
    the CV pool, but is no longer an untouched test set in the overall
    project workflow.

## 3. Preprocessing

The project uses the CXformer-specific official processor rather than a
generic image transform.

Observed loaded processor: `CustomBitImageProcessor`.

Observed behavior/configuration:

-   Histogram equalization enabled.
-   RGB image input.
-   Shortest-edge resize to 518.
-   Center crop to 518 × 518.
-   Bicubic interpolation (`resample=3`) in the actual loaded processor.
-   Rescaling by `1/255`.
-   ImageNet mean and standard deviation normalization.
-   Three channels; output tensor shape `(3, 518, 518)`.

**Metadata discrepancy:** the checkpoint metadata recorded interpolation
as `"bilinear"`, while the actual loaded processor reported bicubic
(`resample=3`). Use and verify the actual official processor for
inference and reproducibility.

Original train split class weights for weighted cross-entropy:

``` text
NORMAL:    1.9363477
PNEUMONIA: 0.6740526
```

For CV, calculate class weights independently from each fold's training
labels only.

## 4. CXformer-Small pretrained model

The pretrained model is from M42 Health and was self-supervised
pretrained on 600K+ chest X-rays. Its license is **CC BY-NC 4.0**;
attribution and non-commercial restrictions must be respected. Check the
original model card/license before distribution or commercial use.

  Property                         Value
  -------------------------------- -----------------------
  Backbone family                  DINOv2 with registers
  Hidden dimension                 384
  Transformer layers               12
  Attention heads                  6
  Intermediate dimension           1,536
  Register tokens                  4
  Input channels                   3
  Patch size                       14
  Input image                      518 × 518
  Patch grid                       37 × 37
  Pretrained backbone parameters   22,058,112
  Classification head parameters   1,538
  Total parameters                 22,059,650

One image produced backbone output shape `(1, 1374, 384)`: one CLS
token, four register tokens, and 1,369 patch tokens (37 × 37).

### Classification architecture

``` text
Chest X-ray (3 × 518 × 518)
          |
          v
Pretrained CXformer-Small backbone
(12 transformer layers; frozen)
          |
          v
CLS token (384-dimensional)
          |
          v
LayerNorm(384)
          |
          v
Dropout(p=0.3)
          |
          v
Linear(384, 2)
          |
          v
Two logits
          |
          v
Softmax: NORMAL / PNEUMONIA probabilities
```

The wrapper is implemented in `src/models/cxformer.py`. It loads the
official backbone through the Hugging Face AutoModel interface,
validates the model directory/config/weights, uses the CLS token,
freezes backbone parameters, runs the frozen backbone under
`torch.no_grad()`, and keeps the backbone in evaluation mode when the
wrapper is training. It has explicit unfreeze methods, but the reported
experiment kept the backbone frozen. An explicitly invalid model
directory raises `FileNotFoundError`.

## 5. Project source structure

The working source was restored to `/kaggle/working/gpu-network/`.

``` text
gpu-network/
├── configs/
│   ├── training_config.json
│   └── cxformer_full_15epochs_v1.json
├── scripts/
│   └── train.py
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   ├── cxformer.py
│   │   └── __init__.py
│   └── training/
│       ├── trainer.py
│       └── metrics.py
└── stage4_kaggle_prep/
    ├── gpu_network_source.zip
    ├── cxformer_t4_verification.ipynb
    ├── KAGGLE_T4_VERIFICATION_GUIDE.md
    └── package_project_for_kaggle.py
```

The backup dataset `cxformer-small-training-files` also contains
training scripts, the full-training config, source modules, checkpoints,
and history JSON.

### Kaggle project restoration

Kaggle working storage can be cleared after a kernel reset. Source and
model/data assets are retained as input datasets and can be copied back:

``` python
from pathlib import Path
import shutil
import sys
import os

source_root = Path(
    "/kaggle/input/datasets/vishwad007v/gpu-network-source"
)
project_root = Path("/kaggle/working/gpu-network")

if not project_root.exists():
    shutil.copytree(source_root, project_root)

os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Project root:", project_root)
print("Current directory:", Path.cwd())
```

## 6. Kaggle GPU environment and training configuration

Observed Kaggle environment:

-   NVIDIA Tesla T4 GPU.
-   CUDA available.
-   Approximately 14.56 GiB GPU memory reported.
-   Two T4 devices were detected; device 0 was used as the primary
    device.

GPU check:

``` python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
    print(
        "GPU memory GiB:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Main full-training configuration in
`configs/cxformer_full_15epochs_v1.json`:

  Setting              Value
  -------------------- ---------------------------
  Architecture         `cxformer_small`
  Epochs               15
  Batch size           8
  Learning rate        0.001
  Weight decay         0.01
  Dropout              0.3
  Optimizer            AdamW
  Scheduler            ReduceLROnPlateau
  Scheduler mode       min
  Scheduler factor     0.5
  Scheduler patience   2
  Backbone             Frozen
  Loss                 Weighted CrossEntropyLoss
  Checkpoint monitor   `val_loss`
  Monitor direction    `min`
  Input size           518 × 518

Optimizer uses only trainable parameters:

``` python
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad
]
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=0.001,
    weight_decay=0.01
)
```

Scheduler and loss:

``` python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
```

## 7. Main training code pattern

The implemented project interfaces are `CXformerSmall`,
`create_cxformer_data_loaders`, `compute_class_weights`, and `Trainer`.
The following is a consolidated reference for the main training flow and
configuration. It reflects the implemented interfaces and documented
experiment, but is not claimed to be a byte-for-byte copy of
`scripts/train.py`. Check the current source for exact keyword
compatibility if reusing it after code changes.

``` python
import torch
from src.models.cxformer import CXformerSmall
from src.data.dataset import create_cxformer_data_loaders
from src.training.trainer import Trainer

DATA_ROOT = (
    "/kaggle/input/datasets/vishwad007v/"
    "chest-xray-prepared/prepared"
)
MODEL_DIR = (
    "/kaggle/input/datasets/vishwad007v/"
    "cxformer-small-files"
)
OUTPUT_DIR = (
    "/kaggle/working/gpu-network/"
    "cxformer_small_full_training_15epochs_v1/checkpoints"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, class_weights = (
    create_cxformer_data_loaders(
        data_root=DATA_ROOT,
        model_dir=MODEL_DIR,
        batch_size=8
    )
)

model = CXformerSmall(
    model_dir=MODEL_DIR,
    num_classes=2,
    freeze_backbone=True,
    dropout_rate=0.3
).to(device)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001,
    weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)

trainer = Trainer(
    model=model,
    model_name="cxformer_small",
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device=device,
    output_dir=OUTPUT_DIR,
    monitor_metric="val_loss",
    monitor_mode="min",
    training_config={
        "architecture": "cxformer_small",
        "model_dir": MODEL_DIR,
        "image_size": 518,
        "batch_size": 8,
        "epochs": 15,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "dropout_rate": 0.3,
        "freeze_backbone": True
    }
)

history = trainer.train(num_epochs=15)
```

### Trainer and metrics behavior

`src/training/trainer.py`:

-   Runs training epochs and validation.
-   Computes loss and classification metrics.
-   Steps `ReduceLROnPlateau` using validation loss.
-   Monitors `val_loss` in `min` mode.
-   Saves best and latest checkpoints.
-   Writes epoch history JSON.
-   Stores model/optimizer/scheduler state, class weights, preprocessing
    metadata, training configuration, metrics, and best metric
    information in checkpoints.

`src/training/metrics.py` computes classification metrics. ROC-AUC is
`None` if undefined (for example, if only one class occurs). Missing
scikit-learn is explicitly reported; the implementation catches relevant
`ValueError` cases.

## 8. Original 15-epoch training run and checkpoint details

Output directory:

``` text
/kaggle/working/gpu-network/
cxformer_small_full_training_15epochs_v1/checkpoints/
```

Artifacts:

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `cxformer_small_best.pth`           Best checkpoint selected by
                                      validation loss

  `cxformer_small_latest.pth`         Latest epoch checkpoint

  `cxformer_small_history.json`       Per-epoch metrics
  -----------------------------------------------------------------------

The two checkpoint files were approximately 84.26 MB each in the backup
dataset. The pretrained backbone weight file was observed as 88,256,200
bytes, containing 224 tensors, all F32.

Best checkpoint metadata observed:

-   Epoch 15.
-   Architecture `cxformer_small`.
-   `class_to_idx`: NORMAL 0, PNEUMONIA 1, plus reverse mapping.
-   Model state dictionary.
-   Optimizer/scheduler state information.
-   Class weights.
-   Preprocessing metadata.
-   Training configuration.
-   Best metric value approximately 0.0536.
-   Monitor `val_loss`, mode `min`.

### Full training history

Rounded values recovered from `cxformer_small_history.json`:

  ---------------------------------------------------------------------------------------------
    Epoch   Train   Train   Train     Val Val acc      Val           Val  Val F1 Val AUC Best
             loss     acc      F1    loss           recall   specificity                 
  ------- ------- ------- ------- ------- ------- -------- ------------- ------- ------- ------
        1   .1602   .9379   .9223   .1028   .9597    .9505         .9856   .9500   .9950 Yes

        2   .0988   .9619   .9517   .0806   .9635    .9590         .9761   .9542   .9956 Yes

        3   .0903   .9664   .9572   .0875   .9635    .9556         .9856   .9545   .9964 No

        4   .0848   .9667   .9575   .0619   .9748    .9744         .9761   .9680   .9968 Yes

        5   .0904   .9676   .9586   .0643   .9711    .9727         .9665   .9632   .9966 No

        6   .0848   .9685   .9597   .0764   .9648    .9608         .9761   .9558   .9968 No

        7   .0911   .9694   .9608   .0949   .9635    .9573         .9809   .9544   .9966 No

        8   .0747   .9717   .9638   .0635   .9761    .9795         .9665   .9694   .9964 No

        9   .0786   .9714   .9635   .0590   .9786    .9795         .9761   .9727   .9968 Yes

       10   .0760   .9717   .9637   .0652   .9723    .9710         .9761   .9649   .9971 No

       11   .0723   .9717   .9636   .0612   .9736    .9727         .9761   .9665   .9970 No

       12   .0729   .9737   .9662   .0581   .9799    .9812         .9761   .9743   .9970 Yes

       13   .0735   .9719   .9640   .0621   .9711    .9693         .9761   .9634   .9973 No

       14   .0687   .9751   .9680   .0600   .9761    .9761         .9761   .9696   .9972 No

       15   .0746   .9705   .9623   .0536   .9799    .9812         .9761   .9743   .9972 Yes
  ---------------------------------------------------------------------------------------------

Best validation loss was 0.0536 at epoch 15.

Validation metrics from the selected checkpoint:

  Metric                   Result
  ---------------------- --------
  Accuracy                 0.9799
  Recall / Sensitivity     0.9812
  Specificity              0.9761
  F1-score                 0.9743
  ROC-AUC                  0.9972
  Loss                     0.0536

Training and validation curves did not show a clear severe classical
overfitting pattern: training loss generally decreased and validation
loss also improved. This does not prove no overfitting; the test
generalization gap is still important.

## 9. Original test-set evaluation

The best checkpoint was restored with strict state-dictionary loading.
The test loader contained 618 images in 78 batches, batch size 8,
`shuffle=False`, `drop_last=False`. Evaluation used
`torch.inference_mode()`, softmax, and argmax.

### Test evaluation code pattern

``` python
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

checkpoint_path = (
    "/kaggle/working/gpu-network/"
    "cxformer_small_full_training_15epochs_v1/"
    "checkpoints/cxformer_small_best.pth"
)
checkpoint = torch.load(
    checkpoint_path, map_location=device, weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(device)
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
y_prob = np.asarray(y_prob)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, pos_label=1, zero_division=0
)
recall = recall_score(
    y_true, y_pred, pos_label=1, zero_division=0
)
f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
auc = roc_auc_score(y_true, y_prob[:, 1])

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()
specificity = tn / (tn + fp)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1:", f1)
print("ROC-AUC:", auc)
print("Confusion matrix:", [[tn, fp], [fn, tp]])
print(classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["NORMAL", "PNEUMONIA"],
    zero_division=0
))
```

### Test results

  Metric                               Result
  ---------------------------------- --------
  Accuracy                             0.8900
  Precision (PNEUMONIA)                0.8521
  Recall / Sensitivity (PNEUMONIA)     0.9974
  Specificity (NORMAL)                 0.7100
  F1-score (PNEUMONIA)                 0.9190
  ROC-AUC                              0.9852

Confusion matrix, rows actual and columns predicted; class order NORMAL
(0), PNEUMONIA (1):

  Actual / Predicted       NORMAL   PNEUMONIA
  -------------------- ---------- -----------
  NORMAL                 164 (TN)     67 (FP)
  PNEUMONIA                1 (FN)    386 (TP)

Classification report:

  Class                Precision   Recall      F1   Support
  ------------------ ----------- -------- ------- ---------
  NORMAL                   .9939    .7100   .8283       231
  PNEUMONIA                .8521    .9974   .9190       387
  Macro average            .9230    .8537   .8737       618
  Weighted average         .9051    .8900   .8851       618

### Validation-to-test gap and interpretation

Validation accuracy was 97.99%, compared with test accuracy 89.00%.
Validation specificity was 97.61%, compared with test specificity
71.00%. This is an accuracy gap of approximately 8.99 percentage points
and a specificity gap of approximately 26.61 percentage points.

The model captures nearly all pneumonia examples in this test set but
has many false positives among NORMAL images. Possible explanations
include distribution shift, split composition, preprocessing
differences, threshold behavior, and unresolved patient independence.
The root cause is unknown; none of these explanations should be
presented as proven.

High ROC-AUC indicates strong ranking discrimination in this test set,
but the fixed argmax/0.5 decision behavior still creates false
positives. Do not tune a threshold or select models using this already
evaluated test set.

## 10. Test probability analysis

  Actual class       N   Mean P(PNEUMONIA)   Median     Min      Max
  -------------- ----- ------------------- -------- ------- --------
  NORMAL           231               .3179    .1355   .0000    .9997
  PNEUMONIA        387               .9901   1.0000   .0351   1.0000

Probability bins:

  P(PNEUMONIA) interval     Actual NORMAL   Actual PNEUMONIA
  ----------------------- --------------- ------------------
  (0.0, 0.1\]                         106                  1
  (0.1, 0.3\]                          35                  0
  (0.3, 0.5\]                          23                  0
  (0.5, 0.7\]                          17                  1
  (0.7, 0.9\]                          18                  9
  (0.9, 1.0\]                          32                376

Of 67 false positives, 32 had P(PNEUMONIA) greater than 0.9. These are
high-confidence errors. This analysis does not identify their cause.

## 11. Five-fold stratified cross-validation

### Design

The CV pool consists of original train + validation only (5,206 images:
NORMAL 1,348; PNEUMONIA 3,858). The original test set is excluded.

``` python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Fold distribution generated:

  ------------------------------------------------------------------------------
        Fold      Train      Train       Train  Val total Val NORMAL         Val
                  total     NORMAL   PNEUMONIA                         PNEUMONIA
  ---------- ---------- ---------- ----------- ---------- ---------- -----------
           1      4,164      1,078       3,086      1,042        270         772

           2      4,165      1,078       3,087      1,041        270         771

           3      4,165      1,078       3,087      1,041        270         771

           4      4,165      1,079       3,086      1,041        269         772

           5      4,165      1,079       3,086      1,041        269         772
  ------------------------------------------------------------------------------

Each fold is an independent run with a fresh classification head, same
pretrained CXformer-Small backbone, frozen backbone, 15 epochs, batch
size 8, AdamW (LR .001, WD .01), dropout .3, ReduceLROnPlateau (min,
factor .5, patience 2), weighted cross-entropy, and fold-specific class
weights from training indices only. Best checkpoint is selected using
fold validation loss.

### Five-fold training code used in Kaggle

``` python
import random
import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from src.data.dataset import (
    ChestXRayDataset,
    get_cxformer_transform,
    compute_class_weights
)
from src.models.cxformer import CXformerSmall
from src.training.trainer import Trainer

SEED = 42
N_FOLDS = 5
EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
NUM_WORKERS = 2

DATA_ROOT = (
    "/kaggle/input/datasets/vishwad007v/"
    "chest-xray-prepared/prepared"
)
MODEL_DIR = (
    "/kaggle/input/datasets/vishwad007v/"
    "cxformer-small-files"
)
OUTPUT_ROOT = Path(
    "/kaggle/working/cxformer_small_5fold_cv"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "CUDA GPU is required."

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

processor_transform = get_cxformer_transform(model_dir=MODEL_DIR)

train_dataset = ChestXRayDataset(
    root_dir=DATA_ROOT,
    split="train",
    transform=processor_transform
)
val_dataset = ChestXRayDataset(
    root_dir=DATA_ROOT,
    split="val",
    transform=processor_transform
)

combined_dataset = ConcatDataset([train_dataset, val_dataset])
labels = np.asarray(
    train_dataset.get_labels() + val_dataset.get_labels()
)

assert len(combined_dataset) == 5206
assert len(labels) == 5206

skf = StratifiedKFold(
    n_splits=N_FOLDS, shuffle=True, random_state=SEED
)
fold_splits = list(skf.split(np.zeros(len(labels)), labels))
fold_results = []

for fold_number, (train_idx, val_idx) in enumerate(
    fold_splits, start=1
):
    print(f"Starting fold {fold_number}/{N_FOLDS}")
    set_seed(SEED + fold_number)

    fold_train_dataset = Subset(
        combined_dataset, train_idx.tolist()
    )
    fold_val_dataset = Subset(
        combined_dataset, val_idx.tolist()
    )

    fold_train_labels = labels[train_idx].tolist()
    class_weights = compute_class_weights(
        train_labels=fold_train_labels,
        num_classes=2,
        device=device
    )

    generator = torch.Generator()
    generator.manual_seed(SEED + fold_number)

    train_loader = DataLoader(
        fold_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=generator
    )
    val_loader = DataLoader(
        fold_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # New model instance: do not load a previously trained head.
    model = CXformerSmall(
        model_dir=MODEL_DIR,
        num_classes=2,
        freeze_backbone=True,
        dropout_rate=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    fold_output = OUTPUT_ROOT / f"fold_{fold_number}"
    fold_output.mkdir(parents=True, exist_ok=True)

    training_config = {
        "architecture": "cxformer_small",
        "cross_validation": True,
        "fold": fold_number,
        "n_folds": N_FOLDS,
        "image_size": 518,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout_rate": DROPOUT,
        "freeze_backbone": True,
        "class_weights": class_weights.cpu().tolist(),
        "random_seed": SEED + fold_number,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "monitor_metric": "val_loss",
        "monitor_mode": "min"
    }

    trainer = Trainer(
        model=model,
        model_name="cxformer_small",
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        output_dir=str(fold_output),
        monitor_metric="val_loss",
        monitor_mode="min",
        training_config=training_config
    )

    history = trainer.train(num_epochs=EPOCHS)

    best_path = fold_output / "cxformer_small_best.pth"
    assert best_path.exists()

    best_checkpoint = torch.load(
        best_path, map_location=device, weights_only=False
    )
    model.load_state_dict(
        best_checkpoint["model_state_dict"], strict=True
    )
    model.eval()

    best_metrics = trainer.evaluate(val_loader)

    result = {
        "fold": fold_number,
        "best_epoch": best_checkpoint["epoch"],
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "specificity": best_metrics["specificity"],
        "f1": best_metrics["macro_f1"],
        "auc": best_metrics.get("auc_roc"),
        "val_loss": best_metrics["loss"],
        "train_samples": len(train_idx),
        "val_samples": len(val_idx)
    }
    fold_results.append(result)

    with open(OUTPUT_ROOT / "fold_results.json", "w") as f:
        json.dump(fold_results, f, indent=2)

    del model, trainer, optimizer, scheduler, criterion
    del best_checkpoint, train_loader, val_loader
    torch.cuda.empty_cache()

results_df = pd.DataFrame(fold_results)
metric_names = [
    "accuracy", "precision", "recall",
    "specificity", "f1", "auc", "val_loss"
]
summary = pd.DataFrame({
    "Metric": metric_names,
    "Mean": [results_df[m].mean() for m in metric_names],
    "Standard Deviation": [
        results_df[m].std(ddof=1) for m in metric_names
    ]
})

results_df.to_csv(
    OUTPUT_ROOT / "five_fold_individual_results.csv",
    index=False
)
summary.to_csv(
    OUTPUT_ROOT / "five_fold_summary.csv",
    index=False
)
with open(OUTPUT_ROOT / "five_fold_summary.json", "w") as f:
    json.dump(summary.to_dict(orient="records"), f, indent=2)

print(results_df)
print(summary)
```

At the latest recorded point, the Kaggle notebook had reached Fold 3.
Final fold metrics had not yet been reported. Do not invent them.
Expected output:

``` text
/kaggle/working/cxformer_small_5fold_cv/
├── fold_1/
│   ├── cxformer_small_best.pth
│   ├── cxformer_small_latest.pth
│   └── cxformer_small_history.json
├── fold_2/
├── fold_3/
├── fold_4/
├── fold_5/
├── fold_results.json
├── five_fold_individual_results.csv
└── five_fold_summary.csv
```

The sample standard deviation (`ddof=1`) is used across the five fold
values. This is image-level stratified CV, not patient-grouped CV.

## 12. Grad-CAM explainability

### Why a special gradient path was needed

The normal `CXformerSmall.forward()` runs the frozen backbone inside
`torch.no_grad()`. That is suitable for training the head but prevents
gradients from flowing through the backbone for Grad-CAM.

For explanation, the trained model/checkpoint is not modified. The
backbone is called directly under `torch.enable_grad()`, and the same
trained classifier head is applied to the CLS token. A forward/backward
hook is attached to the penultimate transformer block:

``` python
model.backbone.encoder.layer[-2]
```

An initial hook on the final transformer block had non-zero gradients
for the full activation, but patch-token gradients were exactly zero
because the classification score uses the CLS token. The penultimate
block yielded non-zero patch-token gradients.

Observed Grad-CAM values:

-   Mean absolute gradient approximately `0.00010172`.
-   Raw CAM range approximately `-0.0061778` to `0.0070773`.
-   Normalized heatmap maximum approximately `0.98842`.

### Grad-CAM flow

``` text
Input X-ray
   |
   v
CXformer-Small backbone
   |
   v
Penultimate transformer block
   |                         |
   v                         v
Activations             CLS classification score
   |                         |
   |                         v
   |                    Backpropagation
   |                         |
   +------------+------------+
                |
                v
       Gradient-weighted patch tokens
                |
                v
          37 × 37 CAM grid
                |
                v
         Upsampled heatmap
                |
                v
     Overlay on original X-ray
```

### Consolidated Grad-CAM code pattern

This reference assumes the loaded wrapper exposes `backbone`,
`layer_norm`, `dropout`, and `classifier`, as described by the project
implementation. Verify those exact attribute names in the current
`src/models/cxformer.py` before reuse.

``` python
import torch
import torch.nn.functional as F
import numpy as np

class TransformerGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self.target_layer = model.backbone.encoder.layer[-2]

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(self, module, inputs, output):
        self.activations = (
            output[0] if isinstance(output, tuple) else output
        )

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output = self.model.backbone(pixel_values=input_tensor)
            tokens = output.last_hidden_state

            cls_features = tokens[:, 0, :]
            features = self.model.layer_norm(cls_features)
            features = self.model.dropout(features)
            logits = self.model.classifier(features)

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            logits[:, target_class].sum().backward()

        # Remove CLS and four register tokens; keep 1,369 patch tokens.
        activations = self.activations[:, 5:, :]
        gradients = self.gradients[:, 5:, :]

        weights = gradients.mean(dim=1, keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=-1))
        cam = cam.reshape(1, 1, 37, 37)

        cam = F.interpolate(
            cam, size=(518, 518),
            mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        probabilities = torch.softmax(
            logits.detach(), dim=1
        )[0].cpu().numpy()

        return {
            "heatmap": cam,
            "logits": logits.detach().cpu().numpy(),
            "probabilities": probabilities,
            "predicted_class": int(logits.argmax(dim=1).item()),
            "target_class": target_class
        }

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()
```

### Four representative cases

  Case               Test index Actual      Predicted   Result
  ---------------- ------------ ----------- ----------- -----------
  True Normal                 0 NORMAL      NORMAL      Correct
  True Pneumonia            231 PNEUMONIA   PNEUMONIA   Correct
  False Positive              3 NORMAL      PNEUMONIA   Incorrect
  False Negative            386 PNEUMONIA   NORMAL      Incorrect

Recorded probabilities, rounded:

  Case                          P(NORMAL)   P(PNEUMONIA)
  --------------------------- ----------- --------------
  True Normal, index 0              .6858          .3142
  True Pneumonia, index 231        \~.008         \~.992
  False Positive, index 3          \~.466         \~.534
  False Negative, index 386        \~.965         \~.035

Grad-CAM files saved under `/kaggle/working/cxformer_gradcam/`:

``` text
true_normal_gradcam.png
true_pneumonia_gradcam.png
false_positive_gradcam.png
false_negative_gradcam.png
four_case_gradcam_summary.png
gradcam_case_summary.csv
```

### Qualitative observations and limits

Heatmaps showed activation in parts of the lungs but also in image
borders, upper corners, and areas below the lungs. This raises a
possible artefact-reliance concern, but does not prove it. Each heatmap
was independently normalized, so brightness should not be compared
quantitatively across examples. Grad-CAM is a qualitative visualization,
not proof of pathology localization, causal reasoning, or clinical
validity. The false-negative example was classified NORMAL with
approximately 96.5% probability.

## 13. Project packaging

A Kaggle preparation package was created under `stage4_kaggle_prep/`:

-   `gpu_network_source.zip`
-   `cxformer_t4_verification.ipynb`
-   `KAGGLE_T4_VERIFICATION_GUIDE.md`
-   `package_project_for_kaggle.py`

The source ZIP contained 21 files and was approximately 45.54 KB. It
excluded datasets, pretrained weights, checkpoints, and Kaggle
credentials. It was checked for `kaggle.json`, data, weights, and
checkpoints.

## 14. Planned Streamlit dashboard

The dashboard is a planned component, not yet confirmed implemented.

Recommended sections:

1.  **Overview:** project summary, model information, dataset
    statistics, training configuration, and evaluation summary.
2.  **X-ray Prediction:** upload PNG/JPG/JPEG, preview, run inference,
    display predicted class and NORMAL/PNEUMONIA probabilities.
3.  **Explainability:** original X-ray, Grad-CAM heatmap, overlay, and
    an explanation that highlighted areas are model activations rather
    than verified disease regions.
4.  **Model Evaluation:** five-fold results with mean and standard
    deviation once available; accuracy, precision, recall, specificity,
    F1, ROC-AUC, confusion matrix; separate original test results.
5.  **About the Model:** CXformer-Small architecture, pretrained
    source/license, frozen backbone/trainable head, preprocessing,
    dataset and limitations.

Proposed technology:

  Component                   Technology
  --------------------------- -------------------------------------
  Interface                   Streamlit
  Inference                   PyTorch
  Image preprocessing         Official CXformer processor
  Explainability              Transformer Grad-CAM
  Charts                      Plotly
  Model artifact              Selected CXformer-Small checkpoint
  Training                    Kaggle Tesla T4
  Initial dashboard runtime   Local machine, CPU or available GPU

Proposed inference flow:

``` text
Upload chest X-ray
       |
       v
Validate image
       |
       v
Official CXformer preprocessing
       |
       v
Load selected CXformer-Small checkpoint
       |
       v
Classification inference
       |
       +--------------------+
       |                    |
       v                    v
Predicted class       Class probabilities
       |                    |
       +---------+----------+
                 |
                 v
             Grad-CAM
                 |
                 v
        Original + heatmap overlay
                 |
                 v
          Display results
```

The dashboard must use the same official processor and selected
checkpoint as the evaluation pipeline. CPU inference may be slower. The
model is a research prototype and must not be presented as a clinically
validated diagnostic system.

## 15. Key artifact paths

``` text
Restored source:
/kaggle/working/gpu-network/

Original training checkpoints:
/kaggle/working/gpu-network/cxformer_small_full_training_15epochs_v1/checkpoints/

Five-fold CV:
/kaggle/working/cxformer_small_5fold_cv/

Grad-CAM:
/kaggle/working/cxformer_gradcam/
```

Original checkpoint files:

``` text
cxformer_small_best.pth
cxformer_small_latest.pth
cxformer_small_history.json
```

Five-fold expected files:

``` text
fold_1/ through fold_5/
  cxformer_small_best.pth
  cxformer_small_latest.pth
  cxformer_small_history.json
fold_results.json
five_fold_individual_results.csv
five_fold_summary.csv
five_fold_summary.json
```

## 16. Project status at the time this document was assembled

  -----------------------------------------------------------------------
  Component                           Status
  ----------------------------------- -----------------------------------
  Prepared dataset and split audit    Completed

  CXformer-Small pretrained model     Completed
  inspection                          

  CXformer-Small wrapper              Implemented and tested

  CXformer-specific processor/data    Implemented and tested
  loader                              

  Training integration and CLI        Implemented and smoke-tested
  support                             

  Kaggle source package and T4        Created and checked
  verification notebook               

  Original 15-epoch training          Completed

  Best/latest checkpoints and history Saved and backed up

  Original validation evaluation      Completed

  Original test evaluation            Completed

  Test probability analysis           Completed

  Grad-CAM implementation and four    Completed
  example cases                       

  Five-fold stratified CV             In progress; Fold 3 had started at
                                      latest recorded update

  Final five-fold summary             Pending completion

  Streamlit dashboard                 Planned; not confirmed implemented
  -----------------------------------------------------------------------

## 17. Overall summary

The project classifies chest X-ray images into NORMAL and PNEUMONIA
using a pretrained CXformer-Small transformer. The 12-layer backbone has
six attention heads, hidden dimension 384, and four register tokens. The
backbone is frozen; a trainable head performs LayerNorm, dropout, and a
two-class linear projection.

The CXformer-specific processor performs histogram equalization, 518 ×
518 resize/center crop, rescaling, and ImageNet normalization. The
original model was trained for 15 epochs on a Kaggle Tesla T4 with batch
size 8, AdamW (learning rate .001, weight decay .01), dropout .3, and
weighted cross-entropy.

The selected original checkpoint achieved validation accuracy .9799, F1
.9743, and ROC-AUC .9972. On the 618-image prepared test set it achieved
accuracy .8900, pneumonia precision .8521, recall .9974, specificity
.7100, F1 .9190, and ROC-AUC .9852. The confusion matrix was TN 164, FP
67, FN 1, TP 386. The false-positive rate among NORMAL cases and the
validation-to-test gap are important limitations.

Grad-CAM was generated using the penultimate transformer block by
running a direct gradient-enabled backbone path for explanation only.
Four correct/incorrect cases were visualized. Some activation appeared
outside lung fields; this warrants further investigation but does not
establish artefact reliance.

Five-fold stratified CV uses the 5,206-image train-plus-validation pool
and excludes the original test set. It uses fresh classification heads
and fold-specific class weights. At the latest recorded status, Fold 3
had started; final CV results were not yet available.

A Streamlit dashboard is planned to provide upload, classification,
probabilities, Grad-CAM, and evaluation views. The project is a research
prototype and is not clinically validated.
