"""
Comprehensive Smoke-Test Script for Medical Image Classification Pipeline.

Validates the full mechanics of the training pipeline in an inexpensive, local run:
1. System and compute hardware diagnostics (CPU / GPU detection)
2. DataLoader batch generation and tensor shape/type integrity
3. Dynamic class-weight calculation strictly from training labels
4. CustomCNN: forward pass, loss calculation, backward pass, optimizer step
5. ResNet-18: forward pass, loss calculation, backward pass, optimizer step
6. Atomic checkpoint saving, verification, reloading, and cleanup

DISCLAIMER:
This smoke test checks execution mechanics, tensor flow, and I/O integrity only.
It does NOT constitute evidence of model convergence or successful training.
"""

import os
import sys
import time
import shutil
import argparse
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ChestXRayDataset,
    create_data_loaders,
    compute_class_weights,
    get_transforms,
)
from src.models import build_custom_cnn, build_resnet18
from src.training.trainer import Trainer
from src.utils.device import get_compute_device, print_device_info


def run_smoke_test(data_dir: str = None):
    print("=" * 70)
    print(" MEDICAL IMAGE CLASSIFICATION PIPELINE: SMOKE TEST SUITE")
    print("=" * 70)
    print("Notice: This test validates pipeline mechanics and tensor operations.")
    print("It does NOT train models or prove statistical convergence.\n")

    t_start = time.time()
    device = get_compute_device()
    print_device_info()

    # Resolve dataset directory
    if data_dir:
        data_root = os.path.abspath(data_dir)
    else:
        default_local = os.path.join(PROJECT_ROOT, "data", "prepared")
        kaggle_default = "/kaggle/working/prepared"
        if os.path.isdir(default_local):
            data_root = default_local
        elif os.path.isdir(kaggle_default):
            data_root = kaggle_default
        else:
            data_root = default_local

    # Validate prepared dataset directory
    is_valid = False
    if os.path.isdir(data_root):
        splits_present = [s for s in ["train", "val", "test"] if os.path.isdir(os.path.join(data_root, s))]
        is_valid = (len(splits_present) == 3)

    if not is_valid:
        print(f"\n[ERROR] Prepared dataset directory is missing or invalid: {data_root}")
        print("Expected structure: train/, val/, test/ each with NORMAL/ and PNEUMONIA/ folders.")
        print("Remediation options:")
        print("  1. Run dataset preparation: python scripts/run_preparation.py")
        print("  2. Specify an existing prepared directory: python scripts/run_smoke_test.py --data_dir <path>")
        sys.exit(1)

    # ------------------------------------------------------------------------
    # STEP 1: DataLoader & Tensor Integrity Check
    # ------------------------------------------------------------------------
    print("\n[1/6] Testing DataLoaders and Tensor Pipeline...")
    t0 = time.time()
    train_loader, val_loader, test_loader = create_data_loaders(
        prepared_root=data_root,
        batch_size=4,
        image_size=224,
        num_workers=0,
    )
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 224, 224), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (4,), f"Unexpected label shape: {labels.shape}"
    assert images.dtype == torch.float32, f"Unexpected image dtype: {images.dtype}"
    assert labels.dtype == torch.int64, f"Unexpected label dtype: {labels.dtype}"
    print(f"  [PASS] DataLoaders initialized in {time.time() - t0:.2f}s")
    print(f"  [PASS] Batch tensor shape: {tuple(images.shape)}, dtype: {images.dtype}")
    print(f"  [PASS] Labels tensor shape: {tuple(labels.shape)}, sample values: {labels.tolist()}")

    # ------------------------------------------------------------------------
    # STEP 2: Dynamic Class Weight Calculation (Training ONLY)
    # ------------------------------------------------------------------------
    print("\n[2/6] Testing Dynamic Class Weight Calculation...")
    train_labels = train_loader.dataset.get_labels()
    weights = compute_class_weights(train_labels, num_classes=2, device=device)
    w0 = float(weights[0].item())
    w1 = float(weights[1].item())
    print(f"  [PASS] Training sample count: {len(train_labels)}")
    print(f"  [PASS] Class 0 (NORMAL) weight    : {w0:.4f}")
    print(f"  [PASS] Class 1 (PNEUMONIA) weight : {w1:.4f}")
    assert w0 > w1, f"Expected Class 0 weight > Class 1 weight due to class imbalance, got {w0} <= {w1}"
    assert w0 > 0.0 and w1 > 0.0, "Class weights must be strictly positive"

    criterion = nn.CrossEntropyLoss(weight=weights)

    # ------------------------------------------------------------------------
    # STEP 3: CustomCNN Mechanics (Forward, Loss, Backward, Optimizer Step)
    # ------------------------------------------------------------------------
    print("\n[3/6] Testing CustomCNN (Trained from Scratch) Mechanics...")
    t0 = time.time()
    cnn_model = build_custom_cnn(num_classes=2, in_channels=3, dropout_rate=0.4).to(device)
    cnn_opt = torch.optim.AdamW(cnn_model.parameters(), lr=1e-3)

    cnn_images = images.to(device)
    cnn_labels = labels.to(device)

    # Forward
    cnn_logits = cnn_model(cnn_images)
    assert cnn_logits.shape == (4, 2), f"Expected logits shape (4, 2), got {cnn_logits.shape}"
    print(f"  [PASS] CustomCNN forward pass successful: output shape {tuple(cnn_logits.shape)}")

    # Loss
    cnn_loss = criterion(cnn_logits, cnn_labels)
    assert torch.isfinite(cnn_loss), "Loss must be finite"
    print(f"  [PASS] CustomCNN weighted cross-entropy loss: {cnn_loss.item():.4f}")

    # Backward
    cnn_opt.zero_grad()
    cnn_loss.backward()
    grad_ok = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in cnn_model.parameters())
    assert grad_ok, "CustomCNN gradients were not computed during backward pass!"
    print(f"  [PASS] CustomCNN backward pass successful: gradients verified across parameters")

    # Optimizer Step
    cnn_opt.step()
    print(f"  [PASS] CustomCNN optimizer step executed in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------------
    # STEP 4: ResNet-18 Mechanics (Forward, Loss, Backward, Optimizer Step)
    # ------------------------------------------------------------------------
    print("\n[4/6] Testing ResNet-18 (Transfer Learning) Mechanics...")
    t0 = time.time()
    # For local test without network dependency, initialize cleanly
    resnet_model = build_resnet18(num_classes=2, pretrained=False, freeze_backbone=False).to(device)
    resnet_opt = torch.optim.AdamW(resnet_model.parameters(), lr=1e-4)

    resnet_logits = resnet_model(cnn_images)
    assert resnet_logits.shape == (4, 2), f"Expected logits shape (4, 2), got {resnet_logits.shape}"
    print(f"  [PASS] ResNet-18 forward pass successful: output shape {tuple(resnet_logits.shape)}")

    resnet_loss = criterion(resnet_logits, cnn_labels)
    assert torch.isfinite(resnet_loss), "Loss must be finite"
    print(f"  [PASS] ResNet-18 weighted cross-entropy loss: {resnet_loss.item():.4f}")

    resnet_opt.zero_grad()
    resnet_loss.backward()
    grad_ok = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in resnet_model.parameters())
    assert grad_ok, "ResNet-18 gradients were not computed during backward pass!"
    print(f"  [PASS] ResNet-18 backward pass successful: gradients verified across parameters")

    resnet_opt.step()
    print(f"  [PASS] ResNet-18 optimizer step executed in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------------
    # STEP 5: Atomic Checkpoint Save, Reload, and Metadata Verification
    # ------------------------------------------------------------------------
    print("\n[5/6] Testing Atomic Checkpoint Saving and Resuming...")
    t0 = time.time()
    test_ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "smoke_test_tmp")
    os.makedirs(test_ckpt_dir, exist_ok=True)

    trainer = Trainer(
        model=cnn_model,
        model_name="custom_cnn",
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=cnn_opt,
        criterion=criterion,
        device=device,
        output_dir=test_ckpt_dir,
        monitor_metric="val_loss",
        monitor_mode="min",
    )

    dummy_val_m = {"loss": 0.45, "accuracy": 0.85, "macro_f1": 0.84, "recall": 0.90, "specificity": 0.80, "auc_roc": 0.88}
    dummy_train_m = {"loss": 0.50, "accuracy": 0.82, "macro_f1": 0.81}

    # Save checkpoint
    trainer.save_checkpoint(epoch=1, val_metrics=dummy_val_m, train_metrics=dummy_train_m, is_best=True)

    best_ckpt = os.path.join(test_ckpt_dir, "custom_cnn_best.pth")
    assert os.path.isfile(best_ckpt), f"Best checkpoint not found: {best_ckpt}"

    # Load and verify checkpoint
    loaded = torch.load(best_ckpt, map_location=device)
    required_keys = [
        "epoch", "model_architecture", "model_state_dict", "optimizer_state_dict",
        "class_to_idx", "idx_to_class", "class_weights", "preprocessing_config",
        "training_config", "val_metrics", "best_metric_value", "timestamp"
    ]
    for k in required_keys:
        assert k in loaded, f"Missing required checkpoint key: {k}"

    print(f"  [PASS] Checkpoint saved and verified with all {len(required_keys)} metadata keys.")
    print(f"  [PASS] Embedded class mapping: {loaded['class_to_idx']}")
    print(f"  [PASS] Embedded class weights: {loaded['class_weights']}")

    # Cleanup temporary directory
    shutil.rmtree(test_ckpt_dir, ignore_errors=True)
    print(f"  [PASS] Temporary test checkpoints cleanly removed.")

    # ------------------------------------------------------------------------
    # STEP 6: Summary and Final Verdict
    # ------------------------------------------------------------------------
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f" SMOKE TEST RESULT: ALL 6 CHECKS PASSED ({total_elapsed:.2f}s)")
    print("=" * 70)
    print("Verified Capabilities:")
    print("  [PASS] PyTorch Dataset and DataLoader batch generation")
    print("  [PASS] Dynamic training class weighting (NORMAL vs PNEUMONIA)")
    print("  [PASS] CustomCNN: Forward, Loss, Backward, Optimizer step")
    print("  [PASS] ResNet-18: Forward, Loss, Backward, Optimizer step")
    print("  [PASS] Atomic checkpointing with complete evaluation metadata")
    print("=" * 70)
    print("REMINDER: This smoke test does NOT constitute evidence of model")
    print("convergence or full training completion. Full model training must be")
    print("conducted in the subsequent project phase after review.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline smoke test suite.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to prepared dataset directory.")
    args = parser.parse_args()
    run_smoke_test(data_dir=args.data_dir)
