"""
Training CLI for Medical Image Classification.

Executes training for CustomCNN or ResNet-18 transfer learning.
Supports resuming, device auto-detection (CPU or CUDA), and smoke-test mode.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import build_custom_cnn, build_resnet18, build_cxformer_small
from src.data.dataset import create_data_loaders, CLASS_TO_IDX
from src.training.trainer import Trainer
from src.utils.device import get_compute_device, print_device_info


def set_seed(seed: int = 42, deterministic_cudnn: bool = True):
    """Sets global random seeds for full experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_prepared_dir(path: str) -> bool:
    """Checks whether a candidate directory contains the required splits and classes."""
    if not path or not os.path.isdir(path):
        return False
    required_splits = ["train", "val", "test"]
    required_classes = ["NORMAL", "PNEUMONIA"]
    for s in required_splits:
        for c in required_classes:
            if not os.path.isdir(os.path.join(path, s, c)):
                return False
    return True


def resolve_data_root(cfg: dict, override_path: str = None) -> str:
    # 1. Explicit CLI override
    if override_path:
        norm_override = os.path.normpath(os.path.abspath(override_path))
        if not is_valid_prepared_dir(norm_override):
            raise FileNotFoundError(
                f"Provided --data_dir does not contain a valid prepared dataset "
                f"(expected train/val/test splits with NORMAL/PNEUMONIA): {override_path}"
            )
        return norm_override

    # 2. Configured Kaggle working directory
    kaggle_p = cfg.get("dataset", {}).get("prepared_root_kaggle", "/kaggle/working/prepared")
    if kaggle_p and os.path.isdir(kaggle_p) and is_valid_prepared_dir(kaggle_p):
        return os.path.abspath(kaggle_p)

    # 3. Auto-discovery for attached dataset in /kaggle/input
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        candidates = []
        try:
            for item in sorted(os.listdir(kaggle_input)):
                cand_sub = os.path.join(kaggle_input, item, "prepared")
                if is_valid_prepared_dir(cand_sub):
                    candidates.append(cand_sub)
                cand_root = os.path.join(kaggle_input, item)
                if is_valid_prepared_dir(cand_root) and cand_root not in candidates:
                    candidates.append(cand_root)
        except Exception:
            candidates = []

        if len(candidates) == 1:
            return os.path.abspath(candidates[0])
        elif len(candidates) > 1:
            raise ValueError(
                f"Multiple prepared datasets detected in {kaggle_input}: {candidates}. "
                "Please specify which dataset to use explicitly via --data_dir."
            )

    # 4. Local path (relative to PROJECT_ROOT or current working directory)
    local_p = cfg.get("dataset", {}).get("prepared_root_local", "data/prepared")
    abs_local = os.path.normpath(os.path.abspath(os.path.join(PROJECT_ROOT, local_p)))
    if is_valid_prepared_dir(abs_local):
        return abs_local
    abs_cwd = os.path.normpath(os.path.abspath(local_p))
    if is_valid_prepared_dir(abs_cwd):
        return abs_cwd

    raise FileNotFoundError(
        f"Could not locate a valid prepared dataset directory. "
        f"Searched local path ('{abs_local}'), Kaggle working ('{kaggle_p}'), and {kaggle_input}. "
        "Please run preparation first or provide the path via --data_dir."
    )


def main():
    parser = argparse.ArgumentParser(description="Train Chest X-Ray Classification Models.")
    parser.add_argument("--model", type=str, choices=["custom_cnn", "resnet18", "cxformer_small"], default="resnet18",
                        help="Model architecture to train (custom_cnn, resnet18, or cxformer_small).")
    parser.add_argument("--config", type=str, default="configs/training_config.json",
                        help="Path to training configuration JSON.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override path to prepared dataset directory.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Override path to pretrained model checkpoint directory (e.g. for CXformer).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override mini-batch size.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run an inexpensive smoke test (1 epoch, 2 batches) and exit.")

    args = parser.parse_args()

    # Load configuration
    cfg_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    cfg = load_config(cfg_path)

    # Set random seed
    seed = cfg.get("reproducibility", {}).get("random_seed", 42)
    set_seed(seed)

    # Print device information
    print_device_info()
    device = get_compute_device()

    # Model settings
    model_cfg = cfg["models"][args.model]
    epochs = args.epochs or model_cfg["epochs"]
    batch_size = args.batch_size or model_cfg["batch_size"]
    lr = args.lr or model_cfg["learning_rate"]
    image_size = model_cfg.get("image_size", cfg["dataset"].get("image_size", 224))
    data_root = resolve_data_root(cfg, args.data_dir)

    print(f"\nConfiguration Summary:")
    print(f"  - Model         : {args.model}")
    print(f"  - Dataset Root  : {data_root}")
    print(f"  - Image Size    : {image_size}x{image_size}")
    print(f"  - Batch Size    : {batch_size}")
    print(f"  - Learning Rate : {lr}")
    print(f"  - Planned Epochs: {epochs}")

    # Determine num_workers
    is_windows = sys.platform.startswith("win")
    num_workers = cfg["hardware"]["num_workers_windows"] if is_windows else cfg["hardware"]["num_workers_linux"]

    # Build DataLoaders
    is_cxformer = args.model in ("cxformer_small", "cxformer")
    model_dir = None
    if is_cxformer:
        if args.model_dir:
            model_dir = args.model_dir
        else:
            cfg_dir = model_cfg.get("model_dir")
            if cfg_dir and (os.path.isdir(cfg_dir) or os.path.isdir(os.path.join(PROJECT_ROOT, cfg_dir))):
                model_dir = cfg_dir
            else:
                model_dir = None  # Triggers dynamic auto-discovery across /kaggle/input

    train_loader, val_loader, test_loader = create_data_loaders(
        prepared_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        model_type="cxformer" if is_cxformer else args.model,
        cxformer_model_dir=model_dir if is_cxformer else None,
    )

    # Build Model
    if args.model == "custom_cnn":
        model = build_custom_cnn(
            num_classes=len(CLASS_TO_IDX),
            in_channels=model_cfg.get("in_channels", 3),
            dropout_rate=model_cfg.get("dropout_rate", 0.4),
        )
    elif args.model == "resnet18":
        model = build_resnet18(
            num_classes=len(CLASS_TO_IDX),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )
    elif args.model in ("cxformer_small", "cxformer"):
        model = build_cxformer_small(
            model_dir=model_dir,
            num_classes=len(CLASS_TO_IDX),
            freeze_backbone=model_cfg.get("freeze_backbone", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
        )

    # Build Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=model_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    output_dir = os.path.join(PROJECT_ROOT, cfg.get("checkpoints", {}).get("output_dir", "checkpoints"))

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        model_name=args.model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        monitor_metric=model_cfg.get("monitor_metric", "val_loss"),
        monitor_mode=model_cfg.get("monitor_mode", "min"),
        training_config=model_cfg,
    )

    # Resume if requested
    if args.resume:
        resume_path = os.path.join(PROJECT_ROOT, args.resume) if not os.path.isabs(args.resume) else args.resume
        trainer.resume_from_checkpoint(resume_path)

    # Execute training or smoke test
    if args.smoke_test:
        print("\n" + "=" * 65)
        print(" RUNNING SMOKE TEST MODE (2 Train Batches, 1 Val Batch)")
        print(" Notice: This test validates execution mechanics only, not convergence.")
        print("=" * 65)
        train_metrics = trainer.train_epoch(epoch=1, max_batches=2)
        val_metrics = trainer.evaluate(val_loader, max_batches=1)
        print(f"Smoke Test Train Results: {train_metrics}")
        print(f"Smoke Test Val Results  : {val_metrics}")
        # Save, verify, reload, and clean up temporary smoke checkpoint
        smoke_ckpt_path = os.path.join(output_dir, f"{args.model}_smoke_test.pth")
        smoke_payload = {
            "smoke_test": True,
            "model_architecture": args.model,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        trainer._save_checkpoint_atomic(smoke_ckpt_path, smoke_payload)
        assert os.path.exists(smoke_ckpt_path), "Smoke checkpoint was not created!"
        reloaded = torch.load(smoke_ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in reloaded, "Missing model_state_dict in reloaded checkpoint!"
        os.remove(smoke_ckpt_path)
        print(f"Smoke Test Completed Successfully: Checkpoint save and load verified.")
        print("=" * 65)
        return

    # Full training loop
    trainer.train(num_epochs=epochs)


if __name__ == "__main__":
    main()
