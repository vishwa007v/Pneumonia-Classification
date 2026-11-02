"""
Training and Validation Engine for Medical Image Classification.

Key Features:
- Dynamic class-weighted cross-entropy loss applied strictly from training labels
- Multi-metric tracking: Loss, Accuracy, Precision, Recall/Sensitivity, Specificity, Macro F1, AUC-ROC
- Validation-based model selection (test set is never touched during training)
- Safe, atomic checkpoint saving (prevents corruption on interrupt)
- Complete evaluation reproducibility metadata embedded in checkpoints
- Fully resumable training from latest or best checkpoints
- Low-overhead smoke test mode for environment and pipeline verification
"""

import os
import json
import time
import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    compute_class_weights,
    ChestXRayDataset,
)
from src.training.metrics import compute_classification_metrics
from src.utils.device import get_compute_device


# Metric canonical mapping and expected optimization direction
METRIC_REGISTRY: Dict[str, Tuple[str, str]] = {
    # alias: (canonical_name, required_mode)
    "loss": ("loss", "min"),
    "val_loss": ("loss", "min"),
    "auc": ("auc_roc", "max"),
    "val_auc": ("auc_roc", "max"),
    "auc_roc": ("auc_roc", "max"),
    "val_auc_roc": ("auc_roc", "max"),
    "roc_auc": ("auc_roc", "max"),
    "val_roc_auc": ("auc_roc", "max"),
    "f1": ("macro_f1", "max"),
    "val_f1": ("macro_f1", "max"),
    "macro_f1": ("macro_f1", "max"),
    "val_macro_f1": ("macro_f1", "max"),
    "acc": ("accuracy", "max"),
    "val_acc": ("accuracy", "max"),
    "accuracy": ("accuracy", "max"),
    "val_accuracy": ("accuracy", "max"),
    "prec": ("precision", "max"),
    "precision": ("precision", "max"),
    "val_precision": ("precision", "max"),
    "rec": ("recall", "max"),
    "recall": ("recall", "max"),
    "val_recall": ("recall", "max"),
    "spec": ("specificity", "max"),
    "specificity": ("specificity", "max"),
    "val_specificity": ("specificity", "max"),
}


class Trainer:
    """
    Manages the training, validation, checkpointing, and metric logging lifecycle.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "checkpoints",
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
        training_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_compute_device()
        self.output_dir = output_dir

        # Validate monitoring metric and mode
        norm_metric = monitor_metric.lower().strip()
        if norm_metric not in METRIC_REGISTRY:
            supported = sorted(list(METRIC_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported monitor_metric '{monitor_metric}'. "
                f"Supported metric names and aliases are: {supported}"
            )

        canonical_metric, expected_mode = METRIC_REGISTRY[norm_metric]
        norm_mode = monitor_mode.lower().strip()
        if norm_mode not in ("min", "max"):
            raise ValueError(f"Invalid monitor_mode '{monitor_mode}'. Expected 'min' or 'max'.")

        if norm_mode != expected_mode:
            raise ValueError(
                f"Metric direction mismatch: metric '{monitor_metric}' (canonical: '{canonical_metric}') "
                f"expects monitor_mode='{expected_mode}', but received '{monitor_mode}'."
            )

        self.monitor_metric = monitor_metric
        self.canonical_metric = canonical_metric
        self.monitor_mode = norm_mode
        self.training_config = training_config or {}

        # Move model to compute device
        self.model.to(self.device)

        # Setup loss function with dynamic class weights
        if criterion is not None:
            self.criterion = criterion
            self.class_weights = getattr(criterion, "weight", None)
        else:
            self.class_weights, self.criterion = self._setup_weighted_criterion()

        # Checkpoint directory
        os.makedirs(self.output_dir, exist_ok=True)

        # State tracking
        self.start_epoch = 1
        self.best_metric_value = float("inf") if self.monitor_mode == "min" else float("-inf")
        self.history: List[Dict[str, Any]] = []

    def _setup_weighted_criterion(self) -> Tuple[torch.Tensor, nn.CrossEntropyLoss]:
        """
        Dynamically calculates class weights strictly from training labels.
        """
        if not hasattr(self.train_loader.dataset, "get_labels"):
            raise AttributeError("Train dataset must provide get_labels() method for class weight calculation.")

        train_labels = self.train_loader.dataset.get_labels()
        weights = compute_class_weights(train_labels, num_classes=len(CLASS_TO_IDX), device=self.device)

        # Diagnostic log
        w0, w1 = float(weights[0].item()), float(weights[1].item())
        print(f"Dynamic training class weights calculated strictly from {len(train_labels)} training samples:")
        print(f"  - Class 0 (NORMAL)    : {w0:.4f}")
        print(f"  - Class 1 (PNEUMONIA) : {w1:.4f}")
        if w0 <= w1:
            print("Warning: Expected NORMAL weight to exceed PNEUMONIA weight due to dataset imbalance.")

        criterion = nn.CrossEntropyLoss(weight=weights)
        return weights, criterion

    def _is_better_metric(self, current: Optional[float], best: float) -> bool:
        """
        Determines if the current validation metric outperforms the best observed so far.
        Returns False if current is None or NaN (undefined metrics cannot beat best).
        """
        if current is None or (isinstance(current, float) and np.isnan(current)):
            return False
        if best is None or (isinstance(best, float) and np.isnan(best)):
            return True
        if self.monitor_mode == "min":
            return current < best
        return current > best

    def _extract_monitored_metric(self, metrics_dict: Dict[str, Any]) -> Optional[float]:
        """
        Extracts the target monitored metric from a metric dictionary using
        the validated canonical metric key.
        """
        if self.canonical_metric not in metrics_dict:
            raise KeyError(
                f"Monitored canonical metric '{self.canonical_metric}' not found in metrics dictionary: {list(metrics_dict.keys())}"
            )
        val = metrics_dict[self.canonical_metric]
        if val is None:
            return None
        return float(val)

    def train_epoch(self, epoch: int, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Executes one full training epoch across the training dataset.
        """
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[float] = []

        batch_count = 0
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            # Record batch loss
            total_loss += loss.item() * images.size(0)
            batch_count += 1

            # Predictions and positive class probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].detach().cpu().numpy().tolist())

        total_samples = len(all_targets)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
        metrics["loss"] = round(avg_loss, 4)
        return metrics

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluates the model without computing gradients.
        Strictly used on validation data during training for checkpoint selection.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[float] = []

        batch_count = 0
        for batch_idx, (images, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            batch_count += 1

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        total_samples = len(all_targets)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
        metrics["loss"] = round(avg_loss, 4)
        return metrics

    def _save_checkpoint_atomic(self, file_path: str, checkpoint_data: Dict[str, Any]):
        """
        Saves a checkpoint atomically using a temporary file and os.replace
        to prevent file corruption during an interrupted run.
        """
        tmp_path = file_path + ".tmp"
        torch.save(checkpoint_data, tmp_path)
        os.replace(tmp_path, file_path)

    def save_checkpoint(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
        train_metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """
        Saves full evaluation-reproducible checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "model_architecture": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS,
            "class_weights": self.class_weights.detach().cpu().numpy().tolist() if self.class_weights is not None else None,
            "preprocessing_config": {
                "image_size": self.training_config.get("image_size", 224),
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "interpolation": "bilinear",
            },
            "training_config": self.training_config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_metric_value": self.best_metric_value,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "device": str(self.device),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # 1. Save latest checkpoint for safe resumption
        latest_path = os.path.join(self.output_dir, f"{self.model_name}_latest.pth")
        self._save_checkpoint_atomic(latest_path, checkpoint)

        # 2. Save best checkpoint if model improved
        if is_best:
            best_path = os.path.join(self.output_dir, f"{self.model_name}_best.pth")
            self._save_checkpoint_atomic(best_path, checkpoint)
            metric_disp = self._extract_monitored_metric(val_metrics)
            disp_str = f"{metric_disp:.4f}" if metric_disp is not None else "N/A"
            print(f"[*] Checkpoint saved: New best {self.monitor_metric} = {disp_str} -> {best_path}")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resumes training state from a saved checkpoint file.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"Resuming training state from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Architecture safety check
        saved_arch = checkpoint.get("model_architecture")
        if saved_arch and saved_arch != self.model_name:
            raise ValueError(f"Checkpoint architecture mismatch: {saved_arch} vs {self.model_name}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_metric_value = checkpoint.get("best_metric_value", self.best_metric_value)
        print(f"Resumed at epoch {self.start_epoch}. Prior best {self.monitor_metric}: {self.best_metric_value}")

    def train(self, num_epochs: int) -> List[Dict[str, Any]]:
        """
        Full training loop over num_epochs.
        """
        print(f"\nStarting training: {self.model_name} for {num_epochs} epochs on {self.device}")
        print(f"Monitoring metric for checkpoint selection: {self.monitor_metric} (mode: {self.monitor_mode})")

        for epoch in range(self.start_epoch, num_epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.evaluate(self.val_loader)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_m.get("loss", 0.0))
                else:
                    self.scheduler.step()

            # Check if current epoch sets a new best validation metric
            monitored_val = self._extract_monitored_metric(val_m)
            is_best = self._is_better_metric(monitored_val, self.best_metric_value)
            if is_best:
                self.best_metric_value = monitored_val

            self.save_checkpoint(epoch, val_m, train_m, is_best=is_best)

            elapsed = time.time() - t0
            epoch_record = {
                "epoch": epoch,
                "elapsed_sec": round(elapsed, 2),
                "train_loss": train_m["loss"],
                "train_acc": train_m["accuracy"],
                "train_f1": train_m["macro_f1"],
                "val_loss": val_m["loss"],
                "val_acc": val_m["accuracy"],
                "val_recall": val_m["recall"],
                "val_specificity": val_m["specificity"],
                "val_f1": val_m["macro_f1"],
                "val_auc": val_m.get("auc_roc"),
                "is_best": is_best,
            }
            self.history.append(epoch_record)

            val_auc_val = val_m.get("auc_roc")
            val_auc_str = f"{val_auc_val:.4f}" if val_auc_val is not None else "N/A"

            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_m['loss']:.4f}, Acc: {train_m['accuracy']:.4f} | "
                f"Val Loss: {val_m['loss']:.4f}, Acc: {val_m['accuracy']:.4f}, Recall: {val_m['recall']:.4f}, "
                f"Spec: {val_m['specificity']:.4f}, F1: {val_m['macro_f1']:.4f}, AUC: {val_auc_str}"
            )

        # Save training history
        history_path = os.path.join(self.output_dir, f"{self.model_name}_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")

        return self.history
