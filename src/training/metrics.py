"""
Medical classification metrics calculation module.

Provides comprehensive diagnostic metrics:
- Overall Accuracy
- Precision (Positive Predictive Value)
- Recall / Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- Macro F1 Score
- Area Under the ROC Curve (AUC-ROC)
"""

from typing import Dict, List, Optional
import numpy as np


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[List[float]] = None,
) -> Dict[str, Optional[float]]:
    """
    Computes diagnostic metrics for binary medical classification.
    
    Assumes binary classes:
      0: NORMAL (Negative)
      1: PNEUMONIA (Positive)
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted discrete labels (0 or 1).
        y_probs: Predicted probabilities for the positive class (class 1: PNEUMONIA).
        
    Returns:
        Dict containing accuracy, precision, recall, specificity, macro_f1, and auc_roc.
        Note: auc_roc is None when undefined (e.g. only one class present or y_probs is None).
    """
    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)

    total = len(y_true_arr)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "macro_f1": 0.0,
            "auc_roc": None,
        }

    # Confusion matrix counts (Class 1 = Positive, Class 0 = Negative)
    tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
    tn = np.sum((y_true_arr == 0) & (y_pred_arr == 0))
    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))

    accuracy = float((tp + tn) / total)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # F1 score for class 1
    f1_pos = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Precision, Recall, F1 for class 0 (Normal)
    prec_neg = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_neg = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * (prec_neg * rec_neg) / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0

    macro_f1 = float((f1_pos + f1_neg) / 2.0)

    # AUC-ROC computation
    auc_roc = None
    if y_probs is not None and len(y_probs) == total:
        unique_classes = np.unique(y_true_arr)
        if len(unique_classes) > 1:
            try:
                from sklearn.metrics import roc_auc_score
            except ImportError as err:
                raise ImportError(
                    "scikit-learn is required to compute ROC-AUC score, but could not be imported. "
                    "Please install scikit-learn>=1.2.0 or pass y_probs=None."
                ) from err

            try:
                auc_roc = float(roc_auc_score(y_true_arr, y_probs))
            except ValueError:
                # Undefined if probability inputs or labels do not conform to ROC domain
                auc_roc = None
        else:
            # Explicitly undefined when fewer than 2 classes are present
            auc_roc = None

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "macro_f1": round(macro_f1, 4),
        "auc_roc": round(auc_roc, 4) if auc_roc is not None else None,
    }
