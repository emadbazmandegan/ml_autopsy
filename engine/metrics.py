"""
Metrics Module for Model Autopsy

Computes classification metrics, calibration, and confidence analysis.
"""
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Compute core classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multiclass ('binary', 'macro', 'micro', 'weighted')
        
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def compute_auc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """
    Compute AUC metrics when probability scores are available.
    
    Args:
        y_true: True labels
        y_score: Probability scores
        
    Returns:
        Dictionary with roc_auc and pr_auc
    """
    metrics = {}
    
    try:
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_score)
        metrics["roc_auc"] = None if np.isnan(roc_auc) else float(roc_auc)
    except ValueError:
        # Can fail if only one class present
        metrics["roc_auc"] = None
    
    try:
        # PR-AUC (Average Precision)
        pr_auc = average_precision_score(y_true, y_score)
        metrics["pr_auc"] = None if np.isnan(pr_auc) else float(pr_auc)
    except ValueError:
        metrics["pr_auc"] = None
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Compute confusion matrix and related counts.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of label values
        
    Returns:
        Dictionary with confusion matrix and counts
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    result = {
        "matrix": cm.tolist(),
        "labels": labels if labels else sorted(list(set(y_true) | set(y_pred))),
    }
    
    # For binary classification, add explicit counts
    if cm.shape == (2, 2):
        result["tn"] = int(cm[0, 0])
        result["fp"] = int(cm[0, 1])
        result["fn"] = int(cm[1, 0])
        result["tp"] = int(cm[1, 1])
    
    return result


def compute_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute calibration curve data.
    
    Args:
        y_true: True labels
        y_score: Probability scores
        n_bins: Number of calibration bins
        
    Returns:
        Dictionary with bin data for calibration curve
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_score, bin_edges[1:-1])
    
    # Compute per-bin statistics
    bins = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_data = {
                "bin_idx": i,
                "bin_lower": float(bin_edges[i]),
                "bin_upper": float(bin_edges[i + 1]),
                "count": int(mask.sum()),
                "mean_predicted": float(y_score[mask].mean()),
                "mean_actual": float(y_true[mask].mean()),
            }
            bins.append(bin_data)
    
    return {
        "n_bins": n_bins,
        "bins": bins,
        "total_samples": len(y_true),
    }


def compute_ece(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well the predicted probabilities match the actual outcomes.
    Lower is better (0 = perfectly calibrated).
    
    Args:
        y_true: True labels
        y_score: Probability scores
        n_bins: Number of calibration bins
        
    Returns:
        ECE value in range [0, 1]
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_score, bin_edges[1:-1])
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = bin_indices == i
        bin_size = mask.sum()
        
        if bin_size > 0:
            avg_confidence = y_score[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += (bin_size / total_samples) * abs(avg_confidence - avg_accuracy)
    
    return float(ece)


def compute_confident_wrong_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    confidence_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Compute the rate of confident but wrong predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Probability scores
        confidence_threshold: Threshold for "confident" predictions
        
    Returns:
        Dictionary with confident-wrong statistics
    """
    # For binary, confidence is max(p, 1-p)
    confidence = np.maximum(y_score, 1 - y_score)
    
    # Find confident predictions
    confident_mask = confidence >= confidence_threshold
    confident_count = confident_mask.sum()
    
    # Find confident but wrong
    wrong_mask = y_true != y_pred
    confident_wrong_mask = confident_mask & wrong_mask
    confident_wrong_count = confident_wrong_mask.sum()
    
    # Rate among confident predictions
    confident_wrong_rate = confident_wrong_count / confident_count if confident_count > 0 else 0.0
    
    return {
        "confidence_threshold": confidence_threshold,
        "total_samples": len(y_true),
        "confident_count": int(confident_count),
        "confident_wrong_count": int(confident_wrong_count),
        "confident_wrong_rate": float(confident_wrong_rate),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    task_type: str = "binary",
    n_calibration_bins: int = 10,
    confidence_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Compute all metrics for a classification task.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Optional probability scores
        task_type: 'binary' or 'multiclass'
        n_calibration_bins: Number of bins for calibration
        confidence_threshold: Threshold for confident-wrong analysis
        
    Returns:
        Dictionary with all computed metrics
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Determine averaging method
    average = "binary" if task_type == "binary" else "weighted"
    
    # Core metrics
    metrics = {
        "task_type": task_type,
        "n_samples": len(y_true),
        **compute_classification_metrics(y_true, y_pred, average=average),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
    }
    
    # Score-dependent metrics
    if y_score is not None:
        y_score = np.asarray(y_score)
        
        # AUC metrics
        auc_metrics = compute_auc_metrics(y_true, y_score)
        metrics.update(auc_metrics)
        
        # Calibration
        metrics["calibration"] = compute_calibration_curve(y_true, y_score, n_calibration_bins)
        metrics["ece"] = compute_ece(y_true, y_score, n_calibration_bins)
        
        # Confident-wrong
        metrics["confident_wrong"] = compute_confident_wrong_rate(
            y_true, y_pred, y_score, confidence_threshold
        )
    
    return metrics


def save_metrics(
    metrics: Dict[str, Any],
    output_dir: Path,
    filename: str = "metrics.json",
) -> Path:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_dir: Directory to save to
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    return output_path
