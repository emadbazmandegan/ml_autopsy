"""
Threshold Module for Model Autopsy

Interactive threshold adjustment and analysis.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score


def recompute_predictions(
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Recompute binary predictions using a new threshold.
    
    Args:
        y_score: Prediction probabilities
        threshold: Classification threshold
        
    Returns:
        Binary predictions array
    """
    return (np.asarray(y_score) >= threshold).astype(int)


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics at a specific threshold.
    
    Args:
        y_true: True labels
        y_score: Prediction probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = recompute_predictions(y_score, threshold)
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Confusion matrix elements
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    total = len(y_true)
    
    # Metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    error_rate = (fp + fn) / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_rate": error_rate,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_flip_samples(
    y_score: np.ndarray,
    old_threshold: float,
    new_threshold: float,
) -> Dict[str, Any]:
    """
    Find samples that change classification between two thresholds.
    
    Args:
        y_score: Prediction probabilities
        old_threshold: Original threshold
        new_threshold: New threshold
        
    Returns:
        Dictionary with flip information
    """
    y_score = np.asarray(y_score)
    
    old_preds = recompute_predictions(y_score, old_threshold)
    new_preds = recompute_predictions(y_score, new_threshold)
    
    # Find flips
    flipped = old_preds != new_preds
    flip_indices = np.where(flipped)[0]
    
    # Categorize flips
    pos_to_neg = (old_preds == 1) & (new_preds == 0)
    neg_to_pos = (old_preds == 0) & (new_preds == 1)
    
    return {
        "flip_count": int(flipped.sum()),
        "flip_indices": flip_indices.tolist(),
        "pos_to_neg_count": int(pos_to_neg.sum()),
        "neg_to_pos_count": int(neg_to_pos.sum()),
        "flip_rate": float(flipped.mean()),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Find optimal threshold that maximizes a given metric.
    
    Args:
        y_true: True labels
        y_score: Prediction probabilities
        metric: Metric to optimize ("f1", "accuracy", "balanced")
        n_thresholds: Number of thresholds to try
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    best_threshold = 0.5
    best_value = 0
    
    for t in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_score, t)
        
        if metric == "f1":
            value = metrics["f1"]
        elif metric == "accuracy":
            value = metrics["accuracy"]
        elif metric == "balanced":
            # Balanced accuracy = (recall + specificity) / 2
            specificity = metrics["tn"] / (metrics["tn"] + metrics["fp"]) if (metrics["tn"] + metrics["fp"]) > 0 else 0
            value = (metrics["recall"] + specificity) / 2
        else:
            value = metrics["f1"]
        
        if value > best_value:
            best_value = value
            best_threshold = t
    
    return best_threshold, best_value


def get_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, List[float]]:
    """
    Get precision-recall curve data for plotting.
    
    Args:
        y_true: True labels
        y_score: Prediction probabilities
        
    Returns:
        Dictionary with precision, recall, and thresholds
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }
