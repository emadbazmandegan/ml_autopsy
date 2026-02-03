"""
Unit tests for the threshold module
"""
import pytest
import numpy as np

from engine.threshold import (
    recompute_predictions,
    compute_metrics_at_threshold,
    compute_flip_samples,
    find_optimal_threshold,
    get_precision_recall_curve,
)


# Test recompute predictions
def test_recompute_predictions_at_threshold():
    """Test that predictions change correctly at different thresholds"""
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    
    # At 0.5, should be [0, 0, 1, 1]
    preds_50 = recompute_predictions(y_score, 0.5)
    assert list(preds_50) == [0, 0, 1, 1]
    
    # At 0.3, should be [0, 1, 1, 1]
    preds_30 = recompute_predictions(y_score, 0.3)
    assert list(preds_30) == [0, 1, 1, 1]
    
    # At 0.7, should be [0, 0, 0, 1]
    preds_70 = recompute_predictions(y_score, 0.7)
    assert list(preds_70) == [0, 0, 0, 1]


# Test metrics update
def test_metrics_update_on_threshold_change():
    """Test that metrics are correctly recalculated"""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    
    # At 0.5, predictions are [0, 0, 1, 1] -> perfect match
    metrics_50 = compute_metrics_at_threshold(y_true, y_score, 0.5)
    assert metrics_50["accuracy"] == 1.0
    assert metrics_50["f1"] == 1.0
    
    # At 0.3, predictions are [0, 1, 1, 1] -> one FP
    metrics_30 = compute_metrics_at_threshold(y_true, y_score, 0.3)
    assert metrics_30["fp"] == 1
    assert metrics_30["accuracy"] < 1.0


# Test optimal threshold
def test_optimal_threshold_in_valid_range():
    """Test that optimal threshold is in [0, 1]"""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.3, 0.6, 0.9, 0.4, 0.7])
    
    optimal, value = find_optimal_threshold(y_true, y_score)
    
    assert 0.0 <= optimal <= 1.0
    assert 0.0 <= value <= 1.0


# Test flip samples
def test_flip_samples_identified():
    """Test that flip samples are correctly identified"""
    y_score = np.array([0.45, 0.55, 0.35, 0.65])
    
    # From 0.5 to 0.4: samples at 0.45 should flip neg->pos
    flips = compute_flip_samples(y_score, 0.5, 0.4)
    
    assert flips["flip_count"] == 1
    assert 0 in flips["flip_indices"]
    assert flips["neg_to_pos_count"] == 1


def test_flip_samples_reverse_direction():
    """Test flips in the opposite direction"""
    y_score = np.array([0.45, 0.55, 0.35, 0.65])
    
    # From 0.5 to 0.6: samples at 0.55 should flip pos->neg
    flips = compute_flip_samples(y_score, 0.5, 0.6)
    
    assert flips["flip_count"] == 1
    assert flips["pos_to_neg_count"] == 1


# Test precision-recall curve
def test_precision_recall_curve_shape():
    """Test that PR curve returns valid data"""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    
    pr_data = get_precision_recall_curve(y_true, y_score)
    
    assert "precision" in pr_data
    assert "recall" in pr_data
    assert "thresholds" in pr_data
    assert len(pr_data["precision"]) > 0


# Test edge cases
def test_all_same_predictions():
    """Test when all predictions are same class"""
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    
    # At 0.9, all predictions are 0
    metrics = compute_metrics_at_threshold(y_true, y_score, 0.9)
    assert metrics["tp"] == 0
    assert metrics["precision"] == 0


def test_optimal_threshold_different_metrics():
    """Test optimal threshold for different metrics"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    
    f1_opt, _ = find_optimal_threshold(y_true, y_score, metric="f1")
    acc_opt, _ = find_optimal_threshold(y_true, y_score, metric="accuracy")
    
    # Both should be valid thresholds
    assert 0.0 <= f1_opt <= 1.0
    assert 0.0 <= acc_opt <= 1.0
