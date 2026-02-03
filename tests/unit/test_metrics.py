"""
Unit tests for the metrics module
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sklearn_confusion_matrix,
)

from engine.metrics import (
    compute_classification_metrics,
    compute_auc_metrics,
    compute_confusion_matrix,
    compute_calibration_curve,
    compute_ece,
    compute_confident_wrong_rate,
    compute_all_metrics,
    save_metrics,
)
from engine.plots import (
    plot_confusion_matrix,
    plot_calibration_curve,
    generate_all_plots,
)


# Fixtures
@pytest.fixture
def binary_data():
    """Sample binary classification data"""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.6, 0.9, 0.3, 0.8, 0.2, 0.7, 0.55, 0.85])
    return y_true, y_pred, y_score


@pytest.fixture
def perfectly_calibrated_data():
    """Data where predictions match true probabilities"""
    np.random.seed(42)
    n = 1000
    y_score = np.random.uniform(0, 1, n)
    y_true = (np.random.uniform(0, 1, n) < y_score).astype(int)
    y_pred = (y_score >= 0.5).astype(int)
    return y_true, y_pred, y_score


# Test classification metrics match sklearn
def test_metrics_accuracy_matches_sklearn(binary_data):
    """Test that our accuracy matches sklearn's"""
    y_true, y_pred, _ = binary_data
    
    our_metrics = compute_classification_metrics(y_true, y_pred)
    sklearn_accuracy = accuracy_score(y_true, y_pred)
    
    assert our_metrics["accuracy"] == pytest.approx(sklearn_accuracy)


def test_metrics_precision_matches_sklearn(binary_data):
    """Test that our precision matches sklearn's"""
    y_true, y_pred, _ = binary_data
    
    our_metrics = compute_classification_metrics(y_true, y_pred)
    sklearn_precision = precision_score(y_true, y_pred, zero_division=0)
    
    assert our_metrics["precision"] == pytest.approx(sklearn_precision)


def test_metrics_recall_matches_sklearn(binary_data):
    """Test that our recall matches sklearn's"""
    y_true, y_pred, _ = binary_data
    
    our_metrics = compute_classification_metrics(y_true, y_pred)
    sklearn_recall = recall_score(y_true, y_pred, zero_division=0)
    
    assert our_metrics["recall"] == pytest.approx(sklearn_recall)


def test_metrics_f1_matches_sklearn(binary_data):
    """Test that our f1 matches sklearn's"""
    y_true, y_pred, _ = binary_data
    
    our_metrics = compute_classification_metrics(y_true, y_pred)
    sklearn_f1 = f1_score(y_true, y_pred, zero_division=0)
    
    assert our_metrics["f1"] == pytest.approx(sklearn_f1)


# Test confusion matrix
def test_confusion_matrix_counts_correct(binary_data):
    """Test that confusion matrix counts match sklearn"""
    y_true, y_pred, _ = binary_data
    
    our_cm = compute_confusion_matrix(y_true, y_pred)
    sklearn_cm = sklearn_confusion_matrix(y_true, y_pred)
    
    assert our_cm["matrix"] == sklearn_cm.tolist()
    
    # Check binary counts
    assert our_cm["tn"] == sklearn_cm[0, 0]
    assert our_cm["fp"] == sklearn_cm[0, 1]
    assert our_cm["fn"] == sklearn_cm[1, 0]
    assert our_cm["tp"] == sklearn_cm[1, 1]


def test_confusion_matrix_sums_to_n(binary_data):
    """Test that confusion matrix sums to total samples"""
    y_true, y_pred, _ = binary_data
    
    our_cm = compute_confusion_matrix(y_true, y_pred)
    total = our_cm["tn"] + our_cm["fp"] + our_cm["fn"] + our_cm["tp"]
    
    assert total == len(y_true)


# Test AUC metrics
def test_auc_computed_when_scores_present(binary_data):
    """Test that AUC metrics are computed when scores present"""
    y_true, _, y_score = binary_data
    
    auc_metrics = compute_auc_metrics(y_true, y_score)
    sklearn_roc_auc = roc_auc_score(y_true, y_score)
    
    assert auc_metrics["roc_auc"] == pytest.approx(sklearn_roc_auc)
    assert auc_metrics["pr_auc"] is not None


def test_auc_handles_single_class():
    """Test that AUC handles single-class edge case gracefully"""
    y_true = np.array([0, 0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    auc_metrics = compute_auc_metrics(y_true, y_score)
    
    # Should return None instead of raising
    assert auc_metrics["roc_auc"] is None


# Test calibration
def test_calibration_bins_sum_to_n(binary_data):
    """Test that calibration bin counts sum to total samples"""
    y_true, _, y_score = binary_data
    
    calibration = compute_calibration_curve(y_true, y_score, n_bins=10)
    total_in_bins = sum(b["count"] for b in calibration["bins"])
    
    assert total_in_bins == len(y_true)


def test_calibration_bin_ranges_valid(binary_data):
    """Test that calibration bins have valid ranges"""
    y_true, _, y_score = binary_data
    
    calibration = compute_calibration_curve(y_true, y_score, n_bins=10)
    
    for bin_data in calibration["bins"]:
        assert 0 <= bin_data["bin_lower"] <= 1
        assert 0 <= bin_data["bin_upper"] <= 1
        assert bin_data["bin_lower"] < bin_data["bin_upper"]
        assert 0 <= bin_data["mean_predicted"] <= 1
        assert 0 <= bin_data["mean_actual"] <= 1


# Test ECE
def test_ece_in_range_0_1(binary_data):
    """Test that ECE is in valid range [0, 1]"""
    y_true, _, y_score = binary_data
    
    ece = compute_ece(y_true, y_score, n_bins=10)
    
    assert 0 <= ece <= 1


def test_ece_perfect_calibration():
    """Test that perfectly calibrated predictions have low ECE"""
    # Perfectly calibrated: predicted probability matches actual rate
    y_score = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    
    ece = compute_ece(y_true, y_score, n_bins=3)
    
    # Should be very low for well-calibrated predictions
    assert ece < 0.2


# Test confident-wrong rate
def test_confident_wrong_rate_valid(binary_data):
    """Test confident-wrong rate is computed correctly"""
    y_true, y_pred, y_score = binary_data
    
    result = compute_confident_wrong_rate(y_true, y_pred, y_score, confidence_threshold=0.7)
    
    assert 0 <= result["confident_wrong_rate"] <= 1
    assert result["confident_wrong_count"] <= result["confident_count"]
    assert result["confident_count"] <= result["total_samples"]


# Test compute_all_metrics
def test_compute_all_metrics_complete(binary_data):
    """Test that all expected metrics are computed"""
    y_true, y_pred, y_score = binary_data
    
    metrics = compute_all_metrics(y_true, y_pred, y_score, task_type="binary")
    
    # Check all expected keys
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics
    assert "calibration" in metrics
    assert "ece" in metrics
    assert "confident_wrong" in metrics


def test_compute_all_metrics_without_scores():
    """Test metrics computation without probability scores"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    
    metrics = compute_all_metrics(y_true, y_pred, y_score=None)
    
    # Should have basic metrics but not score-dependent ones
    assert "accuracy" in metrics
    assert "roc_auc" not in metrics
    assert "calibration" not in metrics


# Test save/load metrics
def test_save_metrics_creates_file(binary_data):
    """Test that metrics are saved to file correctly"""
    y_true, y_pred, y_score = binary_data
    metrics = compute_all_metrics(y_true, y_pred, y_score)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = save_metrics(metrics, Path(tmpdir))
        
        assert output_path.exists()
        assert output_path.name == "metrics.json"
        
        # Verify it's valid JSON
        import json
        with open(output_path, "r") as f:
            loaded = json.load(f)
        
        assert loaded["accuracy"] == metrics["accuracy"]


# Integration tests for plots
def test_pipeline_writes_metrics_and_plots(binary_data):
    """Integration test: metrics and plots are written correctly"""
    y_true, y_pred, y_score = binary_data
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        plots_dir = Path(tmpdir) / "plots"
        
        # Compute metrics
        metrics = compute_all_metrics(y_true, y_pred, y_score)
        
        # Save metrics
        metrics_path = save_metrics(metrics, output_dir)
        assert metrics_path.exists()
        
        # Generate plots
        plot_paths = generate_all_plots(metrics, plots_dir)
        
        # Verify all plots exist and are non-empty
        assert "confusion" in plot_paths
        assert plot_paths["confusion"].exists()
        assert plot_paths["confusion"].stat().st_size > 0
        
        assert "calibration" in plot_paths
        assert plot_paths["calibration"].exists()
        assert plot_paths["calibration"].stat().st_size > 0


def test_confusion_matrix_plot_created(binary_data):
    """Test confusion matrix plot is created"""
    y_true, y_pred, _ = binary_data
    confusion_data = compute_confusion_matrix(y_true, y_pred)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "confusion.png"
        result_path = plot_confusion_matrix(confusion_data, output_path)
        
        assert result_path.exists()
        assert result_path.stat().st_size > 0


def test_calibration_curve_plot_created(binary_data):
    """Test calibration curve plot is created"""
    y_true, _, y_score = binary_data
    calibration_data = compute_calibration_curve(y_true, y_score)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "calibration.png"
        result_path = plot_calibration_curve(calibration_data, output_path)
        
        assert result_path.exists()
        assert result_path.stat().st_size > 0
