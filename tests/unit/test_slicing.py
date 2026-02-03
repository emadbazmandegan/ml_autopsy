"""
Unit tests for the slicing module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from engine.slicing import (
    bin_numeric_feature,
    bin_categorical_feature,
    compute_slice_metrics,
    generate_feature_slices,
    generate_all_slices,
    rank_slices,
    get_slice_subset,
    save_slices,
)


# Fixtures
@pytest.fixture
def sample_canonical_df():
    """Sample canonical dataset with features and error columns"""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        "id": range(n),
        "age": np.random.randint(18, 80, n),
        "income": np.random.randint(20000, 150000, n),
        "education": np.random.choice(["high_school", "bachelors", "masters", "phd"], n),
        "region": np.random.choice(["north", "south", "east", "west", "central"], n),
        "y_true": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "y_pred": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "is_error": np.random.choice([True, False], n, p=[0.2, 0.8]),
        "error_type": np.random.choice(["TN", "FP", "FN", "TP"], n),
    })


@pytest.fixture
def simple_binary_df():
    """Simple dataset for testing exact calculations"""
    return pd.DataFrame({
        "feature_num": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "feature_cat": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
        "is_error": [False, False, True, False, True, True, False, False, False, True],
        "error_type": ["TN", "TN", "FP", "TN", "FN", "FP", "TN", "TN", "TN", "FN"],
    })


# Tests for numeric binning
def test_numeric_quantile_binning_produces_k_bins():
    """Test that numeric binning produces approximately k bins"""
    series = pd.Series(np.random.uniform(0, 100, 1000))
    
    binned, labels = bin_numeric_feature(series, n_bins=10)
    
    # Should have at most 10 unique bins
    assert len(labels) <= 10
    assert len(labels) >= 1
    
    # All values should be binned
    assert binned.notna().all()


def test_numeric_binning_handles_few_unique_values():
    """Test that binning handles data with fewer unique values than bins"""
    series = pd.Series([1, 1, 2, 2, 3, 3])
    
    binned, labels = bin_numeric_feature(series, n_bins=10)
    
    # Should use unique values as bins
    assert len(labels) == 3


def test_numeric_binning_creates_labels():
    """Test that numeric bins have readable labels"""
    series = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    binned, labels = bin_numeric_feature(series, n_bins=5)
    
    # Labels should be strings
    assert all(isinstance(label, str) for label in labels)


# Tests for categorical binning
def test_categorical_other_bucketing():
    """Test that rare categories are bucketed into OTHER"""
    series = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 10 + ["D"] * 5 + ["E"] * 3 + ["F"] * 2)
    
    binned, labels = bin_categorical_feature(series, top_k=3)
    
    # Should have top 3 + OTHER
    assert "OTHER" in labels
    assert len(labels) == 4
    
    # Check OTHER contains rare categories
    assert (binned == "OTHER").sum() == 10  # D + E + F


def test_categorical_no_other_when_few_categories():
    """Test that OTHER is not added when categories fit"""
    series = pd.Series(["A", "B", "C"])
    
    binned, labels = bin_categorical_feature(series, top_k=10)
    
    assert "OTHER" not in labels
    assert len(labels) == 3


# Tests for slice metrics
def test_slice_metrics_support_error_rate_lift(simple_binary_df):
    """Test that slice metrics are computed correctly"""
    df = simple_binary_df
    
    # Slice for category B
    mask = df["feature_cat"] == "B"
    metrics = compute_slice_metrics(df, mask)
    
    # Support should be 3
    assert metrics["support"] == 3
    assert metrics["support_pct"] == 0.3
    
    # Error rate: 2 errors out of 3
    assert metrics["error_rate"] == pytest.approx(2/3)
    
    # Lift: slice error rate / global error rate
    global_error_rate = df["is_error"].sum() / len(df)  # 4/10 = 0.4
    expected_lift = (2/3) / global_error_rate
    assert metrics["lift"] == pytest.approx(expected_lift)


def test_slice_metrics_fp_fn_counts(simple_binary_df):
    """Test that FP/FN counts are tracked correctly"""
    df = simple_binary_df
    
    # Slice for category B has FN and FP
    mask = df["feature_cat"] == "B"
    metrics = compute_slice_metrics(df, mask)
    
    assert metrics["fn_count"] == 1
    assert metrics["fp_count"] == 1


def test_slice_metrics_empty_slice():
    """Test handling of empty slices"""
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "is_error": [True, False, False],
        "error_type": ["FP", "TN", "TN"],
    })
    
    empty_mask = pd.Series([False, False, False])
    metrics = compute_slice_metrics(df, empty_mask)
    
    assert metrics["support"] == 0
    assert metrics["error_rate"] == 0.0
    assert metrics["lift"] == 0.0


# Tests for support filtering
def test_support_filter_applied():
    """Test that slices below minimum support are filtered"""
    slices = [
        {"feature": "A", "value": "1", "support": 100, "support_pct": 0.1, "lift": 1.5},
        {"feature": "A", "value": "2", "support": 5, "support_pct": 0.005, "lift": 3.0},
        {"feature": "B", "value": "1", "support": 50, "support_pct": 0.05, "lift": 2.0},
    ]
    
    ranked = rank_slices(slices, min_support=10, min_support_pct=0.01)
    
    # Should filter out slice with support=5
    assert len(ranked) == 2
    assert all(s["support"] >= 10 for s in ranked)


def test_top_slices_sorted_by_lift_then_support():
    """Test that slices are sorted correctly by lift"""
    slices = [
        {"feature": "A", "value": "1", "support": 100, "support_pct": 0.1, "lift": 1.5},
        {"feature": "B", "value": "1", "support": 80, "support_pct": 0.08, "lift": 3.0},
        {"feature": "C", "value": "1", "support": 90, "support_pct": 0.09, "lift": 2.0},
    ]
    
    ranked = rank_slices(slices, min_support=1, min_support_pct=0.0, sort_by="lift")
    
    # Should be sorted by lift descending
    assert ranked[0]["lift"] == 3.0
    assert ranked[1]["lift"] == 2.0
    assert ranked[2]["lift"] == 1.5
    
    # Should have ranks
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[2]["rank"] == 3


def test_top_n_limit():
    """Test that top_n limits results"""
    slices = [
        {"feature": f"F{i}", "value": "1", "support": 100, "support_pct": 0.1, "lift": float(i)}
        for i in range(10)
    ]
    
    ranked = rank_slices(slices, min_support=1, min_support_pct=0.0, top_n=5)
    
    assert len(ranked) == 5


# Tests for slice drilldown
def test_slice_drilldown_subset_matches_rule(sample_canonical_df):
    """Test that drilldown returns correct subset"""
    df = sample_canonical_df
    
    # Get a categorical slice
    subset = get_slice_subset(df, "education", "masters", top_k_categories=10)
    
    # All rows should have masters education
    assert all(subset["education"] == "masters")


# Integration tests
def test_generate_all_slices_produces_results(sample_canonical_df):
    """Test that slices are generated for all features"""
    df = sample_canonical_df
    
    slices = generate_all_slices(df)
    
    # Should have slices
    assert len(slices) > 0
    
    # Each slice should have required fields
    for s in slices:
        assert "feature" in s
        assert "value" in s
        assert "support" in s
        assert "error_rate" in s
        assert "lift" in s


def test_save_slices_creates_file(sample_canonical_df):
    """Test that slices are saved to CSV"""
    df = sample_canonical_df
    slices = generate_all_slices(df)
    ranked = rank_slices(slices, min_support=5, min_support_pct=0.01)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = save_slices(ranked, Path(tmpdir))
        
        assert output_path.exists()
        assert output_path.name == "slices_single.csv"
        
        # Verify it's readable
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == len(ranked)
        assert "feature" in loaded_df.columns
        assert "lift" in loaded_df.columns
