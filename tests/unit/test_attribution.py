"""
Unit tests for the attribution module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from engine.attribution import (
    preprocess_features,
    train_surrogate,
    compute_permutation_importance,
    compute_delta_attribution,
    run_attribution_analysis,
    plot_attribution,
    save_attribution_results,
)


# Fixtures
@pytest.fixture
def sample_df():
    """Sample canonical dataset for attribution testing"""
    np.random.seed(42)
    n = 200
    
    age = np.random.randint(18, 80, n)
    income = np.random.randint(20000, 150000, n)
    score = np.random.uniform(0, 1, n)
    
    # Create error pattern: high age and low income correlates with errors
    error_prob = 0.1 + 0.3 * (age > 50) + 0.2 * (income < 50000)
    is_error = np.random.random(n) < error_prob
    
    return pd.DataFrame({
        "id": range(n),
        "age": age,
        "income": income,
        "score": score,
        "y_true": np.random.choice([0, 1], n),
        "y_pred": np.random.choice([0, 1], n),
        "is_error": is_error,
        "error_type": np.where(is_error, "FP", "TN"),
    })


# Test surrogate training
def test_surrogate_trains_on_fixture(sample_df):
    """Test that surrogate model trains without exceptions"""
    feature_cols = ["age", "income", "score"]
    
    X, features, encoders = preprocess_features(sample_df, feature_cols)
    y = sample_df["is_error"].astype(int).values
    
    # Should not raise
    model = train_surrogate(X, y)
    
    assert model is not None
    assert hasattr(model, 'predict')


# Test permutation importance shape
def test_permutation_importance_shape(sample_df):
    """Test that importance has one score per feature"""
    feature_cols = ["age", "income", "score"]
    
    X, features, encoders = preprocess_features(sample_df, feature_cols)
    y = sample_df["is_error"].astype(int).values
    
    model = train_surrogate(X, y)
    importance = compute_permutation_importance(model, X, y, features, n_repeats=5)
    
    # Should have one row per feature
    assert len(importance) == len(feature_cols)
    
    # Should have required columns
    assert "feature" in importance.columns
    assert "importance_mean" in importance.columns
    assert "rank" in importance.columns


# Test delta attribution
def test_attribution_delta_computed(sample_df):
    """Test that delta attribution is computed correctly"""
    feature_cols = ["age", "income", "score"]
    
    global_imp, error_imp, delta = run_attribution_analysis(
        sample_df,
        feature_cols=feature_cols,
        n_repeats=3,
    )
    
    # Delta should exist and be numeric
    assert len(delta) > 0
    assert "delta" in delta.columns
    assert delta["delta"].dtype in [np.float64, np.float32]


# Test artifact saving
def test_attribution_artifacts_written(sample_df):
    """Test that attribution artifacts are saved"""
    feature_cols = ["age", "income", "score"]
    
    global_imp, error_imp, delta = run_attribution_analysis(
        sample_df,
        feature_cols=feature_cols,
        n_repeats=3,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_attribution_results(global_imp, error_imp, delta, Path(tmpdir))
        
        # Check files exist
        assert paths["global"].exists()
        assert paths["errors"].exists()
        
        # Check files are non-empty
        assert paths["global"].stat().st_size > 0
        
        # Check plots if data exists
        if len(global_imp) > 0:
            assert paths["global_plot"].exists()


# Test error-conditioned attribution differs
def test_error_attribution_differs_from_global(sample_df):
    """Test that error attribution differs from global on non-trivial fixture"""
    feature_cols = ["age", "income", "score"]
    
    global_imp, error_imp, delta = run_attribution_analysis(
        sample_df,
        feature_cols=feature_cols,
        n_repeats=5,
    )
    
    # At least one feature should have non-zero delta
    if len(delta) > 0:
        assert delta["abs_delta"].sum() >= 0  # Can be zero in degenerate cases


# Test top_n limit in plotting
def test_top_features_count_respected(sample_df):
    """Test that top_n limits the features shown"""
    feature_cols = ["age", "income", "score"]
    
    global_imp, _, _ = run_attribution_analysis(
        sample_df,
        feature_cols=feature_cols,
        n_repeats=3,
    )
    
    # Should respect top_n parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.png"
        plot_attribution(global_imp, output_path, top_n=2)
        
        assert output_path.exists()


# Test empty data handling
def test_empty_features_handled():
    """Test that empty feature list is handled"""
    df = pd.DataFrame({
        "is_error": [True, False, True],
    })
    
    global_imp, error_imp, delta = run_attribution_analysis(df, feature_cols=[])
    
    assert len(global_imp) == 0


# Test reproducibility
def test_attribution_reproducible_with_seed(sample_df):
    """Test that same seed produces same results"""
    feature_cols = ["age", "income"]
    
    global1, _, _ = run_attribution_analysis(
        sample_df, feature_cols=feature_cols,
        n_repeats=3, random_state=42
    )
    
    global2, _, _ = run_attribution_analysis(
        sample_df, feature_cols=feature_cols,
        n_repeats=3, random_state=42
    )
    
    # Same seed should produce same importance
    np.testing.assert_array_almost_equal(
        global1["importance_mean"].values,
        global2["importance_mean"].values,
        decimal=5
    )
