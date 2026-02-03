"""
Unit tests for the clustering module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from engine.clustering import (
    filter_errors,
    preprocess_for_clustering,
    cluster_errors,
    project_to_2d,
    generate_cluster_profiles,
    plot_cluster_map,
    run_error_clustering,
    save_clustering_results,
)


# Fixtures
@pytest.fixture
def sample_canonical_df():
    """Sample canonical dataset with errors"""
    np.random.seed(42)
    n = 100
    
    age = np.random.randint(18, 80, n)
    income = np.random.randint(20000, 150000, n)
    education = np.random.choice(["high_school", "bachelors", "masters"], n)
    
    # Create some errors
    is_error = np.random.choice([True, False], n, p=[0.3, 0.7])
    error_type = np.where(
        is_error,
        np.random.choice(["FP", "FN"], n),
        np.random.choice(["TN", "TP"], n)
    )
    
    return pd.DataFrame({
        "id": range(n),
        "age": age,
        "income": income,
        "education": education,
        "y_true": np.random.choice([0, 1], n),
        "y_pred": np.random.choice([0, 1], n),
        "is_error": is_error,
        "error_type": error_type,
    })


@pytest.fixture
def errors_only_df(sample_canonical_df):
    """Only error rows"""
    return filter_errors(sample_canonical_df)


# Test filter_errors
def test_error_subset_only_clustered(sample_canonical_df):
    """Test that only error rows are in clustering input"""
    errors = filter_errors(sample_canonical_df)
    
    # All rows should have is_error == True
    assert errors["is_error"].all()
    
    # Should have fewer rows than original
    assert len(errors) < len(sample_canonical_df)


# Test cluster labels
def test_cluster_labels_length_matches_errors(sample_canonical_df):
    """Test that cluster labels match number of errors"""
    feature_cols = ["age", "income"]
    
    errors_df, profiles, X_2d = run_error_clustering(
        sample_canonical_df,
        feature_cols=feature_cols,
        n_clusters=3,
    )
    
    # Labels should match error count
    assert "cluster" in errors_df.columns
    assert len(errors_df) == sample_canonical_df["is_error"].sum()


# Test cluster profiles
def test_cluster_profiles_have_required_fields(sample_canonical_df):
    """Test that cluster profiles have all required fields"""
    feature_cols = ["age", "income"]
    
    errors_df, profiles, X_2d = run_error_clustering(
        sample_canonical_df,
        feature_cols=feature_cols,
        n_clusters=3,
    )
    
    required_fields = ["cluster_id", "size", "size_pct", "dominant_error", "top_features"]
    
    for profile in profiles:
        for field in required_fields:
            assert field in profile, f"Missing field: {field}"


# Test reproducibility
def test_clustering_reproducible_with_seed(sample_canonical_df):
    """Test that same seed produces same clusters"""
    feature_cols = ["age", "income"]
    
    errors_df1, profiles1, X_2d1 = run_error_clustering(
        sample_canonical_df,
        feature_cols=feature_cols,
        n_clusters=3,
        random_state=42,
    )
    
    errors_df2, profiles2, X_2d2 = run_error_clustering(
        sample_canonical_df,
        feature_cols=feature_cols,
        n_clusters=3,
        random_state=42,
    )
    
    # Same seed should produce same labels
    np.testing.assert_array_equal(
        errors_df1["cluster"].values,
        errors_df2["cluster"].values
    )


# Test artifact saving
def test_cluster_artifacts_written(sample_canonical_df):
    """Test that clustering artifacts are created"""
    feature_cols = ["age", "income"]
    
    errors_df, profiles, X_2d = run_error_clustering(
        sample_canonical_df,
        feature_cols=feature_cols,
        n_clusters=3,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_clustering_results(
            errors_df, profiles, X_2d,
            errors_df["cluster"].values,
            Path(tmpdir)
        )
        
        # All artifacts should exist
        assert paths["clusters"].exists()
        assert paths["profiles"].exists()
        assert paths["map"].exists()
        
        # Check files are non-empty
        assert paths["clusters"].stat().st_size > 0
        assert paths["profiles"].stat().st_size > 0
        assert paths["map"].stat().st_size > 0


# Test preprocessing
def test_preprocess_handles_categorical(sample_canonical_df):
    """Test that preprocessing handles categorical features"""
    errors = filter_errors(sample_canonical_df)
    feature_cols = ["age", "income", "education"]
    
    X, features, encoders = preprocess_for_clustering(errors, feature_cols)
    
    # Should have encoded education
    assert "education" in encoders
    
    # X should be numeric array
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(errors)


# Test PCA projection
def test_project_to_2d_shape():
    """Test that 2D projection has correct shape"""
    X = np.random.randn(50, 10)
    
    X_2d = project_to_2d(X)
    
    assert X_2d.shape == (50, 2)


# Test empty errors case
def test_empty_errors_handled():
    """Test that empty error set is handled gracefully"""
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "is_error": [False, False, False],
        "error_type": ["TN", "TP", "TN"],
    })
    
    errors_df, profiles, X_2d = run_error_clustering(df)
    
    assert len(errors_df) == 0
    assert len(profiles) == 0
    assert len(X_2d) == 0
