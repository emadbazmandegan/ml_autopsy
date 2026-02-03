"""
Unit tests for the blindspots module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from engine.blindspots import (
    identify_confident_wrong,
    compute_psi,
    compute_jsd,
    compute_feature_divergence,
    run_blindspot_analysis,
    save_blindspot_results,
)


# Fixtures
@pytest.fixture
def sample_df():
    """Sample canonical dataset for blind spot testing"""
    np.random.seed(42)
    n = 200
    
    age = np.random.randint(18, 80, n)
    income = np.random.randint(20000, 150000, n)
    category = np.random.choice(["A", "B", "C"], n)
    
    y_true = np.random.choice([0, 1], n)
    y_score = np.random.uniform(0, 1, n)
    y_pred = (y_score > 0.5).astype(int)
    
    return pd.DataFrame({
        "id": range(n),
        "age": age,
        "income": income,
        "category": category,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    })


# Test confident-wrong identification
def test_confident_wrong_subset_correct(sample_df):
    """Test that threshold selects expected rows"""
    # Create specific case
    df = pd.DataFrame({
        "y_true": [0, 1, 0, 1],
        "y_pred": [1, 1, 0, 0],  # First and last are wrong
        "y_score": [0.9, 0.8, 0.3, 0.1],  # First is confident (0.9), last is confident (0.9 = 1-0.1)
    })
    
    confident_wrong = identify_confident_wrong(df, confidence_threshold=0.8)
    
    # Should have 2 confident-wrong samples (indices 0 and 3)
    assert len(confident_wrong) == 2


# Test PSI bounds
def test_psi_in_reasonable_range():
    """Test that PSI is >= 0 and finite"""
    np.random.seed(42)
    
    # Similar distributions
    baseline = pd.Series(np.random.normal(50, 10, 1000))
    comparison = pd.Series(np.random.normal(52, 10, 500))
    
    psi = compute_psi(baseline, comparison)
    
    assert psi >= 0
    assert np.isfinite(psi)


def test_psi_different_distributions():
    """Test that PSI is higher for different distributions"""
    np.random.seed(42)
    
    baseline = pd.Series(np.random.normal(50, 10, 1000))
    similar = pd.Series(np.random.normal(51, 10, 500))
    different = pd.Series(np.random.normal(80, 10, 500))
    
    psi_similar = compute_psi(baseline, similar)
    psi_different = compute_psi(baseline, different)
    
    # Different should have higher PSI
    assert psi_different > psi_similar


# Test JSD bounds
def test_jsd_in_range_0_1():
    """Test that JSD is in [0, 1]"""
    baseline = pd.Series(["A", "A", "B", "B", "C"])
    comparison = pd.Series(["A", "B", "C", "C", "C"])
    
    jsd = compute_jsd(baseline, comparison)
    
    assert 0 <= jsd <= 1


def test_jsd_identical_distributions():
    """Test that identical distributions have JSD close to 0"""
    data = pd.Series(["A", "A", "B", "B", "C", "C"])
    
    jsd = compute_jsd(data, data)
    
    assert jsd < 0.01  # Very close to 0


# Test divergence ranking
def test_divergence_ranking_sorted_desc(sample_df):
    """Test that divergence results are sorted descending"""
    confident_wrong, divergence = run_blindspot_analysis(
        sample_df,
        confidence_threshold=0.6,
    )
    
    if len(divergence) > 1:
        # Should be sorted by divergence descending
        values = divergence["divergence"].values
        assert all(values[i] >= values[i+1] for i in range(len(values)-1))


# Test artifact saving
def test_blindspot_artifacts_written(sample_df):
    """Test that blind spot artifacts are created"""
    confident_wrong, divergence = run_blindspot_analysis(
        sample_df,
        confidence_threshold=0.6,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_blindspot_results(
            confident_wrong, divergence, sample_df, Path(tmpdir)
        )
        
        # Check main artifacts exist
        assert paths["drift"].exists()
        assert paths["confident_wrong"].exists()
        
        # Check files are valid CSVs
        drift_df = pd.read_csv(paths["drift"])
        assert "feature" in drift_df.columns or len(drift_df) == 0


# Test empty confident-wrong handling
def test_no_confident_wrong_samples():
    """Test handling when no samples are confident-wrong"""
    df = pd.DataFrame({
        "y_true": [0, 1, 0, 1],
        "y_pred": [0, 1, 0, 1],  # All correct
        "y_score": [0.1, 0.9, 0.2, 0.8],
        "feature": [1, 2, 3, 4],
    })
    
    confident_wrong, divergence = run_blindspot_analysis(df)
    
    assert len(confident_wrong) == 0
    assert len(divergence) == 0


# Test PSI edge cases
def test_psi_empty_inputs():
    """Test PSI with empty inputs"""
    psi = compute_psi(pd.Series([]), pd.Series([]))
    assert psi == 0.0


def test_psi_single_value():
    """Test PSI when all values are the same"""
    baseline = pd.Series([50, 50, 50, 50])
    comparison = pd.Series([50, 50])
    
    psi = compute_psi(baseline, comparison)
    assert psi >= 0
    assert np.isfinite(psi)
