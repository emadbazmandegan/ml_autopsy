"""
Unit tests for drilldown module
"""
import pytest
import pandas as pd
import numpy as np
from engine.drilldown import get_sample_details, compute_feature_deviation, find_nearest_neighbors

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [20, 30, 40, 50, 30],
        "income": [20000, 30000, 40000, 80000, 31000],
        "class": ["A", "B", "A", "B", "B"],
        "y_true": [0, 1, 0, 1, 1],
        "y_pred": [0, 1, 1, 1, 0]
    })

def test_get_sample_details(sample_df):
    """Test retrieving sample details."""
    details = get_sample_details(sample_df, 1)
    
    assert details["index"] == 1
    assert details["data"]["age"] == 30
    assert details["data"]["income"] == 30000

def test_get_sample_details_with_filter(sample_df):
    """Test filtering specific features."""
    details = get_sample_details(sample_df, 1, feature_cols=["age"])
    
    assert "age" in details["features"]
    assert "income" not in details["features"]

def test_compute_feature_deviation(sample_df):
    """Test z-score computation."""
    # Mean income = (20+30+40+80+31)/5 = 40200
    # Std income ≈ 23370
    # Val 80000 -> z ≈ (80000 - 40200) / 23370 ≈ 1.7
    
    devs = compute_feature_deviation(sample_df, 3, feature_cols=["age", "income"])
    
    assert not devs.empty
    row = devs[devs["feature"] == "income"].iloc[0]
    assert row["z_score"] > 1.0  # Should be positive and significant

def test_find_nearest_neighbors(sample_df):
    """Test finding nearest neighbors."""
    # Index 1 (30, 30k) should be close to Index 4 (30, 31k)
    neighbors = find_nearest_neighbors(sample_df, 1, feature_cols=["age", "income"], k=2)
    
    assert len(neighbors) == 2
    indices = [n["index"] for n in neighbors]
    assert 4 in indices  # Should find the closest match
    assert 1 not in indices  # Should not find itself

def test_invalid_index_raises_error(sample_df):
    """Test error handling for bad indices."""
    with pytest.raises(ValueError):
        get_sample_details(sample_df, 99)
