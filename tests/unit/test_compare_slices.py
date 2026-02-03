"""
Unit tests for slice comparison
"""
import pytest
import pandas as pd
import numpy as np
from engine.compare import compare_slices

@pytest.fixture
def slice_data():
    # 10 samples
    # Feature 'sex': 5 Male (M), 5 Female (F)
    # y_true: all 1
    
    df = pd.DataFrame({
        "sex": ["M"]*5 + ["F"]*5,
        "y_true": [1]*10,
        "y_pred_a": [1, 1, 1, 0, 0] + [1, 1, 0, 0, 0], # M: 2 err, F: 3 err
        "y_pred_b": [1, 1, 1, 1, 1] + [1, 1, 0, 0, 0], # M: 0 err, F: 3 err
    })
    
    # Model A Error Rates:
    # M: 2/5 = 40%
    # F: 3/5 = 60%
    
    # Model B Error Rates:
    # M: 0/5 = 0%
    # F: 3/5 = 60%
    
    return df

def test_compare_slices_logic(slice_data):
    df = slice_data
    
    # Compare
    comp = compare_slices(df, "y_true", "y_pred_a", "y_pred_b", feature_cols=["sex"])
    
    # Should have 2 rows (sex=M, sex=F)
    assert len(comp) == 2
    
    # Check 'sex=M' row
    m_slice = comp[comp["value"] == "M"].iloc[0]
    assert m_slice["error_rate_a"] == 0.4
    assert m_slice["error_rate_b"] == 0.0
    assert m_slice["improvement"] == 0.4  # 40% improvement
    
    # Check 'sex=F' row
    f_slice = comp[comp["value"] == "F"].iloc[0]
    assert f_slice["error_rate_a"] == 0.6
    assert f_slice["error_rate_b"] == 0.6
    assert f_slice["improvement"] == 0.0

def test_compare_slices_ranking(slice_data):
    df = slice_data
    comp = compare_slices(df, "y_true", "y_pred_a", "y_pred_b", feature_cols=["sex"])
    
    # Should be sorted by improvement descending
    # M slice (0.4 imp) should be first
    assert comp.iloc[0]["value"] == "M"

def test_compare_slices_no_features():
    df = pd.DataFrame({"y_true": [1], "y_pred_a": [1], "y_pred_b": [1]})
    comp = compare_slices(df, "y_true", "y_pred_a", "y_pred_b", feature_cols=[])
    assert comp.empty
