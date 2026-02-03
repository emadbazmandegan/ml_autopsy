"""
Unit tests for explanation module
"""
import pytest
import pandas as pd
from engine.explain import generate_explanation

def test_generate_explanation_non_empty():
    """Test generating a basic explanation."""
    sample = {"data": {"y_true": 1, "y_pred": 0}}
    devs = pd.DataFrame([
        {"feature": "age", "z_score": 2.5, "abs_z_score": 2.5},
        {"feature": "income", "z_score": -0.5, "abs_z_score": 0.5}
    ])
    neighbors = [{"data": {"y_pred": 0}}, {"data": {"y_pred": 0}}]
    
    explanation = generate_explanation(sample, devs, neighbors)
    
    assert "predicted **0**" in explanation
    assert "an ERROR" in explanation
    assert "age" in explanation
    assert "2.5σ" in explanation

def test_explanation_mentions_top_features():
    """Test that explanation highlights significant deviations."""
    sample = {"data": {"y_true": 0, "y_pred": 0}}
    devs = pd.DataFrame([
        {"feature": "feature_A", "z_score": 3.0, "abs_z_score": 3.0},
        {"feature": "feature_B", "z_score": 0.1, "abs_z_score": 0.1}
    ])
    neighbors = []
    
    explanation = generate_explanation(sample, devs, neighbors)
    
    assert "feature_A" in explanation
    assert "feature_B" not in explanation
    assert "CORRECT" in explanation
