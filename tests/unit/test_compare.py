"""
Unit tests for the comparison module
"""
import pytest
import pandas as pd
from engine.compare import validate_models, compute_comparison_metrics, find_disagreements

@pytest.fixture
def comparison_data():
    y_true = pd.Series([1, 0, 1, 0, 1])
    y_pred_a = pd.Series([1, 0, 1, 1, 0]) # 3 correct (60% acc)
    y_pred_b = pd.Series([1, 0, 1, 0, 1]) # 5 correct (100% acc)
    
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_a": y_pred_a,
        "y_pred_b": y_pred_b,
        "feature": [1, 2, 3, 4, 5]
    })
    return df, y_true, y_pred_a, y_pred_b

def test_validate_models_success(comparison_data):
    _, y_true, y_pred_a, y_pred_b = comparison_data
    assert validate_models(y_true, y_pred_a, y_pred_b) is True

def test_validate_models_failure():
    y_true = pd.Series([1, 0])
    y_pred_a = pd.Series([1])
    y_pred_b = pd.Series([1, 0])
    
    with pytest.raises(ValueError, match="Model A length"):
        validate_models(y_true, y_pred_a, y_pred_b)

def test_compute_comparison_metrics(comparison_data):
    _, y_true, y_pred_a, y_pred_b = comparison_data
    
    result = compute_comparison_metrics(y_true, y_pred_a, y_pred_b)
    
    assert result["model_a"]["accuracy"] == 0.6
    assert result["model_b"]["accuracy"] == 1.0
    assert result["delta"]["accuracy"] == 0.4  # 1.0 - 0.6

def test_find_disagreements(comparison_data):
    df, _, _, _ = comparison_data
    
    res = find_disagreements(df, "y_true", "y_pred_a", "y_pred_b")
    
    # Model A: [1, 0, 1, 1, 0]
    # Model B: [1, 0, 1, 0, 1]
    # Disagree: Index 3 (A=1, B=0), Index 4 (A=0, B=1)
    
    assert res["count"] == 2
    assert res["rate"] == 0.4
    assert len(res["df"]) == 2
    
    # Index 3: True=0, A=1 (wrong), B=0 (right) -> B wins
    # Index 4: True=1, A=0 (wrong), B=1 (right) -> B wins
    assert res["b_correct_count"] == 2
    assert res["a_correct_count"] == 0

def test_identical_models_have_zero_delta(comparison_data):
    _, y_true, y_pred_a, _ = comparison_data
    
    result = compute_comparison_metrics(y_true, y_pred_a, y_pred_a)
    
    assert result["delta"]["accuracy"] == 0.0
    assert result["delta"]["f1"] == 0.0
