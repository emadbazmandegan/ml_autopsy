"""
Unit tests for the validation module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from engine.validate import (
    ValidationError,
    load_csv,
    validate_required_columns,
    validate_row_alignment,
    detect_task_type,
    validate_numeric_column,
    apply_threshold,
    load_and_validate_inputs,
    map_columns,
)


# Fixtures
@pytest.fixture
def sample_dataset():
    """Create a sample dataset DataFrame"""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25, 35, 45, 55, 65],
        "income": [30000, 50000, 70000, 90000, 110000],
        "y_true": [0, 0, 1, 1, 0],
    })


@pytest.fixture
def sample_predictions():
    """Create sample predictions DataFrame"""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "y_pred": [0, 1, 1, 0, 0],
        "y_score": [0.2, 0.6, 0.8, 0.4, 0.3],
    })


@pytest.fixture
def sample_labels():
    """Create sample labels DataFrame"""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "y_true": [0, 0, 1, 1, 0],
    })


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory"""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def examples_fixtures_dir():
    """Path to examples fixtures directory"""
    return Path(__file__).parent.parent.parent / "examples" / "fixtures"


# Tests for validate_required_columns
def test_validate_required_columns_happy_path(sample_dataset):
    """Test that validation passes when all required columns exist"""
    # Should not raise
    validate_required_columns(sample_dataset, ["id", "age"], "dataset")


def test_missing_required_columns_raises(sample_dataset):
    """Test that missing columns raise ValidationError"""
    with pytest.raises(ValidationError) as exc_info:
        validate_required_columns(sample_dataset, ["id", "nonexistent"], "dataset")
    
    assert "nonexistent" in str(exc_info.value)
    assert "Missing required columns" in str(exc_info.value)


# Tests for validate_row_alignment
def test_validate_row_alignment_happy_path(sample_dataset, sample_predictions):
    """Test row alignment validation with matching row counts"""
    dfs = {"dataset": sample_dataset, "predictions": sample_predictions}
    # Should not raise
    validate_row_alignment(dfs, "dataset")


def test_mismatched_row_counts_raises(sample_dataset):
    """Test that mismatched row counts raise ValidationError"""
    mismatched_df = pd.DataFrame({"a": [1, 2, 3]})  # 3 rows vs 5 in dataset
    dfs = {"dataset": sample_dataset, "predictions": mismatched_df}
    
    with pytest.raises(ValidationError) as exc_info:
        validate_row_alignment(dfs, "dataset")
    
    assert "Row count mismatch" in str(exc_info.value)


# Tests for detect_task_type
def test_task_type_detection_binary():
    """Test that binary labels are correctly detected"""
    binary_labels = pd.Series([0, 1, 0, 1, 0])
    assert detect_task_type(binary_labels) == "binary"


def test_task_type_detection_binary_single_class():
    """Test that single class is detected as binary"""
    single_class = pd.Series([0, 0, 0, 0])
    assert detect_task_type(single_class) == "binary"


def test_task_type_detection_multiclass():
    """Test that multiclass labels are correctly detected"""
    multiclass_labels = pd.Series([0, 1, 2, 0, 1, 2])
    assert detect_task_type(multiclass_labels) == "multiclass"


# Tests for validate_numeric_column
def test_validate_numeric_column_happy_path(sample_predictions):
    """Test that numeric columns pass validation"""
    # Should not raise
    validate_numeric_column(sample_predictions, "y_score", "predictions")


def test_non_numeric_score_raises():
    """Test that non-numeric score columns raise ValidationError"""
    df = pd.DataFrame({
        "y_score": ["high", "low", "medium"],
    })
    
    with pytest.raises(ValidationError) as exc_info:
        validate_numeric_column(df, "y_score", "predictions")
    
    assert "must be numeric" in str(exc_info.value)


# Tests for apply_threshold
def test_threshold_applies_to_scores():
    """Test that threshold correctly converts scores to predictions"""
    scores = pd.Series([0.3, 0.5, 0.7, 0.4, 0.6])
    
    # Default threshold 0.5
    predictions = apply_threshold(scores, 0.5)
    expected = pd.Series([0, 1, 1, 0, 1])
    
    pd.testing.assert_series_equal(predictions, expected)


def test_threshold_custom_value():
    """Test with a custom threshold value"""
    scores = pd.Series([0.3, 0.5, 0.7, 0.4, 0.6])
    
    # Higher threshold
    predictions = apply_threshold(scores, 0.65)
    expected = pd.Series([0, 0, 1, 0, 0])
    
    pd.testing.assert_series_equal(predictions, expected)


# Tests for schema mapping
def test_schema_mapping_happy_path(sample_dataset, sample_predictions, sample_labels):
    """Test that column mapping produces canonical columns"""
    config = {
        "id_column": "id",
        "y_true_column": "y_true",
        "y_pred_column": "y_pred",
        "y_score_column": "y_score",
        "threshold": 0.5,
        "task_type": "binary",
    }
    
    result = map_columns(sample_dataset, sample_predictions, sample_labels, config)
    
    # Check all expected columns exist
    assert "y_true" in result.columns
    assert "y_pred" in result.columns
    assert "y_score" in result.columns
    assert "age" in result.columns
    assert "income" in result.columns


def test_schema_mapping_derives_y_pred_from_score():
    """Test that y_pred is derived from y_score when not provided"""
    dataset = pd.DataFrame({
        "id": [1, 2, 3],
        "feature": [10, 20, 30],
    })
    
    predictions = pd.DataFrame({
        "id": [1, 2, 3],
        "y_score": [0.3, 0.7, 0.5],
    })
    
    labels = pd.DataFrame({
        "id": [1, 2, 3],
        "y_true": [0, 1, 0],
    })
    
    config = {
        "id_column": "id",
        "y_true_column": "y_true",
        "y_pred_column": None,
        "y_score_column": "y_score",
        "threshold": 0.5,
        "task_type": "binary",
    }
    
    result = map_columns(dataset, predictions, labels, config)
    
    # y_pred should be derived from y_score with threshold 0.5
    assert "y_pred" in result.columns
    expected_y_pred = [0, 1, 1]  # 0.3 < 0.5, 0.7 >= 0.5, 0.5 >= 0.5
    assert result["y_pred"].tolist() == expected_y_pred


# Integration test with fixture files
def test_load_and_validate_with_fixtures(examples_fixtures_dir):
    """Test loading and validating actual fixture files"""
    dataset_path = examples_fixtures_dir / "binary_small.csv"
    predictions_path = examples_fixtures_dir / "binary_preds.csv"
    labels_path = examples_fixtures_dir / "binary_labels.csv"
    
    if not dataset_path.exists():
        pytest.skip("Fixture files not found")
    
    dataset_df, predictions_df, labels_df, config = load_and_validate_inputs(
        dataset_path=dataset_path,
        predictions_path=predictions_path,
        labels_path=labels_path,
        id_column="id",
    )
    
    assert len(dataset_df) == 100
    assert len(predictions_df) == 100
    assert len(labels_df) == 100
    assert config["task_type"] == "binary"


def test_mismatched_files_raises(examples_fixtures_dir):
    """Test that mismatched row count files raise error"""
    dataset_path = examples_fixtures_dir / "binary_small.csv"
    predictions_path = examples_fixtures_dir / "bad_mismatch_rows.csv"
    labels_path = examples_fixtures_dir / "binary_labels.csv"
    
    if not dataset_path.exists() or not predictions_path.exists():
        pytest.skip("Fixture files not found")
    
    with pytest.raises(ValidationError) as exc_info:
        load_and_validate_inputs(
            dataset_path=dataset_path,
            predictions_path=predictions_path,
            labels_path=labels_path,
            id_column="id",
        )
    
    assert "Row count mismatch" in str(exc_info.value)

