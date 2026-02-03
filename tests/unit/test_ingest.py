"""
Unit tests for the ingest module

Tests for reading Streamlit-like file uploads and building canonical datasets.
"""
import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path

from engine.ingest import (
    read_uploaded_csv,
    detect_column,
    auto_detect_feature_columns,
    ingest_uploaded_files,
    is_likely_id_column,
    is_likely_label_column,
    is_likely_score_column,
    smart_detect_columns,
)
from engine.validate import ValidationError



# Fixtures
@pytest.fixture
def sample_dataset_bytes():
    """Create a sample dataset as BytesIO (simulating Streamlit upload)"""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25, 35, 45, 55, 65],
        "income": [30000, 50000, 70000, 90000, 110000],
        "education": ["High School", "Bachelor", "Master", "PhD", "High School"],
    })
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_predictions_bytes():
    """Create sample predictions as BytesIO"""
    df = pd.DataFrame({
        "y_pred": [0, 1, 1, 0, 0],
        "y_score": [0.2, 0.6, 0.8, 0.4, 0.3],
        "y_true": [0, 0, 1, 1, 0],
    })
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_labels_bytes():
    """Create sample labels as BytesIO"""
    df = pd.DataFrame({
        "y_true": [0, 0, 1, 1, 0],
    })
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


@pytest.fixture
def predictions_score_only_bytes():
    """Predictions with only y_score (no y_pred)"""
    df = pd.DataFrame({
        "y_score": [0.2, 0.6, 0.8, 0.4, 0.3],
        "y_true": [0, 0, 1, 1, 0],
    })
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


@pytest.fixture
def mismatched_rows_bytes():
    """Predictions with wrong number of rows"""
    df = pd.DataFrame({
        "y_pred": [0, 1, 1],  # Only 3 rows instead of 5
        "y_score": [0.2, 0.6, 0.8],
        "y_true": [0, 0, 1],
    })
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


# Tests for read_uploaded_csv
class TestReadUploadedCSV:
    def test_read_valid_csv(self, sample_dataset_bytes):
        """Test reading a valid CSV from BytesIO"""
        df = read_uploaded_csv(sample_dataset_bytes, "dataset")
        assert len(df) == 5
        assert "age" in df.columns
        assert "income" in df.columns

    def test_read_invalid_csv_raises(self):
        """Test that invalid CSV raises ValidationError"""
        # Use an empty file which will cause pandas to fail
        buffer = BytesIO(b"")
        
        with pytest.raises(ValidationError):
            read_uploaded_csv(buffer, "empty_file")

    def test_read_resets_file_pointer(self, sample_dataset_bytes):
        """Test that file pointer is reset before reading"""
        # Read once to move pointer
        _ = sample_dataset_bytes.read()
        
        # Should still work (function resets pointer)
        df = read_uploaded_csv(sample_dataset_bytes, "dataset")
        assert len(df) == 5


# Tests for detect_column
class TestDetectColumn:
    def test_detect_exact_match(self):
        """Test detection with exact column name"""
        df = pd.DataFrame({"y_true": [0, 1], "feature": [1, 2]})
        result = detect_column(df, ["y_true", "label"])
        assert result == "y_true"

    def test_detect_case_insensitive(self):
        """Test case-insensitive detection"""
        df = pd.DataFrame({"Y_TRUE": [0, 1], "feature": [1, 2]})
        result = detect_column(df, ["y_true", "label"])
        assert result == "Y_TRUE"

    def test_detect_returns_none_when_not_found(self):
        """Test that None is returned when column not found"""
        df = pd.DataFrame({"other": [0, 1]})
        result = detect_column(df, ["y_true", "label"], required=False)
        assert result is None

    def test_detect_raises_when_required(self):
        """Test that ValidationError raised when required column not found"""
        df = pd.DataFrame({"other": [0, 1]})
        with pytest.raises(ValidationError):
            detect_column(df, ["y_true", "label"], required=True)


# Tests for auto_detect_feature_columns
class TestAutoDetectFeatureColumns:
    def test_excludes_specified_columns(self):
        """Test that specified columns are excluded"""
        df = pd.DataFrame({
            "age": [25], "income": [50000], 
            "y_true": [0], "y_pred": [1], "y_score": [0.5]
        })
        features = auto_detect_feature_columns(df, ["y_true", "y_pred", "y_score"])
        assert "age" in features
        assert "income" in features
        assert "y_true" not in features
        assert "y_pred" not in features

    def test_excludes_common_id_columns(self):
        """Test that common ID columns are excluded"""
        df = pd.DataFrame({
            "id": [1], "age": [25], "row_id": [1], "education": ["PhD"]
        })
        features = auto_detect_feature_columns(df, [])
        assert "age" in features
        assert "education" in features
        assert "id" not in features
        assert "row_id" not in features


# Tests for ingest_uploaded_files
class TestIngestUploadedFiles:
    def test_successful_ingestion(self, sample_dataset_bytes, sample_predictions_bytes):
        """Test successful file ingestion"""
        canonical, config, feature_cols = ingest_uploaded_files(
            dataset_file=sample_dataset_bytes,
            predictions_file=sample_predictions_bytes,
        )
        
        # Check canonical has required columns
        assert "y_true" in canonical.columns
        assert "y_pred" in canonical.columns
        assert "is_error" in canonical.columns
        assert "error_type" in canonical.columns
        
        # Check config
        assert config["n_rows"] == 5
        assert config["task_type"] == "binary"
        
        # Check features detected
        assert "age" in feature_cols
        assert "income" in feature_cols

    def test_ingestion_with_separate_labels(
        self, sample_dataset_bytes, predictions_score_only_bytes, sample_labels_bytes
    ):
        """Test ingestion with labels in separate file"""
        # Remove y_true from predictions for this test
        preds_no_labels = pd.DataFrame({
            "y_score": [0.2, 0.6, 0.8, 0.4, 0.3],
        })
        preds_buffer = BytesIO()
        preds_no_labels.to_csv(preds_buffer, index=False)
        preds_buffer.seek(0)
        
        canonical, config, feature_cols = ingest_uploaded_files(
            dataset_file=sample_dataset_bytes,
            predictions_file=preds_buffer,
            labels_file=sample_labels_bytes,
        )
        
        assert "y_true" in canonical.columns
        assert config["has_labels_file"] is True

    def test_derives_y_pred_from_y_score(self, sample_dataset_bytes):
        """Test that y_pred is derived from y_score when no y_pred column exists"""
        # Use explicit y_true column (not just 'label') to avoid ambiguity
        preds_df = pd.DataFrame({
            "y_score": [0.3, 0.7, 0.5, 0.4, 0.6],
            "ground_truth": [0, 1, 0, 0, 1],  # Use 'ground_truth' to be unambiguous
        })
        preds_buffer = BytesIO()
        preds_df.to_csv(preds_buffer, index=False)
        preds_buffer.seek(0)
        
        canonical, config, _ = ingest_uploaded_files(
            dataset_file=sample_dataset_bytes,
            predictions_file=preds_buffer,
            threshold=0.5,
        )
        
        # y_pred should be derived from y_score: 0.3->0, 0.7->1, 0.5->1, 0.4->0, 0.6->1
        expected = [0, 1, 1, 0, 1]
        assert canonical["y_pred"].tolist() == expected

    def test_custom_threshold(self, sample_dataset_bytes):
        """Test that custom threshold is applied when deriving y_pred"""
        preds_df = pd.DataFrame({
            "y_score": [0.3, 0.7, 0.5, 0.4, 0.6],
            "ground_truth": [0, 1, 0, 0, 1],  # Use 'ground_truth' to be unambiguous
        })
        preds_buffer = BytesIO()
        preds_df.to_csv(preds_buffer, index=False)
        preds_buffer.seek(0)
        
        canonical, _, _ = ingest_uploaded_files(
            dataset_file=sample_dataset_bytes,
            predictions_file=preds_buffer,
            threshold=0.65,
        )
        
        # With threshold 0.65: 0.3->0, 0.7->1, 0.5->0, 0.4->0, 0.6->0
        expected = [0, 1, 0, 0, 0]
        assert canonical["y_pred"].tolist() == expected

    def test_row_mismatch_raises(self, sample_dataset_bytes, mismatched_rows_bytes):
        """Test that mismatched row counts raise ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            ingest_uploaded_files(
                dataset_file=sample_dataset_bytes,
                predictions_file=mismatched_rows_bytes,
            )
        assert "Row count mismatch" in str(exc_info.value)

    def test_missing_predictions_columns_raises(self, sample_dataset_bytes):
        """Test that predictions file with only ID-like columns raises ValidationError"""
        # Create a predictions file with ONLY ID-like columns - no usable predictions
        bad_preds = pd.DataFrame({
            "sample_id": [1, 2, 3, 4, 5],  # ID column
            "row_index": [0, 1, 2, 3, 4],   # Another ID column
        })
        buffer = BytesIO()
        bad_preds.to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Create a dataset with y_true so the error is about predictions
        dataset_with_label = pd.DataFrame({
            "age": [25, 35, 45, 55, 65],
            "y_true": [0, 0, 1, 1, 0],
        })
        dataset_buffer = BytesIO()
        dataset_with_label.to_csv(dataset_buffer, index=False)
        dataset_buffer.seek(0)
        
        with pytest.raises(ValidationError) as exc_info:
            ingest_uploaded_files(
                dataset_file=dataset_buffer,
                predictions_file=buffer,
            )
        # Should raise about predictions
        assert "predictions" in str(exc_info.value).lower() or "y_pred" in str(exc_info.value).lower()


    def test_missing_y_true_raises(self, sample_dataset_bytes):
        """Test that missing y_true with no fallback raises ValidationError"""
        # Create predictions with NO label-like columns at all
        preds_no_true = pd.DataFrame({
            "y_pred": [0, 1, 1, 0, 0],
            "y_score": [0.2, 0.6, 0.8, 0.4, 0.3],
        })
        buffer = BytesIO()
        preds_no_true.to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Also create dataset with NO label columns
        dataset_no_labels = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [10, 20, 30, 40, 50],
        })
        dataset_buffer = BytesIO()
        dataset_no_labels.to_csv(dataset_buffer, index=False)
        dataset_buffer.seek(0)
        
        with pytest.raises(ValidationError) as exc_info:
            ingest_uploaded_files(
                dataset_file=dataset_buffer,
                predictions_file=buffer,
            )
        assert "label" in str(exc_info.value).lower() or "ground truth" in str(exc_info.value).lower()


# Integration test with actual fixture files
class TestIntegrationWithFixtures:
    @pytest.fixture
    def examples_fixtures_dir(self):
        return Path(__file__).parent.parent.parent / "examples" / "fixtures"

    def test_ingest_fixture_files(self, examples_fixtures_dir):
        """Test ingestion with actual fixture files"""
        dataset_path = examples_fixtures_dir / "binary_small.csv"
        predictions_path = examples_fixtures_dir / "binary_preds.csv"
        labels_path = examples_fixtures_dir / "binary_labels.csv"
        
        if not dataset_path.exists():
            pytest.skip("Fixture files not found")
        
        # Read as BytesIO to simulate Streamlit uploads
        with open(dataset_path, "rb") as f:
            dataset_buffer = BytesIO(f.read())
        with open(predictions_path, "rb") as f:
            preds_buffer = BytesIO(f.read())
        with open(labels_path, "rb") as f:
            labels_buffer = BytesIO(f.read())
        
        canonical, config, feature_cols = ingest_uploaded_files(
            dataset_file=dataset_buffer,
            predictions_file=preds_buffer,
            labels_file=labels_buffer,
        )
        
        assert config["n_rows"] == 100
        assert config["task_type"] == "binary"
        assert len(feature_cols) > 0
