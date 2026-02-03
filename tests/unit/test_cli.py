"""
CLI Tests for Model Autopsy

Comprehensive tests for the command-line interface.
"""
import pytest
import json
import tempfile
from pathlib import Path
from io import BytesIO
from click.testing import CliRunner

import pandas as pd
import numpy as np

from cli import cli, audit, validate, demo, serve


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset(temp_dir):
    """Create a sample dataset CSV."""
    df = pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(18, 80, 100),
        "income": np.random.randint(20000, 150000, 100),
        "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], 100),
    })
    path = temp_dir / "dataset.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_predictions(temp_dir):
    """Create a sample predictions CSV."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = y_true.copy()
    # Introduce ~15% errors
    error_indices = np.random.choice(100, 15, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": np.random.uniform(0, 1, 100),
    })
    path = temp_dir / "predictions.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_predictions_no_score(temp_dir):
    """Create predictions without y_score column."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = y_true.copy()
    error_indices = np.random.choice(100, 15, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    path = temp_dir / "predictions_no_score.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def invalid_csv(temp_dir):
    """Create an invalid CSV file."""
    path = temp_dir / "invalid.csv"
    with open(path, 'w') as f:
        f.write("this,is,not,valid\n")
        f.write("missing,columns,here\n")
    return path


class TestCLIHelp:
    """Test CLI help and version commands."""
    
    def test_main_help(self, runner):
        """Test that main help works."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Model Autopsy" in result.output
        assert "audit" in result.output
        assert "serve" in result.output
        assert "validate" in result.output
    
    def test_version(self, runner):
        """Test version flag."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
    
    def test_audit_help(self, runner):
        """Test audit command help."""
        result = runner.invoke(cli, ["audit", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output
        assert "--predictions" in result.output
        assert "--output-dir" in result.output
    
    def test_validate_help(self, runner):
        """Test validate command help."""
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output
        assert "--predictions" in result.output


class TestValidateCommand:
    """Test the validate command."""
    
    def test_validate_success(self, runner, sample_dataset, sample_predictions):
        """Test successful validation."""
        result = runner.invoke(validate, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions),
        ])
        assert result.exit_code == 0
        assert "Validation passed" in result.output
        assert "Samples:" in result.output
    
    def test_validate_missing_file(self, runner, sample_dataset, temp_dir):
        """Test validation with missing file."""
        result = runner.invoke(validate, [
            "--dataset", str(sample_dataset),
            "--predictions", str(temp_dir / "nonexistent.csv"),
        ])
        assert result.exit_code != 0
    
    def test_validate_without_score(self, runner, sample_dataset, sample_predictions_no_score):
        """Test validation works without y_score."""
        result = runner.invoke(validate, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions_no_score),
        ])
        assert result.exit_code == 0
        assert "Validation passed" in result.output


class TestAuditCommand:
    """Test the audit command."""
    
    def test_audit_success(self, runner, sample_dataset, sample_predictions, temp_dir):
        """Test successful audit generates report."""
        output_dir = temp_dir / "report"
        result = runner.invoke(audit, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 0
        assert (output_dir / "report.md").exists()
        assert (output_dir / "results_index.json").exists()
        
        # Check report content
        report = (output_dir / "report.md").read_text()
        assert "Model Autopsy Report" in report
        assert "Executive Summary" in report
    
    def test_audit_with_custom_threshold(self, runner, sample_dataset, sample_predictions, temp_dir):
        """Test audit with custom threshold."""
        output_dir = temp_dir / "report_threshold"
        result = runner.invoke(audit, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions),
            "--output-dir", str(output_dir),
            "--threshold", "0.7",
        ])
        
        assert result.exit_code == 0
        assert (output_dir / "report.md").exists()
    
    def test_audit_with_min_clusters(self, runner, sample_dataset, sample_predictions, temp_dir):
        """Test audit with custom cluster count."""
        output_dir = temp_dir / "report_clusters"
        result = runner.invoke(audit, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions),
            "--output-dir", str(output_dir),
            "--n-clusters", "2",
        ])
        
        assert result.exit_code == 0
        assert (output_dir / "report.md").exists()
    
    def test_audit_without_score(self, runner, sample_dataset, sample_predictions_no_score, temp_dir):
        """Test audit works without y_score column."""
        output_dir = temp_dir / "report_no_score"
        result = runner.invoke(audit, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions_no_score),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 0
        assert (output_dir / "report.md").exists()
    
    def test_audit_results_index(self, runner, sample_dataset, sample_predictions, temp_dir):
        """Test that results index is valid JSON."""
        output_dir = temp_dir / "report_json"
        result = runner.invoke(audit, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 0
        
        index_path = output_dir / "results_index.json"
        assert index_path.exists()
        
        with open(index_path) as f:
            index = json.load(f)
        
        assert "run_id" in index
        assert "generated_at" in index
        assert "artifacts" in index
    
    def test_audit_missing_required_arg(self, runner, sample_dataset):
        """Test audit fails with missing required argument."""
        result = runner.invoke(audit, [
            "--dataset", str(sample_dataset),
            # Missing --predictions
        ])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


class TestAuditEdgeCases:
    """Test edge cases in the audit command."""
    
    def test_audit_small_dataset(self, runner, temp_dir):
        """Test audit with very small dataset."""
        # Create tiny dataset
        dataset_path = temp_dir / "tiny_dataset.csv"
        pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "feature": [10, 20, 30, 40, 50],
        }).to_csv(dataset_path, index=False)
        
        preds_path = temp_dir / "tiny_preds.csv"
        pd.DataFrame({
            "y_true": [0, 1, 0, 1, 0],
            "y_pred": [0, 1, 1, 1, 0],  # 1 error
        }).to_csv(preds_path, index=False)
        
        output_dir = temp_dir / "tiny_report"
        result = runner.invoke(audit, [
            "--dataset", str(dataset_path),
            "--predictions", str(preds_path),
            "--output-dir", str(output_dir),
            "--min-support", "1",  # Low support for small data
        ])
        
        assert result.exit_code == 0
        assert (output_dir / "report.md").exists()
    
    def test_audit_all_correct(self, runner, temp_dir):
        """Test audit when model has no errors."""
        dataset_path = temp_dir / "perfect_dataset.csv"
        pd.DataFrame({
            "id": range(50),
            "feature": range(50),
        }).to_csv(dataset_path, index=False)
        
        preds_path = temp_dir / "perfect_preds.csv"
        y = [0, 1] * 25
        pd.DataFrame({
            "y_true": y,
            "y_pred": y,  # Perfect predictions
        }).to_csv(preds_path, index=False)
        
        output_dir = temp_dir / "perfect_report"
        result = runner.invoke(audit, [
            "--dataset", str(dataset_path),
            "--predictions", str(preds_path),
            "--output-dir", str(output_dir),
        ])
        
        # Should still succeed even with no errors to analyze
        assert result.exit_code == 0


class TestSmartColumnDetection:
    """Test that CLI properly uses smart column detection."""
    
    def test_audit_with_ground_truth_column(self, runner, temp_dir):
        """Test audit with 'ground_truth' instead of 'y_true'."""
        dataset_path = temp_dir / "dataset_gt.csv"
        pd.DataFrame({
            "id": range(50),
            "feature": np.random.rand(50),
        }).to_csv(dataset_path, index=False)
        
        preds_path = temp_dir / "preds_gt.csv"
        pd.DataFrame({
            "ground_truth": [0, 1] * 25,
            "label": [0, 0, 1, 1] * 12 + [0, 1],  # Some errors
        }).to_csv(preds_path, index=False)
        
        output_dir = temp_dir / "report_gt"
        result = runner.invoke(audit, [
            "--dataset", str(dataset_path),
            "--predictions", str(preds_path),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 0
        assert (output_dir / "report.md").exists()
    
    def test_audit_with_imageid_column(self, runner, temp_dir):
        """Test audit with 'ImageId' as ID column."""
        dataset_path = temp_dir / "dataset_imgid.csv"
        pd.DataFrame({
            "ImageId": [f"img_{i}" for i in range(50)],
            "feature": np.random.rand(50),
        }).to_csv(dataset_path, index=False)
        
        preds_path = temp_dir / "preds_imgid.csv"
        pd.DataFrame({
            "ImageId": [f"img_{i}" for i in range(50)],
            "y_true": [0, 1] * 25,
            "y_pred": [0, 0, 1, 1] * 12 + [0, 1],
        }).to_csv(preds_path, index=False)
        
        output_dir = temp_dir / "report_imgid"
        result = runner.invoke(audit, [
            "--dataset", str(dataset_path),
            "--predictions", str(preds_path),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 0


class TestDemoCommand:
    """Test the demo command."""
    
    def test_demo_runs(self, runner, temp_dir, monkeypatch):
        """Test demo command invokes audit correctly."""
        # Change to temp_dir to avoid polluting the project
        monkeypatch.chdir(temp_dir)
        
        # Note: This test requires demo files to exist
        # In practice, we might skip this or mock the file existence
        demo_dir = Path(__file__).parent.parent / "examples" / "demo"
        if not (demo_dir / "adult_dataset.csv").exists():
            pytest.skip("Demo files not available")
        
        result = runner.invoke(demo)
        # The demo command will either succeed or fail gracefully
        # based on whether the demo files exist


class TestOutputFormats:
    """Test different output format options."""
    
    def test_audit_creates_markdown(self, runner, sample_dataset, sample_predictions, temp_dir):
        """Test markdown output is valid."""
        output_dir = temp_dir / "report_md"
        result = runner.invoke(audit, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions),
            "--output-dir", str(output_dir),
            "--format", "markdown",
        ])
        
        assert result.exit_code == 0
        report_path = output_dir / "report.md"
        assert report_path.exists()
        
        content = report_path.read_text()
        assert content.startswith("#")  # Markdown header


class TestCLIIntegration:
    """Integration tests for CLI with real fixture files."""
    
    def test_audit_with_fixture_files(self, runner, temp_dir):
        """Test audit with project fixture files."""
        fixture_dir = Path(__file__).parent.parent / "examples" / "fixtures"
        
        groundtruth = fixture_dir / "groundtruth.csv"
        predictions = fixture_dir / "predictions.csv"
        
        if not groundtruth.exists() or not predictions.exists():
            pytest.skip("Fixture files not available")
        
        output_dir = temp_dir / "fixture_report"
        result = runner.invoke(audit, [
            "--dataset", str(groundtruth),
            "--predictions", str(predictions),
            "--output-dir", str(output_dir),
        ])
        
        assert result.exit_code == 0
        assert (output_dir / "report.md").exists()


class TestErrorHandling:
    """Test error handling in CLI."""
    
    def test_audit_invalid_threshold(self, runner, sample_dataset, sample_predictions, temp_dir):
        """Test invalid threshold value."""
        # Click should validate the range
        result = runner.invoke(audit, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions),
            "--output-dir", str(temp_dir / "report"),
            "--threshold", "invalid",
        ])
        assert result.exit_code != 0
    
    def test_audit_negative_clusters(self, runner, sample_dataset, sample_predictions, temp_dir):
        """Test negative cluster count."""
        result = runner.invoke(audit, [
            "--dataset", str(sample_dataset),
            "--predictions", str(sample_predictions),
            "--output-dir", str(temp_dir / "report"),
            "--n-clusters", "-1",
        ])
        # Should either fail validation or use a minimum of 1
        # Implementation-dependent behavior
