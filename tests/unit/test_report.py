"""
Unit tests for the report module
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

from engine.report import (
    generate_executive_summary,
    generate_slices_section,
    generate_clusters_section,
    generate_attribution_section,
    generate_recommendations,
    generate_report,
    generate_results_index,
    save_report,
)


# Fixtures
@pytest.fixture
def sample_metrics():
    return {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1": 0.85,
        "auc": 0.91,
    }


@pytest.fixture
def sample_slices():
    return pd.DataFrame({
        "rank": [1, 2, 3],
        "rule": ["age > 50", "income <= 30000", "education = high_school"],
        "support": [100, 80, 60],
        "error_rate": [0.45, 0.38, 0.32],
        "lift": [2.5, 2.1, 1.8],
    })


@pytest.fixture
def sample_profiles():
    return [
        {
            "cluster_id": 0,
            "size": 50,
            "size_pct": 0.4,
            "dominant_error": "FP",
            "top_features": [
                {"feature": "age", "z_diff": 1.5},
                {"feature": "income", "z_diff": -0.8},
            ],
        },
        {
            "cluster_id": 1,
            "size": 75,
            "size_pct": 0.6,
            "dominant_error": "FN",
            "top_features": [
                {"feature": "score", "z_diff": -1.2},
            ],
        },
    ]


@pytest.fixture
def sample_importance():
    return pd.DataFrame({
        "feature": ["age", "income", "score"],
        "importance_mean": [0.15, 0.12, 0.08],
    })


@pytest.fixture
def sample_delta():
    return pd.DataFrame({
        "feature": ["age", "income", "score"],
        "delta": [0.05, -0.02, 0.01],
    })


# Test required sections
def test_report_contains_required_sections(
    sample_metrics, sample_slices, sample_profiles, sample_importance, sample_delta
):
    """Test that report contains all required sections"""
    report = generate_report(
        run_id="test123",
        metrics=sample_metrics,
        slices_df=sample_slices,
        profiles=sample_profiles,
        global_importance=sample_importance,
        delta=sample_delta,
        total_samples=1000,
        error_count=150,
    )
    
    # Check required sections
    assert "# Model Autopsy Report" in report
    assert "## Executive Summary" in report
    assert "## Top Failure Slices" in report
    assert "## Error Clusters" in report
    assert "## Feature Attribution" in report
    assert "## Recommendations" in report
    
    # Check run ID
    assert "test123" in report


# Test artifact references
def test_report_references_existing_artifacts():
    """Test that generate_results_index lists existing files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create some artifacts
        (tmpdir / "report.md").write_text("# Report")
        (tmpdir / "slices.csv").write_text("a,b,c")
        
        artifacts = {
            "report": tmpdir / "report.md",
            "slices": tmpdir / "slices.csv",
            "missing": tmpdir / "nonexistent.csv",
        }
        
        index = generate_results_index("run123", tmpdir, artifacts)
        
        # Should include existing files
        assert "report" in index["artifacts"]
        assert "slices" in index["artifacts"]
        
        # Should not include missing files
        assert "missing" not in index["artifacts"]


# Test results index
def test_results_index_lists_all_outputs():
    """Test that results index has correct structure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create artifacts
        (tmpdir / "report.md").write_text("content")
        (tmpdir / "metrics.json").write_text("{}")
        
        artifacts = {
            "report": tmpdir / "report.md",
            "metrics": tmpdir / "metrics.json",
        }
        
        index = generate_results_index("run456", tmpdir, artifacts)
        
        assert index["run_id"] == "run456"
        assert "generated_at" in index
        assert "artifacts" in index
        assert len(index["artifacts"]) == 2
        
        # Check artifact details
        for name, details in index["artifacts"].items():
            assert "path" in details
            assert "filename" in details
            assert "size_bytes" in details


# Test save functionality
def test_save_report_creates_files(sample_metrics, sample_slices):
    """Test that save_report creates both files"""
    report = generate_report(
        run_id="save_test",
        metrics=sample_metrics,
        slices_df=sample_slices,
        total_samples=500,
        error_count=50,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index = {"run_id": "save_test", "artifacts": {}}
        paths = save_report(report, index, Path(tmpdir))
        
        assert paths["report"].exists()
        assert paths["index"].exists()
        
        # Verify content
        assert "save_test" in paths["report"].read_text()
        
        with open(paths["index"]) as f:
            loaded_index = json.load(f)
        assert loaded_index["run_id"] == "save_test"


# Test empty data handling
def test_report_handles_empty_data():
    """Test that report handles empty DataFrames gracefully"""
    report = generate_report(
        run_id="empty_test",
        metrics={},
        slices_df=pd.DataFrame(),
        profiles=[],
        global_importance=pd.DataFrame(),
        delta=pd.DataFrame(),
        total_samples=0,
        error_count=0,
    )
    
    assert "# Model Autopsy Report" in report
    assert "No significant failure slices detected" in report


# Test slices section
def test_slices_section_format(sample_slices):
    """Test that slices section has correct table format"""
    section = generate_slices_section(sample_slices, top_n=3)
    
    assert "| Rank | Slice | Support | Error Rate | Lift |" in section
    assert "age > 50" in section
    assert "2.5" in section  # lift value


# Test clusters section
def test_clusters_section_format(sample_profiles):
    """Test that clusters section displays correctly"""
    section = generate_clusters_section(sample_profiles)
    
    assert "Cluster 0" in section
    assert "Cluster 1" in section
    assert "FP" in section
    assert "FN" in section
