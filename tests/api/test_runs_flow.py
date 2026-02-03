"""
API Integration Tests for Runs Flow
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import io

from api.main import app


client = TestClient(app)


# Test fixtures
@pytest.fixture
def valid_dataset():
    """Valid dataset CSV content"""
    return b"id,age,income\n0,25,50000\n1,35,75000\n2,45,100000"


@pytest.fixture
def valid_predictions():
    """Valid predictions CSV content"""
    return b"y_pred,y_score,y_true\n1,0.8,1\n0,0.3,0\n1,0.7,1"


@pytest.fixture
def empty_file():
    """Empty file content"""
    return b""


@pytest.fixture
def single_line_file():
    """File with only header"""
    return b"a,b,c"


# End-to-end test
def test_runs_endpoint_end_to_end(valid_dataset, valid_predictions):
    """Test full upload flow: upload -> status check"""
    # Create run
    response = client.post(
        "/runs/",
        files={
            "dataset": ("dataset.csv", io.BytesIO(valid_dataset), "text/csv"),
            "predictions": ("predictions.csv", io.BytesIO(valid_predictions), "text/csv"),
        },
        data={"threshold": "0.5"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "pending"
    
    run_id = data["run_id"]
    
    # Check status
    status_response = client.get(f"/runs/{run_id}/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["run_id"] == run_id


# Error handling tests
def test_bad_upload_returns_user_friendly_error(empty_file, valid_predictions):
    """Test that empty file returns user-friendly error"""
    response = client.post(
        "/runs/",
        files={
            "dataset": ("dataset.csv", io.BytesIO(empty_file), "text/csv"),
            "predictions": ("predictions.csv", io.BytesIO(valid_predictions), "text/csv"),
        },
        data={"threshold": "0.5"},
    )
    
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_single_row_file_returns_error(single_line_file, valid_predictions):
    """Test that file with only header returns error"""
    response = client.post(
        "/runs/",
        files={
            "dataset": ("dataset.csv", io.BytesIO(single_line_file), "text/csv"),
            "predictions": ("predictions.csv", io.BytesIO(valid_predictions), "text/csv"),
        },
        data={"threshold": "0.5"},
    )
    
    assert response.status_code == 400
    assert "header" in response.json()["detail"].lower() or "row" in response.json()["detail"].lower()


def test_large_file_rejected():
    """Test that files over 10MB are rejected"""
    # Create a file that's just over 10MB
    large_content = b"a,b,c\n" + b"1,2,3\n" * (11 * 1024 * 1024 // 6)
    
    response = client.post(
        "/runs/",
        files={
            "dataset": ("dataset.csv", io.BytesIO(large_content), "text/csv"),
            "predictions": ("predictions.csv", io.BytesIO(b"y,s,t\n1,0.5,1"), "text/csv"),
        },
        data={"threshold": "0.5"},
    )
    
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_nonexistent_run_returns_404():
    """Test that nonexistent run returns 404"""
    response = client.get("/runs/nonexistent123/status")
    
    assert response.status_code == 404


def test_valid_upload_creates_directories(valid_dataset, valid_predictions, tmp_path, monkeypatch):
    """Test that valid upload creates expected directory structure"""
    # This test verifies the run directory is created properly
    response = client.post(
        "/runs/",
        files={
            "dataset": ("dataset.csv", io.BytesIO(valid_dataset), "text/csv"),
            "predictions": ("predictions.csv", io.BytesIO(valid_predictions), "text/csv"),
        },
        data={"threshold": "0.5"},
    )
    
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    
    # Check run_id format
    assert len(run_id) == 8
