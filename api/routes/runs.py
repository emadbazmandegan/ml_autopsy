"""
Runs API Routes
Endpoints for creating and managing autopsy runs
"""
import uuid
import os
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Base directory for runs
RUNS_DIR = Path("runs")

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes


class RunConfig(BaseModel):
    """Configuration for an autopsy run"""
    run_id: str
    created_at: str
    threshold: float = 0.5
    task_type: Optional[str] = None


class RunStatus(BaseModel):
    """Status of an autopsy run"""
    run_id: str
    status: str  # pending, processing, completed, failed
    message: Optional[str] = None


def validate_file_size(content: bytes, filename: str) -> None:
    """Validate file size is within limits."""
    if len(content) > MAX_FILE_SIZE:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File '{filename}' is too large ({size_mb:.1f}MB). Maximum allowed size is 10MB."
        )


def validate_csv_content(content: bytes, filename: str) -> None:
    """Validate that content is valid CSV."""
    try:
        # Try to decode as text
        text = content.decode('utf-8')
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' is empty."
            )
        # Check for basic CSV structure
        lines = text.strip().split('\n')
        if len(lines) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' must have a header and at least one data row."
            )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' is not a valid CSV file. Please ensure it's UTF-8 encoded."
        )


@router.post("/", response_model=RunStatus)
async def create_run(
    dataset: UploadFile = File(..., description="Dataset CSV with features"),
    predictions: UploadFile = File(..., description="Predictions CSV (y_pred or y_score)"),
    labels: Optional[UploadFile] = File(None, description="Labels CSV (if not in dataset)"),
    threshold: float = Form(0.5, description="Classification threshold for y_score"),
):
    """
    Create a new autopsy run.
    
    Uploads dataset, predictions, and optionally labels.
    Returns run_id for status tracking.
    """
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    run_dir = RUNS_DIR / run_id
    input_dir = run_dir / "input"
    
    try:
        # Read and validate files
        dataset_content = await dataset.read()
        validate_file_size(dataset_content, dataset.filename)
        validate_csv_content(dataset_content, dataset.filename)
        
        predictions_content = await predictions.read()
        validate_file_size(predictions_content, predictions.filename)
        validate_csv_content(predictions_content, predictions.filename)
        
        labels_content = None
        if labels:
            labels_content = await labels.read()
            validate_file_size(labels_content, labels.filename)
            validate_csv_content(labels_content, labels.filename)
        
        # Create run directories
        input_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "intermediate").mkdir(exist_ok=True)
        (run_dir / "output").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "report").mkdir(exist_ok=True)
        
        # Save uploaded files
        dataset_path = input_dir / "dataset.csv"
        predictions_path = input_dir / "predictions.csv"
        
        with open(dataset_path, "wb") as f:
            f.write(dataset_content)
            
        with open(predictions_path, "wb") as f:
            f.write(predictions_content)
            
        if labels_content:
            labels_path = input_dir / "labels.csv"
            with open(labels_path, "wb") as f:
                f.write(labels_content)
        
        # Save run config
        config = RunConfig(
            run_id=run_id,
            created_at=datetime.utcnow().isoformat(),
            threshold=threshold,
        )
        
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
        
        return RunStatus(
            run_id=run_id,
            status="pending",
            message="Run created successfully. Analysis will begin shortly."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred. Please try again or contact support."
        )


@router.get("/{run_id}/status", response_model=RunStatus)
async def get_run_status(run_id: str):
    """Get the status of an autopsy run"""
    run_dir = RUNS_DIR / run_id
    
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    # Check for completion markers
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return RunStatus(run_id=run_id, status="failed", message="Missing config")
    
    canonical_path = run_dir / "intermediate" / "canonical.parquet"
    if canonical_path.exists():
        return RunStatus(run_id=run_id, status="completed")
    
    return RunStatus(run_id=run_id, status="pending")
