"""
Canonical Dataset Module for Model Autopsy

Builds the canonical dataset with error columns and saves artifacts.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import json
import pandas as pd
import numpy as np


def compute_error_columns(
    df: pd.DataFrame,
    task_type: str = "binary",
) -> pd.DataFrame:
    """
    Compute error-related columns: is_error and error_type.
    
    For binary classification:
    - is_error: True if y_pred != y_true
    - error_type: FP, FN, TP, TN
    
    For multiclass:
    - is_error: True if y_pred != y_true
    - error_type: 'correct' or 'mismatch_{true}_{pred}'
    
    Args:
        df: DataFrame with y_true and y_pred columns
        task_type: 'binary' or 'multiclass'
        
    Returns:
        DataFrame with is_error and error_type columns added
    """
    result = df.copy()
    
    # Compute is_error
    result["is_error"] = result["y_true"] != result["y_pred"]
    
    if task_type == "binary":
        # Compute error_type for binary classification
        conditions = [
            (result["y_true"] == 0) & (result["y_pred"] == 0),  # TN
            (result["y_true"] == 0) & (result["y_pred"] == 1),  # FP
            (result["y_true"] == 1) & (result["y_pred"] == 0),  # FN
            (result["y_true"] == 1) & (result["y_pred"] == 1),  # TP
        ]
        choices = ["TN", "FP", "FN", "TP"]
        result["error_type"] = np.select(conditions, choices, default="unknown")
    else:
        # Multiclass
        def get_error_type(row):
            if row["y_true"] == row["y_pred"]:
                return "correct"
            return f"mismatch_{row['y_true']}_{row['y_pred']}"
        
        result["error_type"] = result.apply(get_error_type, axis=1)
    
    return result


def build_canonical_dataset(
    dataset_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build the canonical dataset by combining all inputs and computing error columns.
    
    Args:
        dataset_df: Features DataFrame
        predictions_df: Predictions DataFrame
        labels_df: Labels DataFrame
        config: Configuration dict
        
    Returns:
        Canonical DataFrame with all columns and error information
    """
    from engine.validate import map_columns
    
    # Map columns to unified format
    canonical = map_columns(dataset_df, predictions_df, labels_df, config)
    
    # Compute error columns
    canonical = compute_error_columns(canonical, config.get("task_type", "binary"))
    
    return canonical


def save_canonical(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "canonical.parquet",
) -> Path:
    """
    Save the canonical dataset as a Parquet file.
    
    Args:
        df: Canonical DataFrame
        output_dir: Directory to save to
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    df.to_parquet(output_path, index=False)
    
    return output_path


def save_config(
    config: Dict[str, Any],
    output_dir: Path,
    filename: str = "config.json",
) -> Path:
    """
    Save the run configuration as a JSON file.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save to
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    return output_path


def load_canonical(input_path: Path) -> pd.DataFrame:
    """
    Load a canonical dataset from a Parquet file.
    
    Args:
        input_path: Path to Parquet file
        
    Returns:
        Canonical DataFrame
    """
    return pd.read_parquet(input_path)


def get_error_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of errors in the canonical dataset.
    
    Args:
        df: Canonical DataFrame with is_error and error_type columns
        
    Returns:
        Dictionary with error statistics
    """
    total = len(df)
    errors = df["is_error"].sum()
    error_rate = errors / total if total > 0 else 0
    
    error_type_counts = df["error_type"].value_counts().to_dict()
    
    return {
        "total_samples": total,
        "total_errors": int(errors),
        "error_rate": float(error_rate),
        "error_type_counts": error_type_counts,
    }


def run_ingestion_pipeline(
    dataset_path: Path,
    predictions_path: Path,
    output_dir: Path,
    labels_path: Optional[Path] = None,
    threshold: float = 0.5,
    id_column: Optional[str] = None,
    y_true_column: str = "y_true",
    y_pred_column: str = "y_pred",
    y_score_column: str = "y_score",
) -> Dict[str, Any]:
    """
    Run the complete ingestion pipeline.
    
    Args:
        dataset_path: Path to dataset CSV
        predictions_path: Path to predictions CSV
        output_dir: Directory for output artifacts
        labels_path: Optional path to labels CSV
        threshold: Classification threshold
        id_column: Optional ID column name
        y_true_column: Name of true label column
        y_pred_column: Name of prediction column
        y_score_column: Name of score column
        
    Returns:
        Dictionary with paths to created artifacts and summary
    """
    from engine.validate import load_and_validate_inputs
    
    # Load and validate inputs
    dataset_df, predictions_df, labels_df, config = load_and_validate_inputs(
        dataset_path=Path(dataset_path),
        predictions_path=Path(predictions_path),
        labels_path=Path(labels_path) if labels_path else None,
        id_column=id_column,
        y_true_column=y_true_column,
        y_pred_column=y_pred_column,
        y_score_column=y_score_column,
        threshold=threshold,
    )
    
    # Build canonical dataset
    canonical = build_canonical_dataset(dataset_df, predictions_df, labels_df, config)
    
    # Create output directories
    output_dir = Path(output_dir)
    intermediate_dir = output_dir / "intermediate"
    
    # Save artifacts
    canonical_path = save_canonical(canonical, intermediate_dir)
    config_path = save_config(config, output_dir)
    
    # Get summary
    summary = get_error_summary(canonical)
    
    return {
        "canonical_path": str(canonical_path),
        "config_path": str(config_path),
        "config": config,
        "summary": summary,
    }
