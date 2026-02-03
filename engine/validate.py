"""
Validation Module for Model Autopsy

Handles input validation, schema mapping, and task type detection.
"""
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise ValidationError(f"Failed to load CSV file {file_path}: {str(e)}")


def validate_required_columns(
    df: pd.DataFrame, 
    required_columns: List[str], 
    file_name: str = "file"
) -> None:
    """Validate that required columns exist in the DataFrame"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(
            f"Missing required columns in {file_name}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_row_alignment(
    dfs: Dict[str, pd.DataFrame],
    reference_key: str = "dataset"
) -> None:
    """Validate that all DataFrames have the same number of rows"""
    if reference_key not in dfs:
        raise ValidationError(f"Reference DataFrame '{reference_key}' not found")
    
    reference_count = len(dfs[reference_key])
    mismatched = []
    
    for name, df in dfs.items():
        if len(df) != reference_count:
            mismatched.append(f"{name}: {len(df)} rows")
    
    if mismatched:
        raise ValidationError(
            f"Row count mismatch. Reference ({reference_key}): {reference_count} rows. "
            f"Mismatched: {', '.join(mismatched)}"
        )


def detect_task_type(y_true: pd.Series) -> str:
    """
    Detect if the task is binary or multiclass classification.
    
    Args:
        y_true: Series of true labels
        
    Returns:
        'binary' or 'multiclass'
    """
    unique_values = y_true.dropna().unique()
    
    if len(unique_values) <= 2:
        return "binary"
    else:
        return "multiclass"


def validate_numeric_column(
    df: pd.DataFrame, 
    column: str, 
    file_name: str = "file"
) -> None:
    """Validate that a column contains numeric values"""
    if column not in df.columns:
        raise ValidationError(f"Column '{column}' not found in {file_name}")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        # Try to convert
        try:
            pd.to_numeric(df[column], errors='raise')
        except (ValueError, TypeError):
            raise ValidationError(
                f"Column '{column}' in {file_name} must be numeric. "
                f"Found non-numeric values."
            )


def apply_threshold(
    y_score: pd.Series, 
    threshold: float = 0.5
) -> pd.Series:
    """
    Apply threshold to probability scores to get hard predictions.
    
    Args:
        y_score: Series of probability scores
        threshold: Classification threshold (default 0.5)
        
    Returns:
        Series of predicted labels (0 or 1)
    """
    return (y_score >= threshold).astype(int)


def load_and_validate_inputs(
    dataset_path: Path,
    predictions_path: Path,
    labels_path: Optional[Path] = None,
    id_column: Optional[str] = None,
    y_true_column: str = "y_true",
    y_pred_column: Optional[str] = "y_pred",
    y_score_column: Optional[str] = "y_score",
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load and validate all input files.
    
    Args:
        dataset_path: Path to dataset CSV
        predictions_path: Path to predictions CSV
        labels_path: Optional path to labels CSV (if not in dataset)
        id_column: Optional ID column name
        y_true_column: Name of the true label column
        y_pred_column: Name of the predicted label column
        y_score_column: Name of the probability score column
        threshold: Threshold for converting scores to predictions
        
    Returns:
        Tuple of (dataset_df, predictions_df, labels_df, config)
    """
    # Load dataset
    dataset_df = load_csv(dataset_path)
    
    # Load predictions
    predictions_df = load_csv(predictions_path)
    
    # Check for y_pred or y_score in predictions
    has_y_pred = y_pred_column and y_pred_column in predictions_df.columns
    has_y_score = y_score_column and y_score_column in predictions_df.columns
    
    if not has_y_pred and not has_y_score:
        raise ValidationError(
            f"Predictions file must contain either '{y_pred_column}' or '{y_score_column}' column. "
            f"Found columns: {list(predictions_df.columns)}"
        )
    
    # Validate numeric score if present
    if has_y_score:
        validate_numeric_column(predictions_df, y_score_column, "predictions")
    
    # Load labels
    if labels_path and labels_path.exists():
        labels_df = load_csv(labels_path)
        validate_required_columns(labels_df, [y_true_column], "labels")
    elif y_true_column in dataset_df.columns:
        labels_df = dataset_df[[y_true_column]].copy()
        if id_column and id_column in dataset_df.columns:
            labels_df[id_column] = dataset_df[id_column]
    else:
        raise ValidationError(
            f"Labels not found. Either provide a labels file or include "
            f"'{y_true_column}' column in the dataset."
        )
    
    # Validate row alignment
    dfs = {"dataset": dataset_df, "predictions": predictions_df, "labels": labels_df}
    validate_row_alignment(dfs)
    
    # Detect task type
    task_type = detect_task_type(labels_df[y_true_column])
    
    # Build config
    config = {
        "id_column": id_column,
        "y_true_column": y_true_column,
        "y_pred_column": y_pred_column if has_y_pred else None,
        "y_score_column": y_score_column if has_y_score else None,
        "threshold": threshold,
        "task_type": task_type,
        "n_rows": len(dataset_df),
        "n_features": len([c for c in dataset_df.columns if c not in [id_column, y_true_column]]),
    }
    
    return dataset_df, predictions_df, labels_df, config


def map_columns(
    dataset_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Map columns from separate DataFrames into a unified canonical format.
    
    Args:
        dataset_df: Features DataFrame
        predictions_df: Predictions DataFrame
        labels_df: Labels DataFrame
        config: Configuration dict from load_and_validate_inputs
        
    Returns:
        Unified DataFrame with all columns mapped
    """
    # Start with dataset
    result = dataset_df.copy()
    
    # Add y_true
    y_true_col = config["y_true_column"]
    if y_true_col not in result.columns:
        result["y_true"] = labels_df[y_true_col].values
    else:
        result = result.rename(columns={y_true_col: "y_true"})
    
    # Add y_pred
    if config["y_pred_column"] and config["y_pred_column"] in predictions_df.columns:
        result["y_pred"] = predictions_df[config["y_pred_column"]].values
    
    # Add y_score
    if config["y_score_column"] and config["y_score_column"] in predictions_df.columns:
        result["y_score"] = predictions_df[config["y_score_column"]].values
        
        # If no y_pred, derive from y_score
        if "y_pred" not in result.columns:
            result["y_pred"] = apply_threshold(result["y_score"], config["threshold"])
    
    return result
