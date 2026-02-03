"""
Ingestion Module for Model Autopsy

Handles Streamlit file uploads and builds canonical datasets for analysis.
Wraps the validation and canonical modules to work with UploadedFile objects.

This module includes smart column detection with heuristics to handle
various naming conventions and automatically distinguish between ID,
label, and prediction columns.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
from io import BytesIO
import pandas as pd
import numpy as np

from engine.validate import (
    ValidationError,
    validate_required_columns,
    validate_row_alignment,
    detect_task_type,
    validate_numeric_column,
    apply_threshold,
    map_columns,
)
from engine.canonical import compute_error_columns


def read_uploaded_csv(file_obj: Union[BytesIO, Any], filename: str = "file") -> pd.DataFrame:
    """
    Read a CSV from a Streamlit UploadedFile or BytesIO object.
    
    Args:
        file_obj: Streamlit UploadedFile or BytesIO-like object
        filename: Name for error messages
        
    Returns:
        DataFrame from the CSV
        
    Raises:
        ValidationError: If file cannot be read as CSV
    """
    try:
        # Reset file pointer in case it was read before
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        return pd.read_csv(file_obj)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV from {filename}: {str(e)}")


def is_likely_id_column(series: pd.Series, col_name: str) -> bool:
    """
    Heuristically determine if a column is likely an ID column.
    
    Heuristics:
    - Column name contains 'id', 'index', 'key', 'uuid', 'guid'
    - All values are unique (or nearly unique)
    - Values are sequential integers or strings with common ID patterns
    
    Args:
        series: The column data
        col_name: The column name
        
    Returns:
        True if likely an ID column
    """
    col_lower = col_name.lower()
    
    # Name-based heuristics (strong signal)
    id_patterns = ['_id', 'id_', 'imageid', 'sampleid', 'rowid', 'index', 
                   'uuid', 'guid', 'key', 'idx']
    if any(pattern in col_lower for pattern in id_patterns):
        return True
    
    # Exact match for common ID names
    if col_lower in ['id', 'index', 'key', 'uuid', 'guid', 'idx']:
        return True
    
    # If column ends with 'id' and is not just 'id' in a word
    if col_lower.endswith('id') and len(col_lower) > 2:
        # Check it's not a word like "valid" or "rapid"
        non_id_suffixes = ['valid', 'rapid', 'acid', 'paid', 'said', 'maid', 'grid']
        if not any(col_lower.endswith(suffix) for suffix in non_id_suffixes):
            return True
    
    # Value-based heuristics
    try:
        n = len(series)
        unique_ratio = series.nunique() / n if n > 0 else 0
        
        # If nearly all values are unique, likely an ID
        if unique_ratio > 0.95 and n > 10:
            return True
            
        # Check for sequential integers
        if pd.api.types.is_integer_dtype(series):
            sorted_vals = series.sort_values().values
            if len(sorted_vals) > 1:
                diffs = np.diff(sorted_vals)
                if np.all(diffs == 1):  # Sequential
                    return True
    except Exception:
        pass
    
    return False


def is_likely_label_column(series: pd.Series, col_name: str) -> bool:
    """
    Heuristically determine if a column is likely a label/target column.
    
    Heuristics:
    - Column name matches common label patterns
    - Small number of unique values (classification)
    - Values are 0/1, True/False, or categorical strings
    
    Args:
        series: The column data
        col_name: The column name
        
    Returns:
        True if likely a label column
    """
    col_lower = col_name.lower()
    
    # Name-based heuristics (strong signal)
    label_patterns = ['label', 'target', 'class', 'category', 'y_true', 
                      'y_pred', 'prediction', 'predicted', 'outcome', 
                      'result', 'diagnosis', 'sentiment', 'spam', 'fraud',
                      'churn', 'default', 'survived', 'income', 'species']
    
    if any(pattern in col_lower for pattern in label_patterns):
        return True
    
    # Exact match for common names
    if col_lower in ['y', 'label', 'target', 'class', 'output']:
        return True
    
    # Value-based heuristics
    try:
        n_unique = series.nunique()
        n = len(series)
        
        # Small number of unique values suggests classification labels
        if 2 <= n_unique <= 20 and n > 20:
            # Check for binary-like patterns
            unique_vals = set(series.dropna().unique())
            
            # Binary patterns
            if unique_vals <= {0, 1}:
                return True
            if unique_vals <= {True, False}:
                return True
            if unique_vals <= {'yes', 'no', 'Yes', 'No', 'YES', 'NO'}:
                return True
            if unique_vals <= {'positive', 'negative', 'Positive', 'Negative'}:
                return True
                
            # If few unique values compared to total, likely categorical target
            if n_unique / n < 0.1:
                return True
    except Exception:
        pass
    
    return False


def is_likely_score_column(series: pd.Series, col_name: str) -> bool:
    """
    Heuristically determine if a column is likely a probability/score column.
    
    Heuristics:
    - Column name matches score patterns
    - Values are floats between 0 and 1
    - Many unique float values
    
    Args:
        series: The column data
        col_name: The column name
        
    Returns:
        True if likely a score column
    """
    col_lower = col_name.lower()
    
    # Name-based heuristics
    score_patterns = ['score', 'prob', 'confidence', 'likelihood', 'proba',
                      'probability', 'logit', 'softmax', 'sigmoid']
    
    if any(pattern in col_lower for pattern in score_patterns):
        return True
    
    # Value-based heuristics
    try:
        if not pd.api.types.is_numeric_dtype(series):
            return False
            
        # Check if values are in [0, 1] range with many unique floats
        vals = series.dropna()
        if len(vals) == 0:
            return False
            
        min_val, max_val = vals.min(), vals.max()
        
        # Probability-like: mostly in [0, 1] range
        if 0 <= min_val and max_val <= 1:
            n_unique = vals.nunique()
            # Many unique values suggest continuous probabilities
            if n_unique > 10 and n_unique / len(vals) > 0.5:
                return True
    except Exception:
        pass
    
    return False


def smart_detect_columns(
    df: pd.DataFrame,
    file_type: str = "predictions"
) -> Dict[str, Optional[str]]:
    """
    Smart detection of column roles using heuristics.
    
    Returns a dict with detected column names for:
    - 'id': ID column
    - 'label': Label/target column
    - 'prediction': Predicted class column
    - 'score': Probability/score column
    
    Args:
        df: DataFrame to analyze
        file_type: Type of file ('predictions', 'dataset', 'labels')
        
    Returns:
        Dict mapping role to column name (or None if not found)
    """
    result = {
        'id': None,
        'label': None,
        'prediction': None,
        'score': None,
    }
    
    for col in df.columns:
        series = df[col]
        col_lower = col.lower()
        
        # Check for ID column
        if result['id'] is None and is_likely_id_column(series, col):
            result['id'] = col
            continue
        
        # Check for score column (do this before label since scores are numeric)
        if result['score'] is None and is_likely_score_column(series, col):
            result['score'] = col
            continue
            
        # Check for label/prediction column
        if is_likely_label_column(series, col):
            # Try to distinguish between ground truth and prediction
            if 'true' in col_lower or 'actual' in col_lower or 'ground' in col_lower:
                if result['label'] is None:
                    result['label'] = col
            elif 'pred' in col_lower or 'predicted' in col_lower or 'output' in col_lower:
                if result['prediction'] is None:
                    result['prediction'] = col
            else:
                # Generic label column - could be either
                # In predictions file, prefer as prediction
                if file_type == 'predictions' and result['prediction'] is None:
                    result['prediction'] = col
                elif result['label'] is None:
                    result['label'] = col
    
    return result


def detect_column(
    df: pd.DataFrame, 
    candidates: List[str], 
    required: bool = False,
    filename: str = "file"
) -> Optional[str]:
    """
    Auto-detect a column from a list of candidate names.
    
    Args:
        df: DataFrame to search
        candidates: List of possible column names (case-insensitive)
        required: If True, raise error when not found
        filename: Name for error messages
        
    Returns:
        Found column name or None
        
    Raises:
        ValidationError: If required and not found
    """
    # Normalize column names for comparison
    col_map = {c.lower().strip(): c for c in df.columns}
    
    for candidate in candidates:
        normalized = candidate.lower().strip()
        if normalized in col_map:
            return col_map[normalized]
    
    if required:
        raise ValidationError(
            f"Could not find required column in {filename}. "
            f"Tried: {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def auto_detect_feature_columns(
    df: pd.DataFrame, 
    exclude_columns: List[str]
) -> List[str]:
    """
    Auto-detect feature columns by excluding known non-feature columns.
    
    Args:
        df: DataFrame with all columns
        exclude_columns: Columns to exclude (y_true, y_pred, y_score, id, etc.)
        
    Returns:
        List of feature column names
    """
    exclude_set = {c.lower() for c in exclude_columns if c}
    
    features = []
    for col in df.columns:
        col_lower = col.lower()
        
        # Skip if in explicit exclude list
        if col_lower in exclude_set:
            continue
        
        # Skip if it looks like an ID column
        if is_likely_id_column(df[col], col):
            continue
            
        features.append(col)
    
    return features


def match_columns_between_files(
    dataset_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[str]]:
    """
    Smart matching of columns between files to identify ID, y_true, and y_pred columns.
    
    Logic:
    1. Find common column names between files → these are likely ID/join columns
    2. Find columns that only appear in dataset/labels → these are likely y_true
    3. Find columns that only appear in predictions → these are likely y_pred
    
    Args:
        dataset_df: The dataset/features DataFrame
        predictions_df: The predictions DataFrame
        labels_df: Optional labels DataFrame
        
    Returns:
        Dict with keys: 'id_col', 'y_true_col', 'y_true_source', 'y_pred_col'
    """
    result = {
        'id_col': None,
        'y_true_col': None,
        'y_true_source': None,  # 'dataset', 'predictions', or 'labels'
        'y_pred_col': None,
    }
    
    dataset_cols = set(c.lower() for c in dataset_df.columns)
    pred_cols = set(c.lower() for c in predictions_df.columns)
    labels_cols = set(c.lower() for c in labels_df.columns) if labels_df is not None else set()
    
    # Get original column name mappings
    dataset_col_map = {c.lower(): c for c in dataset_df.columns}
    pred_col_map = {c.lower(): c for c in predictions_df.columns}
    labels_col_map = {c.lower(): c for c in labels_df.columns} if labels_df is not None else {}
    
    # Find common columns between dataset and predictions
    common_cols = dataset_cols & pred_cols
    
    # Common columns are likely IDs
    if common_cols:
        # Prefer columns that look like IDs
        for col_lower in common_cols:
            orig_col = dataset_col_map[col_lower]
            if is_likely_id_column(dataset_df[orig_col], orig_col):
                result['id_col'] = orig_col
                break
        
        # If no clear ID, just pick the first common column
        if result['id_col'] is None:
            first_common = list(common_cols)[0]
            result['id_col'] = dataset_col_map[first_common]
    
    # Find unique columns in predictions (not in dataset or labels)
    pred_only = pred_cols - dataset_cols - labels_cols
    
    # Patterns that indicate ground truth, not predictions
    truth_patterns = ['true', 'truth', 'actual', 'ground', 'gt_', 'target', 'answer']
    # Patterns that indicate predictions
    pred_patterns = ['pred', 'output', 'hat', 'class', 'label']
    
    if pred_only:
        # These are likely predictions - exclude ID columns, score columns, and truth columns
        for col_lower in pred_only:
            orig_col = pred_col_map[col_lower]
            if is_likely_id_column(predictions_df[orig_col], orig_col):
                continue
            if is_likely_score_column(predictions_df[orig_col], orig_col):
                continue  # Skip score columns - they're not predictions
            # Check if this looks like a ground truth column by name
            if any(pattern in col_lower for pattern in truth_patterns):
                # This is likely y_true, not y_pred - set it as y_true
                if result['y_true_col'] is None:
                    result['y_true_col'] = orig_col
                    result['y_true_source'] = 'predictions'
                continue
            # Found a non-ID, non-score, non-truth column
            result['y_pred_col'] = orig_col
            break

    
    # Find unique columns in labels (if provided)
    if labels_df is not None:
        labels_only = labels_cols - pred_cols
        if labels_only:
            for col_lower in labels_only:
                orig_col = labels_col_map[col_lower]
                if not is_likely_id_column(labels_df[orig_col], orig_col):
                    result['y_true_col'] = orig_col
                    result['y_true_source'] = 'labels'
                    break
    
    # If no labels file, check dataset for y_true
    if result['y_true_col'] is None:
        dataset_only = dataset_cols - pred_cols
        if dataset_only:
            for col_lower in dataset_only:
                orig_col = dataset_col_map[col_lower]
                if not is_likely_id_column(dataset_df[orig_col], orig_col):
                    if is_likely_label_column(dataset_df[orig_col], orig_col):
                        result['y_true_col'] = orig_col
                        result['y_true_source'] = 'dataset'
                        break
    
    return result


def ingest_uploaded_files(
    dataset_file: Any,
    predictions_file: Any,
    labels_file: Optional[Any] = None,
    threshold: float = 0.5,
    y_true_column: Optional[str] = None,
    y_pred_column: Optional[str] = None,
    y_score_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Ingest uploaded files and build a canonical dataset for analysis.
    
    This is the main entry point for processing user-uploaded CSV files.
    It handles validation, column detection, and canonical format building.
    Uses smart heuristics to detect column roles when standard names aren't used.
    
    Args:
        dataset_file: Streamlit UploadedFile for features/dataset
        predictions_file: Streamlit UploadedFile for predictions
        labels_file: Optional UploadedFile for labels (if not in dataset/predictions)
        threshold: Classification threshold for deriving y_pred from y_score
        y_true_column: Override for true label column name
        y_pred_column: Override for prediction column name
        y_score_column: Override for score column name
        
    Returns:
        Tuple of (canonical_df, config_dict, feature_columns_list)
        
    Raises:
        ValidationError: If validation fails after all fallbacks
    """
    # 1. Read all CSV files
    dataset_df = read_uploaded_csv(dataset_file, "dataset")
    predictions_df = read_uploaded_csv(predictions_file, "predictions")
    
    labels_df = None
    if labels_file is not None:
        labels_df = read_uploaded_csv(labels_file, "labels")
    
    # 2. Smart column detection with multiple fallbacks
    
    # First, try cross-file matching to understand column relationships
    cross_match = match_columns_between_files(dataset_df, predictions_df, labels_df)
    
    # Extended candidate lists
    y_true_candidates = [
        'y_true', 'label', 'target', 'class', 'y', 'ground_truth', 
        'actual', 'true_label', 'groundtruth', 'answer', 'category',
        'outcome', 'result', 'diagnosis', 'survived', 'species'
    ]
    y_pred_candidates = [
        'y_pred', 'prediction', 'predicted', 'pred', 'y_hat', 
        'output', 'predicted_label', 'pred_label', 'classification',
        'label'  # Fallback: 'Label' in predictions file is often the prediction
    ]
    y_score_candidates = [
        'y_score', 'score', 'probability', 'prob', 'confidence', 
        'proba', 'likelihood', 'logit', 'softmax_score'
    ]
    
    # Detect y_true (multiple fallbacks)
    if y_true_column is None:
        # Try standard candidate matching first
        if labels_df is not None:
            y_true_column = detect_column(labels_df, y_true_candidates)
        if y_true_column is None:
            y_true_column = detect_column(predictions_df, y_true_candidates)
        if y_true_column is None:
            y_true_column = detect_column(dataset_df, y_true_candidates)
        
        # Fallback 1: cross-file matching
        if y_true_column is None and cross_match.get('y_true_col'):
            y_true_column = cross_match['y_true_col']
        
        # Fallback 2: smart detection by content
        if y_true_column is None:
            smart = smart_detect_columns(dataset_df, 'dataset')
            y_true_column = smart.get('label')
    
    # Detect y_pred (multiple fallbacks)
    if y_pred_column is None:
        y_pred_column = detect_column(predictions_df, y_pred_candidates)
        
        # Fallback 1: cross-file matching
        if y_pred_column is None and cross_match.get('y_pred_col'):
            y_pred_column = cross_match['y_pred_col']
        
        # Fallback 2: smart detection in predictions file
        if y_pred_column is None:
            smart = smart_detect_columns(predictions_df, 'predictions')
            y_pred_column = smart.get('prediction') or smart.get('label')
    
    # Detect y_score
    if y_score_column is None:
        y_score_column = detect_column(predictions_df, y_score_candidates)
        
        # Fallback: smart detection
        if y_score_column is None:
            smart = smart_detect_columns(predictions_df, 'predictions')
            y_score_column = smart.get('score')
    
    # 3. Handle the case where same column name appears for both y_true and y_pred
    if y_true_column == y_pred_column and y_true_column is not None:
        # The 'Label' in predictions is the prediction, look for ground truth elsewhere
        if labels_df is not None:
            # Check labels file for ground truth
            for col in labels_df.columns:
                if is_likely_label_column(labels_df[col], col) and not is_likely_id_column(labels_df[col], col):
                    y_true_column = col
                    break
        else:
            # Check dataset file for ground truth
            for col in dataset_df.columns:
                if is_likely_label_column(dataset_df[col], col) and not is_likely_id_column(dataset_df[col], col):
                    y_true_column = col
                    break

    
    # 4. Final validation
    if y_true_column is None:
        # List all non-ID columns as potential options
        non_id_cols = [c for c in list(dataset_df.columns) + list(predictions_df.columns) 
                       if not is_likely_id_column(pd.Series([]), c)]
        raise ValidationError(
            f"Could not find ground truth labels column. "
            f"Please ensure your dataset contains a column with true labels. "
            f"Tried patterns: {y_true_candidates[:5]}... "
            f"Potential columns: {non_id_cols[:10]}"
        )
    
    if y_pred_column is None and y_score_column is None:
        # List non-ID columns from predictions as suggestions
        non_id_cols = [c for c in predictions_df.columns 
                       if not is_likely_id_column(predictions_df[c], c)]
        raise ValidationError(
            f"Could not find predictions in your predictions file. "
            f"Expected a column with predicted labels or probability scores. "
            f"Found columns: {list(predictions_df.columns)}. "
            f"Non-ID columns: {non_id_cols}"
        )
    
    # 5. Validate row alignment
    dfs_to_check = {"dataset": dataset_df, "predictions": predictions_df}
    if labels_df is not None:
        dfs_to_check["labels"] = labels_df
    validate_row_alignment(dfs_to_check, reference_key="dataset")
    
    # 6. Validate numeric columns
    if y_score_column and y_score_column in predictions_df.columns:
        validate_numeric_column(predictions_df, y_score_column, "predictions")
    
    # 7. Build unified DataFrame
    canonical = dataset_df.copy()
    
    # Add y_true
    if labels_df is not None and y_true_column in labels_df.columns:
        canonical["y_true"] = labels_df[y_true_column].values
    elif y_true_column in predictions_df.columns:
        canonical["y_true"] = predictions_df[y_true_column].values
    elif y_true_column in dataset_df.columns:
        if y_true_column != "y_true":
            canonical["y_true"] = dataset_df[y_true_column].values
    else:
        raise ValidationError(f"Column '{y_true_column}' not found in any input file.")
    
    # Add y_score if available
    if y_score_column and y_score_column in predictions_df.columns:
        canonical["y_score"] = predictions_df[y_score_column].values
    
    # Add y_pred (or derive from y_score)
    if y_pred_column and y_pred_column in predictions_df.columns:
        canonical["y_pred"] = predictions_df[y_pred_column].values
    elif y_score_column and y_score_column in predictions_df.columns:
        canonical["y_pred"] = apply_threshold(canonical["y_score"], threshold)
    else:
        raise ValidationError("Cannot determine predictions. Need y_pred or y_score column.")
    
    # 8. Detect task type and compute error columns
    task_type = detect_task_type(canonical["y_true"])
    canonical = compute_error_columns(canonical, task_type)
    
    # 9. Auto-detect feature columns
    exclude_cols = ["y_true", "y_pred", "y_score", "is_error", "error_type"]
    if y_true_column and y_true_column not in exclude_cols:
        exclude_cols.append(y_true_column)
    if y_pred_column and y_pred_column not in exclude_cols:
        exclude_cols.append(y_pred_column)
    
    feature_cols = auto_detect_feature_columns(canonical, exclude_cols)
    
    # 10. Build config with detection info
    config = {
        "y_true_column": y_true_column,
        "y_pred_column": y_pred_column,
        "y_score_column": y_score_column,
        "threshold": threshold,
        "task_type": task_type,
        "n_rows": len(canonical),
        "n_features": len(feature_cols),
        "has_labels_file": labels_df is not None,
        "detected_columns": {
            "y_true": y_true_column,
            "y_pred": y_pred_column,
            "y_score": y_score_column,
        }
    }
    
    return canonical, config, feature_cols
