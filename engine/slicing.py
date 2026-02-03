"""
Slicing Module for Model Autopsy

Identifies systematic failure patterns across data subgroups (slices).
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np


def bin_numeric_feature(
    series: pd.Series,
    n_bins: int = 10,
    labels: Optional[List[str]] = None,
) -> Tuple[pd.Series, List[str]]:
    """
    Bin a numeric feature into quantile-based bins.
    
    Args:
        series: Numeric feature series
        n_bins: Number of quantile bins
        labels: Optional custom labels for bins
        
    Returns:
        Tuple of (binned series, bin labels)
    """
    # Handle edge cases
    if series.nunique() <= n_bins:
        # Fewer unique values than bins, use unique values as bins
        return series.astype(str), sorted(series.dropna().unique().astype(str).tolist())
    
    try:
        # Create quantile bins
        binned, bin_edges = pd.qcut(series, q=n_bins, retbins=True, duplicates='drop')
        
        # Create readable labels
        if labels is None:
            labels = []
            for i in range(len(bin_edges) - 1):
                lower = bin_edges[i]
                upper = bin_edges[i + 1]
                labels.append(f"{lower:.2f}-{upper:.2f}")
        
        # Map intervals to labels
        binned_labels = binned.cat.rename_categories(
            {cat: labels[i] for i, cat in enumerate(binned.cat.categories)}
        )
        return binned_labels.astype(str), labels
        
    except ValueError:
        # Fall back to equal-width bins if quantiles fail
        binned = pd.cut(series, bins=n_bins, duplicates='drop')
        labels = binned.cat.categories.astype(str).tolist()
        return binned.astype(str), labels


def bin_categorical_feature(
    series: pd.Series,
    top_k: int = 10,
    other_label: str = "OTHER",
) -> Tuple[pd.Series, List[str]]:
    """
    Bin a categorical feature, grouping rare categories into "OTHER".
    
    Args:
        series: Categorical feature series
        top_k: Number of top categories to keep
        other_label: Label for grouped rare categories
        
    Returns:
        Tuple of (binned series, category labels)
    """
    value_counts = series.value_counts()
    
    if len(value_counts) <= top_k:
        # All categories fit, no need for OTHER
        labels = value_counts.index.astype(str).tolist()
        return series.astype(str), labels
    
    # Get top k categories
    top_categories = set(value_counts.head(top_k).index)
    
    # Map to category or OTHER
    binned = series.apply(lambda x: str(x) if x in top_categories else other_label)
    
    # Labels include top categories + OTHER
    labels = [str(c) for c in value_counts.head(top_k).index] + [other_label]
    
    return binned, labels


def compute_slice_metrics(
    df: pd.DataFrame,
    slice_mask: pd.Series,
    is_error_col: str = "is_error",
    error_type_col: str = "error_type",
) -> Dict[str, Any]:
    """
    Compute metrics for a single slice.
    
    Args:
        df: Canonical DataFrame
        slice_mask: Boolean mask for rows in this slice
        is_error_col: Column name for error indicator
        error_type_col: Column name for error type
        
    Returns:
        Dictionary with slice metrics
    """
    total_samples = len(df)
    slice_df = df[slice_mask]
    slice_size = len(slice_df)
    
    if slice_size == 0:
        return {
            "support": 0,
            "support_pct": 0.0,
            "error_count": 0,
            "error_rate": 0.0,
            "lift": 0.0,
            "fp_count": 0,
            "fn_count": 0,
            "tp_count": 0,
            "tn_count": 0,
        }
    
    # Basic metrics
    support = slice_size
    support_pct = slice_size / total_samples
    
    # Error metrics
    error_count = slice_df[is_error_col].sum()
    error_rate = error_count / slice_size
    
    # Global error rate for lift calculation
    global_error_rate = df[is_error_col].sum() / total_samples
    lift = error_rate / global_error_rate if global_error_rate > 0 else 0.0
    
    # Error type breakdown
    error_types = slice_df[error_type_col].value_counts().to_dict()
    
    return {
        "support": int(support),
        "support_pct": float(support_pct),
        "error_count": int(error_count),
        "error_rate": float(error_rate),
        "lift": float(lift),
        "fp_count": int(error_types.get("FP", 0)),
        "fn_count": int(error_types.get("FN", 0)),
        "tp_count": int(error_types.get("TP", 0)),
        "tn_count": int(error_types.get("TN", 0)),
    }


def generate_feature_slices(
    df: pd.DataFrame,
    feature_col: str,
    is_numeric: bool,
    n_bins: int = 10,
    top_k_categories: int = 10,
    is_error_col: str = "is_error",
    error_type_col: str = "error_type",
) -> List[Dict[str, Any]]:
    """
    Generate slices for a single feature.
    
    Args:
        df: Canonical DataFrame
        feature_col: Feature column name
        is_numeric: Whether feature is numeric
        n_bins: Number of bins for numeric features
        top_k_categories: Number of top categories to keep
        is_error_col: Column name for error indicator
        error_type_col: Column name for error type
        
    Returns:
        List of slice dictionaries with metrics
    """
    slices = []
    
    # Bin the feature
    if is_numeric:
        binned, labels = bin_numeric_feature(df[feature_col], n_bins=n_bins)
    else:
        binned, labels = bin_categorical_feature(df[feature_col], top_k=top_k_categories)
    
    # Compute metrics for each bin
    for label in labels:
        slice_mask = binned == label
        metrics = compute_slice_metrics(df, slice_mask, is_error_col, error_type_col)
        
        if metrics["support"] > 0:
            slices.append({
                "feature": feature_col,
                "value": label,
                "rule": f"{feature_col} = {label}",
                **metrics,
            })
    
    return slices


def generate_all_slices(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    n_bins: int = 10,
    top_k_categories: int = 10,
    is_error_col: str = "is_error",
    error_type_col: str = "error_type",
) -> List[Dict[str, Any]]:
    """
    Generate slices for all features in the dataset.
    
    Args:
        df: Canonical DataFrame
        feature_cols: Optional list of feature columns (auto-detect if None)
        exclude_cols: Columns to exclude from slicing
        n_bins: Number of bins for numeric features
        top_k_categories: Number of top categories to keep
        is_error_col: Column name for error indicator
        error_type_col: Column name for error type
        
    Returns:
        List of all slice dictionaries
    """
    # Default exclusions
    default_exclude = {
        "y_true", "y_pred", "y_score", is_error_col, error_type_col,
        "id", "index", "_id"
    }
    exclude_set = default_exclude | set(exclude_cols or [])
    
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude_set]
    
    all_slices = []
    
    for col in feature_cols:
        if col in exclude_set or col not in df.columns:
            continue
            
        # Determine if numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        # Generate slices for this feature
        slices = generate_feature_slices(
            df, col, is_numeric,
            n_bins=n_bins,
            top_k_categories=top_k_categories,
            is_error_col=is_error_col,
            error_type_col=error_type_col,
        )
        all_slices.extend(slices)
    
    return all_slices


def rank_slices(
    slices: List[Dict[str, Any]],
    min_support: int = 10,
    min_support_pct: float = 0.01,
    sort_by: str = "lift",
    ascending: bool = False,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Filter and rank slices by metrics.
    
    Args:
        slices: List of slice dictionaries
        min_support: Minimum absolute support
        min_support_pct: Minimum support percentage
        sort_by: Field to sort by ('lift', 'error_rate', 'support')
        ascending: Sort order
        top_n: Limit to top N slices
        
    Returns:
        Filtered and ranked list of slices
    """
    # Filter by support
    filtered = [
        s for s in slices
        if s["support"] >= min_support and s["support_pct"] >= min_support_pct
    ]
    
    # Sort
    sorted_slices = sorted(filtered, key=lambda x: x.get(sort_by, 0), reverse=not ascending)
    
    # Limit
    if top_n is not None:
        sorted_slices = sorted_slices[:top_n]
    
    # Add rank
    for i, s in enumerate(sorted_slices):
        s["rank"] = i + 1
    
    return sorted_slices


def get_slice_subset(
    df: pd.DataFrame,
    feature: str,
    value: str,
    n_bins: int = 10,
    top_k_categories: int = 10,
) -> pd.DataFrame:
    """
    Get the subset of rows matching a slice rule (drilldown).
    
    Args:
        df: Canonical DataFrame
        feature: Feature column name
        value: Slice value to match
        n_bins: Number of bins (for consistent binning)
        top_k_categories: Top categories (for consistent binning)
        
    Returns:
        DataFrame subset matching the slice
    """
    is_numeric = pd.api.types.is_numeric_dtype(df[feature])
    
    if is_numeric:
        binned, _ = bin_numeric_feature(df[feature], n_bins=n_bins)
    else:
        binned, _ = bin_categorical_feature(df[feature], top_k=top_k_categories)
    
    mask = binned == value
    return df[mask]


def save_slices(
    slices: List[Dict[str, Any]],
    output_dir: Path,
    filename: str = "slices_single.csv",
) -> Path:
    """
    Save slices to a CSV file.
    
    Args:
        slices: List of slice dictionaries
        output_dir: Directory to save to
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    if slices:
        df = pd.DataFrame(slices)
        # Reorder columns
        col_order = [
            "rank", "feature", "value", "rule", "support", "support_pct",
            "error_count", "error_rate", "lift", "fp_count", "fn_count"
        ]
        cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
        df = df[cols]
        df.to_csv(output_path, index=False)
    else:
        # Create empty file with headers
        pd.DataFrame(columns=[
            "rank", "feature", "value", "rule", "support", "support_pct",
            "error_count", "error_rate", "lift", "fp_count", "fn_count"
        ]).to_csv(output_path, index=False)
    
    return output_path
